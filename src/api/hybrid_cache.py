import hashlib
import json
import os
import time
import uuid
import sqlite3

import requests

from contextlib import closing
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from src.adaptive.adaptive_controller import AdaptiveCacheController
from src.adaptive.cache_policy import classify_request, choose_cache_policy

# Configuration

CACHE_BENCH_MODE = os.getenv(
    "CACHE_BENCH_MODE",
    "adaptive",
).lower()

VALID_CACHE_MODES = {
    "exact_only",
    "semantic_always",
    "adaptive",
}

if CACHE_BENCH_MODE not in VALID_CACHE_MODES:
    raise ValueError(
        f"Invalid CACHE_BENCH_MODE: {CACHE_BENCH_MODE}"
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SQLITE_DB = os.path.join(BASE_DIR, "exact_cache.db")

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "semantic_cache"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VLLM_URL = "http://localhost:8000/v1/chat/completions"

MODEL_ID = os.getenv(
    "VLLM_MODEL_ID",
    "Qwen/Qwen3-8B-AWQ",
)

SIMILARITY_THRESHOLD = 0.80

# Keep these settings fixed for the initial semantic-cache experiment.
SEMANTIC_TEMPERATURE = 0
SEMANTIC_MAX_TOKENS = 128

app = FastAPI(title="Hybrid LLM Cache Router")

qdrant = QdrantClient(url=QDRANT_URL)

embedder = SentenceTransformer(EMBEDDING_MODEL)

adaptive_controller = AdaptiveCacheController()

# L1: Exact-response cache (SQLite)

def canonicalize_request(payload):
    """
    Produce a deterministic representation of the complete request.

    The representation must match the convention used by the
    existing exact-cache implementation.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def exact_cache_key(payload):
    canonical_request = canonicalize_request(payload)

    return hashlib.sha256(
        canonical_request.encode("utf-8")
    ).hexdigest()


def initialize_sqlite():
    """
    Ensure the expected exact-cache table exists.

    This does not delete or clear existing entries.
    """
    with closing(sqlite3.connect(SQLITE_DB, timeout=10)) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    request TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )


def exact_cache_lookup(payload):
    key = exact_cache_key(payload)

    with closing(sqlite3.connect(SQLITE_DB, timeout=10)) as conn:
        with conn:
            row = conn.execute(
                "SELECT response FROM cache WHERE key = ?",
                (key,),
            ).fetchone()

    if row is None:
        return None

    return json.loads(row[0])


def exact_cache_store(payload, response):
    key = exact_cache_key(payload)

    canonical_request = canonicalize_request(payload)

    serialized_response = json.dumps(
        response,
        separators=(",", ":"),
    )

    with closing(sqlite3.connect(SQLITE_DB, timeout=10)) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO cache (key, request, response, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (
                    key,
                    canonical_request,
                    serialized_response,
                    time.time(),
                ),
            )


# L2: Semantic-response cache (Qdrant)

def generate_embedding(text):
    return embedder.encode(
        text,
        normalize_embeddings=True,
    ).tolist()


def get_semantic_query(payload):
    """
    Return the user query only when semantic reuse is safe
    under the initial experiment's restrictions.

    We only support a single user message, temperature=0,
    and max_tokens=128.
    """
    messages = payload.get("messages")

    if not isinstance(messages, list) or len(messages) != 1:
        return None

    message = messages[0]

    if not isinstance(message, dict):
        return None

    if message.get("role") != "user":
        return None

    query = message.get("content")

    # This initial implementation supports plain-text content.
    if not isinstance(query, str) or not query.strip():
        return None

    if payload.get("temperature", 0) != SEMANTIC_TEMPERATURE:
        return None

    if payload.get("max_tokens", SEMANTIC_MAX_TOKENS) != SEMANTIC_MAX_TOKENS:
        return None

    if payload.get("model", MODEL_ID) != MODEL_ID:
        return None

    return query.strip()


def semantic_cache_lookup(query):
    vector = generate_embedding(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="model",
                    match=models.MatchValue(value=MODEL_ID),
                ),
                models.FieldCondition(
                    key="temperature",
                    match=models.MatchValue(value=SEMANTIC_TEMPERATURE),
                ),
            ]
        ),
        limit=1,
    ).points

    if not results:
        return None, None

    match = results[0]

    if match.score >= SIMILARITY_THRESHOLD:
        return match, match.score

    return None, match.score


def semantic_cache_store(query, answer):
    vector = generate_embedding(query)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "query": query,
                    "response": answer,
                    "model": MODEL_ID,
                    "temperature": SEMANTIC_TEMPERATURE,
                    "created_at": time.time(),
                },
            )
        ],
    )


def make_semantic_response(answer):
    """
    Reconstruct an OpenAI-compatible chat-completion response.

    A semantic hit returns the cached answer, not a new
    LLM-generated completion.
    """
    return {
        "id": f"cached-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
    }


# L3: vLLM inference

def call_vllm(payload):
    response = requests.post(
        VLLM_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


# Hybrid routing

def log_result(layer, elapsed_ms, extra=""):
    print(
        f"[{layer}] "
        f"latency={elapsed_ms:.2f} ms "
        f"{extra}",
        flush=True,
    )


@app.on_event("startup")
def startup():
    initialize_sqlite()

    # Fail early if the expected Qdrant collection is absent.
    collections = qdrant.get_collections().collections

    if not any(
        collection.name == COLLECTION_NAME
        for collection in collections
    ):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
            "Create it before starting the router."
        )

    print("Hybrid cache router initialized.", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "exact_cache": "sqlite",
        "semantic_cache": "qdrant",
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": MODEL_ID,
    }

@app.get("/adaptive/stats")
def adaptive_stats():
    return adaptive_controller.get_stats()

@app.post("/v1/chat/completions")
def hybrid_chat_completions(payload: dict):
    start = time.perf_counter()
    request_type = classify_request(payload)

    # Conservatively bypass persistent response caches for
    # requests that are personalized, time-sensitive, high-risk,
    # conversational, or unsupported by our classifier.
    cache_allowed = request_type not in {
        "personal",
        "time_sensitive",
        "high_risk",
        "conversation",
        "invalid",
        "non_text",
    }
    # L1: Exact cache
    cached_response = None
    if cache_allowed:
        try:
            cached_response = exact_cache_lookup(payload)
        except sqlite3.Error as exc:
            raise HTTPException(
                status_code=500,
                detail=f"SQLite lookup failed: {exc}",
            )

    if cached_response is not None:
        elapsed_ms = (time.perf_counter() - start) * 1000

        log_result("EXACT HIT", elapsed_ms)

        return JSONResponse(
            content=cached_response,
            headers={
                "X-Cache-Layer": "exact",
                "X-Cache-Latency-Ms": f"{elapsed_ms:.3f}",
            },
        )

  
    # L2: Semantic cache
   
    query = get_semantic_query(payload)
    
    if request_type != "general":
        query=None
        
    if query is not None and CACHE_BENCH_MODE != "exact_only":
        try_semantic = (
            CACHE_BENCH_MODE == "semantic_always"
            or adaptive_controller.should_try_semantic()
        )

        if try_semantic:
            semantic_start = time.perf_counter()

            try:
                match, similarity = semantic_cache_lookup(query)

            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Semantic-cache lookup failed: {exc}",
                )

            semantic_latency_ms = (
                time.perf_counter() - semantic_start
            ) * 1000

            # Record both hits and misses.
            adaptive_controller.record(
                was_hit=(match is not None),
                lookup_latency_ms=semantic_latency_ms,
            )

            if match is not None:
                answer = match.payload["response"]
                cached_response = make_semantic_response(answer)

                # Promote semantic result to the exact cache.
                try:
                    exact_cache_store(payload, cached_response)
                except sqlite3.Error as exc:
                    print(
                        f"[WARNING] Exact-cache promotion failed: {exc}",
                        flush=True,
                    )

                log_result(
                    "SEMANTIC HIT",
                    semantic_latency_ms,
                    extra=f"similarity={similarity:.4f}",
                )

                return JSONResponse(
                    content=cached_response,
                    headers={
                        "X-Cache-Layer": "semantic",
                        "X-Cache-Similarity": f"{similarity:.4f}",
                        "X-Cache-Latency-Ms":
                            f"{semantic_latency_ms:.3f}",
                    },
                )

            if similarity is not None:
                print(
                    f"[SEMANTIC MISS] "
                    f"best_similarity={similarity:.4f}",
                    flush=True,
                )

            print(
                "[ADAPTIVE ROUTER] Semantic lookup completed; "
                "continuing to inference.",
                flush=True,
            )

        else:
            print(
                "[ADAPTIVE ROUTER] Skipping semantic lookup "
                "because its expected latency benefit is too low.",
                flush=True,
            )

    # L3: LLM inference

    inference_start = time.perf_counter()

    try:
        response = call_vllm(payload)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"vLLM request failed: {exc}",
        )

    inference_ms = (time.perf_counter() - inference_start) * 1000

    answer = response["choices"][0]["message"]["content"]

    decision = choose_cache_policy(payload, answer)

    print(
        f"[CACHE POLICY] "
        f"request={decision.request_type}, "
        f"answer={decision.answer_type}, "
        f"exact={decision.store_exact}, "
        f"semantic={decision.store_semantic}, "
        f"reason={decision.reason}",
        flush=True,
    )

    # Store the complete response in SQLite only when eligible.
    if decision.store_exact:
        try:
            exact_cache_store(payload, response)
        except sqlite3.Error as exc:
            print(
                f"[WARNING] Exact-cache write failed: {exc}",
                flush=True,
            )

    # Store the generated answer in Qdrant only when eligible.
    if decision.store_semantic:
        try:
            semantic_cache_store(
                payload["messages"][0]["content"].strip(),
                answer,
            )
        except Exception as exc:
            print(
                f"[WARNING] Semantic-cache write failed: {exc}",
                flush=True,
            )

    total_ms = (time.perf_counter() - start) * 1000

    log_result(
        "CACHE MISS",
        total_ms,
        extra=f"inference={inference_ms:.2f} ms",
    )

    return JSONResponse(
        content=response,
        headers={
            "X-Cache-Layer": "miss",
            "X-Inference-Latency-Ms": f"{inference_ms:.3f}",
            "X-Total-Latency-Ms": f"{total_ms:.3f}",
        },
    )
