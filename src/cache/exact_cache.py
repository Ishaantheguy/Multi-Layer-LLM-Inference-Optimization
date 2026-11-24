import hashlib
import json
import sqlite3
import time

import httpx
from fastapi import FastAPI, Request

app = FastAPI()

VLLM_URL = "http://127.0.0.1:8000"

DB_PATH = "exact_cache.db"


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            request TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------
# Cache key
# ---------------------------------------------------------

def make_cache_key(request_data):
    """
    Generate a deterministic key from the complete inference
    request.

    Things such as temperature, model, messages, max_tokens,
    etc. therefore become part of the cache identity.
    """

    canonical = json.dumps(
        request_data,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# Cache operations
# ---------------------------------------------------------

def cache_get(key):
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        "SELECT response FROM cache WHERE key = ?",
        (key,)
    ).fetchone()

    conn.close()

    if row is None:
        return None

    return json.loads(row[0])


def cache_put(key, request_data, response):
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT OR REPLACE INTO cache
        (key, request, response, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            key,
            json.dumps(request_data),
            json.dumps(response),
            time.time(),
        ),
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------
# OpenAI-compatible endpoint
# ---------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    start = time.perf_counter()

    request_data = await request.json()

    cache_key = make_cache_key(request_data)

    # -----------------------------------------------------
    # EXACT CACHE LOOKUP
    # -----------------------------------------------------

    cached_response = cache_get(cache_key)

    if cached_response is not None:

        latency = (
            time.perf_counter() - start
        ) * 1000

        print(
            f"[CACHE HIT] "
            f"latency={latency:.2f} ms "
            f"key={cache_key[:12]}"
        )

        return cached_response

    # -----------------------------------------------------
    # CACHE MISS → vLLM
    # -----------------------------------------------------

    async with httpx.AsyncClient() as client:

        response = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=request_data,
            timeout=300.0,
        )

        response.raise_for_status()

        response_data = response.json()

    # Store result

    cache_put(
        cache_key,
        request_data,
        response_data,
    )

    latency = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"[CACHE MISS] "
        f"latency={latency:.2f} ms "
        f"key={cache_key[:12]}"
    )

    return response_data


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "cache": "exact",
        "backend": VLLM_URL,
    }

@app.delete("/cache")
def clear_cache():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("DELETE FROM cache")

    conn.commit()
    conn.close()

    return {
        "status": "cleared"
        }
