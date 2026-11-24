import os
import time
import uuid

import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "semantic_cache"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.80

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"

# Initialize clients
qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Set this to the model identifier exposed by your vLLM server if different.
MODEL_ID = os.getenv("VLLM_MODEL_ID", MODEL_NAME)


def generate_embedding(text):
    return embedder.encode(
        text,
        normalize_embeddings=True,
    ).tolist()


def call_vllm(query):
    """Generate a response using the existing vLLM server."""
    start = time.perf_counter()

    response = requests.post(
        VLLM_URL,
        json={
            "model": MODEL_ID,
            "messages": [
                {"role": "user", "content": query}
            ],
            "temperature": 0,
            "max_tokens": 128,
        },
        timeout=180,
    )

    response.raise_for_status()
    result = response.json()

    latency_ms = (time.perf_counter() - start) * 1000
    answer = result["choices"][0]["message"]["content"]

    return answer, latency_ms


def search_cache(query):
    """Return the best eligible semantic-cache match, if any."""
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
                    match=models.MatchValue(value=0),
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


def store_response(query, answer):
    """Store the query embedding and its generated response."""
    vector = generate_embedding(query)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "query": query,
                    "response": answer,
                    "model": MODEL_ID,
                    "temperature": 0,
                    "created_at": time.time(),
                },
            )
        ],
    )


def run_query(query):
    start = time.perf_counter()

    match, similarity = search_cache(query)

    if match is not None:
        total_ms = (time.perf_counter() - start) * 1000

        print("\n[SEMANTIC CACHE HIT]")
        print(f"Matched query: {match.payload['query']}")
        print(f"Cosine similarity: {similarity:.4f}")
        print(f"Lookup + embedding latency: {total_ms:.2f} ms")

        return match.payload["response"]

    print("\n[SEMANTIC CACHE MISS]")
    if similarity is not None:
        print(f"Best similarity: {similarity:.4f}")

    answer, inference_ms = call_vllm(query)
    store_response(query, answer)

    total_ms = (time.perf_counter() - start) * 1000

    print(f"Inference latency: {inference_ms:.2f} ms")
    print(f"Total miss-path latency: {total_ms:.2f} ms")
    print("Stored the new response in Qdrant.")

    return answer


if __name__ == "__main__":
    while True:
        try:
            query = input("\nEnter a query (or 'exit'): ").strip()

            if query.lower() == "exit":
                break

            if not query:
                continue

            answer = run_query(query)
            print(f"\nResponse:\n{answer}")

        except requests.RequestException as exc:
            print(f"vLLM request failed: {exc}")
        except Exception as exc:
            print(f"Error: {exc}")

