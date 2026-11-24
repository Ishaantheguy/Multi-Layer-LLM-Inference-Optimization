from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "semantic_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.80

client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBEDDING_MODEL)

query = "Can you explain what machine learning is?"

# Embed the incoming query
query_vector = embedder.encode(
    query,
    normalize_embeddings=True,
).tolist()

# Search Qdrant
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=1,
).points

if results:
    result = results[0]

    print(f"Incoming query: {query}")
    print(f"Matched query: {result.payload['query']}")
    print(f"Cosine similarity: {result.score:.4f}")

    if result.score >= SIMILARITY_THRESHOLD:
        print("\nSEMANTIC CACHE HIT")
        print(f"Cached response: {result.payload['response']}")
    else:
        print("\nSEMANTIC CACHE MISS")
        print("The request should be forwarded to vLLM.")
else:
    print("SEMANTIC CACHE MISS")
    print("No cached entries found.")
