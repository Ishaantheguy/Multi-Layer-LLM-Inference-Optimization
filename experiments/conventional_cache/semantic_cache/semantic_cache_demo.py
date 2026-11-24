from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import uuid

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "semantic_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize clients
client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Create a sample cached response
cached_query = "What is machine learning?"
cached_response = (
    "Machine learning is a branch of artificial intelligence "
    "that enables computers to learn patterns from data."
)

# Generate embedding
embedding = embedder.encode(
    cached_query,
    normalize_embeddings=True,
).tolist()

# Insert into Qdrant
point_id = str(uuid.uuid4())

client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "query": cached_query,
                "response": cached_response,
                "model": "Qwen/Qwen3-8B-AWQ",
            },
        )
    ],
)

print("Inserted semantic-cache entry.")
print(f"Query: {cached_query}")
print(f"Point ID: {point_id}")
print(f"Embedding dimensions: {len(embedding)}")

