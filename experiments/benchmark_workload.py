import csv
import os
import time
import uuid
import statistics
from collections import Counter, defaultdict

import requests


ROUTER_URL = "http://localhost:9100/v1/chat/completions"
MODEL = "Qwen/Qwen3-8B-AWQ"

OUTPUT_CSV = "workload_benchmark.csv"
RUN_ID = uuid.uuid4().hex[:8]

TIMEOUT = 180

HEADERS = {
    "Content-Type": "application/json",
}


# Each pair contains a canonical query and a paraphrase.
# These test whether semantic caching can reuse answers.
SEMANTIC_PAIRS = [
    ("What is a CPU cache?",
     "Can you explain what a CPU cache does?"),

    ("Explain machine learning.",
     "What does machine learning mean?"),

    ("What is a database index?",
     "How does an index in a database help?"),

    ("What is DNS?",
     "How does DNS translate domain names?"),

    ("Explain public-key cryptography.",
     "What is public key cryptography used for?"),

    ("How does an HTTP request work?",
     "Can you describe how an HTTP request works?"),

    ("What is a virtual machine?",
     "Explain what a virtual machine is."),

    ("What is a neural network?",
     "Can you explain neural networks?"),

    ("What is an API?",
     "What does API mean in software?"),

    ("What is a hash function?",
     "Explain the purpose of a hash function."),

    ("How does TCP differ from UDP?",
     "What is the difference between TCP and UDP?"),

    ("What is a compiler?",
     "Explain what a compiler does."),

    ("What is containerization?",
     "What does containerization mean?"),

    ("What is a cache hit?",
     "Explain the meaning of a cache hit."),

    ("What is a load balancer?",
     "What does a load balancer do?"),

    ("What is SQL?",
     "Explain what SQL is used for."),

    ("What is an operating system?",
     "What does an operating system do?"),

    ("What is encryption?",
     "Explain encryption in simple terms."),

    ("What is an embedding in machine learning?",
     "What are embeddings used for in ML?"),

    ("What is a REST API?",
     "Explain REST APIs."),
]


# These queries occur only once in this workload.
# They test the cost of handling queries that do not repeat.
UNIQUE_QUERIES = [
    "Explain the difference between processes and threads.",
    "What is a Merkle tree?",
    "How does garbage collection work?",
    "What is a race condition in concurrent programming?",
    "Explain the purpose of a database transaction.",
    "What is the difference between authentication and authorization?",
    "How does a compiler optimize code?",
    "What is a content delivery network?",
    "Explain the purpose of a message queue.",
    "What is eventual consistency in distributed systems?",
    "How does public-key digital signing work?",
    "What is the purpose of a reverse proxy?",
    "Explain the difference between latency and throughput.",
    "What is a finite-state machine?",
    "How does a bloom filter work?",
    "What is the purpose of a database connection pool?",
    "Explain the difference between symmetric and asymmetric encryption.",
    "What is a distributed hash table?",
    "How does virtual memory work?",
    "What is the purpose of a circuit breaker in distributed systems?",
]


def build_workload():
    """
    Construct a deterministic sequence.

    Each semantic pair produces:
      1. Original query.
      2. Exact repetition.
      3. Paraphrase.
      4. Exact repetition of the paraphrase.

    Unique queries are appended at the end.
    """

    workload = []

    for index, (original, paraphrase) in enumerate(SEMANTIC_PAIRS):
        workload.extend([
            {
                "query_type": "original",
                "pair_id": index,
                "query": original,
            },
            {
                "query_type": "exact_repeat",
                "pair_id": index,
                "query": original,
            },
            {
                "query_type": "paraphrase",
                "pair_id": index,
                "query": paraphrase,
            },
            {
                "query_type": "exact_repeat",
                "pair_id": index,
                "query": paraphrase,
            },
        ])

    for index, query in enumerate(UNIQUE_QUERIES):
        workload.append({
            "query_type": "unique",
            "pair_id": None,
            "query": query,
        })

    return workload


def send_request(item):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": item["query"],
            }
        ],
        "temperature": 0,
        "max_tokens": 128,
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            ROUTER_URL,
            headers=HEADERS,
            json=payload,
            timeout=TIMEOUT,
        )

        client_ms = (time.perf_counter() - start) * 1000

        row = {
            "run_id": RUN_ID,
            "query_type": item["query_type"],
            "pair_id": item["pair_id"],
            "query": item["query"],
            "status_code": response.status_code,
            "cache_layer": response.headers.get(
                "X-Cache-Layer", "unknown"
            ),
            "similarity": response.headers.get(
                "X-Cache-Similarity", ""
            ),
            "client_latency_ms": round(client_ms, 3),
            "router_cache_latency_ms": response.headers.get(
                "X-Cache-Latency-Ms", ""
            ),
            "inference_latency_ms": response.headers.get(
                "X-Inference-Latency-Ms", ""
            ),
            "total_router_latency_ms": response.headers.get(
                "X-Total-Latency-Ms", ""
            ),
            "error": "",
        }

        if not response.ok:
            row["error"] = response.text[:500]

        else:
            # Ensure the response is a valid chat completion.
            body = response.json()
            if not body.get("choices"):
                row["error"] = "Response has no choices"

        return row

    except Exception as exc:
        client_ms = (time.perf_counter() - start) * 1000

        return {
            "run_id": RUN_ID,
            "query_type": item["query_type"],
            "pair_id": item["pair_id"],
            "query": item["query"],
            "status_code": "",
            "cache_layer": "error",
            "similarity": "",
            "client_latency_ms": round(client_ms, 3),
            "router_cache_latency_ms": "",
            "inference_latency_ms": "",
            "total_router_latency_ms": "",
            "error": str(exc),
        }


def percentile(values, percent):
    if not values:
        return 0.0

    values = sorted(values)
    index = round((percent / 100) * (len(values) - 1))
    return values[index]


def summarize(rows):
    successful = [
        row for row in rows
        if row["status_code"] == 200 and not row["error"]
    ]

    latencies = [
        row["client_latency_ms"]
        for row in successful
    ]

    layers = Counter(
        row["cache_layer"] for row in successful
    )

    print("\n========== WORKLOAD SUMMARY ==========")
    print(f"Run ID:             {RUN_ID}")
    print(f"Total requests:     {len(rows)}")
    print(f"Successful:         {len(successful)}")
    print(f"Failed:             {len(rows) - len(successful)}")

    print("\nCache-layer counts:")

    for layer, count in sorted(layers.items()):
        print(f"  {layer:12s}: {count}")

    exact_hits = layers.get("exact", 0)
    semantic_hits = layers.get("semantic", 0)

    if successful:
        print("\nCache effectiveness:")
        print(
            f"  Overall response-cache hit rate: "
            f"{(exact_hits + semantic_hits) / len(successful):.2%}"
        )
        print(
            f"  Exact hit rate: "
            f"{exact_hits / len(successful):.2%}"
        )
        print(
            f"  Semantic hit rate: "
            f"{semantic_hits / len(successful):.2%}"
        )

        print("\nClient-side latency:")
        print(f"  Mean: {statistics.mean(latencies):.3f} ms")
        print(f"  P50:  {percentile(latencies, 50):.3f} ms")
        print(f"  P95:  {percentile(latencies, 95):.3f} ms")
        print(f"  P99:  {percentile(latencies, 99):.3f} ms")

    print("\nResults by query type:")

    groups = defaultdict(list)

    for row in rows:
        groups[row["query_type"]].append(row)

    for query_type, group in sorted(groups.items()):
        successful_group = [
            row for row in group
            if row["status_code"] == 200 and not row["error"]
        ]

        group_layers = Counter(
            row["cache_layer"] for row in successful_group
        )

        print(
            f"  {query_type:14s}: "
            f"requests={len(group):3d}, "
            f"successful={len(successful_group):3d}, "
            f"layers={dict(group_layers)}"
        )

    errors = [row for row in rows if row["error"]]

    if errors:
        print("\nFirst errors:")

        for row in errors[:5]:
            print(
                f"  HTTP {row['status_code']}: "
                f"{row['error'][:200]}"
            )


def main():
    health = requests.get(
        "http://localhost:9100/health",
        timeout=5,
    )
    health.raise_for_status()

    print("Router health:", health.json())

    workload = build_workload()

    print(f"Run ID: {RUN_ID}")
    print(f"Total workload requests: {len(workload)}")
    print("Sending requests sequentially...")

    rows = []

    for index, item in enumerate(workload, start=1):
        row = send_request(item)
        rows.append(row)

        print(
            f"[{index:3d}/{len(workload)}] "
            f"type={item['query_type']:13s} "
            f"layer={row['cache_layer']:8s} "
            f"latency={row['client_latency_ms']:9.3f} ms"
        )

        if row["error"]:
            print(f"    Error: {row['error'][:200]}")

    fieldnames = [
        "run_id",
        "query_type",
        "pair_id",
        "query",
        "status_code",
        "cache_layer",
        "similarity",
        "client_latency_ms",
        "router_cache_latency_ms",
        "inference_latency_ms",
        "total_router_latency_ms",
        "error",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)

    print(f"\nRaw results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
