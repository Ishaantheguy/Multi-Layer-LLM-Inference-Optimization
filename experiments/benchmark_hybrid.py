import csv
import math
import statistics
import time
import uuid

import requests


# ============================================================
# Configuration
# ============================================================

ROUTER_URL = "http://localhost:9100/v1/chat/completions"

MODEL = "Qwen/Qwen3-8B-AWQ"
QUERY = "What is machine learning?"

REQUESTS_PER_TEST = 10000
WARMUP_REQUESTS = 100

OUTPUT_CSV = "hybrid_benchmark.csv"

RUN_ID = uuid.uuid4().hex

BASE_PAYLOAD = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": QUERY,
        }
    ],
    "temperature": 0,
    "max_tokens": 128,
}


# ============================================================
# HTTP request helper
# ============================================================

def send_request(payload, expected_layer):
    start = time.perf_counter()

    response = requests.post(
        ROUTER_URL,
        json=payload,
        timeout=30,
    )

    client_latency_ms = (
        time.perf_counter() - start
    ) * 1000
    
    print("response.text:",response.text)
    response.raise_for_status()

    actual_layer = response.headers.get(
        "X-Cache-Layer",
        "unknown",
    )

    if actual_layer != expected_layer:
        raise RuntimeError(
            f"Expected {expected_layer!r}, "
            f"received {actual_layer!r}. "
            f"Response: {response.text[:300]}"
        )

    server_latency_ms = float(
        response.headers["X-Cache-Latency-Ms"]
    )

    return {
        "client_latency_ms": client_latency_ms,
        "server_latency_ms": server_latency_ms,
        "cache_layer": actual_layer,
    }


# ============================================================
# Statistics
# ============================================================

def percentile(values, p):
    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def summarize(name, results):
    client = [
        result["client_latency_ms"]
        for result in results
    ]

    server = [
        result["server_latency_ms"]
        for result in results
    ]

    summary = {
        "test": name,
        "requests": len(results),
        "client_mean_ms": statistics.mean(client),
        "client_p50_ms": percentile(client, 0.50),
        "client_p95_ms": percentile(client, 0.95),
        "server_mean_ms": statistics.mean(server),
        "server_p50_ms": percentile(server, 0.50),
        "server_p95_ms": percentile(server, 0.95),
    }

    print(f"\n--- {name} ---")
    print(f"Requests:           {summary['requests']}")
    print(
        f"Client mean:        "
        f"{summary['client_mean_ms']:.3f} ms"
    )
    print(
        f"Client p50:         "
        f"{summary['client_p50_ms']:.3f} ms"
    )
    print(
        f"Client p95:         "
        f"{summary['client_p95_ms']:.3f} ms"
    )
    print(
        f"Router mean:        "
        f"{summary['server_mean_ms']:.3f} ms"
    )
    print(
        f"Router p50:         "
        f"{summary['server_p50_ms']:.3f} ms"
    )
    print(
        f"Router p95:         "
        f"{summary['server_p95_ms']:.3f} ms"
    )

    return summary


# ============================================================
# Exact-cache benchmark
# ============================================================

def benchmark_exact():
    print("\nTesting exact-cache hits...")

    results = []

    # Warm-up requests are excluded from statistics.
    for _ in range(WARMUP_REQUESTS):
        send_request(
            BASE_PAYLOAD,
            expected_layer="exact",
        )

    for i in range(REQUESTS_PER_TEST):
        result = send_request(
            BASE_PAYLOAD,
            expected_layer="exact",
        )

        result["test"] = "exact"
        result["iteration"] = i + 1

        results.append(result)

    return results


# ============================================================
# Semantic-cache benchmark
# ============================================================

def benchmark_semantic():
    print("\nTesting semantic-cache hits...")

    results = []

    for i in range(REQUESTS_PER_TEST):
        payload = dict(BASE_PAYLOAD)

        # The OpenAI-compatible 'user' field makes each exact
        # request key different. The semantic query itself
        # remains unchanged.
        #
        # This prevents exact-cache promotion from turning
        # subsequent trials into exact hits.
        payload["user"] = f"benchmark-{RUN_ID}-{i}"

        result = send_request(
            payload,
            expected_layer="semantic",
        )

        result["test"] = "semantic"
        result["iteration"] = i + 1

        results.append(result)

    return results


# ============================================================
# Main
# ============================================================

def main():
    # Confirm the router is reachable.
    health = requests.get(
        "http://localhost:9100/health",
        timeout=10,
    )
    health.raise_for_status()

    print("Router health:", health.json())

    exact_results = benchmark_exact()
    semantic_results = benchmark_semantic()

    exact_summary = summarize(
        "Exact cache",
        exact_results,
    )

    semantic_summary = summarize(
        "Semantic cache",
        semantic_results,
    )

    exact_mean = exact_summary["client_mean_ms"]
    semantic_mean = semantic_summary["client_mean_ms"]

    print("\n--- Comparison ---")
    print(
        f"Mean latency reduction: "
        f"{(1 - semantic_mean / exact_mean) * 100:.2f}% "
        f"(semantic relative to exact)"
    )

    print(
        f"Exact/semantic latency ratio: "
        f"{semantic_mean / exact_mean:.2f}x"
    )

    # Export per-request measurements for later analysis.
    all_results = exact_results + semantic_results

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test",
                "iteration",
                "cache_layer",
                "client_latency_ms",
                "server_latency_ms",
            ],
        )

        writer.writeheader()

        for result in all_results:
            writer.writerow(result)

    print(f"\nRaw measurements saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
