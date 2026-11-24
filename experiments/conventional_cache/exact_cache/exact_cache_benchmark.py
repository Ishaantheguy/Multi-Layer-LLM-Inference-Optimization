#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from urllib.request import Request, urlopen

PROMPT = "Explain the Torional Balance experiment performed by Coloumb"

PAYLOAD = {
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0,
}

def post(url):
    data = json.dumps(PAYLOAD).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(req, timeout=300) as r:
        body = r.read()
    return (time.perf_counter() - start) * 1000, body

def stats(values):
    values = sorted(values)
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": values[max(0, int(0.95 * len(values)) - 1)],
        "min_ms": min(values),
        "max_ms": max(values),
    }

parser = argparse.ArgumentParser()
parser.add_argument("--cache-url", default="http://127.0.0.1:9000/v1/chat/completions")
parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1/chat/completions")
parser.add_argument("--repeats", type=int, default=20)
args = parser.parse_args()

print("=== Exact Cache Benchmark ===")
print(f"Repeats: {args.repeats}")
print()

# First request through cache: expected MISS.
miss, _ = post(args.cache_url)
print(f"Cache MISS: {miss:.2f} ms")

# Repeated requests through cache: expected HITs.
hits = []
for i in range(args.repeats):
    latency, _ = post(args.cache_url)
    hits.append(latency)

print()
print("Cache HIT statistics:")
print(json.dumps(stats(hits), indent=2))

# Direct vLLM baseline.
# Use a unique prompt so we don't accidentally benchmark an existing
# prefix/cache state from the previous exact-cache experiment.
baseline_payload = dict(PAYLOAD)
baseline_payload["messages"] = [{
    "role": "user",
    "content": PROMPT + f" Benchmark baseline {time.time_ns()}"
}]

old_payload = PAYLOAD.copy()
PAYLOAD.clear()
PAYLOAD.update(baseline_payload)

baseline = []
for i in range(args.repeats):
    latency, _ = post(args.vllm_url)
    baseline.append(latency)

print()
print("Direct vLLM statistics:")
print(json.dumps(stats(baseline), indent=2))

hit_mean = statistics.mean(hits)
base_mean = statistics.mean(baseline)

print()
print("=== Comparison ===")
print(f"Mean direct-vLLM latency : {base_mean:.2f} ms")
print(f"Mean exact-cache HIT     : {hit_mean:.2f} ms")
print(f"Mean latency speedup     : {base_mean / hit_mean:.1f}x")
print(f"Mean latency reduction   : {(1 - hit_mean / base_mean) * 100:.2f}%")
