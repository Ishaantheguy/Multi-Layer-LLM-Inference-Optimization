# Multi-Layer LLM Inference Optimization

An experimental project exploring different caching strategies to reduce latency in LLM inference.

The system combines three layers:

- **Exact caching:** Stores request-response pairs in SQLite and serves identical requests without inference.
- **Semantic caching:** Uses sentence embeddings and Qdrant to identify similar queries and reuse previous responses.
- **Prefix caching:** Uses vLLM's prefix/KV caching to avoid recomputing shared prompt prefixes during inference.

An adaptive controller decides whether a semantic cache lookup is worthwhile based on observed hit rates and lookup latency, while accounting for expected prefix-cache reuse.

## Architecture

```text
             Incoming Request
                    |
                    v
               Exact Cache
                 (SQLite)
                    |
                Cache Miss
                    |
                    v
              Semantic Cache
                (Qdrant)
                    |
                Cache Miss
                    |
                    v
               vLLM Inference
              (Prefix / KV)
                    |
                    v
                 Response
```

## Tech Stack

- Python, FastAPI
- SQLite
- Qdrant
- Sentence Transformers
- vLLM
- Qwen3-8B-AWQ

## Benchmarks

The project includes experiments for individual cache layers and the combined system.

- Exact and semantic cache latency measurements.
- Workloads containing repeated, paraphrased and unique queries.
- Mean, and latency measurements.
- Comparison of exact-only, semantic-always and adaptive cache modes.

Benchmark scripts are in `experiments/`, with recorded results in `results/`.

## Setup

Requires Python, Docker, a running Qdrant instance and a vLLM inference server.

```bash
git clone https://github.com/Ishaantheguy/Multi-Layer-LLM-Inference-Optimization.git
cd Multi-Layer-LLM-Inference-Optimization

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Start the router:

```bash
python -m uvicorn src.api.hybrid_cache:app --host 0.0.0.0 --port 9100
```

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Service health |
| `GET /adaptive/stats` | Adaptive controller statistics |
| `POST /v1/chat/completions` | Chat completions |

The chat completions endpoint follows the OpenAI API request format.

## Project Structure

```text
src/            Core implementation
  cache/        Exact and semantic caching
  adaptive/     Adaptive controller and cache policies
  api/          FastAPI router

experiments/    Cache experiments and benchmarks
results/        Benchmark outputs
docs/           Project documentation
```

## Status

This is an ongoing experimental project. Current work focuses on evaluating the trade-offs between cache lookup overhead, cache hit rates and inference latency, and improving the adaptive strategy. Probably will look into how CUDA programming can be used to implement GPU resident semantic cache instead of a seperate database. Then probably look into how GPU KV/prefix caching works and if something can be done to make it more suited for our uses. And so on and on. Quite a bit of scope :)
