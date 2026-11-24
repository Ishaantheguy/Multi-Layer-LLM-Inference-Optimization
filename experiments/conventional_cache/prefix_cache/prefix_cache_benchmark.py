#!/usr/bin/env python3
import argparse,json,statistics,time
from urllib.request import Request,urlopen

VLLM_URL='http://127.0.0.1:8000/v1/chat/completions'
MODEL='Qwen/Qwen3-8B-AWQ'
BASE='''You are analyzing a technical document for a research experiment on language model inference and caching. Treat this synthetic technical knowledge base as the source of truth.\n\nComputer architecture: registers are fastest; L1, L2, L3 caches trade latency and size; temporal locality means recently used data is likely reused; spatial locality means nearby data is likely reused.\n\nOperating systems: virtual addresses map to physical frames through page tables; the TLB caches recent translations.\n\nNetworking: TCP provides reliable ordered byte-stream delivery using sequence numbers, acknowledgements, retransmission, flow control and congestion control; its handshake uses SYN, SYN-ACK and ACK.\n\nDatabases: indexes such as B-trees can accelerate reads but consume storage and can increase write cost.\n\nMachine learning: training learns parameters by optimizing an objective; inference uses learned parameters.\n\nLarge language models: autoregressive models predict the next token. During generation, prior key/value tensors can be retained in a KV cache. Prefix caching can reuse KV blocks for a shared prompt prefix across requests.\n\nLLM caching: an exact response cache stores a complete response; a semantic cache reuses responses for sufficiently similar requests; prefix/KV caching reuses intermediate computation rather than returning a previous response.\n\nExperimental methodology: to study prefix caching, requests should share a substantial prompt prefix while their final questions differ. This isolates prefix reuse from complete-response reuse.\n\nPerformance: TTFT is time until the first token; total latency is complete request duration; prefill processes input tokens and decode produces output tokens. Prefix caching can reduce repeated prefill work while generation remains necessary.\n\nCost: prefix reuse can reduce prompt-processing cost without eliminating generation; exact response caching can avoid both prompt processing and generation for an identical request.'''
QUESTIONS=['Explain temporal versus spatial locality.','Explain the TCP three-way handshake.','Explain why indexes improve reads but can increase write cost.','Explain what the KV cache does during autoregressive generation.','Explain how prefix caching differs from exact response caching.']

def post(payload):
    data=json.dumps(payload).encode(); 
    req=Request(VLLM_URL,data=data,headers={'Content-Type':'application/json'}); 
    t=time.perf_counter()
    with urlopen(req,timeout=300) as r: 
        body=r.read()
    return (time.perf_counter()-t)*1000,json.loads(body)

def stats(v):
    v=sorted(v); return {'n':len(v),'mean_ms':statistics.mean(v),'p50_ms':statistics.median(v),'p95_ms':v[max(0,int(.95*len(v))-1)],'min_ms':min(v),'max_ms':max(v)}
p=argparse.ArgumentParser(); 
p.add_argument('--repetitions',type=int,default=8); 
p.add_argument('--warmup',type=int,default=1); 
a=p.parse_args()
context='\n'.join(f'--- Knowledge block {i+1} ---\n{BASE}' for i in range(a.repetitions))
print('=== vLLM Prefix Cache Benchmark ==='); 
print(f'Shared context repetitions: {a.repetitions}'); 
print(f'Questions: {len(QUESTIONS)}'); 
print(f'Endpoint: {VLLM_URL}\n')
print('Warmup...')
for i in range(a.warmup):
    payload={'model':MODEL,'messages':[{'role':'user','content':context+'\n\nQuestion: '+QUESTIONS[i%len(QUESTIONS)]}],'temperature':0,'max_tokens':128}
    l,_=post(payload); 
    print(f'  warmup {i+1}: {l:.2f} ms')
print('\nMeasured requests:'); results=[]
for i,q in enumerate(QUESTIONS):
    payload={'model':MODEL,'messages':[{'role':'user','content':context+'\n\nQuestion: '+q}],'temperature':0,'max_tokens':128}
    l,r=post(payload); 
    u=r.get('usage',{}); results.append(l); 
    print(f"  request {i+1}: {l:.2f} ms | prompt={u.get('prompt_tokens')} | completion={u.get('completion_tokens')}")
print('\n=== Latency Statistics ==='); 
print(json.dumps(stats(results),indent=2)); 
print('\nCompare this with the same script under prefix OFF and prefix ON. Check vLLM logs for prefix-cache hit rate. This script uses direct non-streaming HTTP, so latency is end-to-end rather than TTFT.')


