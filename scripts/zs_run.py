#!/usr/bin/env python3
"""Zero-shot + throughput for the Ling-3.2-air RCO artifact (quant side)."""
import json, math, sys, time

sys.path.insert(0, "/Users/hermit.dave/Projects/ml/slq_mlx/src")
sys.path.insert(0, "/Users/hermit.dave/.hermes/profiles/helix/cache/scratch")

import numpy as np
import mlx.core as mx
from mlx_lm import load, generate

MODEL = "/Users/hermit.dave/Projects/ml/models/Ling-3.2-air-MLX-RCO"
TASKS = ["arc_challenge", "arc_easy", "hellaswag", "winogrande", "mmlu"]
LIMIT = 400
OUT = "/Users/hermit.dave/.hermes/profiles/helix/cache/scratch/zeroshot_ling_rco.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


from slq_mlx.scripts.run_zero_shot import load_examples, score_task

t0 = time.time()
model, tok = load(MODEL, trust_remote_code=True, lazy=True,
                  tokenizer_config={"trust_remote_code": True})
print(f"loaded {time.time()-t0:.1f}s | {type(model).__module__}.{type(model).__name__} "
      f"| clamped={getattr(model,'ling_clamped_paths','n/a')}", flush=True)

res = {}
for t in TASKS:
    tt = time.time()
    n, ncorr, idx = score_task(tok, model, t, LIMIT)
    lo, hi = wilson(ncorr, n)
    res[t] = dict(n=n, correct=ncorr, acc=ncorr / n, ci95=[lo, hi],
                  examples=sorted(idx), secs=round(time.time() - tt, 1))
    print(f"  {t:14s} acc {ncorr/n:.4f} [{lo:.4f},{hi:.4f}] (n={n}) "
          f"[{time.time()-tt:.0f}s]", flush=True)

# ---------------- throughput ----------------
prompt = ("You are a helpful assistant. " * 120).strip()
ids = tok(prompt, return_tensors="pt").input_ids if hasattr(tok, "__call__") else None
try:
    import torch
    enc = tok(prompt, return_tensors="pt")
    input_ids = mx.array(enc.input_ids.numpy())
except Exception:
    input_ids = mx.array([tok.encode(prompt)])

t0 = time.time()
_ = model(input_ids)
mx.eval(_)
prefill_s = time.time() - t0
ntok = input_ids.shape[-1]
print(f"\nprefill: {ntok} tokens in {prefill_s:.2f}s -> {ntok/prefill_s:.1f} tok/s", flush=True)

dec = []
for i in range(3):
    t0 = time.time()
    out = generate(model, tok, prompt="Explain why hybrid linear attention is useful.",
                   max_tokens=256, verbose=False)
    dt = time.time() - t0
    gen = len(tok.encode(out)) if isinstance(out, str) else 0
    dec.append(gen / dt)
    print(f"decode run {i+1}: {gen} tok in {dt:.1f}s -> {gen/dt:.2f} tok/s", flush=True)

mx.eval()
print(f"\npeak memory: {mx.get_peak_memory()/1e9:.2f} GB")
json.dump(dict(model=MODEL, zero_shot=res,
               throughput=dict(prefill_tok=ntok, prefill_tok_s=round(ntok/prefill_s, 1),
                               decode_tok_s_runs=[round(x, 2) for x in dec],
                               decode_tok_s_median=round(float(np.median(dec)), 2),
                               peak_mem_gb=round(mx.get_peak_memory()/1e9, 2))),
          open(OUT, "w"), indent=2)
print("saved", OUT)
