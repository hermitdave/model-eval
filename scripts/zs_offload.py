#!/usr/bin/env python3
"""Zero-shot scoring with oMLX MoE expert offload.

Lets a checkpoint larger than RAM be scored in-process: only a fraction of each
layer's routed experts stays resident, the rest stream from the artifact's own
safetensors shards on demand. Accuracy is preserved by construction (routing is
unchanged; only *when* an expert's weights are read differs).

Usage:
  python zs_offload.py --model <dir> --resident-fraction 0.2 --limit 400
"""
import argparse, json, math, sys, time

sys.path.insert(0, "/Users/hermit.dave/Projects/ml/slq_mlx/src")
sys.path.insert(0, "/Users/hermit.dave/Projects/web/omlx-mtp")

import mlx.core as mx
from mlx_lm import load
from omlx.patches.moe_expert_offload import (
    apply_moe_expert_offload,
    moe_offload_stats,
)

TASKS = ["arc_challenge", "arc_easy", "hellaswag", "winogrande", "mmlu"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--resident-fraction", type=float, default=0.2)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tasks = [t for t in args.tasks.split(",") if t]
    label = args.label or args.model.rstrip("/").split("/")[-1]

    from slq_mlx.scripts.run_zero_shot import score_task

    t0 = time.time()
    model, tok = load(args.model, trust_remote_code=True, lazy=True,
                      tokenizer_config={"trust_remote_code": True})
    print(f"loaded {time.time()-t0:.1f}s | clamped="
          f"{getattr(model,'ling_clamped_paths','n/a')}", flush=True)

    wrapped = apply_moe_expert_offload(model, args.model, args.resident_fraction)
    print(f"offload: wrapped {wrapped} layers at residency "
          f"{args.resident_fraction} | stats={moe_offload_stats(model)}", flush=True)
    if wrapped == 0:
        print("WARNING: nothing wrapped -- scoring resident (may OOM on wide quants)",
              flush=True)

    res = {}
    for t in tasks:
        tt = time.time()
        n, ncorr, idx = score_task(tok, model, t, args.limit)
        lo, hi = wilson(ncorr, n)
        res[t] = dict(n=n, correct=ncorr, acc=ncorr / n, ci95=[lo, hi],
                      secs=round(time.time() - tt, 1), examples=sorted(idx))
        print(f"  {t:14s} acc {ncorr/n:.4f} [{lo:.4f},{hi:.4f}] (n={n}) "
              f"[{time.time()-tt:.0f}s]", flush=True)

    print(f"\npeak memory: {mx.get_peak_memory()/1e9:.2f} GB")
    if args.out:
        json.dump(dict(label=label, model=args.model, resident_fraction=args.resident_fraction,
                       offload_wrapped=wrapped, results=res,
                       peak_gb=round(mx.get_peak_memory() / 1e9, 2)),
                  open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
