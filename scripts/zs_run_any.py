#!/usr/bin/env python3
"""Parameterized zero-shot scorer using the pinned slq_mlx task loader.

Records per-item correct indices so paired McNemar tests are possible later.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

SLQ_SRC = os.environ.get(
    "SLQ_MLX_SRC",
    str(Path(__file__).resolve().parents[2] / "slq_mlx" / "src"),
)
sys.path.insert(0, SLQ_SRC)

import mlx.core as mx  # pyright: ignore[reportMissingImports]
from mlx_lm import load  # pyright: ignore[reportMissingImports]
from slq_mlx.scripts.run_zero_shot import (  # pyright: ignore[reportMissingImports]
    load_examples, score_task,
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load(args.model, lazy=True)
    print(f"loaded {args.model} in {time.time()-t0:.1f}s", flush=True)

    res = {}
    for task in args.tasks:
        tt = time.time()
        n, ncorr, idx = score_task(tok, model, task, args.limit)
        lo, hi = wilson(ncorr, n)
        res[task] = dict(n=n, correct=ncorr, acc=ncorr / n, ci95=[lo, hi],
                         examples=sorted(idx), secs=round(time.time() - tt, 1))
        print(f"  {task:14s} acc {ncorr/n:.4f} [{lo:.4f},{hi:.4f}] "
              f"(n={n}) [{time.time()-tt:.0f}s]", flush=True)

    mx.eval()
    payload = dict(model=args.model, limit=args.limit, zero_shot=res,
                   peak_mem_gb=round(mx.get_peak_memory() / 1e9, 2))
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print("saved", out, "peak", payload["peak_mem_gb"], "GB", flush=True)


if __name__ == "__main__":
    main()
