#!/usr/bin/env python3
"""Perplexity on a raw-text corpus — no chat template, no task wrapper.

Self-contained apart from mlx / mlx-lm. Expert streaming is optional (``--resident
fraction``), so checkpoints larger than RAM can be scored too.

Usage:
  python ppl.py --model <dir> [--corpus corpora/wikitext-2-test.txt]
                [--context-length 2048] [--max-tokens 200000]
                [--resident-fraction 0.3] [--out results/ppl.json]
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

DEFAULT_CORPUS = str(Path(__file__).resolve().parent.parent / "corpora" / "wikitext-2-test.txt")


def _nll_for_chunk(model, ids):
    """Summed negative log-likelihood and token count for one context window.

    Predictions for positions 1..L-1 against targets 1..L-1; position 0 has no
    predecessor and is never counted.
    """
    import mlx.core as mx

    logits = model(ids[None, :])
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = logits[0].astype(mx.float32)          # (L, V)
    targets = ids[1:]
    logits = logits[:-1]                            # predict 1..L-1 from 0..L-2
    tgt = logits[mx.arange(targets.shape[0]), targets]
    lse = mx.logsumexp(logits, axis=-1)
    nll = lse - tgt
    return float(nll.sum()), int(targets.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=200000)
    ap.add_argument("--resident-fraction", type=float, default=None,
                    help="stream routed experts from disk at this residency")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm import load

    corpus_path = Path(args.corpus)
    raw = corpus_path.read_text()
    digest = hashlib.sha256(raw.encode()).hexdigest()

    t0 = time.time()
    model, tokenizer = load(
        args.model,
        trust_remote_code=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    wrapped = 0
    if args.resident_fraction is not None:
        sys.path.insert(0, str(Path.home() / "Projects" / "web" / "omlx-mtp"))
        from omlx.patches.moe_expert_offload import apply_moe_expert_offload
        wrapped = apply_moe_expert_offload(model, args.model, args.resident_fraction)
        print(f"offload: wrapped {wrapped} layers @ {args.resident_fraction}", flush=True)
    print(f"loaded in {time.time()-t0:.1f}s | clamped="
          f"{getattr(model, 'ling_clamped_paths', 'n/a')}", flush=True)
    load_seconds = time.time() - t0

    ids = tokenizer.encode(raw) if hasattr(tokenizer, "encode") else tokenizer(raw)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(i) for i in ids[: args.max_tokens]]
    ctx = args.context_length
    chunks = [ids[i:i + ctx] for i in range(0, len(ids), ctx)]
    chunks = [c for c in chunks if len(c) > 1]
    print(f"corpus={corpus_path.name} sha256={digest[:12]} tokens={len(ids)} "
          f"chunks={len(chunks)} ctx={ctx}", flush=True)

    import mlx.core as mx
    total_nll, total_tok, t1 = 0.0, 0, time.time()
    for n, chunk in enumerate(chunks, 1):
        nll, tok = _nll_for_chunk(model, mx.array(chunk))
        total_nll += nll
        total_tok += tok
        if n % 10 == 0 or n == len(chunks):
            cur = math.exp(total_nll / total_tok)
            print(f"  chunk {n}/{len(chunks)} ppl_so_far={cur:.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    ppl = math.exp(total_nll / total_tok)
    out = dict(
        model=args.model,
        corpus=corpus_path.name,
        corpus_sha256=digest,
        context_length=ctx,
        tokens_scored=total_tok,
        chunks=len(chunks),
        ppl=round(ppl, 4),
        seconds=round(time.time() - t1, 1),
        load_seconds=round(load_seconds, 1),
        resident_fraction=args.resident_fraction,
        offload_wrapped=wrapped,
        peak_gb=round(mx.get_peak_memory() / 1e9, 2),
    )
    print(f"\nPPL = {ppl:.4f} over {total_tok} tokens "
          f"[{out['seconds']}s, peak {out['peak_gb']} GB]")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print("saved", args.out)


if __name__ == "__main__":
    main()
