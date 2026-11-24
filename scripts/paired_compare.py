#!/usr/bin/env python3
"""Paired comparison of the Ling RCO widths on the shared zero-shot items.

Unpaired Wilson CIs compare two independent proportions and ignore that all
three models answered the *same* items. McNemar's test uses the discordant
pairs (items only one model got right), which removes item difficulty entirely
and is far more sensitive at n=400.

Usage:
    python paired_compare.py BASELINE.json A.json B.json
"""
import json
import math
import sys
from pathlib import Path


def load(path):
    d = json.loads(Path(path).read_text())
    res = d.get("zero_shot") or d.get("results")
    return {t: (set(v["examples"]), v.get("n", 0)) for t, v in res.items() if "examples" in v}


def mcnemar(a_only, b_only):
    """Exact two-sided binomial test on the discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    # two-sided exact p from Binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    base, other_labels = load(sys.argv[1]), sys.argv[2:]
    others = [(Path(p).stem, load(p)) for p in other_labels]

    for label, cur in others:
        print(f"\n=== baseline vs {label} ===")
        print(f"{'task':14s} {'base':>7s} {'cur':>7s} {'delta':>7s} "
              f"{'base-only':>9s} {'cur-only':>8s} {'p(McNemar)':>10s}")
        for task in base:
            if task not in cur:
                continue
            b, nb = base[task]
            c, nc = cur[task]
            n_items = nb or nc or max(len(b | c), 1)
            kb, kc = len(b), len(c)
            db, dc = len(b - c), len(c - b)
            p = mcnemar(db, dc)
            print(f"{task:14s} {kb/n_items:7.4f} {kc/n_items:7.4f} "
                  f"{(kc-kb)/n_items:+7.4f} {db:9d} {dc:8d} {p:10.4f}")


if __name__ == "__main__":
    main()
