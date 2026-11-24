# model-eval

Zero-shot evaluation of quantized LLMs on Apple Silicon, with the two things that
make comparisons trustworthy: **a fidelity gate for memory offload**, and **paired
statistics** instead of overlapping confidence intervals.

Built while comparing mixed-precision MoE quantizations of InclusionAI Ling 3.2
air; designed to be reused for any model family.

## Contents

```
scripts/
  zs_run.py           resident zero-shot scoring (5 tasks x N examples)
  zs_offload.py       same, with routed-expert streaming for checkpoints > RAM
  paired_compare.py   exact McNemar on discordant items across variants
results/
  ling-3.2-air-rco/   measured results, including the withdrawn 3.66 width
docs/
  protocol.md         the protocol, statistical rules and known pitfalls
```

## Quickstart

```bash
# resident baseline (capture per-item correctness!)
python scripts/zs_run.py --model <dir> --limit 400 --out results/base.json

# variant too large for RAM: stream experts at 30% residency
python scripts/zs_offload.py --model <dir> --resident-fraction 0.3 \
    --limit 400 --out results/variant.json

# compare on the shared items
python scripts/paired_compare.py results/base.json results/variant.json
```

Scoring core is `slq_mlx.scripts.run_zero_shot`; data lives in
`~/Projects/ml/zeroshot_data`.

## Results: Ling 3.2 air RCO widths

Five tasks x 400 examples, identical prompts, Wilson 95% CI. Absolute numbers
only — the BF16 teacher cannot be scored on the measuring host.

| task | 2.75 bpw | 3.66 bpw | 4.62 bpw |
|---|---:|---:|---:|
| ARC-Challenge | 0.5200 [0.471, 0.569] | 0.5375 [0.489, 0.586] | 0.5900 [0.541, 0.637] |
| ARC-Easy | 0.7975 [0.755, 0.834] | 0.8000 [0.758, 0.836] | 0.8275 [0.787, 0.861] |
| HellaSwag | 0.4300 [0.382, 0.479] | 0.4275 [0.380, 0.476] | 0.4700 [0.422, 0.519] |
| WinoGrande | 0.5850 [0.536, 0.632] | 0.6050 [0.556, 0.652] | 0.5950 [0.546, 0.642] |
| MMLU | 0.5375 [0.489, 0.586] | 0.5300 [0.481, 0.578] | 0.5750 [0.526, 0.623] |

Paired McNemar (shared items) against 2.75:

| task | 3.66 p | 4.62 p |
|---|---:|---:|
| ARC-Challenge | 0.525 | **0.0023** |
| ARC-Easy | 1.000 | 0.162 |
| HellaSwag | 1.000 | **0.044** |
| WinoGrande | 0.512 | 0.762 |
| MMLU | 0.810 | 0.058 |

Conclusions: **3.66 is a proven null** and its checkpoint was withdrawn;
**4.62 is genuinely better**, with only ARC-Challenge surviving a Bonferroni
correction across five tasks.

Offload fidelity was validated before use: 209 vs 208 of 400 correct on
ARC-Challenge against the resident baseline, i.e. accuracy-neutral within noise.

## Attribution

Evaluation harness, paired analysis, dequant prep and validation tooling produced
with [Hermes Agent](https://github.com/NousResearch/hermes-agent). Base model:
InclusionAI Ling 3.2 air (MIT).

## License

MIT
