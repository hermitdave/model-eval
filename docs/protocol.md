# Evaluation protocol

## Scope

Comparing quantized variants (widths, recipes) of the same base model on Apple
Silicon — including checkpoints too large for resident RAM.

Protocol: **5 tasks x 400 examples** (ARC-Challenge, ARC-Easy, HellaSwag,
WinoGrande, MMLU), identical prompts, temperature 0, Wilson 95% CI. Data in
`~/Projects/ml/zeroshot_data`.

Report absolute accuracy only. If the BF16 teacher cannot be scored on the host,
state that and never imply a recovery ratio against it.

## Order of operations

1. **Resident baseline first.** `zs_run.py`, and make sure per-item correctness
   is captured (`examples=sorted(idx)` in the result JSON).
2. **Validate offload fidelity before trusting any offloaded number.** Re-score
   an already-measured model with `zs_offload.py` on one task and compare correct
   counts against the resident result. A delta of 1-2 out of 400 is kernel
   nondeterminism; anything larger means the wrapper is broken.
3. **Score sequentially.** Never run two models concurrently (RAM and
   contention). Finish every prompt for one model before switching.
4. **Capture per-item indices on every run.** Without them no paired test is
   possible later.
5. **Compare paired**, never by overlapping confidence intervals.

## MoE expert streaming

`zs_offload.py` wraps routed-expert layers so only a fraction stays resident and
the rest stream from the artifact's own safetensors shards on demand.

```
capacity = min(n_experts, max(max(8, num_experts_per_tok),
                              round(n_experts * resident_fraction)))
```

Routing is unchanged; a cache miss changes *when* weights are read, never *which*
expert runs. Accuracy is preserved by construction.

Measured residency guidance (64 GB host):

| fraction | notes |
|---|---|
| 0.30 | validated accuracy-neutral; ~2.2x slower than resident; safe default |
| 0.50 | ~1.3x faster than 0.30; use when peak stays well below RAM |
| > 0.60 | not worth the OOM risk |

Lightning MTP and expert streaming are mutually exclusive in oMLX except for
`deepseek_v41`, `glm5_next`, `qwen4_exp`. Only resident-fitting variants can use
MTP.

## Statistical rules

- Independent Wilson CIs are **not** a comparison when both models answered the
  same items. At n=400 they are underpowered for 3-4 pp differences and overlap
  even for real effects.
- Use exact McNemar on discordant items (`paired_compare.py`). It removes item
  difficulty and is the only defensible test at this sample size.
- Correct for multiple comparisons across tasks (Bonferroni) and say so. A
  single p=0.04 out of five is suggestive, not established.
- Report per-task deltas, direction counts, which tasks survive correction, plus
  residency, peak memory and wall time.

## Pitfalls found the hard way

- Dividing rates by `len(correct_A | correct_B)` instead of `n` inflates every
  number (produced 0.81 for a known 0.52). Assert the harness reproduces a
  recorded accuracy before computing anything downstream.
- oMLX loads a checkpoint-local `model_file` **instead of** its vendored
  architecture module, silently dropping Ling's nextn/MTP layer. Verify head
  attachment in logs ("MTP path activated"); never assume a toggle took effect.
- `HF_HOME` on a removable drive breaks downloads when unmounted; uploads via
  `HfApi.upload_file` still work because they use a temp dir.
- When withdrawing a variant that measured equivalent, replace its table column
  with an explicit withdrawal note rather than deleting rows silently.
