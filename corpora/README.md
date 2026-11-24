# Corpora

## `wikitext-2-test.txt`

Raw WikiText-2 test split, concatenated row-wise with `\n`.

| | |
|---|---|
| upstream | [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) |
| config | `wikitext-2-raw-v1` |
| split | `test` (4358 rows) |
| dataset revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| size | 1,292,013 bytes |
| sha256 | `aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc` |

Source text is Wikipedia-derived and used under the terms published with the
dataset. Re-derive before trusting it:

```bash
sha256sum corpora/wikitext-2-test.txt   # must match the hash above
```

Scoring uses a fixed token prefix of this file (`--max-tokens`, default 200000),
tokenised by each model's own tokenizer. No chat template is applied: perplexity
is measured on raw text. Because tokenisation differs between models, compare PPL
only within one model family.

**Contamination caveat.** WikiText-2 is small and widely used in pretraining, so
absolute PPL on it is optimistically low for models that saw it during training.
Treat these numbers as a *within-family* comparison between quantization widths,
not as an absolute measure of language quality. A held-out corpus would be needed
for the latter.
