# Offline held-out perplexity

`python scripts/evaluate_quality.py --help` exposes a standalone evaluator. It
never downloads a corpus or model, never enables remote/custom Python model or
tokenizer code, and does not install fused dispatch. `quality.py`'s accounting and
NumPy oracle can be imported and tested without MLX.

## Input and scope

Supply an existing UTF-8 JSON **list of strings**, in the intended held-out order.
Blank records are preserved, not filtered. The default separator is two literal
newlines. Records are joined **before** tokenization; the joined text is encoded
once with `add_special_tokens=False`. There are no added BOS/EOS tokens, chat
formats, record-level context resets, or separately tokenized separators.
`--separator` accepts a literal string; for an empty separator use `--separator ''`.

For WikiText, obtain the public **test** split separately and serialize its `text`
records in source order. Do not reuse calibration/training records. The evaluator
records the supplied provenance label and content hashes but cannot independently
verify that the corpus was held out or that a JSON export is the official split.

**All reports are explicitly noncanonical.** A prefix of a WikiText test export is
held-out likelihood on that specific subsample, not canonical full WikiText
perplexity. Even evaluating all records with this tokenization/window policy does
not establish equivalence to another published evaluation protocol. Perplexity
is not task accuracy. The tiny random model acceptance test establishes plumbing,
not language-model quality.

## Deterministic scoring and exact accounting

Let `tokens` have length `N`, the configured window length be `L >= 2`, and stride
be `1 <= S < L`. With cap `C`, score exactly the target positions
`1 .. min(N-1, C)` inclusive. The cap is **targets across the joined corpus**, not
input tokens per record/window. The CLI defaults to 4,096 targets; pass an explicit
cap in experiments. The Python accounting API also accepts `None` for an uncapped
stream. A short corpus produces fewer targets, explicitly recorded.

Starting at target position `t = 1`, each window uses:

```text
end          = min(t + S, min(N, C + 1))   # end-exclusive
start        = max(0, end - L)
input ids    = tokens[start : end - 1]
target ids   = tokens[t : end]
scored logits= logits[0, t - start - 1 : end - start - 1, :]
t            = end
```

`L` counts the **full token slice including the final target**; model input is at
most `L-1` tokens. This convention is recorded in the report and differs from
some external libraries' context-length conventions. Every target has its
preceding input token, each target is scored once, and there are no missing
boundary tokens. The first token is unscored because there is no preceding
context. The final partial window is included, not discarded. Context-only logits
in overlapping windows are excluded. Each forward pass starts afresh with local
positions, no persistent KV cache, batch size one. Context for the earliest target
in a full window is `L-S` preceding tokens; later targets see more preceding tokens.
Changing stride therefore changes the evaluation protocol, not just speed.

The stock MLX path casts logits to float32 and uses stable, maximum-shifted
log-sum-exp to compute per-target NLL on device. Only the scored NLL vector is
transferred to the host and cast to float64. `math.fsum` aggregates per-window
losses and then window totals in host double precision. Mean NLL is
`total_nll / actual_target_count`; perplexity is `exp(mean_nll)`. It does **not**
average window perplexities or unweighted window means. All logits and NLLs must
be finite; invalid target ids, shapes, negative NLL, and nonfinite/overflowing
perplexity are errors. GPU float64 is neither used nor required.

Window audit entries contain global token offsets, local end-exclusive scored
logit offsets, input counts, target counts, and NLL totals. Aggregate accounting
includes stream length, available targets, scored targets, initial unscored token,
tail targets omitted by the cap, sequence count, and whether truncation occurred.

## Local reference/uniform4/mixed comparison

Run from `/Users/jeremylien/Projects/paretoquant-metal`. The following commands are
ready to run **after setting the four absolute paths to the existing local
artifacts**; no remote repository identifiers are accepted:

```bash
CORPUS='/absolute/path/to/wikitext-2-raw-v1-test.json'
REF05='/absolute/path/to/unquantized-Qwen2.5-0.5B'
MIXED05='/absolute/path/to/mixed-Qwen2.5-0.5B'
REF15='/absolute/path/to/unquantized-Qwen2.5-1.5B'
MIXED15='/absolute/path/to/mixed-Qwen2.5-1.5B'

.venv/bin/python scripts/evaluate_quality.py \
  --corpus "$CORPUS" --corpus-name 'wikitext-2-raw-v1:test:prefix4096' \
  --reference "$REF05" --mixed "$MIXED05" \
  --window-length 512 --stride 256 --max-target-tokens 4096 \
  --group-size 64 --output results/quality-qwen2.5-0.5b-wikitext-prefix4096.json

# Run only after the first process exits; do not launch GPU jobs in parallel.
.venv/bin/python scripts/evaluate_quality.py \
  --corpus "$CORPUS" --corpus-name 'wikitext-2-raw-v1:test:prefix4096' \
  --reference "$REF15" --mixed "$MIXED15" \
  --window-length 512 --stride 256 --max-target-tokens 4096 \
  --group-size 64 --output results/quality-qwen2.5-1.5b-wikitext-prefix4096.json
```

`--reference` creates `reference_fp16` and `uniform4`; `--mixed` adds `mixed_stock`.
The FP16 variant casts the unquantized reference's floating weights to float16.
Uniform4 loads the **same reference again**, casts it to FP16, and uses
`mlx_lm.utils.quantize_model(..., bits=4, group_size=..., mode='affine')` in memory.
It applies to framework-eligible modules, not every parameter: dimensions,
architecture predicates, norms, and biases can leave weights unquantized. Actual
quantized module paths, bit widths, group sizes, modes, and parameter dtype counts
are included. Both derived variants retain the same **source artifact hash**;
the recorded transformation and effective configuration distinguish them.
Already-quantized reference sources are rejected, not dequantized and relabeled
as FP16. Native mixed artifacts load from their saved quantization configuration;
any dispatch manifest is hashed as source content but is **not executed**.
Fused-quality comparisons are outside this evaluator's scope.

Named preexisting local models can instead be compared with repeated mappings:

```bash
.venv/bin/python scripts/evaluate_quality.py \
  --corpus "$CORPUS" --corpus-name 'wikitext-2-raw-v1:test:prefix4096' \
  --model "fp16=$REF05" --model "mixed_stock=$MIXED05" \
  --window-length 512 --stride 256 --max-target-tokens 4096 \
  --output results/quality-named.json
```

Mappings load their native saved dtypes. `--uniform-bits 4` adds a fresh
`NAME:uniform4` companion to **each** named mapping; therefore use that flag only
with unquantized input sources. It cannot be combined with `--reference`.

Only one variant is loaded at a time. The previous model, tokenizer, loss callback,
and MLX cached allocations are released before loading the next. No unquantized
model is retained as a comparison baseline. This does not mean quantizing one
variant has no transient memory overhead: its source weights and new packed
weights may temporarily coexist during conversion. Corpus text and token ids
remain on the CPU; model inputs/logits are window-bounded.

The comparison refuses differing full or evaluated token-stream hashes or target
counts. Deltas and perplexity ratios are relative to the **first named variant**.
Do not combine different tokenizers/model families in one comparison if they
produce different token ids; run separate reports instead.

## Reproduction and report interpretation

Reports include:

- Raw corpus file SHA-256 and joined UTF-8 text SHA-256, exact separator, source and
  blank record counts, provenance label, and explicit unverified held-out status.
- Full and evaluated-prefix token SHA-256, using unsigned 64-bit little-endian
  token ids, plus the actual stream lengths.
- Model source content manifest: sorted non-hidden file paths, sizes, and SHA-256
  hashes. The aggregate model hash is SHA-256 of that manifest serialized as
  compact, sorted-key JSON. Snapshot symlinks are dereferenced; absolute directory
  names and hidden Hub/cache metadata do not affect the aggregate hash. Do not
  mutate model/corpus files during evaluation. Non-hidden auxiliary files inside
  the model directory also participate in its hash.
- Transformation, effective model configuration, actual quantized module policy,
  parameter bytes (not runtime peak memory), dtype counts, device, package versions,
  Python/platform versions, method settings, exact metrics, and per-window audits.

These identify the input and method, not a promise of bitwise-identical floating
point results across hardware or framework versions. PPL deltas are descriptive,
not confidence intervals, significance tests, or task-benchmark accuracy.

Errors exit with code 2 and do not write a report. Existing output reports are refused
before model loading to preserve prior evidence. Output cannot overwrite the
corpus or any file inside an evaluated model directory. Without `--output`, the
report is JSON on stdout; quantization messages go to stderr. For useful run
comparisons, keep corpus hashes, tokenizer streams, context length, stride, cap,
and software versions fixed.

## Tests and manual acceptance

The normal test file runs CPU fake-logit accounting, exhaustive boundary/stride
coverage, float64 aggregation, corpus/model hashing, local-only preflight,
comparison compatibility, help, and invalid-argument checks. Real-model tests
are marked `integration` and **skipped unless explicitly opted in**:

```bash
.venv/bin/python -m pytest tests/test_quality.py -q \
  --basetemp=/Users/jeremylien/.hermes/cache/scratch/paretoquant-quality-unit

PARETOQUANT_QUALITY_INTEGRATION=1 .venv/bin/python -m pytest \
  tests/test_quality.py -m integration -q -s \
  --basetemp=/Users/jeremylien/.hermes/cache/scratch/paretoquant-quality-integration
```

Manual acceptance constructs and saves a genuinely tiny, randomly initialized
one-layer Qwen2 model and local byte-level BPE tokenizer, reloads FP16, generated
uniform4, and saved mixed 3/4-bit artifacts through the actual offline loader, and
checks finite metrics and exact seven-target accounting. Weak references verify
that no previous model survives the next load. Tests select the **MLX CPU device**
so they do not compete with full-model GPU runs, and restore the prior device.
A separate test compares actual MLX scored losses against the NumPy float64
logit oracle. Neither test downloads a model or corpus.
