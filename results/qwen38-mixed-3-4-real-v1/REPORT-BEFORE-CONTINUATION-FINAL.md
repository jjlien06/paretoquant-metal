# Qwen3.8-27B real-activation mixed 3/4-bit experiment

## Authorized continuation / amended Stage 1 gate

Jeremy authorized continuation with BF16 fidelity rather than bit-exactness:
all per-position argmax IDs identical, max absolute logit difference <=0.125,
and mean absolute difference <=0.001. The original failed gate remains below
and in its original receipts; no historical result was relabelled.

The first fix, `stream_v2.py`, removes 16-row head chunking: all positions in a
sequence use one call, as do all decoder projections. On the mini, native versus
streamed control-model logits are **bit-exact** at lengths 17, 64 and 256; all
24 layers' hidden-state relative RMSE and max absolute differences are zero in
each case. An instrumented hidden-capture pass also matches the pristine native
logits exactly. BF16 dtype is checked. Thus this candidate passes both the amended
and original gates on these control cases. Receipt: `small-parity-v2.json`.
Real 27B calibration is running under the original resource rules; final results
will replace this continuation status, not the preserved history below.

## Original stopped attempt (historical record)

Owner: Jeremy. Experiment date: 2026-11-06.
Source: `spikes/033-qwen38-mixed-3-4-real/`.

## Verdict

The streamed-forward candidate is **not numerically admitted**. Following the requested
stop-at-first-failed-gate rule, no 27B calibration, real allocation, mixed checkpoint
assembly, or perplexity evaluation was run. There are no measured sensitivity rankings,
quality deltas, or mixed-checkpoint speedup results in this experiment.

### First numerical gate: rejected

On the M4 mini, compare the existing Qwen3.5-0.8B control checkpoint's normal native
`mlx_lm.utils.load_model` forward with the new one-decoder-layer-resident implementation:

| Item | Observed |
|---|---:|
| Input length | 17 tokens |
| Compared logits | 1 × 17 × 248,320 |
| Maximum absolute logit difference | 0.0625 |
| Mean absolute logit difference | 0.0005839725490659475 |
| Per-position argmax IDs | All 17 match |
| Fixed allclose tolerance | atol=1e-5, rtol=1e-5 |
| Numerical gate | **FAIL** |

The planned 64-token second case was not executed after this failure. Matching argmax
IDs is not a substitute for the failed logit gate and is not a generation-quality result.
No tolerances were relaxed and no full-model experiment was started.

Exact result: `remote-evidence/small-parity-r2.json`.
Actor output: `remote-evidence/small-r2.log`.
Watchdog: `remote-evidence/small-guard-r2.jsonl`.

The cause has not been isolated. One concrete diagnostic lead in the candidate source is
that `project()` uses vocabulary-projection chunks of 16 rows, whereas the reference
projects the 17-token input together; this is not yet a controlled comparison of identical
projection shapes. Explicit layer evaluation can also change lazy execution boundaries.
Neither possibility is a proven explanation. Any follow-up should isolate these at the
original tolerance, preserve this rejection, and rerun the small gate before calibration.

## Work actually completed

### Pinned 3-bit source downloaded and verified on the mini

- Repo: `lukaskremla/Qwen3.8-27B-3bit-MLX`
- Revision: `b3503b31b36125107e12a2b57fbe690c37bfc053`
- Downloaded files total: **12,719,810,876 bytes** (11.846247 GiB), including metadata,
  tokenizer assets and all three weight shards. This is download size, not text-only
  resident model memory.
- All three safetensors shards matched pinned HF LFS byte sizes and SHA-256:
  - `model-00001-of-00003.safetensors`: 5,357,835,580 bytes;
    `c7dc04780ba5616be8f00ea9b4669c3e42522205cc511f04d51d56fe46f5355c`
  - `model-00002-of-00003.safetensors`: 5,368,439,377 bytes;
    `4739a929dd9515ac6922aa85d866561427c028e971bc2a82de5fb96792754b79`
  - `model-00003-of-00003.safetensors`: 1,966,597,268 bytes;
    `46b1c69035fd973c06ad8ce438aaf62efa92c86589c8d1a770b107cbe14731e0`
- Retained path on mini:
  `/Users/jeremylien/Projects/paretoquant-dual-host/spikes/033-qwen38-mixed-3-4-real/q3-source/`

Receipts: `remote-evidence/q3-hf-metadata.json`, `q3-download-receipt.json`,
`download.log`, and `download-guard.jsonl`.

The unchanged 4-bit target remains revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.
It was not loaded for a 27B forward in this experiment.

### Calibration/evaluation data frozen, not evaluated

WikiText-2 raw, `Salesforce/wikitext` revision
`b08601e04326c79dfdd32d625aee71d232d685c3`:

- Calibration: first 3,072 tokenizer IDs from the **train** split.
- Held-out evaluation: first 6,144 tokenizer IDs from the separate **test** split.
- Planned evaluation: 24 independent, nonoverlapping 256-token sequences, no BOS,
  reset state/positions per sequence; score 255 next-token targets per sequence
  (6,120 total). This is a capped, short-context NLL/perplexity protocol, not a
  canonical full WikiText score.
- Local tokenizer: `models/qwen3.8-27b-4bit-tokenizer-only/tokenizer.json`.
- Tokenizer SHA-256:
  `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`;
  matches the downloaded q3 tokenizer file.
- Source split isolation is established by distinct train/test parquet artifacts;
  this is not a claim that natural-language phrases never repeat between splits.

Manifest and token IDs: `spikes/033-qwen38-mixed-3-4-real/data/`.
Source parquet and selected-token hashes are recorded in its `manifest.json`.

### Implemented candidates, not full-model-validated functionality

- `stream.py`: native `DecoderLayer`, masks, embedding, norm and quantized head;
  uncached independent-sequence execution, one decoder layer resident at a time.
  For proposed calibration it compares q3 decoder linears against q4 on the actual
  q4 hidden states, retaining q4 floating parameters. Metric: summed squared
  output error divided by summed squared q4 layer output. Planned spread gate:
  interdecile ratio >=1.5 and coefficient of variation >=0.15. **No sensitivities
  were measured**, because small-model admission failed first.
- `allocate_real.py`: the project's unchanged exact allocator, 64 per-layer
  {q3,q4} units, real header costs, target savings 0.5/1.0/1.5/2.0 GiB. Bounded
  frontier; overflow is an explicit failure, not an unlabelled approximation.
  **Not run on real sensitivity data.**
- `build_mixed.py`: bounded CPU raw-tensor assembly, per-module MLX quantization
  overrides, q4 fixed embedding/head/norm, source/output tensor hash checks,
  recursive tokenizer-asset copies, exclusive outputs. **No 27B mixed checkpoint
  was assembled or reloaded.**
- 14 CPU tests passed, covering tiny-fixture allocation against exhaustive search,
  overflow, bounded reads, source selection, config overrides, reference safetensors
  reading, output collisions and corruption detection. These do not admit 27B
  arithmetic or quality. See `cpu-tests.log`.

Planned bandwidth-only prediction: q4 text decode payload divided by mixed decode
payload, including the q4 output head/norm and excluding the embedding read.
**No new predicted plan values exist**, because allocation was not reached.

## Resources, failure preservation and cleanup

Only the mini ran MLX inference. All local work was CPU data preparation, code,
tests and SSH transport; no local model was loaded. The mini watchdog checked
pressure and direct-child RSS, allowing normal1/warning2 and rejecting other or
invalid pressure, excessive gaps, deadlines and RSS >=3 GiB. Per-step
`memory_pressure` output and MLX allocator/RSS high-water data are in the actor log.
The numerical attempt's sampled peak RSS was **530,726,912 bytes**; all its sampled
pressure values were **2**. This is sampled monitoring, not a continuous safety guarantee.
The guard reports the failed actor reaped; the download actor completed and was reaped.

Other preserved setup failures:

1. Local `.venv/bin/python -m pip ...` failed because that interpreter has no pip.
   Used `uv pip --target spikes/033-qwen38-mixed-3-4-real/deps` instead, without
   changing the existing environment. PyArrow 26.0.1 was used for parquet reading.
2. First mini launch failed before inference because the imported builder required
   `paretoquant.allocator`. Preserved in `remote-evidence/small.log` and
   `small-guard.jsonl`. Deployed a byte-identical allocator dependency snapshot
   under the new spike and supplied its `PYTHONPATH`; no existing source changed.
3. Initial cleanup probe had an overescaped relative-script regex; retained that
   probe and reran the corrected `cleanup_check_v2.py`. Final assertions use v2.

Final `pgrep` readbacks found **no experiment workers on either host**:
`cleanup-mini-v2.json`, `cleanup-m2-v2.json`. The pre-existing mini chat rank
PID 22670 was still present; local chat rank PID 91269 still listened on
`127.0.0.1:8780`. The chat was not stopped, restarted or sent inference requests.
Final free disk was about 57 GiB on the mini and 50 GiB locally; both hosts
remained at warning pressure2. No unrelated process was killed.

Remote evidence was copied back with SHA-256 equality checks (`evidence-transfer.json`).
Deployment manifests bind the exact executed source and data. New repository files
are confined to the requested spike/results directories; pre-existing dirty work
was left alone. No git commit was made.

## Remaining stages

First diagnose and pass small-model full-logit parity without weakening its gate.
Then run real-activation calibration and its spread gate, exact allocation, 1–2
mixed builds with real reload/source-byte verification, and identical held-out
perplexity for uniform4/mixed/uniform3. Only after those gates: two-Mac timing,
actual per-rank byte placement and memory admission. No quality or performance
claim is supported yet.
