# 009 — archived 32B trajectory / small-draft acceptance

## Verdict: PARTIAL — real GPU integration, no 32B speedup established

`draft_screen.py` is an **archived-target-trace acceptance diagnostic**, not a
speculative engine. It loads only the local small draft model. It never loads 32B weights,
calculates target logits, performs target batch verification, or establishes
acceleration, full-model performance, or quality. Acceptance alone cannot predict
speedup. CPU fixtures and mocked adapter results are **not model evidence**.

`draft_screen.py` is standalone and imports only the standard library until CLI,
reference, target config/receipt, and small-draft config checks have passed.
No installation, download, network, SSH, or credentials are needed.

## Parent run

Use the parent's existing MLX environment on the admitted small-model host:

```sh
python -B draft_screen.py \
  --draft-model /Users/jeremylien/Projects/paretoquant-dual-host/small-model \
  --target-tokenizer /absolute/path/to/copied-target-metadata \
  --reference-json /absolute/path/to/copied-rank0.record.json \
  --target-receipt /absolute/path/to/target-receipt.json \
  --output /absolute/path/to/NEW-draft-acceptance.json \
  --block-sizes 1 2 4 8 --max-seconds 150
```

`--target-receipt` defaults to `TARGET_METADATA/target-receipt.json`. It must
be a bounded local JSON receipt with these explicit fields:

```json
{
  "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
  "revision": "2938092373e5f97b95538884112085364c2da315",
  "config_sha256": "c027829d800805358d67ac87819a3754fd8240be973f7147840651310fd30ae3",
  "reference_sha256": "ACTUAL_SHA256_OF_THE_COPIED_REFERENCE_FILE"
}
```

The placeholder is not valid data. Parent must calculate the reference digest
with `hashlib.sha256(path.read_bytes()).hexdigest()` and retain the source receipt
it used to assert model/revision. Receipt binding is **not independent verification
of archived target execution**. Target config bytes must match the exact pinned
archived digest above; semantic equality after reformatting is insufficient.
Target directory must contain metadata only, including the actual target tokenizer
files and chat template; no `.safetensors`. No target weight files are read.

The actual existing draft was identified by the parent as ParetoQuant **mixed
3/4-bit 0.5B**, not uniform 4-bit: config SHA256
`43a1c2f3b692e8cca5974f4753514e94ee4c969f8853b6d02acc16c7e64805da`,
260563712 stored tensor bytes, 628 tensors, tied embedding, vocab 151936. Target
config vocab is 152064. Runtime records its own loaded tensor layout/countable
bytes and complete source config rather than assuming this inventory.

## Contract and interpretation

- Reference: `requests[0].token_ids`, nonempty exact integers in target vocabulary,
  genuine `finish_reason="stop"`, one terminal EOS and no earlier EOS. Input must
  be a regular non-symlink file of at most 1 MiB. Maximum reference length: 256.
- Raw prompt comes from original top-level `prompt`; without it an explicit
  `--prompt` is mandatory. An explicit prompt must agree with an existing archived
  prompt. The canonical archived prompt is `What is 2 plus 2? Explain briefly.`
- Existing `formatted_prompt` is authoritative. Regenerating the **target** chat
  template must agree byte-for-byte. No draft chat template is used. Encoding uses
  the same BOS / `add_special_tokens` convention as `phased_generate`; archived
  `prompt_tokens`, when present, must match. Prompt plus reference is capped at 512.
- Every used prompt/reference/proposal ID must be in both model vocabularies and
  have exactly the same tokenizer token string. All common tokenizer IDs are also
  checked. The smaller padded draft vocabulary is allowed; unsupported/mismapped
  IDs fail closed. There is no retokenization or ID repair.
- Each K gets a separate trajectory walk. Each round makes a fresh draft KV cache,
  recomputes `target_prompt_ids + reference_tokens[:cursor]`, then greedily drafts
  up to K tokens (capped by reference remaining and draft EOS). MLX synchronization
  brackets actual draft timing and each selected token is evaluated before `.item()`.
- Matching prefix length is `m`; hypothetical emitted count is
  `min(m + 1, remaining)`, except a matching EOS emits only through EOS. The next
  round advances along archived tokens by that count, **never rejected draft tokens**.
  Every proposal, reference window, accepted prefix, emitted count, prefix length,
  and elapsed sample is retained, plus aggregate actual counts per K.
- These hypothetical rounds are not actual verified target rounds. Times include
  fresh-cache prefix recomputation; they are not optimized draft decode latency.
- Source binding uses bounded regular-file SHA256 inventories before/after the
  run plus stock `mlx_lm.load(lazy=False, trust_remote_code=False)`, with
  `tokenizer_config={"local_files_only": true, "trust_remote_code": false}`.
  This is not a separate extraction/hash of loaded packed arrays. Draft directory
  is capped at 512 MiB/64 root files, target metadata at 32 MiB/64 root files;
  root file symlinks are refused. Loaded draft parameters are also capped at 512 MiB.
- Output includes harness hash, target tokenizer/config inventory, receipt/reference
  digests, draft source inventory/config/layout, Python/library versions, MLX device
  details, memory, and explicit limitations. MLX memory is not OS residency; native
  `ru_maxrss` units are reported without a cross-platform conversion claim.
- Block sizes: default `1 2 4 8`, unique values 1..16, at most eight. Cooperative
  deadline is 150 seconds by default, at most 180; it cannot interrupt an in-flight
  model load/GPU operation. Parent should use its existing external 180-second
  process guard. Deadline/errors leave no completed acceptance JSON. Output must
  be NEW and is created with exclusive `x`; no existing result is overwritten.

## CPU verification / vertical TDD record

```sh
python3 -B test_draft_screen.py -v
python3 -B draft_screen.py --help
```

Tests use stdlib `unittest`, explicit synthetic fixtures and unavoidable runtime
adapter doubles. No real MLX/model/NumPy import or GPU operation is performed by
this worker. The provenance preparation test reads the repository's archived
config/reference **metadata only**, so run the full test suite in the repository;
the standalone harness itself does not depend on the repository package.
Temporary fixtures live under `~/.hermes/cache/scratch` and are removed.

Observed test-first slices (each RED was run before its implementation):

| Slice | Observed RED | Observed GREEN |
|---|---|---|
| Acceptance / bonus / EOS | 1 test, missing utility failure | 1 passed |
| Bounded reference / IDs / prompt / stop | 2 tests, missing validation failure | 2 passed |
| CPU preparation / pinned provenance / CLI bounds | 3 tests, missing preparation failure | 3 passed |
| Archived target template / exact ID compatibility | 4 tests, missing prompt path failure | 4 passed |
| Teacher-forced archived trajectory rounds | 5 tests, missing rounds failure | 5 passed |
| Synchronized greedy runtime adapter / fresh cache | 6 tests, missing adapter failure | 6 passed |
| Bounded source inventory binding | 7 tests, missing inventory failure | 7 passed |
| Integrated draft-only runtime / audit report | 8 tests, missing runtime failure | 8 passed |
| Exclusive CLI output / CPU-safe help and import | 10 tests, 2 expected CLI failures | 10 passed |

Inventory GREEN initially exposed an `OSError` rather than the required explicit
symlink rejection; the rejection was corrected and all seven tests passed.
Final recorded suite: **10 tests passed**, 0.382 seconds, exit 0. Tests actively
block `mlx`, `mlx_lm`, and `numpy` imports for subprocess help/import/invalid CLI.
AST parsing and `ruff check --no-cache --select F draft_screen.py test_draft_screen.py`
also passed. The full repository Ruff style profile flags compact formatting,
long lines, and import ordering; it was not claimed to pass. The harness is 264
lines, kept near the requested roughly 250-line bound rather than expanded into
an engine. No real acceptance result or acceleration claim has been produced by
this worker.

### Parent evidence and opt-in component

Parent completed the real M4 draft diagnostic; evidence is in
`../../results/m4-draft-32b-trace-v1/`. K4 required 13 hypothetical verification
rounds for the 41 archived token events with 0.502 s draft-only work. These are not
actual 32B verification rounds or a measured speedup. Full lint/format passed after
parent style cleanup. Independent diagnostic/head review passed for that narrow
scope; its original worker-only statements above are the historical TDD record.

`speculative_decode.py` adds a separate opt-in fresh-cache verification loop;
`test_speculative_decode.py` covers CPU packet/acceptance/cache contracts and
`test_speculative_gpu.py` is a parent-only actual tiny-GPU fixture, executed from
a stage colocated with the admitted `phased_decode.py`. It is not part of the
original default decoder or guarded32B CLI.

`../../results/m4-speculative-integration-v1/` contains the real integration gate:
80 tiny GPU cases passed. Six authentic 0.5B self-draft cases matched sequential
teacher tokens, stop and offsets, but all six failed the unchanged live-KV numerical
tolerance. The component therefore remains partial, not performance-admitted;
independent component CPU review passed; actual two-rank/32B controls remain pending.
Full32B work remains paused for parent host resource pressure; no acceleration or
quality claim follows from this spike.

### Width-one teacher diagnostic after reconnection

`verification_mode="sequential"` is an exact-string opt-in to run each teacher
verification input at width one; the default remains `"batch"`. It preserves
acceptance, rollback, draft catch-up and EOS handling. Receipts/events identify
the mode; `target_forward_calls` counts individual forwards while
`target_verification_calls` counts rounds. Sequential verification does not
amortize teacher computation and is a diagnostic, not acceleration.

`../../results/m4-sequential-verification-v1/` preserves the physical M4 rerun:
80 tiny packed-GPU cases passed in each mode. On the unchanged authentic
mixed-3/4-bit 0.5B checkpoint, all six width-one self-draft cases passed token IDs,
stop reasons, every-layer offsets, live teacher/draft KV and accounting; maximum
cache difference was zero. The paired six batched cases again matched outputs
but failed the unchanged `.002` live teacher-KV tolerance, with maximum absolute
difference `.125`. This strengthens localization to width-dependent teacher
execution without proving a specific numerical kernel mechanism. CPU gate:
36 passed, one explicit optional GPU skip, 2,653 subtests; lint/format passed.

The original guarded32B sources were read back unchanged. No post-extension
independent review, two-rank speculative deployment, real32B speculation, quality
evaluation or speedup is established. The last post-run M2 preflight had pressure
level2 and only 753 MiB free disk; full32B remained resource-blocked at that check.

On October 6, after resource recovery, the unchanged guarded group8 decoder
completed four physical full32B requests over the upgraded 40 Gb/s cable with
historical token/tensor/source equality and verified cleanup. Evidence is in
`../../results/m2pro-m4-cable40-32b-v1/`: bulk TCP smoke measured 37.552/37.317 Gb/s,
while warmed post-first decode was 6.314 events/s versus the historical 6.270.
This is a separate stock-grouped runtime rerun, not speculative deployment or
admission; the speculative batch numerical gate remains failed.
