# 024 observed RED → GREEN record

All execution used the repository `.venv/bin/python` from
`/Users/jeremylien/Projects/paretoquant-metal`. Every pytest process selected
`mx.cpu` **before collection**. Every new MLX fixture selects CPU; queue workers
also select a fresh CPU stream. All TMPDIR/pytest basetemp paths were under
`/Users/jeremylien/.hermes/cache/scratch`. The inherited GPU parametrization was
deselected in full regressions. No GPU, SSH, deployment, original023 edits,
running batch polling, full checkpoint loading or commit occurred.

Each production slice followed an actual observed RED, not imagined output.
Command suffixes below run under `pytest.main([...,'-q','-p','no:cacheprovider',
'--basetemp=<scratch>/<unique-cycle>'])` inside the explicit CPU wrapper shown
in README. Full failures are retained in session tool results; this document
records concise actual reasons/counts.

| Slice / test | Observed RED | Observed GREEN |
|---|---|---|
| All-position normalized hidden, no projection; `test_hidden_prefill.py` | 1 failed: hidden-only returned `None` before norm | 1 passed in1.20s |
| New bounded command; first control test | 1 failed: `prefill_hidden` rejected by `Command.validate` | Command test passed before the next backend RED |
| Opt-in backend / shifted recurrence; `test_hidden_control.py` | After worker GREEN, backend failed with unexpected `hidden_only_prefill` argument | 7 passed in1.68s |
| Native/reference/optimized hidden/cache/head gate; `test_hidden_gate.py` initial three tests | 3 failed: missing `hidden_prefill_gate` | 3 passed in2.11s |
| Live option-bound admission | `DID NOT RAISE` when reference admission enabled optimization | Combined gate/control:11 passed in8.42s |
| Small checkpoint and loaded full runtime wiring; `test_hidden_runtime.py` | 2 failed: missing hidden receipts | 2 passed in3.26s |
| Explicit both-rank CLI and complete fixed registry; `test_hidden_cli.py` | 2 failed: missing CLI option and registry omitted code/config/kernel files | 2 passed in0.15s |
| Fail-closed receipt validator; `test_hidden_evidence.py` initial two tests | 2 failed: opt-in accepted reference receipts; validator missing | Combined evidence/runtime:4 passed in3.58s |
| Projection phase attestation; two-rank schedule mutation | `DID NOT RAISE` after replacing `hidden_norm_done` with `head_done` | Two real queue-backed rank tests, gate widths4/256:2 passed in7.46s |
| Mandatory native256 full path even with reduced forced chunks | 1 failed: reduced full gate had no256-position chunk | New full wrapper + representative256 receipt, combined17 tests passed in20.98s |
| Per-request trusted mode (parent review steering) | `DID NOT RAISE` for `slot-qmv` receipt when trusted mode was native | Mode mutation rejected; next loop exposed independent wall-TPS RED |
| Whole-request TPS consistency (parent review steering) | `DID NOT RAISE` for mismatched `wall_tokens_per_second` | 1 passed in1.45s after both checks |

One intermediate control-test failure was a test-fixture issue, not a production
change: the inherited cache-metric helper cannot slice an uninitialized one-token
prompt's empty head KV. The test now directly verifies both empty states and
zero offset, then uses complete cache metrics after nonempty prefill.

The first new two-rank CPU test failed because a lazily initialized test head
carried a main-thread CPU stream into workers. Evaluating its parameters before
starting workers fixed the fixture. No runtime arithmetic was changed for this.
The final real two-rank tests run native and reference MTP, then opt-in MTP on
identical retained models/head, with boundary and single-token requests; all
outputs/counters match and both queues drain. Explicit loading/watchdog envelopes
are unit fixtures, never presented as physical source/weight/cleanup receipts.

Formatting occurred only after GREEN. Initial Ruff found import ordering and
line-length issues; `ruff check --select I,F401 --fix` and `ruff format` corrected
these. Final `ruff check` returned **All checks passed!**; `ruff format --check`
reported **55 files already formatted**.

Regression progression:129 passed/1 GPU deselected in20.99s before the final
per-request evidence test; final **130 passed,1 GPU deselected in24.16s**.
Actual final stdout and JUnit results are `cpu-regression.txt` and `cpu-tests.xml`.
`cpu-hidden-prefill-proof.txt` contains separately executed tiny CPU small/full
all-position/cache/head/continuation metrics; it is not physical target admission.
The source/changed manifests contain tool-computed SHA256 values, not hand hashes.

## Independent-review cache-part fix and action-auditor integration

All commands below still forced CPU before collection and used scratch TMPDIR
and basetemp. No physical model, GPU, SSH, deployment, archived batch audit,
commit, or023/result mutation was performed.

| Review slice | Actual RED observed | Actual GREEN verification |
|---|---|---|
| Reproduce only-first-part-per-layer in **real** native CPU gate receipts | `test_review_reproduction_one_part_per_layer_must_not_admit`:1 failed in0.89s, **DID NOT RAISE ValueError** | Same regression1 passed in0.87s with native architecture inventory check |
| Finite/dtype/shape/tolerance consistency despite forged `strict_pass=true`, plus exact offsets | Initial matrix:63 failed,77 passed in1.67s. Missing guards accepted huge errors, integer dtype, nonfinite flag, scaled error >1, negative tolerance, inconsistent exactness, `Inf==Inf`, inexact/nonzero-error offsets | Full cache suite green; unsupported no-op parameter combinations subsequently removed rather than counted as tests |
| Detached both-rank receipts against hash-bound config, without trusting process-local registry | Real two-rank tests at widths4/256:2 failed in5.51s, `Missing live hidden prefill cache layout` on valid deep-copied receipts | Controller derives exact layout from trusted small/target metadata; both width4+native256 and default256 tests pass, including detached receipts and original phase mutations |
| Bind outer width ownership to actual hidden native layout |1 failed in1.76s, **DID NOT RAISE** when outer receipt claimed `[10,14]` for the same four layers | Exact native ownership check added; final suite passes |
| Bind reduced-width native256 receipt to exact live full model too |1 failed in1.58s, **DID NOT RAISE** for detached representative256 substituted into live receipt | Live representative receipt lookup/model/layout binding added; final suite passes |
| Integrate actual parent action-progress auditor, not a replacement | Three new `test_workload_evidence.py` review regressions:3 failed in1.34s, all **DID NOT RAISE** for zero-round/five-output, changed accepted count, wrong verify-current token; both rank/result copies remained consistent | Mandatory `validate_mtp_actions` called per scheduled MTP request on each owner. All three now reject |

`test_hidden_cache_inventory.py` individually deletes **every part** in every
small/full/native256 applicable row at live Admission, and exercises each unique
part's deletion/duplication/unknown/replacement-duplication at the detached
validator. All required KV offsets, both GDN tensors, dtype/shape/numeric metadata
and strict scalar offsets are covered. A valid enabled backend is created before
the negative mutations; another model's live receipt and detached JSON cannot
mint an enabled live admission. Native256 and all original gates remain required.

Additional actual CPU tests verify float32/bfloat16 target caches and both the
small float32 fixture head and target-dtype full head against independently
hash-bound metadata layouts; tampering the config fails its trusted digest.
The producer registry stores no arrays or strong model references. It binds the
expected inventory to native `make_cache()` classes and architecture dimensions,
not to claimed metric parts; warm native reference supplies activation dtypes.

The actual parent's `mtp_audit.py`/`test_mtp_audit.py` were copied only after its
`prefill_hidden` spelling fix. Their Python ASTs are verified identical after
local formatting. Independent execution before copying:74 unittest tests, OK,
1 archived-evidence test skipped. Copied tests remain in the full pytest suite.
The auditor checks recorded progress/grammar/counters/EOS, **not teacher greedy
prediction**, and never substitutes for physical numerical gates.

Final real output:

- Original130 CPU regressions isolated: **130 passed,4 deselected in18.85s**
  (GPU plus3 newly added review tests excluded), `cpu-original-regression.txt`.
- Expanded suite: **331 passed,1 skipped,1 GPU deselected in22.70s**,
  `cpu-cache-regression.txt` and `cpu-cache-tests.xml`. The skip is the explicitly
  opt-in archived physical evidence audit. No GPU test was run.
- Ruff check: **All checks passed!**; format check: **58 files already formatted**.
- Current standalone actual CPU receipts/layouts/source hashes:
  `cpu-cache-inventory-proof.txt`; small lengths256/257/511/512/513, full4 with two
  chunks and mandatory representative256. These are tiny CPU proof only.
- Registry contains **60 exact source entries**;22 changed/new relative to023.
  Both SHA256 manifests were recomputed after all registered code changes.
  Eight inherited arithmetic/kernel files remain byte-identical to023.

The original130-test snapshot artifacts above remain preserved, not relabeled as
current-source proof. Physical both-host small and same-loaded full gates,
headroom/deadline checks, old/native output parity, and paired whole-request
timing remain parent-owned and unperformed. The registered schedule stays empty.

