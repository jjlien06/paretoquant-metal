# 012: Qwen3.8 request-interleaved evaluation scheduling

Copied standalone runtime and tests from 011. This changes evaluation scheduling
only: identical stock quantized operators, tensor precision, pipeline ownership,
CPU handoffs and fresh native KV/SSM caches per request. No full-model or physical
host performance claim is made by this spike's tiny-fixture tests.

## API

Both `probe.py` and `supervise.py` accept:

- `--eval-group-size {1,2,4,8,16,32}`: default 1; preserved as the top-level
  `eval_group_size` even when a schedule overrides individual requests.
- `--eval-group-schedule [GROUP ...]`: optional space-separated integer list.
  A nonempty list requires phased execution and exactly `--repeats` entries.
  Empty or omitted means use the default group size for every request.
- `--repeats`: 1..32. The supervisor retains its maximum 240-second deadline.

Example arguments for a same-process interleaved comparison (append to the
existing admitted 011-style launch configuration, but deploy this 012 source):

```sh
--execution phased --eval-group-size 4 --repeats 8 \
  --eval-group-schedule 4 8 8 4 4 16 16 4
```

Use `--phased` instead of `--execution phased` for the probe. Both ranks receive
exactly the same arguments. Each request receipt and request phase records its
actual `eval_group_size`; the top-level `eval_group_schedule` records the supplied
list (or `[]`). Each request creates a fresh native prompt cache and resets the
same RNG seed. First requests may include compilation/cache warmup; analyze
interleaved timings accordingly rather than presenting cold time as steady state.

The controller admits the schedule before any job is launched. Final verification
binds every request's index and group to that trusted schedule, checks rank
execution and default metadata, requires ordered request boundary phases, checks
every repeat-tagged phase's group/rank, and binds process-start source hashes to
the admitted runtime sources. Existing inventory, token agreement, metadata,
actor-source, ownership, deadline/reserve and strict owned cleanup checks remain.
No helper file was added; `remote_actor.SOURCES` is unchanged.

## Verification

```sh
.venv/bin/python -m pytest spikes/012-qwen38-scheduling -q
.venv/bin/ruff check spikes/012-qwen38-scheduling
.venv/bin/ruff format --check spikes/012-qwen38-scheduling
```

Tests use tiny 4-bit quantized hybrid F32/BF16 GPU fixtures: all logits, greedy
tokens, all native KV and recurrent SSM state, chunked prefill/decode, groups
1/2/4/8/16/32 and ragged ownership. Handoffs are simulated locally; no remote
execution or heavy checkpoint loading is needed. Schedule admission, invalid CLI,
source registry and tampered request/phase/source receipts have separate tests.

See `../011-qwen38-dual-host/README.md` for the base architecture and previously
verified physical-host result; those measurements are not 012 measurements.

## Parent physical-host validation

The parent subsequently passed the two-Mac small-checkpoint gate and full-model
trials. See `../../results/qwen38-speed-v1/README.md` and `fast-profile.json` there.
A 36/28 split with group32 observed 12.8% higher warm decode throughput and 10.1%
lower request wall in a short native/fast/native counting pilot; all nine
completions had identical 70 event IDs including EOS and unchanged tensor hashes.
This is a modest separate-process pilot, not statistical significance or a
universal speed/quality claim. Larger local shards require fresh memory admission.

The parent also added a real 35-layer BF16 hybrid fixture for complete/ragged
16/32-layer block coverage; the scheduling suite now passes 83 tests.

Keep evidence volume bounded. A 12-request full-model schedule completed on both
GPUs but exceeded the inherited 8 MiB per-log retrieval limit; the original
controller failure is preserved, with separately hash-verified disk-log recovery
and original admission/cleanup checks. Shorter three-request physical trials
completed normal controller verification. Do not silently remove retrieval limits.

