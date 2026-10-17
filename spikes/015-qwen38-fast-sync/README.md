# 015: opt-in MLX Metal fast synchronization

Standalone byte-exact copy of 012 before edits, with minimal runtime changes only
in `supervise.py`, `remote_actor.py`, and `probe.py`. No 012 file or archived trial
was modified. Initial copy hashes, red/green logs, final hashes, and runtime diff
are under `results/qwen38-speed-v2/fast-sync-unit-baseline/`.

**CPU contracts verified; Metal execution and physical trials are not verified by
this implementation. No throughput improvement or >=10 tokens/s claim is made.**

## Upstream basis and risk

Pinned MLX 0.32.3 upstream references, archived in the evidence directory:

- `https://github.com/ml-explore/mlx/blob/v0.32.3/docs/src/usage/distributed.rst`
- `https://github.com/ml-explore/mlx/blob/v0.32.3/docs/src/usage/environment_variables.rst`
- `https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/utils.h`
- `https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/fence.cpp`

The distributed documentation describes faster CPU/GPU synchronization, applicable
beyond JACCL, but explicitly warns about deadlock and a wedged GPU (issue 3142),
recommending leaving the variable unset. The parent's primary-source issue 4438
review also warns that this path is not guaranteed and is not recommended by the
Metal team. Treat it only as a bounded risky experiment. The environment accessor defaults to 0
and caches the value. The Metal fence implementation checks GPU-family support
and macOS 15/iOS 18 availability before using that accessor. This experiment sets
the environment before child process execution, never by mutating it after MLX
initialization. Environment readback does **not** attest which internal Metal
path was selected, nor that the driver remains healthy after a timeout.

## CLI and environment contract

`supervise.py` adds only `--fast-synch`, an opt-in `store_true` flag. Omit it for
the default, **unset** environment; do not pass `false`, `0`, or `=false`.
All existing 012 options, scheduling rules, split/weight behavior, fresh caches,
stock operators, and bounds are preserved. Example argument suffix for the parent
(after separate memory admission and all ordinary trusted launch arguments):

```sh
--execution phased --split 36 28 --eval-group-size 32 --repeats 3 --fast-synch
```

There is no corresponding probe CLI flag. The supervisor explicitly places
`MLX_METAL_FAST_SYNCH=1` in both `rank0` and `rank1` jobs, never the weight server
job. The actor permits that exact value only for a probe with explicit rank and
owned hostfile. Values such as `0`, `false`, unknown names, or non-string values
fail admission. The existing inherited-environment filter is unchanged: a setting
in the supervisor or SSH shell cannot enable a default-off rank accidentally.
Direct standalone probes admit only unset or exactly `1`.

The admitted boolean is stored as `startup.json.fast_synch` (also in `args`) and
`result.json.fast_synch`, including failed trials. Each probe reads its actual
`os.environ` before any MLX import, emitting in `process_start`:

```json
{"fast_synch": true, "environment": {"MLX_METAL_FAST_SYNCH": "1"}}
```

It reads the actual environment again for the final rank result, overriding any
claimed fields returned by the runtime. Default records use `false` and JSON
`null`, not an environment value of `0`. The controller passes its **trusted**
boolean to `verify_trial`; rank agreement alone is insufficient. Both rank
results, and their source-bound process-start records on the live controller
path, must match it exactly. Missing/unknown/false/forged values fail before
success. Successful verification includes `rank_environment_verified: true`.
Archived 012 receipts lack this new evidence and are not valid 015 receipts.

No runtime dependency was added. `remote_actor.SOURCES` remains the same seven
files. The trusted inventory, metadata, source, and closed-listener helpers remain
unchanged. Tensor-name/digest/replication checks, ordered scheduling phases, token
agreement, ownership, deadline/reserve checks, nonce-owned cleanup, and strict SSH
host verification are preserved. Log retrieval remains **8 MiB per actor log**,
controller responses **20 MiB**; source and job bounds are unchanged. Keep physical
trial request counts short rather than weakening these limits.

## CPU verification (safe without MLX)

Run only the explicitly selected CPU files, not the whole directory:

```sh
export TMPDIR=/Users/jeremylien/.hermes/cache/scratch
.venv/bin/python -m pytest -q \
  spikes/015-qwen38-fast-sync/test_fast_sync.py \
  spikes/015-qwen38-fast-sync/test_schedule.py \
  spikes/015-qwen38-fast-sync/test_supervise.py \
  spikes/015-qwen38-fast-sync/test_remote_actor.py \
  spikes/015-qwen38-fast-sync/test_progress.py
.venv/bin/ruff check spikes/015-qwen38-fast-sync
.venv/bin/ruff format --check spikes/015-qwen38-fast-sync
```

CPU tests include actual actor/exec subprocesses using explicitly labeled stdlib
substitute probes, plus synthetic supervisor transports/receipts. They are not
MLX, distributed-network, numerical, or benchmark evidence. The inherited 007
supervision suite was copied and adapted only for canonical Qwen3.5 names, 012
request fields/group range, and the new environment receipt fields. The copied
012 schedule fixture was extended with the same environment fields. No existing
safety assertion was removed. The two original GPU test files remain byte-exact.

## Parent-only deployment and physical acceptance

1. Deploy all seven `remote_actor.SOURCES` files plus `supervise.py` to a new,
   immutable `015-qwen38-fast-sync` directory on the mini; read back exact hashes
   for all eight. The sources must come from 015, including unchanged files.
2. Re-run the copied `test_qwen38.py` and `test_long_blocks.py` Metal fixtures in
   fresh processes on each actual GPU, first unset and then with
   `MLX_METAL_FAST_SYNCH=1`. Preserve packed F32/BF16 logits, greedy tokens,
   recurrent/KV-state checks and unchanged numerical tolerances. These files
   require a working Metal GPU; the implementing subagent did not run them.
3. Gate real two-rank small-checkpoint generation against native single-host
   controls before loading the full requested Qwen3.8-27B 4-bit checkpoint. Recheck
   memory/disk headroom, trusted checkpoint inventory and all deployed sources.
4. Use a fresh output directory and existing durable bounded supervisor command,
   pointing `--remote-root` at deployed 015. Add `--fast-synch` only for the
   candidate. Use contemporaneous default-off controls and unchanged prompts,
   split/grouping/token limits; separate cold and warmed fresh-cache requests.
5. Read back `startup.json`, both `rank*.job.json`, both authoritative actor
   receipts (stdout final result and stderr process-start record), source hashes,
   `result.json`, and cleanup/listener receipts. Require true/`"1"` on both ranks
   for the candidate and false/null for the control, together with
   `rank_environment_verified`, original inventory/tokens/schedule gates, and
   verified cleanup. Do not infer health or success from a transport exit alone.

Deadlock/GPU-wedge failures must be retained as failures even if the actor is
reaped; the process/group receipt explicitly does not inspect GPU driver state.
The parent serializes hardware jobs and owns all deployment/SSH/physical trials.
