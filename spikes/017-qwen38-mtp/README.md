# 017 — isolated native Qwen3.8 MTP reference prototype

**Status: tested CPU head + greedy engine + native phased adapter, BLOCKED as a
real two-host accelerator. No speed or real-target losslessness claim.** Parent's
physical M4 small-checkpoint native width2 gate FAILED (strict all-cache/logit and
secondary BF16 logit diagnostics), despite matching short greedy IDs. M2 passing
does not override that failure. Do not deploy native width2 MTP on both hosts;
a separately validated batch-invariant verifier is required. All changes live
here. No GPU, SSH, new weight download, target-weight edit or archive edit.

## Run CPU tests

From the repository root:

```sh
.venv/bin/python -m pytest spikes/017-qwen38-mtp -q
```

Every feature slice was first exercised failing for its missing module, then
implemented and rerun green. The final CPU suite includes:

- Native full-attention head equation, shared target embedding/output object
  identity, position/KV offset checks, explicit raw-delta versus absolute RMSNorm
  conversion, strict Q4 packed load round-trip/name/shape/dtype coverage.
- Trusted exact sidecar config/revision/SHA binding and mismatched target rejection.
- Real packed **tiny hybrid Qwen** CPU phased/native logits and every live cache;
  opt-in **normalized** final hidden return; native width2 and phased width2 versus
  sequential native width1 at unchanged `atol=1e-6, rtol=1e-5`.
- Reference accept/reject/EOS/cap/fresh-cache cases with two mutable rank fixtures;
  real packed tiny hybrid head+engine generation versus native greedy, identical
  final cache convention (generated final token is emitted, not consumed).
- Native `ArraysCache` recurrent arrays, conv state, lengths, padding and native
  `KVCache` keys/values/offset snapshots survive in-place mutation and restoration.
- Ordered bounded rank commands and exact fixed source-registry admission.

CPU tiny hybrid fixtures use `model.train()` solely to select upstream CPU
recurrent arithmetic. This is NOT the GPU inference kernel and not a GPU gate.

## Exact head identity and primary contract

Target remains `mlx-community/Qwen3.8-27B-4bit` revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.

Sidecar: `mlx-community/Qwen3.8-27B-MTP-4bit` revision
`b643c01b6d3b094e325edb6ebd832e16c486c575`, 238934137 file bytes,
SHA256 `76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6`.
`sidecar-header-evidence.json` is an independent CPU file-hash/header inspection
of the parent's downloaded file: 31 tensors, no private embedding/lm_head.
`load_trusted_head` checks the exact config digest and file digest before loading.
It does **not** independently attest already-loaded target tensor bytes: preserve
012's full trusted target inventory/config/replicated tensor attestation.

Primary immutable sources are copied into `primary_sources/`; their URLs and
hashes are in `registry.json`. The standalone mlx-vlm head consumes **normalized
final target hidden**, embedding first in concat; seven standalone norm vectors
are already absolute. Do NOT apply +1 again. MTPLX's pre_norm default belongs to
a different loader contract and is not adopted. The cloned phased function only
adds boolean `return_hidden`; defaults and all target operators stay unchanged.

Prefill processes the native chunked prefix and separately the final prompt
input. Each prefix hidden row at position i is paired with prompt token i+1;
head offsets start at zero. The last prompt hidden is paired with first teacher
token y only when drafting. After accepted `[y,d]`, catch up the head with
`(normalized hidden_y, d)`; next draft pairs `(normalized hidden_d, bonus)`.
No speculative draft hidden substitutes for verified target hidden.

## Minimal usage seam (caller selects device and already loads target)

```python
from mtp_head import load_trusted_head, TARGET_REVISION
from mlx_backend import MTPBackend
from speculative import generate

head = load_trusted_head(sidecar_directory, rank0_target,
                        target_config=trusted_target_config,
                        target_revision=TARGET_REVISION)
backend = MTPBackend(rank0_target, head, group=group, peer=control_peer,
                     prefill_step_size=256, eval_group_size=admitted_group)
result = generate(backend, exact_prompt_ids, max_tokens=64,
                  eos_ids=official_target_tokenizer.eos_token_ids)
backend.finish()
```

This is **not** a runnable two-host CLI: `control_peer` transport/actor integration
is deliberately not supplied. Moreover `MTPBackend` now **unconditionally rejects
size=2** because the parent's M4 gate failed; there is no bypass flag. The sample
above is a future integration sketch, not an executable distributed recipe.
`group=None, peer=None` on the tiny full-ownership CPU fixture is exercised by tests.
There is no generic model download or replacement teacher in this module.

## Parent integration plan / remaining gates

1. Clone 012's actor/supervisor/probe/streaming files into a **new** isolated
   deployment, preserving all bounded deadlines/log retrieval/source and tensor
   admission/cleanup rules. Do not run production 012 with only extra hash keys:
   its `run_owned` snapshots a fixed `SOURCES` list and would omit these helpers.
   Merge **every path in `source_registry.SOURCES`** plus all cloned 012 actor,
   streaming and planner dependencies into the isolated fixed registry. Bind
   startup/process source hashes and include `supervise.py` as usual. Keep original
   8MiB/20MiB evidence limits; start with few short requests and compact round data.
2. Rank0/M2 alone owns `BoundHead` and target output projection. Rank1/M4 constructs
   `Rank1Target(model,group)` and `rank_control.Worker(target.dispatch)`. Implement
   a dedicated bounded control socket (or explicit CPU distributed frames) carrying
   `Command(request,sequence,op,tokens)` and exact ACK. `peer.begin(command)` sends
   before rank0 executes local work; `peer.finish()` waits **after** local work.
   ACK-after-forward plus wait-before-forward would deadlock. Rank1 invokes phased
   forward on each `forward` with the identical IDs and width; it computes no MTP
   or vocabulary logits. Validate request ID, sequence, bounded payload and both
   ranks' admitted schedule, and preserve nonce-owned timeout/cleanup.
3. `reset` initializes both rank target caches and rank0 head KV. `snapshot` saves
   all target state on each rank and head KV **after** `(last_hidden,y)` drafting.
   This head pair is valid even if proposed d is wrong. `forward([y,d])` returns
   all logits/normalized hidden only on rank0. Rejection sends `restore`, restoring
   both target ranks and head to that snapshot, then `forward([y])` on BOTH ranks.
   Native width1 replay prediction is authoritative; do not merely trim recurrent
   cache or retain width2 h_y after rejection. Terminal accepted d uses the same
   restore/replay y path so d is emitted but not consumed. `finish` ends request.
4. Before speed runs, execute packed-GPU tiny fixtures, then the existing genuine
   small checkpoint on **both physical GPUs**, then exact27B forced-token width2
   gates under this actual partition. `width_gate.compare_width2` is the local
   single-host comparison helper; distributed equivalents must collect each rank's
   owned conv/recurrent state and live KV/offset, not logits/rank agreement alone.
   Use identical native chunked prefill and forced tokens. Require same-width
   phased/native AND width2/sequential-width1 all-vocabulary logits, exact argmax,
   every layer's live cache/metadata. ORIGINAL tolerances remain 1e-6/1e-5; retain
   failures, with BF16 diagnostics separate. Parent's small-M2 gate is useful but
   does not admit M4 or27B. No tolerance override exists here.
5. Exercise integrated all-accept, forced reject, EOS at y/d/bonus, cap at each
   slot, repeated different-length fresh requests, per-rank restore acknowledgments,
   exception/abort and cleanup before real measurements. Reference restore copies
   complete state and width1 replay redoes the owned target layers: correct but
   potentially slow. Efficient recurrence-only rollback is a separate TDD/gated
   optimization, not implemented here.
6. Only after all gates pass: keep exact target digests, metric/EOS convention,
   warm/fresh caches and prompt fixed; measure actual generated token count,
   acceptance, teacher calls, round wall, memory and post-first throughput. This
   prototype has no evidence of >=10 tok/s and must not be labeled lossless on
   the real target until width2/logits/every-cache gates pass.

## Files

Runtime: `mtp_head.py`, `speculative.py`, `mlx_backend.py`, `rank_control.py`,
`cache_state.py`, `numerical_gate.py`, `width_gate.py`, `source_registry.py`, and
cloned/opt-in `phased_decode.py`. Tests and primary-source/header/CPU receipts are
local to this directory. No changes to 015,016,012 or existing archives.
