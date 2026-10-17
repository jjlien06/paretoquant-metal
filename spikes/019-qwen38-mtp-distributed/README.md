# 019 — bounded two-rank native MTP integration

Standalone clone of 012 supervisor/probe/actor/streaming/planner plus 017 native
head, engine and full hybrid-cache snapshot/replay. No changes to those spikes.
**CPU integration tested; physical 27B admission and throughput remain unmeasured.**

## Launch (hardware owner only)

Deploy **every file in `remote_actor.SOURCES` plus `supervise.py`**, byte-exactly,
to a NEW immutable mini directory. The fixed registry includes sidecar_config.json,
slot_qmv.py and BOTH .metal files; actor snapshots support these ordinary bounded
files without extension exceptions. Read back hashes before launching. This agent
has not deployed or launched any SSH/GPU job.

From repository root, after the parent's live bridge/memory admission and deployment:

```sh
.venv/bin/python spikes/019-qwen38-mtp-distributed/supervise.py \
  --remote-root /Users/jeremylien/Projects/paretoquant-dual-host/spikes/019-qwen38-mtp-distributed-v1 \
  --remote-source /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
  --local-metadata /Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only \
  --output /Users/jeremylien/Projects/paretoquant-metal/results/qwen38-speed-v2/019-full-gate-v1 \
  --ssh-key /Users/jeremylien/.ssh/paretoquant_mini_ed25519 \
  --ssh-target "${MINI_SSH_TARGET:?set freshly verified pinned mini target}" \
  --ssh-host-key-alias 'fe80::56:c507:c02a:a630%bridge0' \
  --local-address "${LOCAL_BRIDGE_IP:?set verified bridge IPv4}" \
  --remote-address "${MINI_BRIDGE_IP:?set verified bridge IPv4}" \
  --split 36 28 --execution phased --eval-group-size 32 \
  --max-tokens 96 --repeats 1 --deadline 240 \
  --rank0-verifier native --rank1-verifier slot-qmv \
  --target-revision 10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
  --target-config-sha256 14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85 \
  --target-index-sha256 13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8 \
  --sidecar /Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-mtp-4bit-b643c01b6d3b094e325edb6ebd832e16c486c575 \
  --small-model-local /Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/small-control \
  --small-model-remote /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462 \
  --small-config-sha256 ba7770da23eae5ebd6827571f086e331956b33f4442a9e876fb4aa10969a6772 \
  --small-index-sha256 6e48f2fa5d6f033a6d77bf833abfa9698ca24d1715ecea4c67447bcfaee44650 \
  --prompt 'Write the integers from 1 through 20, separated by a comma and a space, with no other text.' \
  --verification-gate-only
```

Remove `--verification-gate-only` and choose a **fresh** output directory for the
actual benchmark. It ALWAYS repeats small + full numerical gates in that process;
a saved passing receipt is not accepted as authorization. Optionally use
`--rank1-verifier singleton-qmm` for the serial numerical control. Rank0 also
admits singleton-qmm/slot-qmv explicitly if its target gate rejects native width2.
No fallback from a failed full-model gate is automatic. Use an outer tool deadline
at least 360 seconds so the internal 240-second watchdog can clean up.

The listed metadata hashes were measured locally, not invented. They bind exact
config/index bytes; they do not independently prove a Hugging Face revision's
weight identity. The revision is an **explicit trusted operator binding**, not a
path-derived attestation. The source checkpoint must already have passed the
parent's pinned download/file-hash admission. Original 012 exact tensor inventory,
actual per-tensor SHA256, replicated-digest agreement and server/source ownership
checks remain mandatory. The sidecar independently checks its pinned config,
238934137-byte size and full SHA256 before rank0 loading.

## Runtime admission and transport

* Each physical rank first loads the independently hash-bound genuine small
  checkpoint, runs three forced two-token windows versus native-width1 arithmetic,
  and frees that model before target loading. It gates every distinct admitted
  evaluation-group size. No head is loaded for the small control.
* The already-loaded full target shards run the same three windows after identical
  chunked prefix/final-prompt-token prefill. Receipts contain full-vocabulary
  max-absolute/exact/strict metrics, exact argmax IDs, all local recurrent/conv/live
  KV numerical parts and offsets. Rank0 owns target logits; rank1 receives the
  actual rank0 logit/argmax metrics and reports its own cache metrics separately.
* Fixed `atol=1e-6`, `rtol=1e-5`. Failure is retained and blocks speculation. A
  live model-bound Admission capability is required by both target adapters.
* `native`, `singleton-qmm`, and `slot-qmv` are explicit per-rank CLI choices.
  The latter two modify QMM only inside a width2 **verify** command, never a
  two-token prefill. Slot-QMV uses the byte-copied 018 kernel for eligible shapes;
  unsupported dtype/packing/N<8/N%8/group/transpose cases use singleton native QMM.
  Kernel exceptions do not silently fall back or turn into a passing gate.
* Bounded length + uint8 JSON frames use the SAME CPU-stream MLX group as hidden
  handoffs; no extra listener. Command-before-forward, ACK-after-forward. Sequence,
  request, op and token bounds are enforced. Reset/snapshot/restore/width1 replay
  cover BOTH target caches; rank0 alone owns the MTP head/cache.
* Supervisor consumes actual rank_action telemetry, compares both action streams,
  validates all owned-layer completion for each actual forward/verify and binds
  numerical admission before request execution. It preserves 012 source/tensor,
  schedule, nonce-owned cleanup, disk reserve, 8MiB log/20MiB response bounds and
  closed-listener checks. Caps: 3 requests, 96 tokens/request, 240 seconds.
* Full-state snapshots + replay favor correctness over speed. Timings report
  first token, whole-request and post-first throughput, include emitted EOS and
  finish ACK, and exclude setup/loading/gates/head loading. No >=10 tok/s claim.

## CPU validation

```sh
.venv/bin/python -m pytest spikes/019-qwen38-mtp-distributed -q
```

The CPU suite includes genuine tiny packed-hybrid logits/cache gates and head +
engine generation; a two-rank integration fixture uses actual CPU models/head and
framing with ONLY the physical MLX link replaced by bounded queues. It verifies
matching native greedy IDs and actual reset/verify/restore/replay/finish actions.
The CPU slot-dispatch test substitutes a CPU singleton operation for Metal; it is
not a GPU kernel gate. CPU recurrent fixtures use training arithmetic, not GPU
inference kernels. TDD missing-feature failures preceded the new feature slices;
`cpu-tests.xml` records the final executed suite. Physical small/full gate outcomes
and actual throughput must be collected by the hardware owner.
