# NEW021 — opt-in native GDN prefix rollback

Isolated clone of 019; **019 sources and archived evidence are untouched**. Default
(no `--prefix-rollback`) retains full-snapshot/restore/native-width1 replay.
Target revision, native BF16, quantization, standalone MTP head and tensor digest
admission are unchanged. CPU tests are not physical numerical or speed admission.

## What the opt-in does

Only explicit width2 verify enters scoped thin wrappers around native GDN call,
native Conv1d call and the qwen3_5 module's actual `gated_delta_update`. No model
forward is reimplemented; no module/parameter names or tensors are replaced.
Capture covers the current process's owned layers, not the weight server.
It retains prestate, convolution input and actual q/k/v/a/b/A_log/dt_bias/state/
mask/use_kernel arguments. Only unpadded batch1 native ArraysCache/KVCache is
admitted; unsupported cache metadata fails closed.

On rejection or emitted terminal draft (EOS/cap), both ranks execute one bounded
`commit_prefix [1]` command. Full-attention KV trims one live position. GDN conv
state becomes `conv_input[:, keep:keep+kernel-1, :]`; only native recurrence runs
again on captured prefix q/k/v/a/b from prestate, preserving original use_kernel.
All live caches are evaluated together once. Rank0 reuses verified slot0 prediction
and **normalized** hidden; there is no full-target width1 rejection/terminal replay.
Snapshots/full restore remain available for reference and abort. Prefix0 restores
prestate; prefix2 retains the width2 result. Every capture is single-use.

**Capture may alter lazy fusion; prior 019 numerical admission is NOT inherited.**
Before bench, every physical rank/evaluation-group must pass same-process small
and already-loaded-full-target gates, at original `atol=1e-6, rtol=1e-5`:
* Existing three windows: native singleton full logits/argmax and all live caches.
* Instrumented versus uninstrumented identical width2, including normalized hidden.
* Prefix0/1/2 versus forced native singleton caches/metadata and available full
  logits/normalized hidden; native singleton continuation after each prefix.

Gate receipts bind the opt-in, all owned layers, strict metrics and same live model;
supervisor checks prefix evidence, trusted option, numerical phase before requests
and matching bounded command streams. Peer capability exchange contains only
bounded summaries, not full per-layer metric trees. Actor fixed source registry
includes `prefix_rollback.py`, sidecar config and both slot-QMV Metal files.

## CPU verification

```sh
.venv/bin/python -m pytest spikes/021-qwen38-prefix-rollback -q
```

Genuine tiny packed native hybrid models test all prefixes and greedy rejection,
EOS, cap and repeated fresh-cache resets. Two-rank fixtures use real CPU models
and bounded framing, replacing only the physical transport with queues. CPU GDN
uses training recurrence; GPU inference kernel admission remains the hardware
owner's responsibility. No GPU or SSH was run for this implementation.

## Hardware-owner launch

First deploy **all `remote_actor.SOURCES` plus `supervise.py`** to a NEW immutable
mini directory; read back exact hashes. Preserve the existing pinned host trust,
live memory/bridge admission, native BF16 and exact target/head digests.
Use the prior trusted 019 supervisor options with ONLY these changes:

```sh
.venv/bin/python spikes/021-qwen38-prefix-rollback/supervise.py \
  --remote-root /Users/jeremylien/Projects/paretoquant-dual-host/spikes/021-qwen38-prefix-rollback-v1 \
  --remote-source /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
  --local-metadata /Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only \
  --output /Users/jeremylien/Projects/paretoquant-metal/results/qwen38-speed-v2/021-prefix-gate-v1 \
  --ssh-key /Users/jeremylien/.ssh/paretoquant_mini_ed25519 \
  --ssh-target "${MINI_SSH_TARGET:?freshly verified pinned mini target}" \
  --ssh-host-key-alias 'fe80::56:c507:c02a:a630%bridge0' \
  --local-address "${LOCAL_BRIDGE_IP:?verified bridge IPv4}" \
  --remote-address "${MINI_BRIDGE_IP:?verified bridge IPv4}" \
  --split 28 36 --execution phased --eval-group-size 32 \
  --max-tokens 96 --repeats 1 --deadline 240 \
  --rank0-verifier native --rank1-verifier slot-qmv \
  --target-revision 10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
  --target-config-sha256 14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85 \
  --target-index-sha256 13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8 \
  --sidecar /Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-mtp-4bit-b643c01b6d3b094e325edb6ebd832e16c486c575 \
  --small-model-local "${SMALL_MODEL_LOCAL:?trusted local physical control path}" \
  --small-model-remote /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462 \
  --small-config-sha256 ba7770da23eae5ebd6827571f086e331956b33f4442a9e876fb4aa10969a6772 \
  --small-index-sha256 6e48f2fa5d6f033a6d77bf833abfa9698ca24d1715ecea4c67447bcfaee44650 \
  --prompt 'Explain briefly how a rainbow forms.' \
  --prefix-rollback --verification-gate-only
```

Keep an outer deadline >=360s for the internal <=240s cleanup. To benchmark,
remove `--verification-gate-only` and use a fresh output directory; the numerical
gates still rerun in that process. For reference omit `--prefix-rollback`.
Measure actual fresh prose requests and compare contemporaneous 019 controls.
No >=10 tok/s claim or projection is made; physical speed remains unmeasured.
