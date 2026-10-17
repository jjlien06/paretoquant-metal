# 011: Qwen3.8 hybrid inference across two Macs

## Verdict: VALIDATED for bounded text generation

Qwen3.8-27B 4-bit completed four short text requests on a 16 GB M2 Pro MacBook
and 16 GB M4 mini over Thunderbolt Bridge. All 64 original layers and exact text
tensor coverage were verified; both ranks agreed, and worker/listener cleanup
passed. This is a standalone stock-operator experiment, not custom-kernel
acceleration or a production inference service.

Full measurements, preserved initial network-initialization failure, limitations,
checkpoint identity and a runnable launch command are in
[`results/qwen38-27b-dual-host-v1/README.md`](../../results/qwen38-27b-dual-host-v1/README.md).

## Architecture and admission

- Qwen3.8 uses upstream MLX-LM's built-in `qwen3_5` wrapper with nested text args.
  Preserve the actual model identity; do not spoof Qwen2 to bypass admission.
- Use native hybrid `ArraysCache`/`KVCache` factories, separate SSM/full-attention
  masks and all recurrent/KV states. Original layer numbering is retained.
- Stream only the exact retained `language_model.*` tensors. Known vision roots
  are excluded for text-only execution; unknown names and incomplete shapes,
  storage dtypes, hashes or architecture parameters fail closed.
- Admit already MLX-sanitized checkpoints. Unsanitized convolution layouts are
  rejected; this is not a converter for arbitrary raw official checkpoints.
- The largest target packed embedding/head tensor is 635,699,200 bytes. Transport
  remains bounded at 768 MiB per tensor, not the old 512 MiB. Count replicated
  embedding/head bytes separately from activation/cache/system overhead.
- Phased GPU execution evaluates owned local blocks before CPU-stream network
  boundaries. Grouping changes scheduling only, not operators or precision.
- Durable actors pin source hashes, record ownership, enforce deadlines/disk
  reserve and verify cleanup. `remote_actor.SOURCES` covers all runtime helpers.
- `--ssh-host-key-alias` preserves a prior pinned host identity after bridge
  address changes. Keys live outside the repository; never publish session tokens.
- Chat prompts explicitly disable thinking. `tokens:[1,7,3,4,8,9]` is a fixture-only
  raw-token prompt path; it does not prove pretrained generation quality.

## Tests

Run from the repository root with the native environment:

```sh
.venv/bin/python -m pytest spikes/011-qwen38-dual-host -q
.venv/bin/ruff check spikes/011-qwen38-dual-host
.venv/bin/ruff format --check spikes/011-qwen38-dual-host
```

The 32 packed-GPU tests cover F32/BF16 serialization, complete hybrid state and
logit/token parity, chunked prefill/decode, grouping 1/2/4/8, ragged ownership,
source inventory, prompt formatting and fail-closed checkpoint admission.
Local handoff tests simulate communication; the result directory contains the
separate physical-host controls and actual full-model trials.

`fixture.py --output NEW_DIRECTORY` creates a tiny deterministic random checkpoint.
Tokenizer assets must be copied separately without replacing its config/index.
Do not mistake that fixture for the actual pretrained 27B checkpoint.
