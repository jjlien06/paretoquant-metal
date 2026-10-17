# Qwen3.8-27B across two 16 GB Macs

## Verdict: VALIDATED for bounded text generation

On October 16, 2026, `mlx-community/Qwen3.8-27B-4bit`, immutable revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, completed **four genuine requests**
across the M2 Pro MacBook Pro and M4 Mac mini over Thunderbolt Bridge.
Two prompts were each requested twice, reusing loaded weights with fresh caches.
Both ranks agreed on every generated token, reached EOS and shut down cleanly.

The target really is **Qwen3.8**; its built-in architecture is named `qwen3_5`.
The new isolated ParetoQuant adapter uses **stock MLX quantized operators**,
not ParetoQuant's custom fused MLP or mixed-precision recalibration. This is
text-only feasibility evidence, not a distributed speedup or quality benchmark.

## Observed results

Prompt: `Why do two Macs connected by Thunderbolt not behave like a single shared-memory GPU? Explain in exactly two short sentences.`

| Loaded-model request | First use | Warm repeat |
|---|---:|---:|
| Prompt tokens | 37 | 37 |
| Generated token events, including EOS | 49 | 49 |
| First-token latency | 5.288 s | 1.025 s |
| Request wall time | 12.043 s | 7.842 s |
| Whole-request token events/s | 4.069 | 6.248 |
| Remaining 48 events after first, events/s | 7.106 | 7.041 |

Loading is excluded from request timers. The entire supervised trial, including
setup/loading/two requests/cleanup, took **37.723 seconds**. These are short
observations, not statistical performance guarantees.

Raw model output, not an independently validated technical reference:

> Thunderbolt links provide high-bandwidth data transfer but do not create a unified physical address space, preventing direct peer-to-peer memory access. Consequently, the operating system treats them as separate devices, requiring explicit data copying rather than shared memory operations.

The arithmetic prompt produced `2 plus 2 equals 4.` twice: nine events including
EOS. Its warm wall time was 1.664 seconds and post-first throughput 7.299 events/s.

### Capacity and identity

| Host | Original layers | Stored parameters | Peak MLX allocation |
|---|---|---:|---:|
| M2 Pro MacBook, 16 GB | 40–63 (24 layers) | 6.118 GiB | 6.219 GiB |
| M4 mini, 16 GB | 0–39 (40 layers) | 9.308 GiB | 9.449 GiB |

MLX allocation is **not total process or OS residency**. Each rank replicates
1,430,333,440 non-layer bytes. The hosts are not one contiguous 32 GB pool.
Post-trial pressure was normal on both hosts, but swap existed/grew; this is
not a no-swap capacity guarantee.

All 64 original layers completed. Execution covered all **1,847 text tensors**,
with seven replicated tensor digests agreeing across ranks. The per-rank digests
were unchanged between full-model trials. The unchanged downloaded checkpoint
also contains vision tensors: 2,180 total tensors, 16,081,490,933 file bytes
(14.977 GiB). All 15 files passed pinned Hugging Face Git/LFS hash checks and
three safetensors headers matched pre-download inventory.

### Gates and limitations

- 32 tiny packed-GPU tests pass, with clean Ruff lint/formatting. Independent
  review found no blocking security or logic issues.
- Real physical-small generation matched native controls on both GPUs. That
  separate control uses Qwen3.5-0.8B's matching architecture, not 27B target weights.
- The first small distributed trial stalled in initialization before weights.
  TCP/ring diagnostics passed, and an **unchanged** retry succeeded. The cause
  remains unresolved; no reliability fix is claimed. The failure is preserved.
- All trials verified owned-process/group and listener cleanup. **No inference
  server remains running.** Downloaded models were retained.
- Thinking was disabled. Prompt lengths were 26/37 tokens; maximum observed
  output was 49 events. Long-context, vision, task quality, concurrency, a
  single-host 27B numerical reference and production reliability are untested.

## Evidence and reuse

- `verified-generation-summary.json`: outputs, exact tokens, timings and peaks.
- `full27b-two-host-v1/`, `full27b-explanation-v1/`: full receipts and frozen sources.
- `small-two-host-v1/`: preserved initialization failure; `small-two-host-v2/`: passing gate.
- `preparation/`: immutable download checks and native small-model controls.
- `independent-review.json`, `cross-trial-verification.json`: review and final checks.
- Adapter: `spikes/011-qwen38-dual-host/`; existing 007 source was preserved.
- Durable local metadata: `models/qwen3.8-27b-4bit-tokenizer-only/`.
- Full weights remain only on the mini under the checkpoint path below.

Actual trials used scratch metadata. Its durable copy was SHA-256 readback
verified (`durable-metadata-readback.json`). Addresses may change after a reboot
or reconnect: rediscover bridge neighbors and verify routes. Keep strict
verification against the existing pinned SSH identity; do not accept an unknown
key just to bypass an address change. Recheck memory/swap/disk before loading.

```sh
cd /Users/jeremylien/Projects/paretoquant-metal
.venv/bin/python -u spikes/011-qwen38-dual-host/supervise.py \
  --remote-root /Users/jeremylien/Projects/paretoquant-dual-host/spikes/011-qwen38-20261016-v1 \
  --remote-source /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
  --local-metadata "$PWD/models/qwen3.8-27b-4bit-tokenizer-only" \
  --output "$PWD/results/qwen38-manual-$(date +%s)" \
  --ssh-key /Users/jeremylien/.ssh/paretoquant_mini_ed25519 \
  --ssh-target 'jeremylien@fe80::14b2:4360:a750:94e5%bridge0' \
  --ssh-host-key-alias 'fe80::56:c507:c02a:a630%bridge0' \
  --local-address 169.254.47.213 --remote-address 169.254.214.155 \
  --split 24 40 --max-tokens 96 --repeats 2 --deadline 240 \
  --reserve-bytes 8589934592 --execution phased --eval-group-size 4 \
  --prompt 'Why do two Macs connected by Thunderbolt not behave like a single shared-memory GPU? Explain in exactly two short sentences.'
```

Use a terminal timeout longer than the watchdog plus cleanup allowance. Do not
run through a shorter-lived controller. No Git commit/push, global firewall
change or unrelated process termination was performed.
