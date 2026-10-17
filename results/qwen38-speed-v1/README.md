# Qwen3.8-27B: verified, modest warm-generation improvement

## Result

The unchanged 4-bit checkpoint ran **12.8% faster in warmed post-first-token
throughput**, with **10.1% lower warmed request wall time**, on the matched-output
counting pilot below. This is a useful modest improvement, not a 2× breakthrough
or a universal performance guarantee.

Use the new `spikes/012-qwen38-scheduling/` runner with:

- **36 layers on the M2 Pro / 28 on the M4 mini**, instead of 24/40.
- **32-layer evaluation blocks**, instead of four, reducing synchronization.
- **No GDN compilation, tensor parallelism, requantization or weight changes.**

Exact model: `mlx-community/Qwen3.8-27B-4bit`, revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`. Both ranks' union of all 1,847 text
tensor hashes remained identical across configurations. The model uses upstream
stock MLX operators; this is scheduling/placement tuning, not custom-kernel gain.

## Matched-output measurement

Prompt: `Write the integers from 1 through 20, separated by a comma and a space, with no other text.`

Three separately supervised process starts ran baseline → candidate → baseline.
Each loaded its weights once and made three fresh-cache greedy requests. The
first request of each process is excluded from the warm comparison. **All nine
requests returned identical 70 token IDs including EOS**, and both ranks agreed.

| Warm metric | Original 24/40, group 4 | Faster 36/28, group 32 |
|---|---:|---:|
| Post-first token events/s, median | 7.205 | **8.129** |
| Full request wall, median | 10.606 s | **9.530 s** |
| Warm observations | 4 | 2 |
| Peak MLX allocation, MacBook | 6.218 GiB | 8.712 GiB |
| Peak MLX allocation, mini | 9.448 GiB | 7.081 GiB |

The post-first metric counts 69 remaining events after the first. Whole-request
time includes prefill and all 70 events, but excludes loading. The resulting
ratios are 1.1282× decode throughput and 1.1129× whole-request throughput.

This is a **short, separate-process bracketing pilot**, not randomized in-process
pairing or statistical significance. First-token latency did not improve.
First-use/cold loading is **not** accelerated: the candidate's first request took
15.04 seconds versus 12.51/14.10 seconds in the two controls. Keep cold and warm
behavior separate. No task-accuracy, long-context or concurrency benchmark was run.

A second explanation prompt showed about 12.6% higher decode throughput
(6.906 → 7.775 events/s), but changed wording/length (49 → 50 events) when layers
moved between different GPU architectures. That pilot is **not** an identical-output
comparison. Identical serialized weights do not guarantee identical floating-point
arithmetic or greedy trajectories across every placement and prompt.

## What else was tested

### Grouping alone: a small gain

A single loaded model alternated groups 4/8/16/32 for 12 full requests. Outputs
matched the original 49-event response exactly. Excluding first-use per mode,
the two warm samples per mode had median decode rates:

| Group | Events/s |
|---|---:|
| 4 | 6.895 |
| 8 | 7.001 |
| 16 | 7.114 |
| 32 | 7.209 |

Thus grouping alone was about 4.6% higher; redistribution supplied the rest of
the combined pilot's improvement. These are different experiments, not additive
operator cost attribution or statistically separable causal percentages.

**Preserved controller failure:** both GPU actors completed those 12 requests,
but the controller rejected evidence retrieval because a log exceeded its inherited
8 MiB bound. SSH forwarding also dropped bytes. Exact actor disk logs were then
retrieved and remotely SHA-256 checked. Original source/checkpoint/schedule
verification and explicit listener closure passed in separate post-hoc recovery.
The original failed controller result is unchanged. See
`same-process-group-screen-v1/recovery/`. Keep full-model repeat counts small
with this transport; do not remove retrieval bounds to hide the limitation.

### Compiled recurrent/GDN runs: not adopted

`spikes/014-qwen38-compiled-gdn/` passed tiny F32/BF16 and pretrained small-model
logit/cache/token gates. The physical full-model candidate used all 16 compiled
functions (six on the MacBook, ten on the mini), with stable traces and growing
per-request call counts. Weights and 49-event outputs matched native controls.

A native → compiled → native bracket measured 6.965 versus 7.008 median post-first
events/s, roughly 0.6% difference: **no useful demonstrated acceleration**. This
flag stays off in the recommended profile. Eligibility is width-one-based;
the separately processed last prompt token can compile as well as decode tokens.

### Tensor parallelism: rejected at numerical admission

`spikes/013-qwen38-tensor-parallel/` uses native upstream sharding and bounded
packed slices. Actual two-local-Metal-process F32 gates passed, but native BF16
logits failed the original tolerance on three of five steps. An attempted F32
partial-row reduction variant still failed and introduced one recurrent-convolution
cache failure. Tolerances were not relaxed. **Neither BF16 TP variant was admitted
or deployed for full-model generation.** Matched toy token IDs did not override
failed logits/cache gates. Original failures and source archives remain intact.

Small-message ring diagnostics were also retained. The constant-input reduction
screen was optimistic; a dependent GPU-producer/evaluation/reduction chain cost
substantially more. Neither diagnostic is a measured TP generation speedup.

## Saved configuration and running again

`fast-profile.json` records the exact recommended parameters, paths, source hashes,
checkpoint identity and observed memory. Defaults in the older runner were not
silently changed. The larger MacBook shard needs healthy memory headroom: recheck
both machines' pressure, swap and disk immediately before loading. Do not close
unrelated apps or kill unrelated automation. Use the smaller 24/40 partition if
local headroom is inadequate. MLX peak allocation is not total OS residency.

Bridge addresses can change after reconnect/reboot. Rediscover neighbors and verify
routes through `bridge0`, preserving strict SSH checks against the prior pinned
host identity. With the addresses used in this experiment:

```sh
cd /Users/jeremylien/Projects/paretoquant-metal
.venv/bin/python -u spikes/012-qwen38-scheduling/supervise.py \
  --remote-root /Users/jeremylien/Projects/paretoquant-dual-host/spikes/012-qwen38-20261016-v1 \
  --remote-source /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
  --local-metadata "$PWD/models/qwen3.8-27b-4bit-tokenizer-only" \
  --output "$PWD/results/qwen38-fast-$(date +%s)" \
  --ssh-key /Users/jeremylien/.ssh/paretoquant_mini_ed25519 \
  --ssh-target 'jeremylien@fe80::14b2:4360:a750:94e5%bridge0' \
  --ssh-host-key-alias 'fe80::56:c507:c02a:a630%bridge0' \
  --local-address 169.254.47.213 --remote-address 169.254.214.155 \
  --split 36 28 --eval-group-size 32 --execution phased \
  --max-tokens 96 --repeats 3 --deadline 240 --reserve-bytes 8589934592 \
  --prompt 'Write the integers from 1 through 20, separated by a comma and a space, with no other text.'
```

Use a terminal lifetime longer than the watchdog plus cleanup allowance. No resident
chat server was created; every owned physical trial worker/listener was verified
closed. Model downloads remain intact. Post-run pressure was normal on both Macs,
but swap existed/grew; this is not a no-swap guarantee. No commits/pushes were made.

## Evidence

- `counting-comparison.json`: matched-output measurements and raw trial references.
- `counting-control-before-v1/`, `counting-fast-v1/`, `counting-control-after-v1/`:
  exact rank receipts, source snapshots, trusted tensor coverage and cleanup.
- `combined-scheduling-pilot.json`: explanation prompt, with wording difference disclosed.
- `same-process-group-screen-v1/recovery/`: exact recovered logs and post-hoc checks.
- `gdn-pilot-comparison.json` and its three trial directories: negative compilation result.
- `rejected-tp-summary.json`: copied numerical rejection summary; complete original
  reports/source archives remain under spike013.
- `final-validation.json`: **83 scheduling tests / 47 compiled-GDN tests passed**,
  with clean Ruff checks. TP's retained BF16 tests intentionally remain failed;
  it is not an admitted backend.
