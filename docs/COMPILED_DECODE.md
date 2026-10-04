# Compiled single-token decoding

An opt-in full-model decode path for the exact, unsharded MLX-LM Qwen2 model.
The saved mixed-precision weights and the selected Metal MLP kernels are unchanged.
Native eager prefill creates a normal KV cache; the live prefix is padded once to
fixed capacity. One-token calls then pass keys, values, and an array-valued offset
explicitly through `mx.compile`. No Python cache mutation is captured as hidden
persistent state.

## Run it

```sh
uv run paretoquant generate --model artifacts/m2pro-saved-retune-v1/model \
  --prompt 'Explain binary search briefly.' --max-tokens 64 --compiled
# Same compilation/cache policy, without custom MLP kernels:
uv run paretoquant generate --model artifacts/m2pro-saved-retune-v1/model \
  --prompt 'Explain binary search briefly.' --max-tokens 64 --compiled --stock
```

`--compiled` is opt-in; default generation is unchanged. It uses greedy sampling,
respects the loaded tokenizer's chat template and EOS IDs, and returns generated
text. Fusion still requires a compatible bound manifest and loaded precision;
invalid dispatch falls back explicitly to stock as before. Unsupported compiled
model/cache configurations fail instead of silently substituting another model.

Capacity must fit the model's context. Each request has native prefill and a cold
compilation cost. A reused `FixedDecoder` can reuse its graph across prefill resets;
the command above constructs a new decoder for each process. Warmed decode results
are not a promise about short cold requests or time to first token.

## Cache correctness

The offset is an `int32` scalar array, used both for RoPE and `slice_update`.
A Boolean attention mask permits positions through the newly appended token and
excludes all future storage. Attention receives fixed-capacity keys/values, not
unmasked padded tokens. Python checks reject overflow before calling a compiled
step. Keys and values are evaluated along with logits on every measured step.

Tests compare native and fixed-cache logits and the live KV prefix at every token,
including reset with a different prompt length, poisoned future storage, capacity
exhaustion, and EOS handling. Opt-in Metal tests cover quantized 3/4/6-bit Qwen2
with and without installed custom fusion. `trace_count` means Python graph traces,
not runtime kernel invocations; compiled calls do not update Python wrapper
counters once per GPU execution.

```sh
PARETOQUANT_COMPILED_GPU=1 uv run pytest tests/test_decode.py tests/test_decode_cli.py -q
```

Do not mutate model weights/modules after constructing a decoder. Compilation
captures that model; install fusion before creating the decoder. The API is single
sequence, nonquantized KV state, no sliding-window attention, no distributed model.

## Six-path paired experiment

```sh
uv run python scripts/benchmark_decode.py \
  --model artifacts/m2pro-saved-retune-v1/model \
  --output artifacts/my-compiled-comparison/comparison.json \
  --decode-steps 64 --repeats 24 --warmup 2
```

The runner validates bound weight/config/kernel/runtime metadata and the loaded
projection precision before fusion. It compares:

1. Stock MLPs, native dynamic KV cache, eager full-model decode.
2. Stock MLPs, fixed KV capacity, eager decode.
3. Stock MLPs, the same fixed KV capacity, compiled decode.
4. Retuned custom/stock MLP dispatch, native dynamic KV, eager decode.
5. That same dispatch, fixed KV, eager decode.
6. That same dispatch and fixed KV, compiled decode.

All use identical packed checkpoint bytes, prompts and a greedy stock-generated
teacher-forcing schedule. Every fixed/compiled candidate checks all 64 decode
logits against its own backend's native-cache execution before timing: finite
outputs, identical argmax, maximum absolute error <=0.0625 and RMSE <=0.005.
These thresholds are fixed in the runner, not fit after observing a run. These are
compilation/cache checks, not a new claim that stock and custom arithmetic are
identical for every input. Separate autoregressive generation checks require the
native/compiled token sequences to match within each backend (32 generated tokens
for these evidence runs).

Each context rotates the six paths through 24 paired trials after two warmups.
Every path occupies every order position equally. First compilation and correctness
checks are outside the warmed decode timer. Native prefill plus fixed-cache setup
is recorded separately, not folded into decode speed. The final JSON is published
with exclusive creation; a competing or existing result is never truncated.

## Measured results

M2 Pro, 16 GB, Qwen2.5-0.5B-Instruct, unchanged 260,563,712 parameter bytes per model,
18 selected fused gate/up pairs. Three prompt lengths: 35, 109, 448 tokens.
Two separately executed runs, each with 432 measured path trials and 36 warmups.
The later run is shown below; the earlier run is retained rather than pooled or
replaced. Raw results and SHA-verified executed source:

- `results/m2pro-compiled-paired-v1/`
- `results/m2pro-compiled-paired-v2/`

| Context | Stock eager tok/s | Fused eager tok/s | Stock compiled tok/s | Fused compiled tok/s |
|---|---:|---:|---:|---:|
| short | 207.99 | 217.61 | 248.33 | 267.28 |
| medium | 208.43 | 214.96 | 246.09 | 262.83 |
| long | 203.15 | 211.63 | 238.21 | 256.51 |

| Context | Combined vs stock eager ratio | Paired bootstrap 95% interval |
|---|---:|---|
| short | 1.285094x | [1.267641, 1.301666] |
| medium | 1.261018x | [1.238254, 1.286979] |
| long | 1.262617x | [1.249584, 1.275418] |

The later run's compiled custom path is 26.1–28.5% higher warmed teacher-forced
throughput than stock eager and 21.2–22.8% higher than the existing retuned eager
path. Custom fusion contributes a further 6.8–7.7% over equally compiled stock
under the same fixed-cache policy. This separates the custom-kernel contribution
from the combined result instead of attributing all gains to Metal fusion.

The first run observed combined ratios 1.274048x / 1.261557x / 1.236711x,
with intervals [1.251239,1.282440] / [1.245344,1.285919] /
[1.223880,1.256414]. Its compiled-fusion ratios were 1.082549x /
1.085561x / 1.051709x. Both runs used the same weights and precision map.
All checked per-token logits within each backend matched exactly in the archived
0.5B runs, and native/compiled autoregressive sequences matched. This is not a
full quality evaluation of the compiled backend or proof across all prompts.

## Actual autoregressive requests (separate protocol)

`results/m2pro-compiled-autoregressive-v2/` retains a separate real-generation
verification against native MLX-LM `stream_generate`: 12 rotated paired requests
per path and context, two warmups, 64 actually generated tokens in every measured
request. All 144 measured requests and 9,216 generated tokens are audited in
`archive.json`. Native and compiled sequences matched **within each backend**;
stock and fused trajectories are not asserted identical to each other.

The CLI now matches native MLX-LM prefill: 2048-token chunks excluding the final
prompt token, then that final token separately. This matters because that one-token
call can activate custom fusion. An initial verifier with whole-prompt prefill
failed the medium-context fused token-equality check; it did not publish a result.
The native-prefill regression and the subsequent successful verifier retain this
correctness distinction. Teacher-forced controls still use whole-prompt prefill
consistently in all six paths, matching the earlier replay protocol.

For actual generation the sampled-token graph incorporates greedy argmax. Token
arrays and KV remain on device, and the next dependent token is queued before the
current token is read on CPU. This is ordinary asynchronous scheduling, not
speculative decoding or an auxiliary draft model. EOS can leave one prefetched
step; outstanding state is evaluated before returning/resetting. Logit-output
and sampled-token-output graphs have separate trace observations.

| Context | Native stock tok/s | Native fused tok/s | Compiled stock tok/s | Compiled fused tok/s |
|---|---:|---:|---:|---:|
| short | 268.73 | 287.95 | 270.09 | 294.14 |
| medium | 254.10 | 272.74 | 253.98 | 274.84 |
| long | 205.71 | 217.06 | 203.84 | 217.91 |

| Context | Compiled fused / native fused | Paired bootstrap 95% interval |
|---|---:|---|
| short | 1.021511x | [1.014905,1.027051] |
| medium | 1.007712x | [1.004304,1.011335] |
| long | 1.003929x | [0.998911,1.006193] |

Thus compilation adds only 0.4–2.2% over the existing fused generator in this
protocol, with long including parity. Compiled stock versus native stock is
1.005090x / 0.999544x / 0.990885x; the long stock case is slower. Combined custom
fusion + compilation versus native stock is 1.094577x / 1.081632x / 1.059324x.
**The 26–29% synchronous cached-decode result is not practical request speed.**
Native MLX-LM already overlaps token work, so removing synchronous Python overhead
has less room to help actual generation.

These timers include prefill/cache setup, sampling, EOS handling and text decoding,
but exclude model loading/process startup. Compiled decoders are reused between
warmed requests; the CLI constructs a new decoder in each process. Single loaded-
model first-call observations are archived, not treated as cold-process benchmarks.
The standalone executed verifier is historical evidence with its original local
paths, not a portable live entry point. These are practical implementation comparisons,
not GPU-only timing or proof of universally equal outputs/quality.

## Limits

These are synchronized wall-clock teacher-forced cached decode times, not GPU-only
times, sampled request throughput, or cold process latency. Sampling, text rendering,
KV setup and first compilation are excluded. Fixed-capacity masked attention can
be slower when capacity greatly exceeds the live prefix. Background swap pressure
was substantial and is recorded in both environment snapshots; no user apps were
closed. Intervals cover sampled trials only. Previous positive, parity, quality
and slower 1.5B custom-kernel evidence remains intact. No new vLLM/TensorFlow
comparison, batching result, distributed acceleration, or larger-model compiled
speedup is established by these 0.5B runs.
