# Qwen3.8-27B mixed 3/4-bit allocator plan (v1, PROXY — not a quality or speed result)

Spike source: `spikes/026-qwen38-mixed-3-4/`. Nothing was downloaded to disk; every tensor was
streamed by HTTP range from pinned Hugging Face revisions, one tensor at a time (≤~1.5 GB process
footprint after cache limiting). No full-model inference, no two-host runs.

## Identity (inventory.json, provenance-check.json)

| role | repo | revision |
|---|---|---|
| 4-bit (deployed) | mlx-community/Qwen3.8-27B-4bit | 10c35caafbb80f7dc6a7a432cdd11af10a6d4818 |
| 3-bit | lukaskremla/Qwen3.8-27B-3bit-MLX (community; no mlx-community 3-bit exists) | b3503b31b36125107e12a2b57fbe690c37bfc053 |
| BF16 source (checks only) | mlx-community/Qwen3.8-27B-bf16 | 6f265714824f3c38d4452baa1628aef3d9b9aae9 |

- Both quant configs: `qwen3_5`, affine, group 64, no per-layer overrides. Same text config/layer_types;
  same 2,180 tensor names; 1,840 text-layer tensors match in name/dtype/shape except packed U32 widths.
- `mx.quantize(BF16, g64, bits=4|3)` reproduced **both** published packings bit-exactly (weight, scales,
  biases) on 3 sampled tensors (L0 linear_attn.out_proj, L3 self_attn.o_proj, L63 mlp.down_proj).
  Not verified for the other 493 quantized text tensors.

## Measurement (per-tensor-error.jsonl, 496/496 quantized text linears)

PROXY: reference = published 4-bit dequantized W4; candidate = published 3-bit W3; inputs = fixed
synthetic Gaussian X (64 rows, seed 0). `rel_out = ||X(W3−W4)ᵀ|| / ||XW4ᵀ||`. **Not real activations.**

- rel_out min / median / max = 0.2176 / 0.2269 / 0.2885. Big tensors (MLP, qkv, z, out) sit in
  0.222–0.237; spread comes mostly from tiny in_proj_a/b (48×5120) and full-attn k/v.
- bf16-crosscheck.json (4 tensors): q4-vs-BF16 ≈ 0.092–0.104, q3-vs-BF16 ≈ 0.187–0.220,
  q3-vs-q4 ≈ 0.219–0.253; the proxy ordering matched the q3-vs-BF16 ordering on those 4.
- Under isotropic inputs this proxy is ≈ weight-noise ratio, so it is nearly uniform: the allocator
  has little real signal to exploit.

## Allocator (exact `paretoquant.allocator.allocate`, units × {q4, q3}, real header bytes)

Uniform 4-bit quantized text-linear bytes 13,697,187,840; uniform 3-bit 10,653,368,320
(max saving 3,043,819,520 B = 2.835 GiB). Text stored total at uniform 4-bit 15,132,802,048 B
(incl. embed 715,161,600 + lm_head 715,161,600 + 5,280,768 non-quantized layer bytes).

Predicted decode speedup = bandwidth-model only: per-token bytes = quantized layers + non-quantized
layer params + lm_head (+ final norm); **not measured**.

| target saving | layer/unweighted | layer/param-weighted | block (attn/mlp)/unweighted | predicted speedup |
|---|---|---|---|---|
| 10% | 0.303 GiB, 7 layers (L03,07,11,15,19,23,47) | 0.303 GiB, L03,07,11,15,19,23,39 | 0.311 GiB, 10 MLP blocks | 1.023× |
| 25% | 0.739 GiB, 17 layers | 0.709 GiB, 16 layers (L02–05,07–17,56) | 0.716 GiB, 23 MLP blocks | 1.056–1.058× |
| 50% | 1.417 GiB, 32 layers | 1.417 GiB, 32 layers | 1.432 GiB, 46 blocks | 1.118× |
| 75% | 2.126 GiB, 48 layers | 2.126 GiB, 48 layers | 2.126 GiB, 75 blocks | 1.188× |
| 90% | 2.567 GiB, 58 layers | 2.568 GiB, 58 layers | 2.552 GiB, 107 blocks | 1.236× |
| 100% | 2.835 GiB, all 64 | same | same | 1.268× |

Full chosen lists: `allocator-plans-*.json`. Caveats:
- Unweighted loss sums rel_out² per tensor, so tiny in_proj_a/b count as much as an MLP; that is
  why full-attention layers (no a/b projections) and MLP blocks are picked first. Param-weighted
  loss fraction tracks byte-saving fraction almost 1:1, i.e. the proxy does not discriminate.
- block/param-weighted did not finish in 300 s (pure-Python frontier too large); no result.

## Not done / next

1. Real-activation sensitivity (calibration prompts, layer-streamed forward or Fisher/KL per layer);
   the synthetic proxy cannot justify which layers to demote.
2. Assemble a mixed checkpoint (per-layer `quantization` overrides in config) and verify packed-array
   equality after reload.
3. Quality eval (held-out perplexity / tasks) vs uniform 4-bit and uniform 3-bit.
4. Two-Mac timing with the admitted runner; re-derive the [24,40] split from new per-rank bytes.
