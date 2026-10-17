# External acceleration research: exact Qwen3.8-27B, two 16 GB Macs

Research date: 2026-10-16. Research only: no GPU inference, GPU probes, SSH, installs, model downloads, or repository code edits were performed. Only metadata, code and bounded safetensors **headers**, not tensor payloads, were fetched. This note is the sole intended project write. Existing baseline/evidence is not modified.

## Decision

**Highest-probability bounded next implementation: native MTP with the standalone `mlx-community/Qwen3.8-27B-MTP-4bit` head, kept on the final/head-owning M2 rank, plus per-rank efficient GDN prefix rollback.** Start with **one proposed token / verification width 2**, then width 3; do not transplant all of MTPLX or replace the target weights. The head's 238,934,137-byte file is independently confirmed by the HF blobs API, with 238,930,944 tensor-data bytes in its header. Its card explicitly pairs it with this exact target repository, and its source checkpoint revision matches official Qwen3.8-27B.[35][64][53]

This is an engineering recommendation, **not a verified >=10 tok/s result**. Given the supplied 8.13 tok/s baseline, the goal requires 1.2300123x throughput (+23.0012%); baseline token cost is 123.001 ms and target is 100 ms/output token. For a round committing E tokens, total draft + verifier + communication + rollback + synchronization time must be <=100E ms. An illustrative E=1.8 allows 180 ms/round, not a claim that this head achieves that acceptance.

**Second choice / higher ceiling:** port `ARahim3/mlx-dspark`'s DFlash2 target/tap/rollback engine with `incoai/Qwen3.8-27B-DFlash2`. It has direct Qwen3.8 registration and actual hybrid rollback code, but needs cross-rank intermediate taps, roughly 1.36 GB additional quantized head parameters, BF16-to-Q4 conversion peak handling, and broader numerical validation. Native MTP needs only final normalized target hidden states, no early-layer taps.[5][6][7][1]

**No discovered source establishes >=10 tok/s on this exact M2 Pro 16 GB + M4 mini 16 GB TB4 pipeline.** M4 Pro, M5 Max, M3 Ultra and H200 numbers below are leads, not extrapolated two-host measurements. The primary barrier is verified cost/equivalence of multi-row teacher verification, not merely locating a compatible head.

## Source access, currency and X/Twitter

Requested `web.run` is not exposed in this subagent's callable tools. The available `web_search` tool was used for explicit indexed `site:x.com` queries; its configured openai-native backend failed client-side and automatically fell back to search backends. Full X-body `web_extract` also failed because its configured backend is search-only. **X evidence below is indexed post text/snippets, not independently extracted full threads.** Code/HF metadata were fetched directly over HTTPS and pinned by revision; no promotional throughput is treated as a local result.

Concrete indexed primary posts:

- Abdur Rahim, 2026-08-14: Qwen3.8-27B about 3x faster on Apple Silicon with mlx-dspark. Lead to inspect the engine, not proof for two 16 GB hosts.[45]
- Abdur Rahim, 2026-08-19: reply announcing DFlash2 support in mlx-dspark. Confirms the implementation lineage rather than a benchmark guarantee.[46]
- Zhijian Liu / Inco AI, 2026-08-18: DFlash2 announcement, indexed claim 70 tok/s on M5 Max / up to 4.6x. Validated against actual DFlash2 configs, selector/conv code and public checkpoint; hardware claim is not our hardware.[47]
- Youssof Al Toukhi, 2026-08-29: MTPLX 2.10 improvements, including Qwen3.8-27B context-dependent boosts. Flash-Next MoE and mmap claims in the same post are **not** Qwen3.8-27B dense evidence.[48]
- Paolo Rosson, 2026-08-29: same-prompt M3 Max comparison board including mlx-dspark. Independent comparison lead, but indexed excerpt alone does not certify timings, quant identity or a distributed implementation.[49]

Dates above were derived from post IDs' UTC Snowflake timestamps where the indexed snippet omitted a date. Search also returned current oMLX and DGX Spark reports, but these were excluded from the recommendation because they change hardware, quantization and/or engine. Queries included `site:x.com "mlx-dspark" "Qwen3.8"`, `site:x.com "ARahim3" "DFlash"`, `site:x.com "MTPLX" "Qwen3.8"`, `site:x.com "Qwen3.8" "DFlash2"`, `site:x.com "Qwen3.8" "MTP"`, and follow-up developer-handle queries. No useful direct SpecForge-on-this-two-Mac X evidence was found.

Pinned code snapshots actually inspected:

| Project | Inspected HEAD |
|---|---|
| ARahim3/mlx-dspark | `d6042f38f0aa7dd3e03a2a406111d725056d45a3` |
| Blaizzy/mlx-vlm | `4623cf9d2633c5ee2b1ac125eeae51423447c41f` |
| youssofal/MTPLX | `9882703f3105363ddc37eca9f97aa09a1d387112` |
| sgl-project/SpecForge | `d80daf31aacfd2193283ae36f36864e4dde199bc` |
| z-lab/dflash | `07ebd93db9f472af339b644bb70221ad8428328a` |
| ml-explore/mlx-lm | `93e6d4a6447b3080861985947f0820e76fcb957d` |

These are retrieved source snapshots, not attestations of the locally installed package version. Links below pin those revisions.

## Compatible heads: actual repos, byte counts and contracts

All byte counts are exact HF `siblings[].size` values for the weight file, not card-rounded model sizes. Header inspections confirmed actual shapes/dtypes and absence of private target embedding/head tensors in these external heads. All use the **Qwen3.8 248,320-token vocabulary**; a Qwen3.5/3.6 head cannot be assumed interchangeable just because the architecture is shared.

| Head | Revision | Weight-file bytes | Important contract |
|---|---|---:|---|
| `mlx-community/Qwen3.8-27B-MTP-4bit` | `b643c01b6d3b094e325edb6ebd832e16c486c575` | 238,934,137 | `qwen3_5_mtp`; block_size=3 = two proposed tokens; one full-attention native head; affine Q4/g64; BF16 scales/norms |
| `incoai/Qwen3.8-27B-DFlash2` | `015e795645c74b1a0eeef3b570031fb62e769bc5` | 3,848,817,896 | 5-layer width5120 GQA32/8, hd128; block8 = seven proposals; taps5,19,33,47,61; noncausal block, sliding2048; rank256/top16 selector; grouped conv16/kernel2; mask248070 |
| `RedHatAI/Qwen3.8-27B-speculator.dspark` | `87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce` | 3,976,869,890 | 5-layer width5120 GQA20/4, hd256; block8; taps4,12,20,28,36,44,52,60; causal sliding2048; Markov256+confidence; anchor sampling/readout at slot0; mask248077 |
| `DimInfer/Qwen3.8-27B-Dspark-v1` | `10b6bccfcc109bda0666d0aed4b7871aac357b99` | 3,714,723,322 | block15; taps1,16,31,46,61; gated q-proj; full attention; mask248200 |
| `RadixArk/Qwen3.8-27B-DSpark` | `b9a5dbdf03bc999c6c73c426b19c2d9041cea393` | 3,714,723,322 | block7/runtime versus training_block_size16; taps5,19,33,47,61; full attention; YaRN32; mask248070 |

Each row is backed by the fetched HF byte API, pinned config and safetensors header.[35][36][64]
DFlash2/RedHat details and bytes:[27][28][30]
Legacy DSpark alternatives and bytes:[31][32][34]
RedHat byte API:[29]; Radix byte API:[33].

### Resident and loading budget, not measured peaks

- Native MTP standalone has 31 tensors totaling **238,930,944 data bytes**, about 228 MiB. Matrices are U32 Q4-packed; scales/biases and seven norm vectors are BF16. It shares the target embedding and lm_head, so no new 248k-vocabulary dense head is needed. Its one-layer KV is additional: at BF16 4 KV heads x256 dim x K/V, about 4096 bytes/token before allocation rounding. Original official MTP consists of 15 BF16 tensors totaling **849,398,784 data bytes**, all in official final shard18 (3,392,197,344-byte file); do not download that shard when the 239 MB standalone head exists.[64][63][56]
- `load_dflash` defaults Q4/group64 for ordinary linears, **excluding `_conv` and candidate_selector**, and leaving selector codebooks raw. Header-shape arithmetic gives **1,361,717,760 bytes (~1.268 GiB)** after that policy; 388,508,160 bytes remain unquantized. This is parameter storage estimate only; add activation, draft-window KV, target tap buffers and conversion temporaries. File download remains 3.85 GB and constructing BF16 then quantizing can transiently require BF16+Q4 residency. Do not claim a 16 GB host fits from serialized size alone.[5][58]
- Applying DSpark default Q4/g64 Linear/Embedding policy to header shapes gives approximately 1,118,584,786 bytes for RedHat and 1,044,854,226 for DimInfer/Radix. These estimates assume norm arrays stay BF16 and exclude Python/allocator/cache overhead; they are not actual loaded allocations. The exact HF weights remain several GB and on-load conversion is a separate peak risk.[5][59][60]

Do not load a complete target on either 16 GB host. Current target HF revision still resolves to supplied `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, with three files totaling 16,054,541,349 bytes including vision. Two separate 16 GB memories are not one addressable 32 GB device.[41]

## Native mlx-vlm MTP: smallest useful source route

**Primary implementation**, not MTPLX marketing: `mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py`, `config.py`, and `mlx_vlm/speculative/drafters/__init__.py`. `load_drafter(path, kind="mtp")` returns `(model, resolved_kind)`, not just the module. Original checkpoints can route through the splitter; a standalone `qwen3_5_mtp` config loads through the ordinary model loader.[1][2][3]

Minimal conceptual loading recipe (not executed here):

```python
from mlx_vlm.speculative.drafters import load_drafter
head, kind = load_drafter(admitted_local_standalone_head, kind="mtp")
assert kind == "mtp"
head.bind(final_rank_target_wrapper)
head.reset(final_rank_target_wrapper)
```

Only the admitted ~239 MB head should be loaded; no generic `mlx_lm.load` of the base target, no second full model. The existing final rank has replicated embedding/head per the supplied pipeline; `bind` accepts wrapper.language_model.model.embed_tokens and wrapper.language_model.lm_head. The standalone header keys are **already unprefixed/sanitized** (`fc.*`, `layers.0.*`, `norm.weight`, etc.). `sanitize` adds +1 to RMSNorm vectors only for raw `mtp.*` HF keys; never re-add +1 to the converted standalone norms. Quantize module structure from `.scales` keys before strict packed loading, preserving U32 weights and BF16 scales/biases. The native loader provides that pattern.[1][50][64]

Head contract: hidden5120, fc10240->5120, one **full-attention** decoder with Q24/KV4 and hd256; q projection has output gate (12288 output rows), o projection takes6144, MLP17408. Native target text attention partial RoPE factor0.25 and the checkpoint's exact rope_parameters must be retained. This is not the external DFlash32/8/128 backbone. `Qwen3_5MTPConfig.from_dict` defaults block_size to mtp_depth+2=3; `draft_block` emits block_size-1 proposals, so width2 is a deliberately smaller first experiment.[36][1][2]

Correct session alignment is not optional: `prefill_from_target_hidden` shifts input IDs by one and appends the bonus target token; it uses matching normalized target hidden rows to prefill the head. `draft_block` may use a cached seed token/hidden, then recurrently apply the one-layer head; `accept_verified_tokens` trims speculative head KV and catches up rejected/missing slots with verified target hidden states. Use those semantics, not “run head on last token repeatedly” with no teacher forcing/cache alignment.[1]

**Existing 012 adapter changes (read-only inspected):** `phased_decode.py` already has `all_logits=True`, but only returns logits; rank0 computes core.norm then head, rank1 CPU-sends hidden and returns None. Add an opt-in final normalized-hidden return alongside all-row logits; capture those hidden rows during native chunked prefill including the separately processed final prompt token. Head reset/catch-up must match fresh request boundaries. Keep rank1 free of the drafter and avoid moving full logits over TB4. Broadcast only draft IDs and accepted-count/correction information.

MLX-LM `qwen3_5.LanguageModel.sanitize` explicitly filters out `mtp.*`; its normal class has no native MTP execution path. Finding `mtp_num_hidden_layers=1` in text_config does **not** enable drafting. Parent's reported absence of built-in LM support is confirmed in primary code.[4]

## Efficient hybrid rollback: prefer captured recurrence prefix, not full trunk replay

### Bounded route for the existing mlx-lm ArraysCache

mlx-dspark has the smallest directly adaptable source: `Target.verify`, `_capture_linear`, `_rollback_gated_delta` in `target.py`. It records pre-round GDN state, already computed post-conv recurrence inputs q/k/v/a/b/A_log/dt_bias and conv input, calling original operators unchanged. On partial reject, keep **anchor + accepted proposals**; full-attention KV trims the rejected tail, conv state becomes `conv_input[:, keep:keep+3, :]`, and only `gated_delta_update` reruns accepted rows from pre-round state using the original use_kernel path. No attention, MLP or input/output projection replay is needed.[6]

Place capture/rollback **on both ranks over only their owned original layers**. Final rank decides the common accepted prefix, sends it to the first rank, and both commit before the next verification. Discard one-round captures on full acceptance/reset/exception. Keep existing grouped local evaluation and evaluated CPU send/recv boundaries; do not introduce unsynchronized GPU work across network calls. Captured block-local arrays and pre-round state must survive evaluation until the acceptance decision.

This source's comment asserts bit-identical prefix recurrence **relative to the already-computed batched verify**, not proof that the batched verify matches the existing width-one teacher. The latter remains a separate strict checkpoint gate. Scoped global monkeypatches are also not a concurrency-safe server design; bind capture per actor/owned layer or otherwise ensure no overlapping requests.[6]

### Upstream mlx-vlm temporal-cache alternative

Current mlx-vlm has a more complete cache-owned transaction API: `start_speculative_cache` starts capable recurrent caches, `SpeculativeCacheTransaction.commit(lengths)` commits accepted **input positions**, `rollback_speculative_cache` converts accepted draft counts to `accepted+1`, and abort restores initial cache state/metadata. `ArraysCache.update_recurrent` requests length-1 intermediate states during speculation; `update_window` records temporal conv views; `commit_speculation` selects states and restores length/padding metadata.[15][16][17]

This is **not drop-in** with the present mlx-lm ArraysCache and explicit state-return GDN API. It depends on mlx-vlm's cache-aware recurrence and gated_delta_with_states kernels. Full-state tapes cost about 3,145,728 bytes per token per layer (48x128x128 F32), **150,994,944 bytes per extra saved position over all 48 GDN layers**, before conv/copy overhead. A width3 tape adds about 302 MB of intermediate states globally. For tight 16 GB hosts, mlx-dspark's recurrence-prefix replay is the smaller integration and avoids saving every matrix state; accept slightly more rollback compute instead.[16][17][6]

## DFlash / DSpark / SpecForge comparison

mlx-dspark current registry selects DFlash2 for Qwen3.8 **both 4-bit and 8-bit**, and RedHat for explicit DSpark, replacing older DimInfer/Radix defaults. Its current source cites roughly2.7x on 4-bit M4 Pro with newer small-M kernels; public cards cite H200 results. Those are author measurements and code comments, not reproduced raw local tests. No per-Qwen JSON benchmark artifact was found in the fetched mlx-dspark Git tree.[5][54]

Real DFlash2 support needs its grouped dynamic convolutions plus candidate-selector predecessor/successor codebooks and hidden projection, not just the original DFlash backbone. The selector uses top16 candidates and rank256 pairwise scores. Quantizing it wholesale or silently treating DFlash2 as DFlash1 changes the proposal distribution/acceptance; sampling must use q for the actual selected proposal path.[7]

With the supplied36/28 layer split (M4 early36, M2 late28), DFlash2 taps5/19/33 are on M4 and47/61 on M2. RedHat taps4/12/20/28 are early and36/44/52/60 late. A final-rank drafter therefore needs **extra hidden-tap payloads** per verification/prefill block, original global layer-index ordering, and cache-trim/catch-up for rejected context. Native MTP needs none of these intermediate-layer gathers. Both modes need broadcast of proposal IDs/accept counts; neither repository supplies an already integrated version of this custom CPU-handoff two-rank runtime.[5][6][1]

SpecForge is **a training framework**, not an MLX distributed inference engine. Its primary current Qwen3.8 recipe has a matching DFlash2 configuration (five layers, taps5/19/33/47/61, selector256/top16, conv16/kernel2). “Disaggregated” here means **SGLang target feature extraction + Mooncake + GPU draft training**, measured on 8/16 H200/B300 GPUs. It does not mean a ready-made two-Mac inference backend. Use the released compatible head; do not start draft training to solve a23% decode gap.[11][10]

Original z-lab/dflash now also contains MLX DFlash2 classes and a GDN capture rollback in `dflash/model_mlx.py`, not merely CUDA-only reference code. Its nontrimmable-cache rollback explicitly assumes GatedDeltaNet layers. It is another useful reference, not a finished integration with the present phased distributed actor.[26]

## MTPLX: useful code, unsafe blanket adoption

MTPLX has actual Qwen3.5-family head injection and explicit GDN capture-commit support, not just model-card promises. Its qwen3_5 MTP loader detects raw delta-encoded norms and shifts them; its source documents the failure mode of successful injection with near-zero acceptance if norms remain wrong. **Do not mix this sidecar convention with the already-sanitized mlx-vlm standalone head.** MTPLX contract fields include base/mtp hidden variants, embedding_hidden concat and position mode; its released packs carry post_norm/local contracts while the module's unconfigured defaults differ.[12][38]

MTPLX “Bare Speed” and “Optimized Speed” are **different target packs**, not proof of equivalence to the exact mlx-community target. HF lists respectively 16,293,474,987 and20,683,241,105 total safetensors bytes including vision and 238,934,249-byte MTP sidecar. Optimized Speed is mixed precision (including higher-bit embedding/head/projections). Do not swap those targets for an exact-checkpoint experiment. If extracting code, use standalone native head above and retain every existing target tensor digest.[37][39][40]

Current MTPLX defaults require32 GB for the Qwen3.8-27B tier; M1/M2 automatically select FP16 sibling packs. This is a **dtype-changing product default**, not evidence of exact BF16 compatibility on this M2/M4 split. Qwen3.8 tests cap public native depth at3 and explicitly preserve a D4 crash receipt. Do not assume arbitrarily deep chaining is supported.[14][52]

### Fused GDN inspection: no immediately safe transplant found

- `gdn_step_fused.py` is compact: one-dispatch conv+silu+Q/K norm+decay/beta+GDN update+gated norm; matching10240 conv,16/48 heads,128 dims,3-row conv tail,F32 matrix state;256-thread groups. **But its epilogue is SigmoidRMSNormGated (sigmoid(z)*norm), whereas stock mlx-lm Qwen3.8 uses Qwen3NextRMSNormGated -> precise_swiglu -> silu(z)*norm.** Missing z multiplication is semantic, not merely a rounding issue. It excludes verification width>1. Do not paste it into exact target decoding unchanged.[20][25][65]
- `gdn_conv_norm.py` is a smaller independent conv+silu+Q/K normalization candidate, retaining library GEMVs. It launches1024-thread groups and its source explicitly records M2/M3 compiled-pipeline limits of896, with a real dispatch capability probe and eager fallback. Thus it is **not immediately universal on M2**. Also preserve every stock intermediate cast/rounding; expression-level equivalence is not full checkpoint parity.[21]
- `gdn_out_fused.py` is specialized to **DMODEL2560** and sigmoid gating, not this5120-wide swish-gated27B; wrong shape/semantics as-is.[22]
- `gdn_gated_norm.py` lookup-table sigmoid path has a sound installed-MLX table/selfcheck design and claims bit identity, but targets **Flash-Next sigmoid**, not this target's silu gate; no free direct adoption.[23][65]
- `native_gdn_tail.py` retains a swish-style stock tail but requires8-bit QuantizedLinear out_proj and a compiled C++/Metal extension, so the target's4-bit out_proj does not meet its eligibility gate.[24]

Most bounded fused experiment, if pursued separately: adapt only the pointwise/GDN middle with the exact SiLU gate, preserve library GEMVs and full cache outputs, support installed dtype explicitly, and gate genuine target qL1 before timing. No claim that it reaches10 tok/s follows from kernel existence.

## Numerical exactness is the hard acceptance gate

1. **Speculative sampler exactness is relative to the implemented verifier distribution.** It does not establish the verifier equals the stock width-one teacher. mlx-dspark `small_m_qmm` explicitly changes accumulation, allows1-2 BF16 ulps, and its skinny kernel describes about0.005 relative differences. MTPLX Turbo explicitly says it is not bit-exact to stock and should not be used for exactness QA. Neither upstream “lossless” label overrides the project's strict cache/logit gates.[8][9][13]
2. Current mlx-vlm supplies `Qwen3_5BatchInvariantForward` / `Qwen3_5ExactSpeculativeVerifier`: singleton-oriented projections/attention and temporal cache transactions. Its linear adapter tries optimized affine kernels then singleton fallbacks; attention has width-specific routes (including a width2 route), not an automatic proof against the installed mlx-lm runtime. Porting the full verifier entails quantized_verifier, shared ops, cache, attention/RoPE and GDN dependencies; reference it surgically rather than importing a replacement teacher and declaring equivalence.[18][19]
3. First test native MTP as a **draft-only** component (proposal errors reduce acceptance, do not authorize target drift). Gate true teacher widths2/3 against sequential width1 at identical forced inputs after identical chunked prefill. Compare full logits, every accepted live KV prefix/offset, conv tails and F32 recurrence matrices, plus multiple greedy continuations/stop reasons. Gate rollback for every possible accepted count, EOS/cap truncation, all-accept, zero-accept, fresh reset and exception abort on **each physical GPU** later; none were run in this research task.
4. Preserve the existing failed BF16 TP/verification evidence and original tolerances. If multi-row verifier is rejected, recurrence-only rollback cannot repair earlier projection or attention drift. Width-one teacher verification is a diagnostic fallback, not a speculative speedup claim.

## Bounded implementation order (proposal only)

1. Admit the small native head's exact revision/config/header and map31 tensors strictly; no base-weight changes. Reuse final-rank embedding/lm_head, retain teacher hashes. Resolve normalized-hidden/position/seed semantics against native reference.
2. Add one-proposal/width2 speculation with per-rank capture/recompute-prefix rollback patterned on mlx-dspark, not full trunk catch-up. Existing all_logits is a useful hook; add normalized hidden only as an opt-in output. Preserve CPU handoff eval boundaries and source-manifest attestation.
3. Pass tiny packed fixtures, then genuine small architecture controls and exact27B forced-prefix gates **before** physical speed claims. Initially leave skinny_qmm/MTPLX Turbo/GDN custom kernels disabled. Screen width3 only if width2 passes; add a fixed source registry for all new dependencies.
4. Measure actual commit lengths and synchronized wall-per-round components. Require <=100 ms/output-token for>=10 tok/s using the original metric, EOS convention and fresh caches. A narrow head plus width2 teacher may suffice for23%; if no cost headroom, stop rather than announcing a hypothetical factor.
5. Only if native MTP is admitted but insufficient, consider DFlash2 (higher acceptance/parallel drafting, more memory/taps) and a separately gated small-M kernel. Do not infer scaling by multiplying a M4 Pro speedup by8.13.

## Remaining uncertainty

No GPU operation or complete integration was executed. Head parameter/header compatibility is verified; actual local import/API compatibility, per-host headroom, numerical gates, native acceptance on exact quantized teacher, width2/3 verify cost and two-host throughput remain unverified. X full-body extraction was blocked by configured search-only extraction. Research found concrete code routes, **not** a confirmed10 tok/s solution.

## Sources
[1] Blaizzy/mlx-vlm: mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py
https://raw.githubusercontent.com/Blaizzy/mlx-vlm/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py

[2] Blaizzy/mlx-vlm: mlx_vlm/speculative/drafters/qwen3_5_mtp/config.py
https://raw.githubusercontent.com/Blaizzy/mlx-vlm/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/speculative/drafters/qwen3_5_mtp/config.py

[3] Blaizzy/mlx-vlm: mlx_vlm/speculative/drafters/__init__.py
https://raw.githubusercontent.com/Blaizzy/mlx-vlm/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/speculative/drafters/__init__.py

[4] ml-explore/mlx-lm: mlx_lm/models/qwen3_5.py
https://raw.githubusercontent.com/ml-explore/mlx-lm/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen3_5.py

[5] ARahim3/mlx-dspark: src/mlx_dspark/load.py
https://raw.githubusercontent.com/ARahim3/mlx-dspark/d6042f38f0aa7dd3e03a2a406111d725056d45a3/src/mlx_dspark/load.py

[6] ARahim3/mlx-dspark: src/mlx_dspark/target.py
https://raw.githubusercontent.com/ARahim3/mlx-dspark/d6042f38f0aa7dd3e03a2a406111d725056d45a3/src/mlx_dspark/target.py

[7] ARahim3/mlx-dspark: src/mlx_dspark/dflash_model.py
https://raw.githubusercontent.com/ARahim3/mlx-dspark/d6042f38f0aa7dd3e03a2a406111d725056d45a3/src/mlx_dspark/dflash_model.py

[8] ARahim3/mlx-dspark: src/mlx_dspark/small_m_qmm.py
https://raw.githubusercontent.com/ARahim3/mlx-dspark/d6042f38f0aa7dd3e03a2a406111d725056d45a3/src/mlx_dspark/small_m_qmm.py

[9] ARahim3/mlx-dspark: src/mlx_dspark/skinny_qmm.py
https://raw.githubusercontent.com/ARahim3/mlx-dspark/d6042f38f0aa7dd3e03a2a406111d725056d45a3/src/mlx_dspark/skinny_qmm.py

[10] sgl-project/SpecForge: docs/recipes/qwen3.8-27b-dflash2-disaggregated.md
https://raw.githubusercontent.com/sgl-project/SpecForge/d80daf31aacfd2193283ae36f36864e4dde199bc/docs/recipes/qwen3.8-27b-dflash2-disaggregated.md

[11] sgl-project/SpecForge: configs/qwen3.8-27b-dflash2.json
https://raw.githubusercontent.com/sgl-project/SpecForge/d80daf31aacfd2193283ae36f36864e4dde199bc/configs/qwen3.8-27b-dflash2.json

[12] youssofal/MTPLX: mtplx/qwen3_5_mtp_patch.py
https://raw.githubusercontent.com/youssofal/MTPLX/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/qwen3_5_mtp_patch.py

[13] youssofal/MTPLX: docs/turbo-verify.md
https://raw.githubusercontent.com/youssofal/MTPLX/9882703f3105363ddc37eca9f97aa09a1d387112/docs/turbo-verify.md

[14] youssofal/MTPLX: mtplx/default_models.py
https://raw.githubusercontent.com/youssofal/MTPLX/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/default_models.py

[15] Blaizzy/mlx-vlm: mlx_vlm/speculative/cache_state.py
https://raw.githubusercontent.com/Blaizzy/mlx-vlm/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/speculative/cache_state.py

[16] Blaizzy/mlx-vlm: mlx_vlm/models/cache.py
https://raw.githubusercontent.com/Blaizzy/mlx-vlm/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/models/cache.py

[17] Blaizzy/mlx-vlm: mlx_vlm/models/qwen3_5/gated_delta.py
https://raw.githubusercontent.com/Blaizzy/mlx-vlm/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/models/qwen3_5/gated_delta.py

[18] Blaizzy/mlx-vlm: mlx_vlm/models/qwen3_5/speculative_verifier.py
https://raw.githubusercontent.com/Blaizzy/mlx-vlm/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/models/qwen3_5/speculative_verifier.py

[19] Blaizzy/mlx-vlm: mlx_vlm/speculative/ops/linear.py
https://raw.githubusercontent.com/Blaizzy/mlx-vlm/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/speculative/ops/linear.py

[20] youssofal/MTPLX: mtplx/kernels/gdn_step_fused.py
https://raw.githubusercontent.com/youssofal/MTPLX/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/kernels/gdn_step_fused.py

[21] youssofal/MTPLX: mtplx/kernels/gdn_conv_norm.py
https://raw.githubusercontent.com/youssofal/MTPLX/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/kernels/gdn_conv_norm.py

[22] youssofal/MTPLX: mtplx/kernels/gdn_out_fused.py
https://raw.githubusercontent.com/youssofal/MTPLX/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/kernels/gdn_out_fused.py

[23] youssofal/MTPLX: mtplx/kernels/gdn_gated_norm.py
https://raw.githubusercontent.com/youssofal/MTPLX/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/kernels/gdn_gated_norm.py

[24] youssofal/MTPLX: mtplx/kernels/native_gdn_tail.py
https://raw.githubusercontent.com/youssofal/MTPLX/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/kernels/native_gdn_tail.py

[25] ml-explore/mlx-lm: mlx_lm/models/qwen3_next.py
https://raw.githubusercontent.com/ml-explore/mlx-lm/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen3_next.py

[26] z-lab/dflash: dflash/model_mlx.py
https://raw.githubusercontent.com/z-lab/dflash/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py

[27] incoai/Qwen3.8-27B-DFlash2: file bytes API (retrieved 2026-10-16)
https://huggingface.co/api/models/incoai/Qwen3.8-27B-DFlash2?blobs=true

[28] incoai/Qwen3.8-27B-DFlash2: pinned configuration
https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/015e795645c74b1a0eeef3b570031fb62e769bc5/config.json

[29] RedHatAI/Qwen3.8-27B-speculator.dspark: file bytes API (retrieved 2026-10-16)
https://huggingface.co/api/models/RedHatAI/Qwen3.8-27B-speculator.dspark?blobs=true

[30] RedHatAI/Qwen3.8-27B-speculator.dspark: pinned configuration
https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark/raw/87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce/config.json

[31] DimInfer/Qwen3.8-27B-Dspark-v1: file bytes API (retrieved 2026-10-16)
https://huggingface.co/api/models/DimInfer/Qwen3.8-27B-Dspark-v1?blobs=true

[32] DimInfer/Qwen3.8-27B-Dspark-v1: pinned configuration
https://huggingface.co/DimInfer/Qwen3.8-27B-Dspark-v1/raw/10b6bccfcc109bda0666d0aed4b7871aac357b99/config.json

[33] RadixArk/Qwen3.8-27B-DSpark: file bytes API (retrieved 2026-10-16)
https://huggingface.co/api/models/RadixArk/Qwen3.8-27B-DSpark?blobs=true

[34] RadixArk/Qwen3.8-27B-DSpark: pinned configuration
https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/raw/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/config.json

[35] mlx-community/Qwen3.8-27B-MTP-4bit: file bytes API (retrieved 2026-10-16)
https://huggingface.co/api/models/mlx-community/Qwen3.8-27B-MTP-4bit?blobs=true

[36] mlx-community/Qwen3.8-27B-MTP-4bit: pinned configuration
https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit/raw/b643c01b6d3b094e325edb6ebd832e16c486c575/config.json

[37] Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed: file bytes API (retrieved 2026-10-16)
https://huggingface.co/api/models/Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed?blobs=true

[38] Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed: pinned configuration
https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed/raw/b59d7002368575f08a58ba0d26686b53a9c162d6/config.json

[39] Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed: file bytes API (retrieved 2026-10-16)
https://huggingface.co/api/models/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed?blobs=true

[40] Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed: pinned configuration
https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed/raw/1d5087d2062c02b279180a53e4016cf9cd7a3d7e/config.json

[41] mlx-community/Qwen3.8-27B-4bit: file bytes API (retrieved 2026-10-16)
https://huggingface.co/api/models/mlx-community/Qwen3.8-27B-4bit?blobs=true

[45] Indexed primary X post (full extraction unavailable)
https://x.com/_ARahim_/status/2088373457784619202

[46] Indexed primary X post (full extraction unavailable)
https://x.com/_ARahim_/status/2090082453700686033

[47] Indexed primary X post (full extraction unavailable)
https://x.com/zhijianliu_/status/2089836737132650504

[48] Indexed primary X post (full extraction unavailable)
https://x.com/Youssofal_/status/2093676290985328905

[49] Indexed primary X post (full extraction unavailable)
https://x.com/redp314/status/2093588456396210479

[50] Blaizzy/mlx-vlm: mlx_vlm/utils.py
https://raw.githubusercontent.com/Blaizzy/mlx-vlm/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/utils.py

[52] youssofal/MTPLX: tests/test_qwen38_family.py
https://raw.githubusercontent.com/youssofal/MTPLX/9882703f3105363ddc37eca9f97aa09a1d387112/tests/test_qwen38_family.py

[53] mlx-community/Qwen3.8-27B-MTP-4bit: pinned model card
https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit/raw/b643c01b6d3b094e325edb6ebd832e16c486c575/README.md

[54] incoai/Qwen3.8-27B-DFlash2: pinned model card
https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/015e795645c74b1a0eeef3b570031fb62e769bc5/README.md

[56] Qwen/Qwen3.8-27B: pinned tensor index
https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json

[58] incoai/Qwen3.8-27B-DFlash2: safetensors header inspected with bounded HTTP Range
https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/015e795645c74b1a0eeef3b570031fb62e769bc5/model.safetensors

[59] RedHatAI/Qwen3.8-27B-speculator.dspark: safetensors header inspected with bounded HTTP Range
https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark/resolve/87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce/model.safetensors

[60] DimInfer/Qwen3.8-27B-Dspark-v1: safetensors header inspected with bounded HTTP Range
https://huggingface.co/DimInfer/Qwen3.8-27B-Dspark-v1/resolve/10b6bccfcc109bda0666d0aed4b7871aac357b99/model.safetensors

[63] Qwen/Qwen3.8-27B: safetensors header inspected with bounded HTTP Range
https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00018-of-00018.safetensors

[64] mlx-community/Qwen3.8-27B-MTP-4bit: safetensors header inspected with bounded HTTP Range
https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit/resolve/b643c01b6d3b094e325edb6ebd832e16c486c575/model.safetensors

[65] ml-explore/mlx-lm: mlx_lm/models/activations.py
https://raw.githubusercontent.com/ml-explore/mlx-lm/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/activations.py
