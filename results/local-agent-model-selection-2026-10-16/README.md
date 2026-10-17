# Model selection for a two-Mac autonomous coding and computer-use agent

**Research date: October 16, 2026.** Hardware: 16 GiB M2 Pro MacBook Pro +16 GiB M4 Mac mini, private Thunderbolt Bridge, external M.2 SSD on the mini. Goal: maximum useful local coding, terminal/tool and visual computer-use capability, eventually available 24/7 and able to finish authorized tasks without corrective prompts.

## Recommendation

**Qwen3.8-27B is the best-supported all-round model choice for this hardware and combined job—not because it is the easiest model, but because the capability evidence does not justify replacing it with an older, much more aggressively compressed 80B model.**[1][5]

Keep the exercised 4-bit model as the control.[9] Treat 6-bit as a quality-preservation candidate only if the full two-host vision/context/runtime budget fits and measured task quality justifies its cost.[19] Neither choice is yet a qualified hands-off service.

**Full Qwen3.8-Flash-Next is the higher-ceiling research challenger.** It has stronger published coding results.[2] However, it is not presently a source-verified, ready-made two-Mac SSD-streamed visual-agent deployment. The SSD removes the download-capacity obstacle; it does not solve residency, transport, backend support or reliability.[3][6][30]

**Ornith-1.5-35B-A3B4-bit is the most interesting additional sparse coding/terminal challenger in this search.** Test it for faster successful tasks rather than infer greater intelligence from its total parameter count. Its first-party MLX export is text-only; a full visual deployment is a separate qualification.[17][18]

These are evidence-weighted recommendations, not a proof that one exact local quantization wins every task. No candidate has demonstrated this user's full no-steering/24-hour requirement on these Macs.

## 1. What the attached M.2 actually changes

Read-only remote `diskutil`, `df`, `ioreg` and `sysctl` discovery found a **Samsung SSD990EVO Plus2TB**, through a **Ugreen USB enclosure negotiated at10,000,000,000bit/s**. This is not a40Gb/s Thunderbolt SSD link. The writable ExFAT volumes had926,501,699,584 and520,171,028,480bytes free. No drive formatting, writes or throughput benchmark was performed. Exact observations are preserved in `hardware-discovery.json`.

The raw interface ceiling is1.25 decimal GB/s before protocol, filesystem and random-read overhead. This is a calculation from negotiated link rate, not a measured storage result. For illustration only, at10 generated tokens/s this link could carry at most125 decimal MB of new disk reads per token before any overhead. Actual expert-cache misses are unknown.

**Resident two-host execution:** the external drive principally helps provisioning/loading. **Expert streaming:** uncached weights repeatedly consume storage bandwidth during inference. The two ExFAT volumes share one physical drive; they are not independent I/O devices. Published internal-SSD streaming results cannot be transferred to this enclosure.[6]

## 2. The candidate shortlist

Weight sizes below are exact publisher-file sums converted toGiB. They exclude cache, prompt/image workspaces, runtime allocation and two operating systems. A capacity-plausible file is not an admitted runtime.

| Candidate | Capability fit | Actual weight footprint | Decision |
|---|---|---:|---|
| **Qwen3.8-27B 4-bit** | Native vision-language; coding, tools and computer-use evidence |14.952GiB, including stored vision weights | **Primary reference and deployment target.** The current custom runtime exercises text only; full vision is not already enabled.[1][9] |
| **Qwen3.8-27B 6-bit** | Same model family with less aggressive weight quantization |21.214GiB | Two-host quality candidate, not an assumed fit/speed win. Replication, vision, context and workspaces must fit per host. Existing4-bit custom kernels/MTP proof does not automatically transfer.[19] |
| **Full Flash-Next GSQ-RCO Q2_0** | Native vision-language; stronger published coding ceiling |35.040GiB transformer +26.822GiB disk-table artifact; vision projector extra | Transformer alone exceeds combined physical32 GiB. Requires genuine expert offload/streaming, not just disk mapping the table.[2][3][10] |
| **DASLab pruned Flash Coder** | Coding-specialized; vision represented during pruning, but GUI retention unmeasured |27.575GiB transformer +26.822GiB table +0.845GiB vision projector | Capacity-edge/offload experiment, not proven best autonomous all-round model.[4][11] |
| **Qwen3-Coder-Next80B** | Coding/terminal specialist; text-only |MLX4-bit41.764GiB; Unsloth low-bit GGUF21.711/23.248GiB; sampled MLX2-bit23.226GiB | Larger original model does not establish better deployed coding. Low-bit quality and headroom are unproven; cannot interpret screenshots by itself.[5][12] |
| **Ornith-1.5-35B-A3B** | Strong sparse coding/terminal/tool challenger |First-party MLX4-bit18.169GiB | Worth testing for successful-task throughput. First-party MLX export is text-only; do not claim ready screenshot support.[17][18] |
| **Qwen3.6-35B-A3B** | Native multimodal sparse alternative |MLX4-bit19.001GiB | Potential compute/latency trade, not a demonstrated intelligence upgrade over3.8-27B.[7][21] |
| **Gemma4-31B** | Multimodal and native function calling |MLX4-bit17.148GiB | Legitimate alternate-family comparator; the inspected card does not establish superior terminal/GUI autonomy.[13][14] |
| **GLM-4.7-Flash** | Text-only sparse model, not the visual GLM variant |MLX4-bit15.695GiB | Secondary tool-use comparator, not the best-supported one-model answer to all three jobs.[8][22] |

Devstral-Small-2-24B and KAT-Coder-V2.5-Dev were also reviewed as coding-specialist comparisons; neither supplied stronger relevant evidence sufficient to displace the recommendation. Do not confuse a conditional-generation architecture name with retained vision: KAT explicitly describes its release as text-only.[23][24]

The27B 8-bit artifact is27.475GiB before runtime/OS overhead, so it is not a sensible default resident target on these two16 GiB hosts. By contrast,6-bit is worth budgeting, not declaring safe merely because its file sum is below32 GiB.[20][19]

## 3. Why the existing27B remains a serious intelligence choice

The official Qwen3.8-27B card reports **Terminal-Bench2.1(Terminus)73.0**, **SWE-bench Pro61.7**, **OSWorld-Verified84.3**, and **WebArena-Verified64.8**. Those are original-model/harness results—not scores measured for the community4-bit model, our distributed runner, or pure screenshot-only interaction.[1]

Flash's direct comparison table reports **SWE-bench Pro62.5 versus61.7** and **DeepSWE1.1 58.7 versus42.2** for Flash versus27B. The DeepSWE scaffold/best-harness handling differs, so the latter is not an identical-execution comparison everywhere. Flash's OSWorld2.0 table gives binary19.4 for both and partial52.3 versus48.0. **OSWorld2.0 is not the same evaluation as OSWorld-Verified**; comparing19.4 to84.3 would be invalid.[2]

DASLab's pruned Flash Coder reports SWE-bench Verified75.60 versus82.80 for its BF16 Flash reference, and LiveCodeBenchv6 86.28 versus87.43, atxhigh reasoning effort. It does not publish preserved OSWorld, terminal-agent or long-horizon unattended success. Its effective1.89bpw is pruning plus retained-weight quantization, not every original weight stored uniformly at1.89bits.[4]

Ornith reports SWE Verified79.0, SWE Pro59.6 and Terminal-Bench2.1/Terminus-2 67.8, with different OpenHands/context/time-budget choices. These are useful challenger evidence, not a direct leaderboard win against Qwen3.8. Its modified template/reasoning handling must be preserved in any actual trial.[17]

Gemma4 and the GLM/Qwen3.6 specialist alternatives support additional family comparisons, but parameter counts, isolated function-call scores and published full-precision results cannot decide the exact low-bit local winner.[13][21][22]

## 4. Coding, terminal tools and visual computer use are different requirements

A text-only coding model can use shell commands, file tools and browser DOM/accessibility tools. It cannot itself understand screenshot pixels. Adding a random vision projector does not turn Coder-Next into a VLM. A correctly deployed Qwen3.8 VLM, however, already has the relevant vision encoder; it does not inherently require a different helper model.[1][5]

Screen grounding is not complete desktop-task success. OSWorld supports multiple observation/action configurations; the model cards do not establish every score as screenshot-only mouse/keyboard operation. WebArena's official grading under an OSWorld scaffold also should not be casually labeled either pure-visual or DOM-only. Nothing here establishes that an autonomous agent can operate arbitrary applications indefinitely without intervention.[1][2]

Our current27B physical evidence is **text-only generation**, with vision weights intentionally excluded from the exercised loading inventory. Reusing that proof as evidence of screenshot handling would be incorrect. The existing broad suite measures generation behavior, not applied code-patch success, browser task completion or autonomous task reliability.

## 5. Actual runtime paths and gaps

### MLX-LM

InstalledMLX0.32.3/MLX-LM0.32.0 already contain Coder-Next pipeline hooks and distributed serving. Current pinned upstream source likewise supports a distributed text-model pipeline and prefix caching/tool parsing. A new distributed engine is not inherently required for that candidate.[25][26]

The audited80B affine2bit shard calculation is11.7034GiB per host at24/24layers, including replicated embedding/norm/head, before KV/SSM, graph, scratch, runtime and macOS. It is marginal capacity evidence, not a tested peak fit. Current upstream distributed single-generation cancellation raises `NotImplementedError`; this was independently verified in source and needs admission/fixing before unattended use.[26]

### Full multimodal two-Mac execution

The audited MLX-VLM loader checks for `model.shard()` or an inner `pipeline()` method. At pinned commitb8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0, the inspected Qwen3.5-family VLM wrapper/language ancestry lacks those methods. Generic distributed-loader documentation therefore does not establish ready Qwen3.8 multimodal sharding. Parent independently checked class ancestry/method inventories.[27][28]

This is an implementation/validation gap, not a claim that two-Mac vision is impossible. Likewise, our custom4-bit MTP integration is not automatically a stock persistent API server, a6-bit verifier or a visual-language serving path.

### llama.cpp and Strata

llama.cpp provides cross-host RPC with Metal devices. Its RPC backend remains explicitly experimental/insecure and needs private-link restriction and physical model/operation admission. A local Metal implementation alone does not certify every remotely executed novel Flash graph.[29]

Strata's Metal expert streaming is single-host, not distributed across these Macs. Mac docs explicitly leave16 GiB hosts and external-SSD streaming untested, serialize model requests, and do not set up Swift/Coder variants. Its narrow vision evidence and short32 GiB soaks do not qualify our hardware for unattended service.[6]

Full Flash uses architecture identifier `qwen4_exp`; it must not be silently treated as `qwen3_next` or the dense27B's `qwen3_5` reuse. Actual model architecture and backend must match.[30]

## 6. Important research reconciliation: two different Flash Coder releases

One engine sub-audit examined **Jab1718's160-expert, text-only, no-PLE subnet**, not **DASLab's256-of512-expert pruned VLM with a separate n-gram shard and vision projector**. Its text-only conclusion and26.444GiB Q4 figure apply to that Jab release, not DASLab's release. We explicitly reject transferring those conclusions across the shared informal name “Flash Coder.”[31][4]

Similarly, the engine audit's67.564GiB smallest fetched Unsloth full-Flash download is not the global minimum: the separately verified DASLabQ2_0 pair totals61.862GiB before vision. The final capacity table uses exact candidate identities rather than the conflicting shorthand.[3][10]

## 7. The no-steering,24/7 requirement

No retrieved evidence establishes arbitrary, indefinite, fully hands-off computer work from any candidate on this setup. A selected model plus an API endpoint is not the complete system. Reliable authorized autonomy also requires durable task state, correct tool-result continuation, bounded retries, tests/completion checks, resource limits, safe stop conditions and process/link recovery. Hermes can connect to a local compatible endpoint, but its framework features do not prove the chosen local model's competence.[15][16]

Interpret “no additional prompting” as: after receiving an authorized task/queue once, perform routine planning, tools, recovery and verification without corrective nudges. If the task is impossible, ambiguous beyond its assigned scope, requires inaccessible credentials, or exceeds its permissions, fail clearly rather than invent success or bypass safeguards. An unattended service is not permission to make unrelated destructive changes or external commitments.

Proposed qualification, **not executed**:

1. Frozen real repository bugfix/test/multi-file tasks, plus terminal build/debug tasks and reversible screenshot-based GUI tasks. Count all attempts and every human intervention.
2. Use the exact deployed quantization, template, reasoning mode, tool parser and agent harness. Judge applied diffs and actual test outcomes, not self-reported success. Compare4-bit/6-bit27B and a sparse challenger only after live memory admission.
3. Measure complete successful-task time, first uncached response, repeated-prefix turns, tool-call validity, recovery/loops, context growth, memory pressure and swap-out. Optimize **verified tasks per elapsed time**, not just warm decode tokens/s.
4. Exercise cancellation, queue limits, context exhaustion, peer/link loss, restart/resume and idle-to-request behavior. Then run a declared 24–48-hour mixed idle/load soak while normal development apps are open.
5. Configure persistent supervised startup only after passing the bounded tests. Current AC sleep settings permit awake operation, but the mini's automatic restart after power loss was disabled at inspection; no setting was changed.

## Final decision and status

**Model choice:** Qwen3.8-27B remains the primary all-round target across BOTH Macs. Start with the known4-bit control; consider6-bit only if full-stack memory and measured quality/performance justify it. **Performance challenger:** Ornith1.5-35B-A3B4-bit, with vision caveats. **Higher-ceiling experiment:** full Flash-Next, only after a concrete two-host offload/vision design and measured external-I/O budget.[1][2][17]

**Research complete; setup and all additional autonomy criteria are NOT complete.** No new weights were downloaded, model inference executed, services installed, power settings modified or volumes reformatted during this research. The previous27B runtime is still the only locally exercised candidate, and its evidence does not cover full visual-agent or24/7 readiness.

Supporting files record exact publisher revisions, source-check dispositions, per-variant sizes, hardware discovery and proposed autonomy acceptance. The source list below is generated from the retrieval ledger rather than invented from memory.

## Sources

[1] https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/README.md — Official Qwen3.8-27B card
[2] https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/README.md — Official Flash-Next card
[3] https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/raw/main/README.md — DASLab full Flash quantization results
[4] https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/raw/main/README.md — DASLab pruned Flash Coder results
[5] https://huggingface.co/Qwen/Qwen3-Coder-Next/raw/main/README.md — Official Coder-Next card
[6] https://raw.githubusercontent.com/Niko1221/Strata/296b59d2f73c8f04a805df29c2fa97655adace78/docs/MAC.md — Strata pinned Mac documentation
[7] https://huggingface.co/api/models/mlx-community/Qwen3.6-35B-A3B-4bit?blobs=true — mlx-community/Qwen3.6-35B-A3B-4bit exact file sizes
[8] https://huggingface.co/api/models/mlx-community/GLM-4.7-Flash-4bit?blobs=true — mlx-community/GLM-4.7-Flash-4bit exact file sizes
[9] https://huggingface.co/api/models/mlx-community/Qwen3.8-27B-4bit?blobs=true — mlx-community/Qwen3.8-27B-4bit exact file sizes
[10] https://huggingface.co/api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF?blobs=true — ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF exact file sizes
[11] https://huggingface.co/api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF?blobs=true — ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF exact file sizes
[12] https://huggingface.co/api/models/unsloth/Qwen3-Coder-Next-GGUF?blobs=true — unsloth/Qwen3-Coder-Next-GGUF exact file sizes
[13] https://huggingface.co/google/gemma-4-31B/raw/main/README.md — Official Gemma4 model card
[14] https://huggingface.co/api/models/mlx-community/gemma-4-31b-it-4bit?blobs=true — Gemma4 exact4bit weights
[15] https://hermes-agent.nousresearch.com/docs/user-guide/local-models — Hermes local-model runtime and context management
[16] https://hermes-agent.nousresearch.com/docs/guides/local-llm-on-mac — Hermes Mac serving guide
[17] https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/raw/main/README.md — Official Ornith1.5-35B-A3B card
[18] https://huggingface.co/api/models/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit?blobs=true — Ornith first-party MLX4bit exact size
[19] https://huggingface.co/api/models/lmstudio-community/Qwen3.8-27B-MLX-6bit?blobs=true — Qwen3.8-27B MLX6bit exact size
[20] https://huggingface.co/api/models/mlx-community/Qwen3.8-27B-8bit?blobs=true — Qwen3.8-27B MLX8bit exact size
[21] https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md — qwen36 official card
[22] https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/README.md — glm47 official card
[23] https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/raw/main/README.md — devstral official card
[24] https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev/raw/main/README.md — kat official card
[25] https://raw.githubusercontent.com/ml-explore/mlx-lm/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen3_next.py — MLX-LM pinned CoderNext pipeline
[26] https://raw.githubusercontent.com/ml-explore/mlx-lm/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/server.py — MLX-LM pinned server/cancellation gap
[27] https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/utils.py — MLX-VLM pinned sharded loader
[28] https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/qwen3_5/language.py — MLX-VLM pinned Qwen3.5-family language implementation
[29] https://raw.githubusercontent.com/ggml-org/llama.cpp/ea0c4c2c85fb183ec0fbd30a1af4e4275a18a064/tools/rpc/README.md — llama.cpp pinned RPC support/warnings
[30] https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json — Official full Flash config
[31] https://huggingface.co/Jab1718/qwen3.8-flash-coder-85gb-bf16/raw/main/config.json — Distinct Jab1718 text-only Flash subnet config
