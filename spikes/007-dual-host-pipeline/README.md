# 007: Two-host pipeline inference over Thunderbolt Bridge

## Question

Given an M2 Pro MacBook Pro and an M4 Mac mini, each with 16 GB unified memory,
can a real quantized Qwen model execute across their GPUs over the existing
Thunderbolt Bridge, with explicit per-host layer ownership and genuine output?
Can a larger-than-one-host-resident checkpoint then generate safely?

## Verdict: VALIDATED for a short request — full 32B generation completed

Real two-host numerical communication and 0.5B generation are verified, including
loading the MacBook's tensors directly from the mini without local weight files.
On October 5, 2026, a guarded phased run completed genuine full 32B generation on
both physical GPUs: all 64 layers executed, both ranks returned the same 41 token
IDs including EOS, exact trusted-index coverage and replicated digests passed,
and owned-process/listener cleanup passed. The earlier GPU timeouts and stalled
trial are retained below, not rewritten as successes.

No distributed speedup or usable contiguous 32 GB memory pool is claimed. This
remains a standalone spike, not integration of the project's custom Metal fusion
or compiled decoder. One short completion does not establish production reliability
or full-model numerical/quality parity.

### Successful guarded 32B trial

The `[26,38]` reverse split placed layers 38..63 on the M2 Pro and 0..37 on the M4.
Stored parameter bytes were 8,007,776,256 and 11,299,411,968 respectively; peak
MLX allocations were 8,044,165,296 and 11,348,383,920 bytes. Those are not OS
residency or total system memory measurements.

For `What is 2 plus 2? Explain briefly.`, the actual response began
`2 plus 2 equals 4.` and completed with EOS. Rank-0 request wall time was
64.878 seconds, including 56.523 seconds to the first token. The remaining 40
token events took 8.355 seconds (4.788 events/second); whole-request throughput
was 0.632 events/second. Total trial elapsed time including loading and cleanup
was 89.179 seconds, below its 180-second deadline. These are one smoke request,
not a warmed or paired performance benchmark.

Before it, a guarded physical 0.5B phased request matched prior control tokens
exactly and passed cleanup. The durable supervisor/ownership repair passed
independent review; the later closed-port timeout-only adjustment was parent
reviewed and regression tested. Exact executed source and this distinction are
preserved with the results.

Evidence: `results/m2pro-m4-dual-host-phased-32b-v1/`, including redacted rank
receipts, all-layer phase logs, trusted metadata, frozen source, review receipts,
programmatically verified summaries and a SHA-256 inventory.

## Verified physical-host results

- Both runtimes use Python 3.13.2, MLX 0.32.3 and MLX-LM 0.32.0.
- A strict two-rank ring passed all-sum, all-gather and bidirectional send/receive
  checks over the cable's bridge addresses.
- The saved Qwen2.5-0.5B mixed checkpoint and tokenizer were copied to the mini;
  all 11 files were SHA-256 checked against the MacBook.
- Layers 12..23 ran on the MacBook and 0..11 on the mini. Three requests for each
  single-host control and each pipeline rank produced identical greedy token IDs
  and `2 plus 2 is 4.` text. The nine events include EOS; the 32-token cap is not
  an actual generation count.
- Full-model stored parameters: 260,563,712 bytes. Pipeline stored parameters:
  171,839,232 bytes on the MacBook and 165,302,016 on the mini. Replicated
  embedding/norm/head arrays make their sum larger than the full-model payload.
- A subsequent physical two-host trial streamed only rank-0 tensors to a
  tokenizer-only MacBook directory. Both ranks' three requests matched each
  other and the earlier single-host controls; the receiver had no `.safetensors`
  files. Per-tensor digests and complete retained architectural keys were checked.
- These are short sequential correctness/smoke runs, not paired performance
  evidence. The earlier small pipeline was slower than the single-host controls.

Evidence:

- `results/m2pro-m4-dual-host-small-v1/`: initial controls and collectives.
- `results/m2pro-m4-dual-host-streamed-small-v1/`: actual disk-free receiving
  trial, raw rank records, exact executed loader/probe source, integrity receipt.

## Larger checkpoint: pinned download and historical trials

Model: `mlx-community/Qwen2.5-32B-Instruct-4bit`, immutable revision
`2938092373e5f97b95538884112085364c2da315`.

The full checkpoint is stored only on the mini. All 12 downloaded files were
verified against the pinned metadata: SHA-256 for LFS weight files and Git blob
hashes for the small assets. Copied tokenizer/config assets were separately
SHA-256 checked. Four safetensors files total 18,431,478,459 bytes; their 1,671
tensor payloads total 18,431,289,344 bytes. Both exceed one host's
17,179,869,184 physical bytes before runtime overhead.

The plan was recomputed from the actual downloaded headers, not only HTTP range
preflight data. Reverse split `[28, 36]` gives:

| Host / rank | Original layer indices | Stored parameter bytes |
|---|---|---:|
| M2 Pro / 0 | 36..63 | 8,556,382,208 |
| M4 / 1 | 0..35 | 10,750,806,016 |

Non-layer replication is 875,898,880 bytes per rank. Layer coverage is disjoint
and exhaustive. A header plan does not itself prove all architectural parameters
or runtime residency; the streamed loader separately checks the retained model's
complete names and shapes before generation.

### Preserved failed trials

1. Stock pipeline communication: mini reported Metal GPU timeout during native
   generation; neither rank produced its final request record.
2. Recommended MLX process-local wired limit applied before loading: same mini
   GPU timeout. This did **not** fix the failure. This attempt lacks an archived
   byte-exact source snapshot; its provenance limitation is explicit.
3. Eager CPU-stream send/receive/all-gather boundaries: the controller tool
   expired at 300 seconds, before its internally configured 600-second deadline.
   Both final records were absent. Surviving task-owned processes were explicitly
   killed and subsequently checked gone. The cause of the stall is unresolved.

The third trial also exposed a supervisor bug: tool death can outlive an inline
Python controller and bypass its cleanup. Do not repeat a large run through a
300-second `execute_code` cell with a longer internal deadline. A durable bounded
supervisor must persist startup hashes, remote process IDs and phase records,
and handle rank failure and controller termination before another large trial.

During the stalled trial, MacBook swap reached 18,788 MiB and boot-volume free
space fell to about 7 GiB; after termination, free space recovered to about
13 GiB. These are system snapshots, not model-only residency measurements.
Those failed trials did not establish complete runtime allocation or throughput.
The later guarded completion above reports actual peaks and timings, not task
accuracy or safe residency under arbitrary background workloads.

### Bounded kernel diagnostic

A separate mini-only diagnostic loaded one actual 32B checkpoint layer plus the
replicated non-layer tensors and changed the diagnostic model's layer count to
one. Prefill `[1,32,152064]` and cached-decode `[1,1,152064]` logits were finite,
with float16 outputs and 1,175,580,948 bytes peak MLX allocation. This establishes
that those large-shape operators execute in isolation. It is **not** full 32B
inference, a capacity result, or evidence of correct full-model logits.

Download verification, actual-header plan, three failure artifacts, available
executed sources and the bounded diagnostic are preserved in
`results/m2pro-m4-dual-host-32b-attempts-v1/`.

## Probe and loader

`probe.py` disables remote model/tokenizer code and admits built-in Qwen2 only.
Distributed mode requires `MLX_HOSTFILE` and `MLX_RANK`, strict `ring`
initialization and exactly two ranks. Both ranks must use the same checkpoint,
split, prompt, greedy sampling and generation limits.

- `--stream-local`: load only owned tensors directly from local source files.
- `--weight-server ADDRESS:PORT`: load owned tensors from the one-shot mini
  service into a local tokenizer/config-only directory. Use an ephemeral
  `PARETOQUANT_WEIGHT_TOKEN`; never publish its value.
- `--cpu-communication`: explicit diagnostic scheduling alternative. It
  materializes GPU outputs before CPU-stream communication and restores the
  process-global wrappers on exit. It is single-threaded scope, not a general
  distributed runtime. This mode passed local two-rank small-model regression
  tests but did not validate 32B generation.
- Ordinary full-checkpoint lazy loading is retained for small controls only.
  Do not use it to load the entire 32B checkpoint on the MacBook.

`stream_server.py` binds the specified cable address, admits one specified peer
and one session, serves only the planned rank and exits. The connection is not
TLS-encrypted: use only the trusted direct cable link, never an untrusted LAN.
Tensor frames are bounded at 512 MiB, metadata at 2 MiB, and payloads are hashed.
`materialize.py` is an optional bounded-chunk disk-shard path; no large MacBook
shard was written. Stored-byte admission is not a peak memory guarantee.

## Tests and remaining gates

The initial native spike gate passed **112 tests**. After fixing three independent
review findings (packed runtime dtypes, cumulative disk-reserve checks and
configuration/asset copy stability), the full native gate passed **136 tests**,
with clean Ruff checks and formatting. The final guarded-run native gate passed
188 tests and 48 subtests, with one optional CPU F16 skip and clean Ruff checks.
Native packed-F16 GPU parity and CPU default-device restoration are also tested.
Tests cover strict header parsing, native null metadata, packing/partition
coverage, disk protections, F32/BF16 streamed logits, real saved-generation
parity, two-rank numerical communication and optional CPU-boundary restoration.

Run from the repo root with the native project environment:

    python -m pytest spikes/007-dual-host-pipeline -q

Completed: durable bounded supervision, phase/source/PID evidence, physical
small-model control parity and a full 32B request with verified cleanup.
Still required before broader claims: repeated and longer prompts, fuller
per-host residency/pressure observations, independent full-model quality controls
and paired performance measurements. No 32B task-accuracy, distributed speedup
or custom fused-kernel claim may be inferred from this experiment.

SSH keys stay outside the repository. Public evidence contains no session
credentials, private keys or checkpoint weight payloads. Downloaded public
model files are retained; the experimental code/evidence are not yet published.
