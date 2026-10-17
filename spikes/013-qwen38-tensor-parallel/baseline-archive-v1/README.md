# 013 — native Qwen3.5-family two-way tensor-parallel prototype

**Admission is BLOCKED by the retained BF16 numerical failure.** This is not a
speedup claim, exact-token claim, or an approved 27B loader deployment.

## Executed result

MLX 0.32.3 / mlx-lm 0.32.0, local Metal GPU, **two real ring processes** (not
simulated reduction). Random 4-bit affine hybrid fixture: three layers, one full
attention layer, two GDN layers, six value heads versus two key heads. The GDN
Q/K/V channel segments have lengths 32/32/192; a rank gets 16/16/96, not a simple
half of the concatenation. One KV head exercises upstream KV repetition.

- Exact serialized parameter bytes/dtypes/shapes match a genuinely upstream
  `Model.shard(group)` model on both ranks, for F32 and BF16 floating storage.
- Planned span union has complete coverage: exactly once for sharded source
  bytes, exactly twice for replicated source bytes. Chunk caps, digests,
  corruption, missing coverage, budget rejection and framed transport are tested.
- F32 prefill/decode logits and **every** live native cache state pass the
  predeclared `atol=2e-5, rtol=2e-4`; maximum logit error is 1.55e-6.
- BF16 every-cache comparisons pass, but logits fail 3/5 steps under the
  predeclared `atol=0.03, rtol=0.03`; maximum absolute difference is 0.046875.
  **Tolerance was not relaxed, the test is not xfailed, and the runner exits 2.**
  Different reduction rounding is plausible, not an established diagnosis.
- Both ranks broadcast the same observed greedy IDs `[1,89,63]`. That does not
  establish equality to single-host greedy generation: the numerical comparison
  is teacher-forced on the same broadcast continuation.
- Each rank executes all three layers at each of five forwards, with 30 native
  inference all-reduces total. `sum_gradients` adds none in inference.

Final execution: **8 passed, 1 failed**. Raw per-rank source tensor hashes,
selected tensor/chunk hashes, cache/logit rows and source-code fingerprints are
retained in `evidence/`. There were no remote jobs or large-model loads.

## Test command

From repository root:

```sh
.venv/bin/python -m pytest spikes/013-qwen38-tensor-parallel/test_tp.py -q \
  --basetemp /Users/jeremylien/.hermes/cache/scratch/tp013-review
```

The command intentionally currently fails at the BF16 gate. Test children have
45-second waits; finally kills/waits only their owned Popen children. No parent
supervisor, 011, or 012 files are changed.

## Physical small-checkpoint API (not physically tried here)

Set an explicit two-host `MLX_HOSTFILE` and distinct `MLX_RANK=0` / `1`. The host
file format is `[["rank0-address:port"],["rank1-address:port"]]`. A caller-owned
supervisor must bound process lifetime, hold process ownership, and verify
cleanup. The standalone runner does not create remote jobs or a listener.

For an already-local checkpoint on each host (one request, <=4 generated IDs):

```sh
MLX_HOSTFILE=/absolute/ring.json MLX_RANK=0 .venv/bin/python \
  spikes/013-qwen38-tensor-parallel/runner.py \
  --model /absolute/checkpoint --budget-bytes 2147483648 \
  --prompt-tokens '[1,7,3,4,8,9]' --max-tokens 3 \
  --output /absolute/new-rank0.json
```

Use rank 1 and a distinct output on the peer. Use real tokenizer-produced prompt
IDs for a pretrained checkpoint. No tokenizer/EOS/chat wrapper is implemented;
this is deliberately a fixed-cap greedy diagnostic. Output publication is
exclusive. Stdout is a single small summary; full receipts go to the output file.
`--reference-small` permits a full single-host reference only for <=64 MiB of
serialized text weights. Do not pass it for 0.8B or 27B.

### Disk-free receiver integration with an owned actor

`tp.py` exposes the bounded binary-stream API so the peer need not have the
checkpoint. The actor must send an admitted config + inventory + source hashes
out-of-band (bound to its trusted checkpoint identity), authenticate the channel,
set socket/read deadlines, and supervise both rank processes. This prototype
**does not implement an authenticated server or adapt 011's actor**.

On the source, using canonical, already-sanitized `language_model.*` tensors:

```python
tensors = inventory(checkpoint)
sources = source_attestation(tensors, max_chunk_bytes=1024*1024)
plans = plan_rank(config, tensors, target_rank)
records = (record for p in plans.values()
           for record in read_chunks(p, max_chunk_bytes=1024*1024))
send_stream(actor_owned_binary_channel, records, max_chunk_bytes=1024*1024)
```

On the receiver, `tensors` is admitted metadata only; the source paths need not
exist locally. There are no filesystem reads inside `plan_rank` or `load_rank`:

```python
plans = plan_rank(config, tensors, group.rank())
model, receipt = load_rank(
    config, tensors, plans, group,
    receive_stream(actor_owned_binary_channel, max_chunk_bytes=1024*1024),
    budget_bytes=explicit_admitted_budget,
    max_chunk_bytes=1024*1024,
)
cache = model.make_cache()
with cpu_collectives(group):
    logits = forward(model, identical_prompt_ids, cache)
    token = broadcast_token(logits, group)
```

Compare the **exact** replicated digest key sets/values across ranks, as the
runner does, and retain `sources`, the plan, and receipts with the actor's
admitted config/inventory identities. The framed transport hashes detect damaged
chunks; they are not authentication or a replacement for trusted source binding.
The local-checkpoint runner additionally records full-source tensor hashes using
bounded reads, which adds source disk I/O and is not timed as inference.

## Loading and execution constraints

- Uses installed `qwen3_5.Model.shard`, native quantized projections, GDN kernels,
  native KVCache and ArraysCache. No new Metal kernels or fake layer ownership.
- Full-shaped constructor and quantization graphs remain **lazy**. Rank slices
  replace every parameter before any final model evaluation. Only tiny tests
  deliberately materialize full random/reference models.
- Row/column plans preserve packed U32 words and matching affine scales/biases.
  Native sharded shapes are an admission oracle. Unknown parameter sets, role
  dtypes, shapes, MoE, and unsupported model types fail closed.
- Payload reads are capped by the explicit chunk size (default 1 MiB); a chunk
  may temporarily have a bounded number of Python copies. Receiver chunks are
  immediately evaluated into MLX arrays. Final assembly can temporarily hold one
  additional selected tensor; `budget_bytes` must cover total retained parameter
  bytes **plus the largest selected tensor**. This is an allocation budget, not
  proof of available physical RAM. Cache/activations/runtime require headroom.
- A tensor larger than a transport chunk is supported without full CPU tensor
  residency. Full source tensor hashing is also chunked. Source size/mtime is
  checked before and after reading; hashes are retained separately.
- Every layer and its complete native state is evaluated. Native all_sum is
  wrapped only inside a locked, process-global single-threaded scope, evaluates
  its GPU producer before explicit CPU-stream reduction, evaluates the result,
  and restores the original function in `finally`. Do not run other model
  threads concurrently with this diagnostic scope.

**Next gate:** independently review the source/receipts and diagnose the BF16
failure before any physical pretrained or 27B admission. Neither this small
fixture nor constant-input collective timing predicts end-to-end acceleration.
