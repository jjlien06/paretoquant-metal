# 014: opt-in compiled GDN decode runs

**Parity passed; subsequent full-model timing found no useful acceleration.**
Isolated clone of 011. Nothing under 011/012/013 is modified by this spike.

The parent completed physical-small and full27B native/compiled/native gates.
The same weights and output tokens were retained, with stable six/ten traces
on the MacBook/mini. Warm decode differed by only about0.6% from pooled native
controls, so compilation was not adopted. Exact trials and limitations are in
`../../results/qwen38-speed-v1/README.md`. The local-gate evidence below remains
separately labeled; it is not the full-model benchmark.

## API and execution scope

```python
from compiler import GDNCompiler
from phased_decode import phased_generate, phased_forward

# Construct after loading/quantization/eval; weights remain immutable thereafter.
compiler = GDNCompiler(model)
# Reuse the same object across requests; each generate still creates fresh caches.
result = phased_generate(
    model, tokenizer, prompt, eval_group_size=4, compiler=compiler, max_tokens=16
)
# Explicit-cache calls also accept compiler=compiler.
```

- No compiler object means the original native phased path. `GDNCompiler` belongs
  to exactly one loaded model; `phased_forward` rejects another model's compiler.
  Do not replace weights or layer ownership while reusing it.
- Only initialized batch-one, width-one GDN runs with two native SSM arrays per
  layer and `left_padding is lengths is None` are compiled. Masked, empty,
  uninitialized and multi-token calls take native arithmetic. Training layers
  are ineligible. No fixed-capacity KV scheme or custom arithmetic is introduced.
- Within an existing evaluation block, maximal consecutive GDN runs are compiled
  separately, stopping before full attention and the existing block boundary.
  Ragged runs and pipeline-owned intervals are supported. Group one compiles
  eligible singleton GDNs, without moving its per-layer evaluations.
- Each graph takes `(hidden, tuple(tuple(native_cache.cache), ...))` and returns
  `(hidden, new_states)`. A trace-local `ArraysCache(size=2)` adapts native layers;
  only returned arrays are published to live caches. No live request cache is
  captured and no `nn.Module` weights or global class methods are patched.
- Embedding, multi-token prefill, full attention/KV, head, CPU communication,
  `mx.eval` boundaries and layer-completion publication remain outside compilation.
  The separately processed last prompt token, and any initialized width-one
  prefill remainder, can compile: eligibility is width-based, not phase-based.
  Completion events follow successful evaluation of hidden state and every live
  cache. Compiler and GPU exceptions propagate; no silent runtime fallback.
- `compiler.receipt()` reports Python `trace_count`, `compiled_calls` and
  `compiled_functions`; traces are not GPU execution counts. Request receipts
  contain cumulative snapshots, not per-request deltas.

## CLI and admission

Both `probe.py` and `supervise.py` support **`--compile-gdn`**, default off.
It requires phased execution. Keep `--eval-group-size 4` in both comparison
trials. This minimal spike uses separate trials, **not an interleaved compile-mode
schedule**. It does not inherit 012's larger group sizes/scheduling extensions.

The supervisor forwards the flag to both ranks and binds it in admission args.
Rank and request receipts record the mode/counters; startup records include the
mode and exact source hashes. Supervisor rank verification rejects a different
mode or absent/zero compiled execution evidence for a compiled trial. Thus an
all-fallback trial is not certified as compiled acceleration. `compiler.py` is
explicitly included in this spike's fixed `remote_actor.SOURCES` registry.

Single-host CLI parity smoke (already run, two fresh-cache requests):

```sh
.venv/bin/python spikes/014-qwen38-compiled-gdn/probe.py \
  --mode generate --single-host --phased --compile-gdn --eval-group-size 4 \
  --model /Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/small-control \
  --prompt 'Say hi.' --max-tokens 4 --repeats 2
```

Parent source review and physical-small gates passed before the subsequent
27B trials. This directory retains the 011 transport, inventory, source hashing,
cleanup and admission mechanisms. Networking was not part of the original local
gates below; the later physical trial evidence is in the linked result directory.

## Verification results (Apple M2 Pro, MLX 0.32.3)

- **47 tests passed**, including the explicitly enabled local pretrained gate;
  ordinary runs skip only that opt-in pretrained test (46 pass, one skip).
- Tiny 4-bit F32 and BF16 fixtures compare full native prefill/decode logits,
  every SSM array and live KV prefix/offset, and greedy tokens, using unchanged
  **`atol=1e-6, rtol=1e-5`**. Runs include three-GDN groups, ragged singleton
  tails, different-length fresh-cache resets, stable graph traces, clipped
  pipeline ownership at four splits, completion events, flag admission,
  source registry, model binding and propagated compiler errors.
- Nine-layer tiny fixture: **3 functions / 3 traces / 36 compiled calls** per
  dtype across three requests; parameter array identities unchanged. Observed
  test-process peak MLX allocation: **2,493,358 bytes** (shared suite peak).
- Actual local pretrained 0.8B architecture control: three requests with prompt
  lengths **18, 15, 18**, native chunked prefill and four greedy output tokens
  each. **Maximum absolute logit difference 0.0**; every live cache passes the
  same tolerances. **6 functions / 6 stable traces**, cumulative calls
  **30 / 54 / 84**. Final peak MLX allocation **569,477,628 bytes**, while both
  reference and compiled cache sets coexist. This is allocator accounting, not
  system residency.
- Real `probe.py` CLI: exit zero, tokens `[9419, 0, 2500, 628]` in both requests,
  six traces throughout, cumulative compiled calls **24 / 48**, peak MLX
  allocation **464,927,748 bytes**. Startup attests all eight source files.
- Ruff check and format check pass. No tolerances were relaxed, no numerical or
  GPU failure was suppressed, no full checkpoint was loaded, and no benchmark
  was performed. A parity gate is not performance evidence or 27B validation.

Evidence:
- `small-control-parity-final.json`: final-source local parity/counter/memory gate.
- `small-control-parity.json`: preserved earlier pre-format run, with its own
  original hashes; not an attestation of final sources.
- `small-cli.stdout.jsonl` / `small-cli.stderr.jsonl`: actual CLI final receipt and
  startup/phase/source/memory records. Timing fields are incidental, not a paired
  performance experiment.

Reproduce:

```sh
.venv/bin/python -m pytest spikes/014-qwen38-compiled-gdn -q
.venv/bin/ruff check spikes/014-qwen38-compiled-gdn
.venv/bin/ruff format --check spikes/014-qwen38-compiled-gdn
PARETOQUANT_SMALL_CONTROL=/Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/small-control \
  .venv/bin/python -m pytest spikes/014-qwen38-compiled-gdn/test_pretrained.py -qs
```

Set `PARETOQUANT_SMALL_REPORT` to a **new** filename to publish another report;
exclusive creation refuses overwriting archived evidence. The pretrained test
uses a local directory only and checks the 1024-hidden/24-layer control shape.
