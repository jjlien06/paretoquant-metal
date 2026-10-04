# Reproducible measured-profile budget sweep

`scripts/sweep_profile.py` reuses the exact allocator on an **existing** hardware-measured
profile. It uses only Python's standard library and `paretoquant.allocator`: no MLX,
model loading, downloads, new GPU trials, or plotting dependencies. It does not modify
its input profile or the `results/` archive.

From the repository root:

```sh
python3 -S scripts/sweep_profile.py \
  --profile artifacts/m2pro-mixed-v2/profile.json \
  --output artifacts/portfolio-sweep-0.5b-v1 \
  --memory-fractions .90 .94 .97 1.0 1.03 \
  --latency-factors .9 1.0 1.15 \
  --max-states 10000
```

Both space-separated and comma-separated lists work. The output directory must be
absent or empty; even a hidden file makes it nonempty. Existing artifacts are never
overwritten. To reproduce the example after its first run, choose a **new** output
directory. Exit status is 0 for completed sweeps (including infeasible rows), 1 when
any row has an allocator error, and 2 for invalid input or output errors.

## Budgets and scope

The parser constructs `Option` and `Unit` directly, retaining the same measured costs
as `pipeline.units_from_profile` without importing its MLX dependencies. It shares the
allocator's identifier/cost validation. The profile must declare
`scope = coupled_gate_up_pairs`, nonnegative integer `fixed_parameter_bytes`, a positive
integer `uniform4_parameter_bytes`, and exactly one stock 4-bit option per unit.
The uniform4 bytes must equal fixed bytes plus all stock 4-bit option bytes.

For **both** strategies the common reference is uniform4 stock:

- Total model parameter budget = floor(memory fraction × uniform4 parameter bytes).
  Fractions are interpreted via their decimal string as exact rational numbers for
  integer byte rounding, independent of the process's Decimal context.
- Gate/up byte budget = total budget − `profile.fixed_parameter_bytes`.
- Gate/up latency budget = latency factor × sum of uniform4 stock option latencies,
  summed in canonical unit-name order as in the profile allocation pipeline.
- `stock` considers stock options only. `fusion_aware` considers all stock and fused
  options; it does **not** force every pair onto the fused backend.

Reported solution parameter bytes include the fixed non-gate/up model parameters
and each selected option's profiled resident bytes (packed weights, scales, and
quantization metadata). These are not runtime working-set or KV-cache bytes. A budget
below the fixed bytes is infeasible, not malformed input.

Every feasible row comes from `allocate`, with its original exact objective and
strict floating-point budget semantics. The reported `surrogate_loss` is the **sum of
local calibration activation relative MSE**, not measured model quality, NLL, or
perplexity. `predicted_gate_up_latency_sum_ms` is a **predicted sum of isolated
hardware-profiled gate/up latencies**, not measured full-model latency or throughput.
The exact solver cannot repair a noisy profile or validate an additive performance
model. Backend histograms count coupled pairs, not individual projections.

## Output and error semantics

The deterministic `sweep.json` includes the SHA256 of the exact input profile bytes,
its absolute path, the fixed-byte source (`profile.fixed_parameter_bytes`), reference
costs, sorted input grid, frontier cap, interpretation notes, and every budget row.
Each feasible solution includes selected options, actual parameter bytes, predicted
latency sum, surrogate loss, exactness, considered states, and bits/backend histograms.
There is no timestamp or stochastic sampling in the output.

Rows are ordered by ascending memory fraction, ascending latency factor, then stock
and fusion-aware. Unit and option ordering is canonical. Lists must be nonempty,
positive, finite, and numerically distinct (`1` and `1.0` are duplicates). Booleans,
strings in the Python API, NaN, infinity, duplicate JSON keys, duplicate units/options,
unknown backends, and contradictory byte metadata are rejected before solving.

- `status=feasible`, `feasible=true`: exact solution populated, no error.
- `status=infeasible`, `feasible=false`: `InfeasibleBudget`, no solution.
- `status=error`, `feasible=null`: feasibility **unknown**, no solution.
  `error.kind=capped_frontier` identifies `FrontierOverflow` separately from
  `allocation_error` (for example, nonfinite accumulated surrogate loss).
  The exception type and original message are preserved.

A capped frontier is never silently pruned or relabeled as infeasible. Increase
`--max-states` and use a new output directory to retry it. This cap bounds the retained
exact frontier, not all candidate expansion or total process memory.

`frontier.svg` is a self-contained memory/latency scatter of feasible optima only.
It is not a claim that all plotted rows are globally nondominated across the grid.
Blue circles represent stock, orange squares fusion-aware. Open it directly in a
browser to toggle memory fractions (click or keyboard Enter/Space) and hover/focus
points for surrogate-loss labels and complete selections. Static SVG viewers display
all feasible points. Repeated optima can overlap; the plot does not infer a connecting
performance curve. Text and attributes are escaped, and no external assets are loaded.

## Executed 0.5B example

The command above was executed CPU-only over the existing 24-pair profile, producing
**30 unique rows: 15 per strategy, 14 feasible, 16 infeasible, zero errors**. Stock had
6 feasible / 9 infeasible; fusion-aware had 8 feasible / 7 infeasible. All feasible
solutions were exact and satisfied both recorded budgets.

The reference is 277,996,288 resident parameter bytes with 160,326,400 fixed bytes;
its predicted uniform4 stock gate/up latency sum is 5.682353 ms. At memory fraction
1.03 / latency factor 1.0, stock selected 23 four-bit pairs and one six-bit pair
(surrogate loss 0.4102568843); fusion-aware selected 21 four-bit and three six-bit pairs
(surrogate loss 0.3617993575). This is a difference in the allocator's calibration
surrogate under the supplied profile, **not evidence of measured quality improvement**.

Artifacts:

- `artifacts/portfolio-sweep-0.5b-v1/sweep.json`
- `artifacts/portfolio-sweep-0.5b-v1/frontier.svg`

Tests are synthetic CPU-only fixtures, including a CLI run with Python `-S` to prove
that no installed MLX/NumPy/plotting dependencies are required:

```sh
.venv/bin/python -m pytest tests/test_sweep.py tests/test_allocator.py -q
```
