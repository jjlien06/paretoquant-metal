# Attribution and pinned source

The helper portion of `slot_qmv_header.metal` is derived from Apple MLX's
MIT-licensed `mlx/backend/metal/kernels/quantized.h` (Copyright © 2023–2024 Apple
Inc.). `LICENSE` reproduces the complete upstream license without modification.
The source snapshots were already provided locally by the parent; this task did
not fetch anything. These are content pins, not an assertion about a release tag.
The reported installed runtime is MLX 0.32.3 / MLX-LM 0.32.0; neither was imported.

| Provided upstream file | SHA-256 |
| --- | --- |
| `quantized.h` | `8e2ca65d8d8dafe67b97120d1127d54ce57032f621bdba161e54a0bc4a948b8c` |
| `quantized.cpp` | `36b79607ee01bc213511dcdad590b2dc3b96099300551cbc3e7e6c7e13ba5982` |
| `LICENSE` | `ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01` |

Local source directory:
`/Users/jeremylien/.hermes/cache/scratch/paretoquant-dual-host/width-stable-v1/upstream/`.

## Mechanical helper adaptation

- Copy `quantized.h` lines 17–392, inclusive: packing helpers, `load_vector`,
  `load_vector_safe`, `qdot`, and `qdot_safe`.
- Prefix the four vector/dot helper names with `slot_` to avoid collisions with
  generated kernel support code. Packing helper names remain unchanged.
- **Only in the dot helpers**, replace `const device` with `const thread`, including
  their uint16 views. Vector loaders still read device input. No arithmetic,
  loop ordering, explicit scaling, grouping, or casts were otherwise rewritten.
- Keep unused upstream 2/5/6/8-bit helper branches intact. They are **not admitted**
  by the Python interface or candidate kernel's bit guard.
- The exact adapted helper section, between its BEGIN/END markers, has SHA-256
  `66cfc54321acdc2a2da5e5ba4f99d430ae25430e4fdee0d2356f7899edc8bb6b`.
  A CPU source-contract test pins this entire section, not just selected lines.

## New scheduling, derived from QMV bodies

The candidate's slot scheduling derives from `qmv_fast_impl` (lines 756–821)
and the admitted N>=8 aligned-output branch of `qmv_impl` (lines 823–982, relevant
branch 927–980). Dispatch was read in `quantized.cpp` lines 149–153 and 465–539.
Two SIMD groups, four rows per SIMD, float accumulation, original vector-loading
sum expressions, pack counts, the slow safe final step, `simd_sum`, and final
T cast are retained at source level. Each row's raw bytes are staged into aligned
thread uint32 storage outside its slot dot loop. Affine scale/bias are shared too.

Slot dimension, staging, explicit coordinate mapping, template guards, lazy
Python wrapper, metadata planning, and CPU tests are new experiment code.
Stock output-overlap/ragged branches are excluded by admission, not silently
approximated. Stock `qmv_wide` is deliberately not used. Source resemblance and
host tests are **not** Metal compilation, numerical parity, byte-traffic, or speed
proof; compiler transformations, register pressure, and spills remain unmeasured.
