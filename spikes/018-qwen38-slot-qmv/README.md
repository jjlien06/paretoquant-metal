# 018: BF16 slot-shared QMV

Isolated extension of 010. The copied Apple-derived Metal helpers/body and license
are unchanged. Only host metadata admission adds `bfloat16` alongside F16/F32.
Existing archives and 010 are unchanged. The initial implementation delegation
failed at provider rate limiting and created nothing; the parent implemented it.

## Verified scope

- Strict TDD: BF16 metadata test failed for the old F16/F32 guard, then passed.
- 313 CPU metadata/source-contract tests passed.
- Actual Qwen3.8-27B packed layer-0 MLP gate/up/down fixtures, widths 1/2/3:
  **bit-exact to separately executed stock width-one QMM on both physical GPUs**.
- M4 genuine Qwen3.5-0.8B architecture-control full model, width2 teacher against
  forced singleton teacher: all vocabulary logits and all 48 cache parts were
  exact across three windows with this scoped projection substitution.
- The original native M4 width2 failure remains preserved separately.

The actual-weight fixture is `models/qwen38-operator-fixture-v2`, copied read-only
from the pinned target checkpoint. All ten source payload hashes were checked
against the already trusted full-generation receipt, then checked after transfer.
The fixed random operator inputs are not calibration inputs or quality evidence.

## Operator measurements (not generation speedups)

Warmed interleaved 24-pair operator samples on M4 measured width2 at 1.51–1.60x
versus serial stock singleton projections, and width3 at 1.76–1.77x. Width1 was
approximately parity/slower. M2 width2 gain versus serial singleton projections
was smaller, and native M2 batched QMM can remain faster. These results do not
establish 27B verifier/cache equivalence, speculative acceptance or >=10 tokens/s.
No default decoder/backend is changed. Use only after full model admission.

Evidence in `results/qwen38-speed-v2/`:

- `018-operator-m2-v1.json`, `018-operator-m4-v1.json`
- `width2-slot-small-m4-v2.json`: successful complete small-model numerical gate.
- `width2-slot-small-m4-v1.json`: diagnostic call-signature failure, preserved.
- `screen_slot_qmv.py`, `diagnose_width2_controls.py`: executed diagnostics.

## Commands

```sh
.venv/bin/python -m pytest -q spikes/018-qwen38-slot-qmv
.venv/bin/python results/qwen38-speed-v2/screen_slot_qmv.py
```

The wrapper supports batch1, width1..9, affine3/4-bit, group32/64/128, complete
quantization groups, matching floating input/scales/bias dtype, U32 packed words,
N>=8 and N divisible by8, and bounded signed32-bit indexing. It is not a generic
QMM replacement. Any distributed integration must register the Python helper and
both Metal sources in the immutable actor snapshot/source registry.
