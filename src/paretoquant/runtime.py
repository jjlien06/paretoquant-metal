"""Named-module integration with conservative fallback for non-decode shapes."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from .adapters import require_supported_mlp
from .metal import SUPPORTED_BITS, compiled_fused_gate_up


class FusedMLP(nn.Module):
    def __init__(self, original, rows_per_group=4):
        super().__init__()
        require_supported_mlp(original)
        if rows_per_group not in (1, 2, 4, 8):
            raise ValueError("rows_per_group must be 1, 2, 4, or 8")
        if not all(
            isinstance(getattr(original, p, None), nn.QuantizedLinear)
            for p in ("gate_proj", "up_proj")
        ):
            raise ValueError("fusion requires compatible quantized gate/up projections")
        gate, up = original.gate_proj, original.up_proj
        if (
            gate.bits != up.bits
            or gate.bits not in SUPPORTED_BITS
            or gate.group_size != up.group_size
            or gate.mode != "affine"
            or up.mode != "affine"
            or gate.weight.shape != up.weight.shape
            or gate.scales.dtype not in (mx.float16, mx.float32)
            or gate.scales.dtype != up.scales.dtype
            or "bias" in gate
            or "bias" in up
        ):
            raise ValueError("fusion requires compatible affine gate/up shapes and precision")
        self.gate_proj = gate
        self.up_proj = up
        self.down_proj = original.down_proj
        self.rows_per_group = rows_per_group
        # Invoke the verified unbound forward on current fields, never a stale original module.
        object.__setattr__(self, "_original_forward", type(original).__call__)
        # Counters are not model parameters and are deliberately not saved as weights.
        object.__setattr__(self, "stats", {"fused_calls": 0, "stock_calls": 0})

    def __call__(self, x):
        if x.ndim > 0 and x.size == x.shape[-1] and x.dtype == self.gate_proj.scales.dtype:
            self.stats["fused_calls"] += 1
            activation = compiled_fused_gate_up(
                x,
                (self.gate_proj.weight, self.gate_proj.scales, self.gate_proj.biases),
                (self.up_proj.weight, self.up_proj.scales, self.up_proj.biases),
                bits=self.gate_proj.bits,
                group_size=self.gate_proj.group_size,
                rows_per_group=self.rows_per_group,
            )
        else:
            self.stats["stock_calls"] += 1
            return self._original_forward(self, x)
        return self.down_proj(activation)


def install_fusion(model, dispatch):
    modules = dict(model.named_modules())
    unknown = set(dispatch) - set(modules)
    if unknown:
        raise ValueError(f"Unknown module paths in fusion manifest: {sorted(unknown)}")
    replacements = []
    for name, config in dispatch.items():
        backend = config.get("backend", "stock")
        if backend not in ("stock", "fused"):
            raise ValueError(f"Unknown backend {backend!r} for {name}")
        require_supported_mlp(modules[name])
        if backend == "fused":
            replacements.append((name, FusedMLP(modules[name], config.get("rows_per_group", 4))))
    if replacements:
        model.update_modules(tree_unflatten(replacements))
    return [name for name, _ in replacements]
