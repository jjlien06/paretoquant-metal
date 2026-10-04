"""Exact-type support for the gated MLP semantics verified with pinned mlx-lm."""

import mlx.nn as nn
from mlx_lm.models.qwen2 import MLP as Qwen2MLP


def require_supported_mlp(module):
    """Reject lookalike modules and subclasses with unverified forward semantics."""
    if type(module) is not Qwen2MLP:
        raise ValueError("unsupported MLP: only exact mlx_lm.models.qwen2.MLP is verified")
    for projection in ("gate_proj", "up_proj"):
        if type(getattr(module, projection, None)) not in (nn.Linear, nn.QuantizedLinear):
            raise ValueError(
                f"unsupported MLP projection: {projection} must be an exact MLX linear"
            )
