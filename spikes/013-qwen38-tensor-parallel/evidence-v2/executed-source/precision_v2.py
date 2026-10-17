"""Opt-in precision candidate; exact parameter values, different arithmetic.

Only row projections use F32 input/scales/affine-bias views. Native packed
weights remain untouched. F32 partials are summed before casting to input dtype.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import QuantizedShardedToAllLinear
from mlx_lm.models.qwen3_5 import GatedDeltaNet


def partial_f32(module,x):
    biases=module.get('biases')
    return mx.quantized_matmul(x.astype(mx.float32),module.weight,
        scales=module.scales.astype(mx.float32),
        biases=None if biases is None else biases.astype(mx.float32),
        transpose=True,group_size=module.group_size,bits=module.bits,mode=module.mode)


class PreciseRow(QuantizedShardedToAllLinear):
    def __call__(self,x):
        y=mx.distributed.all_sum(partial_f32(self,x),group=self.group).astype(x.dtype)
        if 'bias' in self:y=y+self.bias
        return y


class PreciseGDNPartial(nn.QuantizedLinear):
    def __call__(self,x):
        if 'bias' in self:raise ValueError('GDN output bias is not admitted')
        return partial_f32(self,x)


class PreciseGDN(GatedDeltaNet):
    def __call__(self,inputs,mask=None,cache=None):
        # Native GDN, including its all_sum; only final cast is added here.
        return super().__call__(inputs,mask=mask,cache=cache).astype(inputs.dtype)


def high_precision_row_reductions(model):
    for layer in model.layers:
        if not isinstance(layer.mlp.down_proj,QuantizedShardedToAllLinear):
            raise ValueError('Expected native quantized TP row projection')
        layer.mlp.down_proj.__class__=PreciseRow
        if layer.is_linear:
            gdn=layer.linear_attn
            if gdn.sharding_group is None:raise ValueError('Expected sharded GDN')
            gdn.out_proj.__class__=PreciseGDNPartial
            gdn.__class__=PreciseGDN
        else:
            layer.self_attn.o_proj.__class__=PreciseRow
