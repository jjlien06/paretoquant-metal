import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from test_tp import checkpoint,Group


def test_opt_in_changes_only_row_arithmetic_preserves_parameters(checkpoint):
    from precision_v2 import high_precision_row_reductions
    from tp import inventory,skeleton
    path,cfg,full=checkpoint
    model=skeleton(cfg,inventory(path),Group(0))
    full.shard(Group(0))
    model.load_weights(list(tree_flatten(full.parameters())))
    mx.eval(model.parameters())
    before=dict(tree_flatten(model.parameters()))
    high_precision_row_reductions(model)
    after=dict(tree_flatten(model.parameters()))
    assert before.keys()==after.keys()
    for k,a in before.items():
        assert a.dtype==after[k].dtype
        assert mx.array_equal(a,after[k]).item()
    layer=model.layers[0].linear_attn.out_proj
    x=mx.ones((1,1,layer.weight.shape[-1]*8),mx.bfloat16)
    assert layer(x).dtype==mx.float32
