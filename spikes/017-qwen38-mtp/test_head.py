import copy
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModelArgs

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).parent


def args():
    return TextModelArgs(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         vocab_size=128, full_attention_interval=4,
                         rope_parameters={'type':'default', 'rope_theta':10000,
                                          'partial_rotary_factor':0.5})


def test_head_matches_explicit_native_equation_and_uses_shared_objects():
    from mtp_head import MTPHead, BoundHead
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.base import create_attention_mask
    a = args()
    mx.random.seed(93)
    module = MTPHead(a)
    embed, out = nn.Embedding(128,64), nn.Linear(64,128,bias=False)
    bound = BoundHead(module, embed, out)
    assert bound.embedding is embed and bound.output is out
    hidden = mx.random.normal((1,3,64)); tokens = mx.array([[2,3,4]])
    cache = KVCache(); reference = KVCache()
    actual = bound.forward(hidden, tokens, cache, position=0)
    mixed = module.fc(mx.concatenate([module.pre_fc_norm_embedding(embed(tokens)),
                                      module.pre_fc_norm_hidden(hidden)], axis=-1))
    expected = out(module.norm(module.layers[0](mixed,
                   mask=create_attention_mask(mixed,reference),cache=reference)))
    assert mx.allclose(actual, expected, atol=1e-6,rtol=1e-5).item()
    assert cache.offset == reference.offset == 3
    with pytest.raises(ValueError, match='position'):
        bound.forward(hidden[:,:1], tokens[:,:1], cache, position=0)


def test_norm_conversion_is_explicit_never_a_value_heuristic():
    from mtp_head import convert_norms
    weights = {'mtp.norm.weight':mx.array([0.0,0.1]),
               'mtp.layers.0.self_attn.q_norm.weight':mx.array([0.0,0.2]),
               'mtp.fc.weight':mx.array([[1.0]])}
    absolute = convert_norms(weights, encoding='absolute')
    delta = convert_norms(weights, encoding='delta')
    assert mx.array_equal(absolute['norm.weight'], mx.array([0.,0.1])).item()
    assert mx.array_equal(delta['norm.weight'], mx.array([1.,1.1])).item()
    assert mx.array_equal(delta['layers.0.self_attn.q_norm.weight'],mx.array([1.,1.2])).item()
    assert mx.array_equal(delta['fc.weight'], weights['mtp.fc.weight']).item()
    with pytest.raises(ValueError): convert_norms(weights, encoding='guess')


def test_trusted_sidecar_binds_exact_config_and_target_architecture():
    from mtp_head import validate_binding, SIDECAR_SHA256
    config = json.loads((ROOT/'primary_sources/sidecar_config.json').read_text())
    text = copy.deepcopy(config['text_config'])
    validate_binding(config, text, target_revision='10c35caafbb80f7dc6a7a432cdd11af10a6d4818',
                     sidecar_sha256=SIDECAR_SHA256)
    for field,value in [('hidden_size',64),('num_hidden_layers',4),('rope_parameters',{})]:
        bad = copy.deepcopy(text); bad[field]=value
        with pytest.raises(ValueError):
            validate_binding(config,bad,target_revision='10c35caafbb80f7dc6a7a432cdd11af10a6d4818',sidecar_sha256=SIDECAR_SHA256)
    config['block_size']=99
    with pytest.raises(ValueError):
        validate_binding(config,text,target_revision='10c35caafbb80f7dc6a7a432cdd11af10a6d4818',sidecar_sha256=SIDECAR_SHA256)


def test_quantized_head_strict_roundtrip_does_not_shift_absolute_norms(tmp_path):
    from mtp_head import MTPHead, load_head_weights
    a=args(); mx.random.seed(5)
    source=MTPHead(a); nn.quantize(source,group_size=32,bits=4)
    weights=dict(tree_flatten(source.parameters()))
    p=tmp_path/'tiny.safetensors'; mx.save_safetensors(str(p),weights)
    loaded=load_head_weights(a,p,group_size=32,bits=4,norm_encoding='absolute')
    actual=dict(tree_flatten(loaded.parameters()))
    assert actual.keys()==weights.keys()
    for name in actual: assert mx.array_equal(actual[name],weights[name]).item(),name
    bad=dict(weights); bad['norm.weight']=mx.zeros((3,))
    mx.save_safetensors(str(p),bad)
    with pytest.raises(ValueError): load_head_weights(a,p,group_size=32,bits=4,norm_encoding='absolute')
