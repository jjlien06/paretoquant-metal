import copy
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3_5 import Model,ModelArgs
mx.set_default_device(mx.cpu)


def tiny_model():
    config=dict(model_type='qwen3_5',hidden_size=64,intermediate_size=128,
                num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
                head_dim=16,vocab_size=128,linear_num_value_heads=4,
                linear_num_key_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,
                linear_conv_kernel_dim=4,full_attention_interval=2,
                rope_parameters={'type':'default','partial_rotary_factor':0.5,'rope_theta':10000})
    mx.random.seed(77); model=Model(ModelArgs.from_dict(config));
    nn.quantize(model,group_size=32,bits=4)
    # Training selects source-defined CPU recurrent implementation, not GPU kernel.
    model.train()
    mx.eval(model.parameters())
    return model


def test_optin_hidden_is_normalized_default_output_unchanged():
    from phased_decode import phased_forward
    from cache_state import assert_cache_equal
    from numerical_gate import assert_logits_equal
    model=tiny_model(); x=mx.array([[1,2,3]])
    c1,c2,c3=model.make_cache(),model.make_cache(),model.make_cache()
    native=model(x,cache=c1)
    default=phased_forward(model,x,c2,all_logits=True)
    logits,hidden=phased_forward(model,x,c3,all_logits=True,return_hidden=True)
    assert_logits_equal(native,default); assert_logits_equal(native,logits)
    expected=model.language_model.lm_head(hidden)
    assert_logits_equal(logits,expected)
    assert_cache_equal(c1,c2); assert_cache_equal(c1,c3)
    with pytest.raises(ValueError): phased_forward(model,x,c3,return_hidden='yes')


def test_width_gate_runs_both_slots_against_native_sequential_and_every_hybrid_state():
    from width_gate import compare_width2
    model=tiny_model()
    report=compare_width2(model,[1,2,3,4,5],[6,7],prefill_step_size=2)
    assert report['atol']==1e-6 and report['rtol']==1e-5
    assert report['logits_pass'] and report['cache_pass']
    assert report['widths']==[1,2]
