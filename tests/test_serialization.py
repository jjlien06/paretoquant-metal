"""Explicit real MLX checkpoint roundtrip of mixed 3/6-bit projections."""
from dataclasses import asdict

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


@pytest.mark.metal
def test_mixed_three_six_bit_checkpoint_roundtrip(tmp_path):
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model, ModelArgs
    from mlx_lm.utils import load_model, save_config, save_model

    from paretoquant.allocator import Option
    from paretoquant.pipeline import apply_plan
    from paretoquant.runtime import install_fusion

    args = ModelArgs(model_type="qwen2", hidden_size=128, num_hidden_layers=2,
                     intermediate_size=256, num_attention_heads=4, num_key_value_heads=2,
                     rms_norm_eps=1e-6, vocab_size=128)
    mx.random.seed(71)
    model = Model(args)
    model.set_dtype(mx.float16)
    choices = {
        "model.layers.0.mlp": Option("q3:fused:rpg4", 3, 1, 1.0, 0.01, "fused"),
        "model.layers.1.mlp": Option("q6:fused:rpg2", 6, 1, 1.0, 0.01, "fused"),
    }
    model, config, dispatch = apply_plan(model, asdict(args), choices)
    ids = mx.array([[1, 2, 3, 4]])
    original_logits = model(ids)
    original_decode = model(mx.array([[4]]))
    mx.eval(original_logits, original_decode, model.parameters())
    save_model(tmp_path, model)
    save_config(config, tmp_path / "config.json")
    restored, _ = load_model(tmp_path)
    assert restored.model.layers[0].mlp.gate_proj.bits == 3
    assert restored.model.layers[0].mlp.up_proj.bits == 3
    assert restored.model.layers[1].mlp.gate_proj.bits == 6
    assert restored.model.layers[1].mlp.up_proj.bits == 6
    before = dict(tree_flatten(model.parameters()))
    after = dict(tree_flatten(restored.parameters()))
    assert set(before) == set(after)
    for key in before:
        np.testing.assert_array_equal(np.array(before[key]), np.array(after[key]))
    assert len(install_fusion(restored, dispatch)) == 2
    restored_logits = restored(ids)
    restored_decode = restored(mx.array([[4]]))
    mx.eval(restored_logits, restored_decode)
    np.testing.assert_allclose(np.array(restored_logits), np.array(original_logits), rtol=0, atol=0)
    np.testing.assert_allclose(np.array(restored_decode), np.array(original_decode), rtol=0, atol=0)
