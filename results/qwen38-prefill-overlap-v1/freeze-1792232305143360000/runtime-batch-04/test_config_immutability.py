"""Regression for upstream ModelArgs mutating caller-owned rope metadata."""

import copy
import json

import mlx.core as mx
from mlx.utils import tree_flatten


def test_stream_loader_preserves_exact_admitted_rope_metadata(tmp_path):
    from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records
    from test_phased_cpu import tiny_model

    mx.set_default_device(mx.cpu)
    native = tiny_model()
    config = copy.deepcopy(native.args.text_config)
    rope = config["rope_parameters"]
    rope["rope_type"] = rope.pop("type")
    config["quantization"] = {"group_size": 32, "bits": 4}
    admitted = copy.deepcopy(config)
    path = tmp_path / "weights"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(native.parameters())))
    tensors, provenance = checkpoint_inventory(path)

    class Group:
        def rank(self):
            return 0

        def size(self):
            return 1

    loaded, _ = load_from_tensors(
        config, Group(), [4], tensors, tensor_records(tensors, provenance), budget_bytes=8 * 1024**2
    )
    assert config == admitted, "MLX constructor must not mutate trusted source metadata"
    assert loaded.language_model.args.rope_theta == native.language_model.args.rope_theta
