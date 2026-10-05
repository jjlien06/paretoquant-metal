"""Regression for an actual MLX-emitted optional metadata representation."""

from shard_plan import parse_safetensors_header


def test_native_mlx_null_optional_metadata_is_accepted():
    header = b'{"__metadata__":null,"x":{"dtype":"U32","shape":[1],"data_offsets":[0,4]}}'
    assert parse_safetensors_header(header, data_size=4) == {
        "x": {"dtype": "U32", "shape": [1], "data_offsets": [0, 4]}
    }
