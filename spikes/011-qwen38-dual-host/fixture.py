"""Create a deterministic, bounded built-in Qwen3.5 hybrid checkpoint.

Tokenizer assets are deliberately separate. Copy real tokenizer.save_pretrained
assets into the output (no config.json overwrite), and use --prompt tokens:[...]
for the 128-entry fixture vocabulary. Random weights are NOT a pretrained model.
"""

import argparse
import json
from pathlib import Path


def create(output):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    config = dict(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=2,
        tie_word_embeddings=False,
        max_position_embeddings=512,
        rope_parameters=dict(type="default", partial_rotary_factor=0.5, rope_theta=10000),
        quantization=dict(group_size=32, bits=4),
        eos_token_id=[],
    )
    output = Path(output)
    output.mkdir(mode=0o700)
    mx.set_default_device(mx.gpu)
    mx.random.seed(731)
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(model, group_size=32, bits=4)
    model.eval()
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(output / "model.safetensors"), weights)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(
                metadata=dict(total_size=sum(a.nbytes for a in weights.values())),
                weight_map={k: "model.safetensors" for k in weights},
            ),
            indent=2,
        )
    )
    return dict(
        output=str(output.resolve()),
        tensor_count=len(weights),
        parameter_bytes=sum(a.nbytes for a in weights.values()),
        prompt="tokens:[1,7,3,4,8,9]",
        split=[2, 3],
        tokenizer_assets="copy separately",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(create(parser.parse_args().output)))
