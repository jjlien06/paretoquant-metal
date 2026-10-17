"""Explicit local 0.8B gate, never downloads or runs the large checkpoint.

PARETOQUANT_SMALL_CONTROL=/absolute/small-control python -m pytest ... -s
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import pytest


def test_local_pretrained_parity():
    root = os.environ.get("PARETOQUANT_SMALL_CONTROL")
    if not root:
        pytest.skip("Opt-in local 0.8B gate")
    from compiler import GDNCompiler
    from mlx_lm import load
    from phased_decode import phased_forward
    from remote_actor import source_hashes
    from test_qwen38 import assert_caches

    root = Path(root).resolve(strict=True)
    config = json.loads((root / "config.json").read_text())
    assert config["text_config"]["hidden_size"] == 1024
    assert config["text_config"]["num_hidden_layers"] == 24
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024**2)
    mx.reset_peak_memory()
    model, tokenizer = load(
        str(root),
        trust_remote_code=False,
        tokenizer_config={"local_files_only": True, "trust_remote_code": False},
    )
    compiler = GDNCompiler(model)
    rows = []
    for prompt in ["Reply with the number four.", "Say hi.", "Reply with the number four."]:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        left, right = model.make_cache(), model.make_cache()
        chunks = [ids[i : min(i + 16, len(ids) - 1)] for i in range(0, len(ids) - 1, 16)]
        chunks.append([ids[-1]])
        tokens = []
        max_diff = 0.0
        for step in range(len(chunks) + 3):
            current = chunks[step] if step < len(chunks) else [tokens[-1]]
            x = mx.array([current], mx.int32)
            expected = model(x, cache=left)
            mx.eval(expected, [c.state for c in left])
            actual = phased_forward(
                model, x, right, compiler=compiler, eval_group_size=4, all_logits=True
            )
            diff = float(
                mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))).item()
            )
            max_diff = max(max_diff, diff)
            assert mx.allclose(expected, actual, atol=1e-6, rtol=1e-5).item(), (
                f"logits step={step} max_abs={diff}"
            )
            assert_caches(left, right)
            a, b = mx.argmax(expected[:, -1, :], -1), mx.argmax(actual[:, -1, :], -1)
            assert mx.array_equal(a, b).item()
            if step >= len(chunks) - 1:
                tokens.append(int(a.item()))
        assert compiler.trace_count == 6
        rows.append(
            dict(
                prompt=prompt,
                prompt_tokens=len(ids),
                token_ids=tokens,
                text=tokenizer.decode(tokens),
                max_abs_logits=max_diff,
                compiler=compiler.receipt(),
                active_mlx_bytes=mx.get_active_memory(),
                peak_mlx_bytes=mx.get_peak_memory(),
                cache_mlx_bytes=mx.get_cache_memory(),
            )
        )
    report = dict(
        checkpoint=str(root),
        atol=1e-6,
        rtol=1e-5,
        rows=rows,
        source_sha256=source_hashes(Path(__file__).parent),
        device=mx.device_info(),
        memory_accounting="MLX allocator bytes, not system residency",
    )
    print(json.dumps(report))
    output = os.environ.get("PARETOQUANT_SMALL_REPORT")
    if output:
        with Path(output).open("x") as handle:
            json.dump(report, handle, indent=2)
