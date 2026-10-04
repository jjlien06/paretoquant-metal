"""CPU accounting first; opt-in tiny real-model acceptance last."""

import math
import os

import numpy as np
import pytest


def test_windows_score_every_target_once_with_bounded_left_context():
    from paretoquant.quality import iter_windows

    for size in range(2, 30):
        for length in range(2, 9):
            for stride in range(1, length):
                windows = list(iter_windows(size, window_length=length, stride=stride))
                targets = [t for w in windows for t in range(w.target_start, w.target_end)]
                assert targets == list(range(1, size))
                for w in windows:
                    assert 0 <= w.start < w.target_start < w.target_end == w.end <= size
                    assert w.end - w.start <= length
                    assert w.start == max(0, w.end - length)


def test_uniform_fake_logits_exact_count_nll_and_cap():
    from paretoquant.quality import evaluate_tokens, numpy_logit_losses

    seen = []

    def losses(inputs, targets, score_start):
        seen.append((inputs.copy(), targets.copy(), score_start))
        logits = np.zeros((1, len(inputs), 8), dtype=np.float32)
        return numpy_logit_losses(logits, targets, score_start)

    result = evaluate_tokens(list(range(8)), losses, window_length=4, stride=2, max_target_tokens=5)
    assert result["target_token_count"] == 5
    assert result["total_nll"] == pytest.approx(5 * math.log(8))
    assert result["perplexity"] == pytest.approx(8)
    assert result["available_target_token_count"] == 7
    assert result["truncated"] is True
    assert result["window_count"] == 3
    assert seen == [([0, 1], [1, 2], 0), ([1, 2, 3], [3, 4], 1), ([2, 3, 4], [5], 2)]


def test_report_exposes_mask_positions_and_unscored_tail_count():
    from paretoquant.quality import evaluate_tokens

    result = evaluate_tokens(
        list(range(8)),
        lambda inputs, targets, offset: [1.0] * len(targets),
        window_length=4,
        stride=2,
        max_target_tokens=5,
    )
    assert result["unscored_tail_target_token_count"] == 2
    assert result["scored_sequence_count"] == 1
    assert [
        (w["input_token_count"], w["scored_logit_start"], w["scored_logit_end"])
        for w in result["windows"]
    ] == [(2, 0, 2), (3, 1, 3), (3, 2, 3)]


def test_masked_nonuniform_logits_follow_the_previous_token_at_every_boundary():
    from paretoquant.quality import evaluate_tokens, numpy_logit_losses

    tokens = [0, 2, 1, 2, 0, 1, 0, 2, 1]

    def losses(inputs, targets, score_start):
        logits = np.full((1, len(inputs), 3), -1000.0)
        for index, previous in enumerate(inputs):
            logits[0, index, previous] = -998.0
        return numpy_logit_losses(logits, targets, score_start)

    expected = sum(
        math.log(math.exp(2) + 2) - (2 if a == b else 0) for a, b in zip(tokens[:-1], tokens[1:])
    )
    for length in range(2, 7):
        for stride in range(1, length):
            result = evaluate_tokens(tokens, losses, window_length=length, stride=stride)
            assert result["total_nll"] == pytest.approx(expected)
            assert result["target_token_count"] == len(tokens) - 1


@pytest.mark.parametrize(
    "settings",
    [
        {"window_length": 1},
        {"window_length": True},
        {"window_length": 4.5},
        {"stride": 0},
        {"stride": 4, "window_length": 4},
        {"stride": -1},
        {"max_target_tokens": 0},
        {"max_target_tokens": 1.5},
    ],
)
def test_invalid_window_settings_fail_before_scoring(settings):
    from paretoquant.quality import evaluate_tokens

    def unexpected(*args):
        pytest.fail("invalid settings reached scorer")

    with pytest.raises(ValueError):
        evaluate_tokens([0, 1], unexpected, **({"window_length": 4, "stride": 2} | settings))


@pytest.mark.parametrize("tokens", [[], [0], [0, -1], [0, 1.5], [0, True]])
def test_invalid_token_stream_rejected(tokens):
    from paretoquant.quality import evaluate_tokens

    with pytest.raises(ValueError):
        evaluate_tokens(tokens, lambda *args: [0.0], window_length=4, stride=2)


@pytest.mark.parametrize("losses", [[float("nan")], [float("inf")], [-1.0], [], [[1.0]], [1, 2]])
def test_invalid_losses_cannot_produce_a_report(losses):
    from paretoquant.quality import evaluate_tokens

    with pytest.raises(ValueError):
        evaluate_tokens([0, 1], lambda *args: losses, window_length=2, stride=1)


def test_perplexity_overflow_rejected_as_nonfinite():
    from paretoquant.quality import evaluate_tokens

    with pytest.raises(ValueError, match="finite"):
        evaluate_tokens([0, 1], lambda *args: [1000.0], window_length=2, stride=1)


@pytest.mark.parametrize(
    "logits,targets,offset",
    [
        (np.array([[[0.0, np.nan]]]), [0], 0),
        (np.zeros((1, 1, 2)), [2], 0),
        (np.zeros((1, 1, 2)), [-1], 0),
        (np.zeros((1, 1, 2)), [0], 1),
        (np.zeros((1, 2)), [0], 0),
    ],
)
def test_logit_oracle_validates_finiteness_shape_and_target_bounds(logits, targets, offset):
    from paretoquant.quality import numpy_logit_losses

    with pytest.raises(ValueError):
        numpy_logit_losses(logits, targets, offset)


def test_corpus_join_preserves_order_blanks_and_content_hashes(tmp_path):
    import hashlib
    import json

    from paretoquant.quality import load_corpus, tokenize_corpus

    path = tmp_path / "texts.json"
    raw = json.dumps(["ab", "", "café"], ensure_ascii=False).encode()
    path.write_bytes(raw)
    corpus = load_corpus(path, label="held-out test subset", separator="\n\n")
    assert corpus.text == "ab\n\n\n\ncafé"
    assert corpus.metadata["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert corpus.metadata["joined_text_sha256"] == hashlib.sha256(corpus.text.encode()).hexdigest()
    assert corpus.metadata["source_record_count"] == 3
    assert corpus.metadata["blank_record_count"] == 1
    assert corpus.metadata["sequence_count"] == 1
    assert corpus.metadata["canonical_benchmark"] is False

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            assert text == corpus.text
            return [ord(c) for c in text]

    tokens, metadata = tokenize_corpus(corpus, Tokenizer(), max_target_tokens=2)
    assert tokens == [ord(c) for c in corpus.text]
    assert metadata["stream_token_count"] == len(tokens)
    assert metadata["evaluated_stream_token_count"] == 3
    assert metadata["stream_sha256"] != metadata["evaluated_stream_sha256"]
    assert metadata["hash_encoding"] == "unsigned_64bit_little_endian_token_ids"


@pytest.mark.parametrize("raw", ["{}", "[]", "[1]", '[" ", ""]', '["a", null]', "invalid"])
def test_invalid_corpus_rejected_without_loading_models(tmp_path, raw):
    from paretoquant.quality import load_corpus

    path = tmp_path / "texts.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_corpus(path, label="test")


def test_model_fingerprint_is_content_addressed_not_path_addressed(tmp_path):
    import shutil

    from paretoquant.quality import fingerprint_model

    original = tmp_path / "first"
    original.mkdir()
    (original / "config.json").write_text('{"model_type": "qwen2"}')
    (original / "model.safetensors").write_bytes(b"test fixture, not a real model")
    (original / "tokenizer.json").write_text("{}")
    copied = tmp_path / "copy"
    shutil.copytree(original, copied)
    first = fingerprint_model(original)
    assert first["sha256"] == fingerprint_model(copied)["sha256"]
    assert [f["path"] for f in first["files"]] == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]
    (copied / "model.safetensors").write_bytes(b"different weights")
    assert first["sha256"] != fingerprint_model(copied)["sha256"]
    with pytest.raises(ValueError, match="local"):
        fingerprint_model(tmp_path / "remote-repo-id")


def test_sequential_comparison_rejects_changed_token_stream(tmp_path):
    from paretoquant.quality import Corpus, Variant, evaluate_variants

    calls = []
    corpus = Corpus("ab", {"canonical_benchmark": False})

    def runner(variant, corpus, **settings):
        calls.append(variant.name)
        token_hash = "same" if variant.name != "changed" else "different"
        return {
            "name": variant.name,
            "tokenization": {"stream_sha256": token_hash, "evaluated_stream_sha256": token_hash},
            "metrics": {"target_token_count": 1, "mean_nll": 2.0, "perplexity": math.exp(2)},
        }

    variants = [Variant("first", tmp_path), Variant("second", tmp_path)]
    report = evaluate_variants(
        variants, corpus, runner=runner, window_length=4, stride=2, max_target_tokens=1
    )
    assert calls == ["first", "second"]
    assert report["execution"] == "sequential_one_model_resident_at_a_time"
    assert report["results"][1]["delta_mean_nll_from_first"] == 0
    assert report["scope"] == "noncanonical_held_out_perplexity_not_task_accuracy"
    with pytest.raises(ValueError, match="token"):
        evaluate_variants([variants[0], Variant("changed", tmp_path)], corpus, runner=runner)


def test_existing_quality_report_is_preserved_before_loading(tmp_path, monkeypatch):
    from paretoquant import quality

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    corpus = tmp_path / "corpus.json"
    corpus.write_text('["held out"]')
    output = tmp_path / "quality.json"
    output.write_text("prior evidence")

    def unexpected(*args, **kwargs):
        pytest.fail("must refuse existing output before evaluating models")

    monkeypatch.setattr(quality, "evaluate_variants", unexpected)
    with pytest.raises(SystemExit) as error:
        quality.main(["--corpus", str(corpus), "--corpus-name", "heldout",
                      "--reference", str(model), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_text() == "prior evidence"


def test_quality_report_created_during_evaluation_is_preserved(tmp_path, monkeypatch, capsys):
    from paretoquant import quality

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    corpus = tmp_path / "corpus.json"
    corpus.write_text('["held out"]')
    output = tmp_path / "quality.json"
    competing = b'{"completed_run": "competing evidence"}\n'

    def evaluate(*args, **kwargs):
        assert not output.exists(), "preflight must finish before the competing publication"
        output.write_bytes(competing)
        return {"completed_run": "later evaluation"}

    monkeypatch.setattr(quality, "evaluate_variants", evaluate)
    try:
        code = quality.main(
            [
                "--corpus", str(corpus), "--corpus-name", "heldout",
                "--reference", str(model), "--output", str(output),
            ]
        )
    except SystemExit as error:
        code = error.code
    assert output.read_bytes() == competing
    assert code == 2
    assert "File exists" in capsys.readouterr().err


def test_cli_reference_expands_variants_and_writes_report(tmp_path, monkeypatch):
    import json

    from paretoquant import quality

    reference = tmp_path / "reference"
    mixed = tmp_path / "mixed"
    for path in (reference, mixed):
        path.mkdir()
        (path / "config.json").write_text("{}")
    corpus = tmp_path / "test.json"
    corpus.write_text('["held out"]')
    output = tmp_path / "quality.json"
    received = []

    def evaluate(variants, data, **settings):
        received.extend(variants)
        assert settings == {
            "window_length": 8,
            "stride": 3,
            "max_target_tokens": 5,
            "group_size": 64,
        }
        assert data.metadata["label"] == "WikiText test subset"
        return {"verified": True}

    monkeypatch.setattr(quality, "evaluate_variants", evaluate, raising=False)
    assert (
        quality.main(
            [
                "--corpus",
                str(corpus),
                "--corpus-name",
                "WikiText test subset",
                "--reference",
                str(reference),
                "--mixed",
                str(mixed),
                "--window-length",
                "8",
                "--stride",
                "3",
                "--max-target-tokens",
                "5",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert [(v.name, v.path, v.uniform_bits, v.dtype) for v in received] == [
        ("reference_fp16", reference, None, "float16"),
        ("uniform4", reference, 4, "float16"),
        ("mixed_stock", mixed, None, "native"),
    ]
    assert json.loads(output.read_text()) == {"verified": True}


def test_named_models_uniform_companions_are_explicit(tmp_path, monkeypatch, capsys):
    from paretoquant import quality

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    corpus = tmp_path / "test.json"
    corpus.write_text('["text"]')
    seen = []

    def evaluate(variants, *args, **kwargs):
        seen.extend(variants)
        return {"ok": True}

    monkeypatch.setattr(quality, "evaluate_variants", evaluate, raising=False)
    assert (
        quality.main(
            [
                "--corpus",
                str(corpus),
                "--corpus-name",
                "test",
                "--model",
                f"base={model}",
                "--uniform-bits",
                "4",
            ]
        )
        == 0
    )
    assert [(v.name, v.uniform_bits, v.dtype) for v in seen] == [
        ("base", None, "native"),
        ("base:uniform4", 4, "float16"),
    ]
    assert '"ok": true' in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--model", "bad-mapping"],
        ["--model", "x=/does/not/exist"],
        ["--reference", "/does/not/exist"],
        ["--mixed", "/does/not/exist"],
        ["--model", "x=PLACEHOLDER", "--stride", "8", "--window-length", "8"],
        ["--model", "x=PLACEHOLDER", "--max-target-tokens", "0"],
        ["--model", "x=PLACEHOLDER", "--model", "x=PLACEHOLDER"],
        ["--model", "x=PLACEHOLDER", "--reference", "PLACEHOLDER"],
        ["--model", "x=PLACEHOLDER", "--uniform-bits", "3"],
    ],
)
def test_cli_invalid_arguments_exit_two_without_loading(tmp_path, monkeypatch, extra):
    from paretoquant import quality

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    corpus = tmp_path / "test.json"
    corpus.write_text('["text"]')
    monkeypatch.setattr(
        quality, "evaluate_variants", lambda *a, **k: pytest.fail("loaded"), raising=False
    )
    with pytest.raises(SystemExit) as exc:
        quality.main(
            ["--corpus", str(corpus), "--corpus-name", "test"]
            + [arg.replace("PLACEHOLDER", str(model)) for arg in extra]
        )
    assert exc.value.code == 2


def test_script_help_works_without_mlx_import():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/evaluate_quality.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-target-tokens" in result.stdout
    assert "--reference" in result.stdout


@pytest.mark.parametrize(
    "config,variant_options",
    [
        ({"model_file": "unsafe.py"}, {}),
        ({"auto_map": {"AutoModel": "unsafe.Model"}}, {}),
        ({"quantization": {"bits": 4}}, {"dtype": "float16"}),
        ({"text_config": {"quantization_config": {"bits": 4}}}, {"uniform_bits": 4}),
    ],
)
def test_unsafe_or_quantized_reference_rejected_before_framework_import(
    tmp_path, config, variant_options
):
    import json

    from paretoquant.quality import Corpus, Variant, evaluate_local_model

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        evaluate_local_model(
            Variant("test", model, **variant_options),
            Corpus("text", {}),
            window_length=4,
            stride=2,
            max_target_tokens=3,
            group_size=64,
        )


manual_integration = pytest.mark.skipif(
    os.environ.get("PARETOQUANT_QUALITY_INTEGRATION") != "1",
    reason="manual tiny real MLX acceptance; set PARETOQUANT_QUALITY_INTEGRATION=1",
)


@pytest.mark.integration
@manual_integration
def test_real_mlx_window_losses_match_float64_cpu_oracle():
    import mlx.core as mx

    from paretoquant.quality import mlx_model_losses, numpy_logit_losses

    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        logits = np.array([[[0.0, 2.0, -1.0], [3.0, 1.0, 0.0], [-2.0, 0.0, 4.0]]], dtype=np.float32)

        def model(inputs):
            return mx.array(logits)

        actual = mlx_model_losses(model, [0, 1, 2], [0, 2], 1)
        expected = numpy_logit_losses(logits, [0, 2], 1)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    finally:
        mx.set_default_device(old_device)


@pytest.mark.integration
@manual_integration
def test_tiny_real_local_model_reference_uniform_and_mixed_cli(tmp_path, monkeypatch):
    import gc
    import json
    import weakref

    import mlx.core as mx
    import mlx_lm.utils as utils
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model, ModelArgs
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from paretoquant.quality import main

    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        reference = tmp_path / "reference"
        mixed = tmp_path / "mixed"
        reference.mkdir()
        mixed.mkdir()
        config = dict(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
        mx.random.seed(7)
        model = Model(ModelArgs(**config))
        model.set_dtype(mx.float16)
        mx.eval(model.parameters())
        mx.save_safetensors(
            str(reference / "model.safetensors"), dict(tree_flatten(model.parameters()))
        )
        (reference / "config.json").write_text(json.dumps(config))
        # Use a genuine byte-level BPE fixture: Transformers may select the
        # Qwen2 tokenizer class from model_type instead of a generic WordLevel.
        vocab = {
            token: index
            for index, token in enumerate(
                ["[UNK]", "t", *"0123456789", "Ġ", "Ċ", *"abcdefghijklmnopqr"]
            )
        }
        tokenizer = Tokenizer(models.BPE(vocab, merges=[], unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
        fast.save_pretrained(reference)
        fast.save_pretrained(mixed)
        model, mixed_config = utils.quantize_model(
            model,
            config,
            32,
            4,
            quant_predicate=lambda path, module: (
                {"bits": 3, "group_size": 32, "mode": "affine"}
                if path.endswith("gate_proj")
                else True
            ),
        )
        mx.eval(model.parameters())
        mx.save_safetensors(
            str(mixed / "model.safetensors"), dict(tree_flatten(model.parameters()))
        )
        (mixed / "config.json").write_text(json.dumps(mixed_config))
        del model, tokenizer, fast
        gc.collect()
        corpus = tmp_path / "test.json"
        corpus.write_text(json.dumps(["t1 t2 t3 t4 t5", "", "t6 t7 t8 t9 t10"]))
        output = tmp_path / "quality.json"
        weak_models = []
        real_load = utils.load_model

        def load_checked(*args, **kwargs):
            assert all(ref() is None for ref in weak_models), "previous model retained"
            loaded, cfg = real_load(*args, **kwargs)
            weak_models.append(weakref.ref(loaded))
            return loaded, cfg

        monkeypatch.setattr(utils, "load_model", load_checked)
        assert (
            main(
                [
                    "--corpus",
                    str(corpus),
                    "--corpus-name",
                    "random tiny acceptance only",
                    "--reference",
                    str(reference),
                    "--mixed",
                    str(mixed),
                    "--window-length",
                    "5",
                    "--stride",
                    "2",
                    "--max-target-tokens",
                    "7",
                    "--group-size",
                    "32",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        assert all(ref() is None for ref in weak_models)
        report = json.loads(output.read_text())
        assert report["canonical_benchmark"] is False
        assert [r["name"] for r in report["results"]] == [
            "reference_fp16",
            "uniform4",
            "mixed_stock",
        ]
        assert len({r["tokenization"]["stream_sha256"] for r in report["results"]}) == 1
        for result in report["results"]:
            assert result["metrics"]["target_token_count"] == 7
            assert result["metrics"]["window_count"] == 4
            assert math.isfinite(result["metrics"]["perplexity"])
            assert sum(w["target_token_count"] for w in result["metrics"]["windows"]) == 7
        assert report["results"][0]["model"]["sha256"] == report["results"][1]["model"]["sha256"]
        assert set(report["results"][0]["weight_dtype_counts"]) == {"mlx.core.float16"}
        assert sum(report["results"][0]["weight_dtype_counts"].values()) > 0
        assert {q["bits"] for q in report["results"][1]["quantized_modules"]} == {4}
        assert {q["bits"] for q in report["results"][2]["quantized_modules"]} == {3, 4}
        print(
            "TINY_ACCEPTANCE",
            json.dumps({r["name"]: r["metrics"]["perplexity"] for r in report["results"]}),
        )
    finally:
        mx.set_default_device(old_device)
        mx.clear_cache()


def test_float64_aggregation_preserves_small_losses_and_is_target_weighted():
    from paretoquant.quality import evaluate_tokens

    values = [700.0, 1e-6, 2e-6, 3e-6, 4e-6]

    def losses(inputs, targets, offset):
        return np.asarray([values[t - 1] for t in targets], dtype=np.float32)

    result = evaluate_tokens(list(range(6)), losses, window_length=4, stride=2)
    expected = math.fsum(float(np.float32(v)) for v in values)
    assert result["total_nll"] == expected
    assert result["mean_nll"] == expected / 5
    assert result["truncated"] is False
    assert sum(w["target_token_count"] for w in result["windows"]) == 5


@pytest.mark.parametrize(
    "config", [{"text_config": []}, {"text_config": None}, {"max_position_embeddings": "bad"}]
)
def test_malformed_model_metadata_is_a_clear_value_error(tmp_path, config):
    import json

    from paretoquant.quality import Corpus, Variant, evaluate_local_model

    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        evaluate_local_model(
            Variant("bad", tmp_path), Corpus("text", {}), window_length=4, stride=2
        )


def test_module_entry_point_and_script_remain_cpu_only():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import builtins, runpy, sys; "
        f"sys.path.insert(0, {str(root / 'src')!r}); "
        "original = builtins.__import__; "
        "builtins.__import__ = lambda name, *a, **k: "
        "(_ for _ in ()).throw(AssertionError('unexpected MLX import')) "
        "if name.startswith('mlx') else original(name, *a, **k); "
        "sys.argv = ['paretoquant.quality', '--help']; "
        "runpy.run_module('paretoquant.quality', run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--reference" in result.stdout


@pytest.mark.parametrize("stride", [1, 2])
def test_finite_losses_whose_sum_overflows_are_rejected_cleanly(stride):
    from paretoquant.quality import evaluate_tokens

    with pytest.raises(ValueError, match="finite"):
        evaluate_tokens(
            [0, 1, 2],
            lambda inputs, targets, offset: [1e308] * len(targets),
            window_length=3,
            stride=stride,
        )


@pytest.mark.parametrize("where", ["corpus", "model"])
def test_output_cannot_overwrite_evaluation_inputs(tmp_path, monkeypatch, where):
    from paretoquant import quality

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    corpus = tmp_path / "texts.json"
    corpus.write_text('["text"]')
    output = corpus if where == "corpus" else model / "config.json"
    monkeypatch.setattr(quality, "evaluate_variants", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(SystemExit) as exc:
        quality.main(
            [
                "--corpus",
                str(corpus),
                "--corpus-name",
                "test",
                "--model",
                f"local={model}",
                "--output",
                str(output),
            ]
        )
    assert exc.value.code == 2
    assert corpus.read_text() == '["text"]'
    assert (model / "config.json").read_text() == "{}"
