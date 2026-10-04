"""Preflight failures must happen before requiring GPU dependencies or writing files."""

import builtins

import pytest

from paretoquant.cli import _prepare_output, main


@pytest.mark.parametrize("command", ["run", "generate", "replay"])
def test_missing_source_checked_before_gpu_import(monkeypatch, command):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("mlx"):
            raise AssertionError("GPU import happened before source validation")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    args = [command, "--model", "/path/that/does/not/exist"]
    if command == "generate":
        args += ["--prompt", "fixture"]
    assert main(args) == 2


def test_output_refuses_nonempty_directory(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "important.txt").write_text("preserve this")
    with pytest.raises(ValueError, match="empty"):
        _prepare_output(out, tmp_path / "source")
    assert (out / "important.txt").read_text() == "preserve this"


def test_output_refuses_overwriting_source_model(tmp_path):
    with pytest.raises(ValueError, match="source"):
        _prepare_output(tmp_path, tmp_path / "model")


def test_fresh_output_is_created(tmp_path):
    output = tmp_path / "out"
    _prepare_output(output, tmp_path / "source")
    assert output.is_dir()
