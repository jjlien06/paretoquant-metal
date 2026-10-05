"""CPU-only tests of bounded, byte-exact per-rank checkpoint writing."""

import json
import struct
from pathlib import Path

import pytest


def fixture_checkpoint(root):
    root.mkdir()
    names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.fixture.weight",
        "model.layers.1.fixture.weight",
    ]
    header = {}
    data = b""
    for i, name in enumerate(names):
        payload = struct.pack("<ff", float(i), float(i + 10))
        header[name] = {"dtype": "F32", "shape": [2], "data_offsets": [len(data), len(data) + 8]}
        data += payload
    encoded = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 2,
                "tie_word_embeddings": False,
            }
        )
    )
    (root / "tokenizer_config.json").write_text("{}")
    templates = root / "additional_chat_templates"
    templates.mkdir()
    (templates / "named.jinja").write_text("{{ messages }}")
    return names, header, data


def test_materialize_rank_preserves_exact_tensor_payloads_and_templates(tmp_path):
    source = tmp_path / "source"
    _, header, data = fixture_checkpoint(source)
    assert (Path(__file__).parent / "materialize.py").exists(), "materializer is not implemented"
    from materialize import materialize_rank
    from shard_plan import read_safetensors_header

    target = tmp_path / "rank0"
    record = materialize_rank(source, target, split=[1, 1], rank=0, reserve_bytes=0)
    copied = read_safetensors_header(target / "model.safetensors")
    assert set(copied) == {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.1.fixture.weight",
    }
    assert record["parameter_bytes"] == 32
    assert record["complete"] is True
    assert (target / "additional_chat_templates/named.jinja").read_text() == "{{ messages }}"
    with (target / "model.safetensors").open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        for name, descriptor in copied.items():
            handle.seek(8 + size + descriptor["data_offsets"][0])
            actual = handle.read(descriptor["data_offsets"][1] - descriptor["data_offsets"][0])
            lo, hi = header[name]["data_offsets"]
            assert actual == data[lo:hi]
    index = json.loads((target / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(copied)
    assert index["metadata"]["total_size"] == 32


def test_materialize_rank_does_not_overwrite_existing_destination(tmp_path):
    source = tmp_path / "source"
    fixture_checkpoint(source)
    from materialize import materialize_rank

    target = tmp_path / "rank0"
    target.mkdir()
    original = target / "existing"
    original.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        materialize_rank(source, target, split=[1, 1], rank=0, reserve_bytes=0)
    assert original.read_bytes() == b"do not overwrite"


def test_reserve_rechecked_across_many_small_tensors(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import materialize

    source = tmp_path / "source"
    _, header, data = fixture_checkpoint(source)
    payload = b"x" * materialize.CHUNK_BYTES
    offset = len(data)
    for i in range(65):
        header[f"model.layers.1.small_{i}.weight"] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    encoded = json.dumps(header).encode()
    with (source / "model.safetensors").open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + data)
        for _ in range(65):
            handle.write(payload)
    target = tmp_path / "rank0"

    def disk_usage(path):
        return SimpleNamespace(free=0 if Path(path) == target else 1024**3)

    monkeypatch.setattr(materialize.shutil, "disk_usage", disk_usage)
    with pytest.raises(ValueError, match="Disk reserve exhausted"):
        materialize.materialize_rank(source, target, split=[1, 1], rank=0, reserve_bytes=1)
    assert 64 * 1024**2 <= (target / "model.safetensors").stat().st_size < 65 * 1024**2
    assert not (target / "rank_manifest.json").exists()


@pytest.mark.parametrize("asset_count, chunks_per_asset", [(1, 65), (65, 1)])
def test_reserve_rechecked_during_streamed_assets(
    tmp_path, monkeypatch, asset_count, chunks_per_asset
):
    from types import SimpleNamespace

    import materialize

    source = tmp_path / "source"
    fixture_checkpoint(source)
    payload = b"a" * materialize.CHUNK_BYTES
    assets = [source / f"asset_{i:02}.model" for i in range(asset_count)]
    for asset in assets:
        with asset.open("wb") as handle:
            for _ in range(chunks_per_asset):
                handle.write(payload)
    target = tmp_path / "rank0"

    def disk_usage(path):
        return SimpleNamespace(free=0 if Path(path) == target else 1024**3)

    monkeypatch.setattr(materialize.shutil, "disk_usage", disk_usage)
    with pytest.raises(ValueError, match="Disk reserve exhausted"):
        materialize.materialize_rank(source, target, split=[1, 1], rank=0, reserve_bytes=1)
    copied = [target / asset.name for asset in assets if (target / asset.name).exists()]
    assert sum(asset.stat().st_size for asset in copied) == 64 * 1024**2
    assert all(
        asset.read_bytes() == payload * (asset.stat().st_size // len(payload)) for asset in copied
    )
    assert not (target / "rank_manifest.json").exists()


def test_reserve_rechecked_before_manifest_publication(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import materialize

    source = tmp_path / "source"
    fixture_checkpoint(source)
    target = tmp_path / "rank0"

    def disk_usage(path):
        exhausted = Path(path) == target and (target / "model.safetensors.index.json").exists()
        return SimpleNamespace(free=0 if exhausted else 1024**3)

    monkeypatch.setattr(materialize.shutil, "disk_usage", disk_usage)
    with pytest.raises(ValueError, match="Disk reserve exhausted"):
        materialize.materialize_rank(source, target, split=[1, 1], rank=0, reserve_bytes=1)
    assert (target / "model.safetensors.index.json").exists()
    assert (target / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert (target / "additional_chat_templates/named.jinja").read_bytes() == b"{{ messages }}"
    assert not (target / "rank_manifest.json").exists()


def test_config_mutation_between_validation_and_copy_blocks_manifest(tmp_path, monkeypatch):
    import materialize

    source = tmp_path / "source"
    fixture_checkpoint(source)
    config_path = source / "config.json"
    admitted = config_path.read_bytes()
    changed = admitted.replace(b'"qwen2"', b'"other"')
    target = tmp_path / "rank0"
    original_open = Path.open
    config_reads = 0

    def mutate_on_copy(path, mode="r", *args, **kwargs):
        nonlocal config_reads
        if path == config_path and mode == "rb":
            config_reads += 1
            if config_reads == 3:
                config_path.write_bytes(changed)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", mutate_on_copy)
    with pytest.raises(ValueError, match="config changed"):
        materialize.materialize_rank(source, target, split=[1, 1], rank=0, reserve_bytes=0)
    assert config_reads == 3
    assert (target / "config.json").read_bytes() == changed
    assert not (target / "rank_manifest.json").exists()


@pytest.mark.parametrize(
    "asset_name", ["tokenizer_config.json", "additional_chat_templates/named.jinja"]
)
@pytest.mark.parametrize("when", ["before_copy", "after_copy"])
def test_source_asset_mutation_blocks_manifest(tmp_path, monkeypatch, asset_name, when):
    import os

    import materialize

    source = tmp_path / "source"
    fixture_checkpoint(source)
    asset = source / asset_name
    admitted = asset.read_bytes()
    changed = b"x" * len(admitted)
    original_stat = asset.stat()
    target = tmp_path / "rank0"
    original_open = Path.open
    mutated = False

    def mutate_asset(path, mode="r", *args, **kwargs):
        nonlocal mutated
        trigger = asset if when == "before_copy" else target / "model.safetensors"
        if not mutated and path == trigger and mode == "rb":
            mutated = True
            asset.write_bytes(changed)
            os.utime(asset, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 10**9))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", mutate_asset)
    with pytest.raises(ValueError, match="asset changed"):
        materialize.materialize_rank(source, target, split=[1, 1], rank=0, reserve_bytes=0)
    assert mutated
    assert (target / asset_name).read_bytes() == (changed if when == "before_copy" else admitted)
    assert not (target / "rank_manifest.json").exists()


@pytest.mark.parametrize(
    "asset_name", ["tokenizer_config.json", "additional_chat_templates/named.jinja"]
)
def test_copied_asset_mutation_blocks_manifest(tmp_path, monkeypatch, asset_name):
    import materialize

    source = tmp_path / "source"
    fixture_checkpoint(source)
    target = tmp_path / "rank0"
    original_open = Path.open
    mutated = False

    def mutate_copied_asset(path, mode="r", *args, **kwargs):
        nonlocal mutated
        if not mutated and path == target / "model.safetensors" and mode == "rb":
            mutated = True
            asset = target / asset_name
            asset.write_bytes(b"x" * asset.stat().st_size)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", mutate_copied_asset)
    with pytest.raises(ValueError, match="Copied asset changed"):
        materialize.materialize_rank(source, target, split=[1, 1], rank=0, reserve_bytes=0)
    assert mutated
    assert not (target / "rank_manifest.json").exists()
