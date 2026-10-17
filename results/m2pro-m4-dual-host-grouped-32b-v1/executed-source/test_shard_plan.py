"""CPU-only tests; fixtures live in memory, not on disk."""

import importlib
import io
import json
import os
import struct
from pathlib import Path

import pytest


def planner():
    return importlib.import_module("shard_plan")


def safetensors(header, payload=b"", *, encoded=None):
    encoded = json.dumps(header).encode() if encoded is None else encoded
    return io.BytesIO(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_read_header_counts_packed_storage_not_quantization_bits():
    header = {
        "__metadata__": {"format": "mlx", "bits": "4"},
        "model.layers.0.weight": {"dtype": "U32", "shape": [2, 3], "data_offsets": [0, 24]},
    }
    result = planner().read_safetensors_header(safetensors(header, bytes(24)))
    assert result == {"model.layers.0.weight": header["model.layers.0.weight"]}


@pytest.mark.parametrize(
    "metadata",
    [
        {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 4]},
        {"dtype": "U32", "shape": [1], "data_offsets": [0, 3]},
        {"dtype": "U32", "shape": [-1], "data_offsets": [0, 4]},
        {"dtype": "U32", "shape": [True], "data_offsets": [0, 4]},
        {"dtype": "U32", "shape": [1.0], "data_offsets": [0, 4]},
        {"dtype": "U32", "shape": "1", "data_offsets": [0, 4]},
        {"dtype": "U32", "shape": [1], "data_offsets": [-1, 3]},
        {"dtype": "U32", "shape": [1], "data_offsets": [4, 0]},
        {"dtype": "U32", "shape": [1], "data_offsets": [False, 4]},
        {"dtype": "U32", "shape": [1], "data_offsets": [0.0, 4]},
        {"dtype": "U32", "shape": [1], "data_offsets": [0, 4, 4]},
        {"dtype": "U32", "shape": [1]},
        [],
    ],
)
def test_rejects_invalid_tensor_storage_metadata(metadata):
    with pytest.raises(ValueError):
        planner().read_safetensors_header(safetensors({"x": metadata}, bytes(4)))


@pytest.mark.parametrize(
    "dtype,width",
    [
        ("BOOL", 1),
        ("U8", 1),
        ("I8", 1),
        ("F8_E4M3", 1),
        ("F8_E5M2", 1),
        ("U16", 2),
        ("I16", 2),
        ("F16", 2),
        ("BF16", 2),
        ("U32", 4),
        ("I32", 4),
        ("F32", 4),
        ("U64", 8),
        ("I64", 8),
        ("F64", 8),
    ],
)
def test_validates_storage_width_and_scalar_shape(dtype, width):
    meta = {"dtype": dtype, "shape": [], "data_offsets": [0, width]}
    assert planner().read_safetensors_header(safetensors({"x": meta}, bytes(width))) == {"x": meta}
    with pytest.raises(ValueError):
        planner().read_safetensors_header(safetensors({"x": meta}, bytes(width - 1)))


@pytest.mark.parametrize(
    "encoded",
    [
        b"[]",
        b"null",
        b"{} garbage",
        b"\xff",
        b"{",
        b" {}",
        b'{"__metadata__": []}',
        b'{"__metadata__": {"bits": 4}}',
        b'{"x": {"dtype":"U8","shape":[0],"data_offsets":[0,0]},'
        b'"x": {"dtype":"U8","shape":[0],"data_offsets":[0,0]}}',
        b'{"x": {"dtype":"U8","dtype":"U8","shape":[0],"data_offsets":[0,0]}}',
        b'{"x": {"dtype":"U8","shape":[NaN],"data_offsets":[0,0]}}',
    ],
)
def test_rejects_malformed_json_headers(encoded):
    with pytest.raises(ValueError):
        planner().read_safetensors_header(safetensors({}, encoded=encoded))


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"\0" * 7,
        struct.pack("<Q", 0),
        struct.pack("<Q", 100) + b"{}",
        struct.pack("<Q", 2**64 - 1) + b"{}",
    ],
)
def test_rejects_truncated_or_oversized_headers(blob):
    with pytest.raises(ValueError):
        planner().read_safetensors_header(io.BytesIO(blob))


@pytest.mark.parametrize(
    "offsets,payload_size",
    [
        ([[0, 2], [1, 3]], 3),  # overlap
        ([[0, 2], [3, 5]], 5),  # internal gap
        ([[1, 3]], 3),  # leading gap
        ([[0, 2]], 3),  # trailing unindexed bytes
    ],
)
def test_rejects_non_exhaustive_payload_layout(offsets, payload_size):
    header = {
        str(i): {"dtype": "U8", "shape": [end - start], "data_offsets": [start, end]}
        for i, (start, end) in enumerate(offsets)
    }
    with pytest.raises(ValueError):
        planner().read_safetensors_header(safetensors(header, bytes(payload_size)))


def test_parse_header_bytes_accepts_unordered_tensors_empty_shapes_and_padding():
    header = {
        "last": {"dtype": "U8", "shape": [2], "data_offsets": [2, 4]},
        "empty": {"dtype": "F16", "shape": [0, 20], "data_offsets": [2, 2]},
        "first": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
    }
    assert (
        planner().parse_safetensors_header(json.dumps(header).encode() + b"  ", data_size=4)
        == header
    )
    assert planner().parse_safetensors_header(b"{}", data_size=0) == {}


def test_header_read_is_bounded_and_never_reads_payload():
    class HeaderOnly(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= header_end - self.tell()
            return super().read(size)

    encoded = b'{"x":{"dtype":"U8","shape":[20],"data_offsets":[0,20]}}'
    header_end = 8 + len(encoded)
    stream = HeaderOnly(struct.pack("<Q", len(encoded)) + encoded + bytes(20))
    result = planner().read_safetensors_header(stream, max_header_bytes=len(encoded))
    assert result["x"]["shape"] == [20]
    with pytest.raises(ValueError):
        planner().read_safetensors_header(stream, max_header_bytes=len(encoded) - 1)


def test_local_path_is_opened_read_only_without_creating_files():
    # The already-owned test file is deliberately not a safetensors container.
    for path in (Path(__file__), str(Path(__file__))):
        with pytest.raises(ValueError, match="header length"):
            planner().read_safetensors_header(path)


@pytest.mark.parametrize("valid", [True, False])
def test_borrowed_stream_position_is_restored_even_on_failure(valid):
    stream = safetensors({}) if valid else io.BytesIO(b"bad")
    stream.seek(1)
    if valid:
        assert planner().read_safetensors_header(stream) == {}
    else:
        with pytest.raises(ValueError):
            planner().read_safetensors_header(stream)
    assert stream.tell() == 1
    assert not stream.closed


def qwen_tensors(num_layers=6, *, tied=False):
    # Descriptors may come from separate files: offsets are file-local.
    result = {
        "model.embed_tokens.weight": {"dtype": "U32", "shape": [2, 3], "data_offsets": [0, 24]},
        "model.embed_tokens.scales": {"dtype": "F16", "shape": [2], "data_offsets": [24, 28]},
        "model.embed_tokens.biases": {"dtype": "F16", "shape": [2], "data_offsets": [28, 32]},
        "model.norm.weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
    }
    if not tied:
        for suffix in ("weight", "scales", "biases"):
            result[f"lm_head.{suffix}"] = dict(result[f"model.embed_tokens.{suffix}"])
    for index in range(num_layers):
        size = (index + 1) * 4
        result[f"model.layers.{index}.self_attn.q_proj.weight"] = {
            "dtype": "U32",
            "shape": [index + 1],
            "data_offsets": [0, size],
        }
        result[f"model.layers.{index}.self_attn.q_proj.scales"] = {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [size, size + 2],
        }
    return result


def test_rank_zero_selects_last_layers_and_all_nonlayer_quantization_parameters():
    tensors = qwen_tensors()
    rank = planner().select_pipeline_rank(
        tensors, model_type="qwen2", num_hidden_layers=6, split=[2, 4], rank=0
    )
    assert rank["rank"] == 0
    assert rank["start_idx"] == 4
    assert rank["end_idx"] == 6
    assert rank["layer_indices"] == [4, 5]
    replicated = sorted(key for key in tensors if not key.startswith("model.layers."))
    layer_keys = sorted(
        key for key in tensors if key.startswith(("model.layers.4.", "model.layers.5."))
    )
    assert rank["replicated_keys"] == replicated
    assert rank["layer_keys"] == layer_keys
    assert rank["selected_keys"] == sorted(replicated + layer_keys)
    assert rank["layer_bytes"] == sum(
        tensors[key]["data_offsets"][1] - tensors[key]["data_offsets"][0] for key in layer_keys
    )
    assert rank["replicated_bytes"] == sum(
        tensors[key]["data_offsets"][1] - tensors[key]["data_offsets"][0] for key in replicated
    )
    assert rank["selected_bytes"] == rank["layer_bytes"] + rank["replicated_bytes"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_type": "qwen3"},
        {"model_type": None},
        {"num_hidden_layers": 0},
        {"num_hidden_layers": -1},
        {"num_hidden_layers": True},
        {"num_hidden_layers": 6.0},
        {"split": []},
        {"split": None},
        {"split": [0, 6]},
        {"split": [-1, 7]},
        {"split": [True, 5]},
        {"split": [2.0, 4]},
        {"split": "24"},
        {"split": [2, 3]},
        {"split": [2, 5]},
        {"split": {0: 2, 1: 4}},
        {"rank": -1},
        {"rank": 2},
        {"rank": True},
        {"rank": 0.0},
        {"tie_word_embeddings": "false"},
    ],
)
def test_rejects_unsupported_or_inconsistent_pipeline_configuration(overrides):
    options = {"model_type": "qwen2", "num_hidden_layers": 6, "split": [2, 4], "rank": 0}
    options.update(overrides)
    with pytest.raises(ValueError):
        planner().select_pipeline_rank(qwen_tensors(), **options)


@pytest.mark.parametrize("split", [(6,), (1, 2, 3), (2, 1, 1, 2)])
def test_uneven_multirank_splits_follow_reverse_numbering(split):
    for rank_index, count in enumerate(split):
        rank = planner().select_pipeline_rank(
            qwen_tensors(), model_type="qwen2", num_hidden_layers=6, split=split, rank=rank_index
        )
        start = sum(split[rank_index + 1 :])
        assert rank["layer_indices"] == list(range(start, start + count))


@pytest.mark.parametrize(
    "remove,add",
    [
        ([key for key in qwen_tensors() if key.startswith("model.layers.0.")], {}),
        (["model.embed_tokens.weight"], {}),
        (["model.norm.weight"], {}),
        (["lm_head.weight"], {}),
        ([], {"model.layers.6.weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
        ([], {"model.layers.-1.weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
        ([], {"model.layers.01.weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
        ([], {"model.layers.x.weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
        ([], {"model.layers.1.": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
        ([], {"unexpected.weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
        ([], {"model.embed_tokens.": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
        ([], {1: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
        # Invalid metadata on rank 1's layer must not escape rank 0 validation.
        (
            [],
            {
                "model.layers.0.self_attn.q_proj.weight": {
                    "dtype": "U32",
                    "shape": [1],
                    "data_offsets": [0, 3],
                }
            },
        ),
    ],
)
def test_rejects_incomplete_or_unrecognized_model_parameter_maps(remove, add):
    tensors = qwen_tensors()
    for key in remove:
        del tensors[key]
    tensors.update(add)
    with pytest.raises(ValueError):
        planner().select_pipeline_rank(
            tensors, model_type="qwen2", num_hidden_layers=6, split=[2, 4], rank=0
        )


def test_tied_embeddings_have_no_separate_head_replica():
    rank = planner().select_pipeline_rank(
        qwen_tensors(tied=True),
        model_type="qwen2",
        num_hidden_layers=6,
        split=[2, 4],
        rank=0,
        tie_word_embeddings=True,
    )
    assert not any(key.startswith("lm_head.") for key in rank["replicated_keys"])
    with pytest.raises(ValueError, match="tied"):
        planner().select_pipeline_rank(
            qwen_tensors(),
            model_type="qwen2",
            num_hidden_layers=6,
            split=[2, 4],
            rank=0,
            tie_word_embeddings=True,
        )


@pytest.mark.parametrize("split", [[6], [2, 4], [1, 2, 3]])
def test_plan_reports_exhaustive_disjoint_layers_and_replication_overhead(split):
    tensors = qwen_tensors()
    plan = planner().plan_qwen2_pipeline(
        tensors, model_type="qwen2", num_hidden_layers=6, split=split
    )
    assert plan["model_type"] == "qwen2"
    assert plan["num_hidden_layers"] == 6
    assert plan["world_size"] == len(split)
    assert plan["split"] == split
    assert plan["tie_word_embeddings"] is False
    assert plan["coverage"]["disjoint"] is True
    assert plan["coverage"]["exhaustive"] is True
    layer_owners = plan["coverage"]["layer_owner_by_index"]
    selected_layers = []
    selected_keys = set()
    for rank_index, rank in enumerate(plan["ranks"]):
        assert rank == planner().select_pipeline_rank(
            tensors, model_type="qwen2", num_hidden_layers=6, split=split, rank=rank_index
        )
        selected_layers.extend(rank["layer_indices"])
        selected_keys.update(rank["selected_keys"])
        for layer in rank["layer_indices"]:
            assert layer_owners[layer] == rank_index
        assert rank["replicated_keys"] == plan["replicated_keys"]
        assert rank["replicated_bytes"] == plan["replicated_bytes_per_rank"]
    assert sorted(selected_layers) == list(range(6))
    assert len(set(selected_layers)) == len(selected_layers)
    assert selected_keys == set(tensors)
    assert plan["replicated_parameters"] == {
        "embedding": sorted(key for key in tensors if key.startswith("model.embed_tokens.")),
        "norm": ["model.norm.weight"],
        "untied_head": sorted(key for key in tensors if key.startswith("lm_head.")),
    }
    assert plan["unique_tensor_bytes"] == sum(
        metadata["data_offsets"][1] - metadata["data_offsets"][0] for metadata in tensors.values()
    )
    assert plan["aggregate_selected_bytes"] == sum(r["selected_bytes"] for r in plan["ranks"])
    assert plan["replication_overhead_bytes"] == (
        (len(split) - 1) * plan["replicated_bytes_per_rank"]
    )
    assert plan["aggregate_selected_bytes"] == (
        plan["unique_tensor_bytes"] + plan["replication_overhead_bytes"]
    )
    assert "packed" in plan["storage_note"]
    assert "KV" in plan["storage_note"]
    assert json.loads(json.dumps(plan)) == plan


def test_merge_independent_file_metadata_maps_without_confusing_local_offsets():
    tensors = qwen_tensors()
    first = {key: meta for key, meta in tensors.items() if key.startswith("model.layers.")}
    second = {key: meta for key, meta in tensors.items() if key not in first}
    merged = planner().merge_safetensors_headers(iter([first, second]))
    assert merged == tensors
    with pytest.raises(ValueError, match="duplicate"):
        planner().merge_safetensors_headers([first, first])
    with pytest.raises(ValueError):
        planner().merge_safetensors_headers(
            [{"broken": {"dtype": "U8", "shape": [1], "data_offsets": [0, 2]}}]
        )
    with pytest.raises(ValueError):
        planner().merge_safetensors_headers([[]])


def test_real_qwen32_archived_headers_match_two_host_budget():
    root = Path(
        os.environ.get(
            "PARETOQUANT_QWEN32_HEADER_DIR",
            Path.home() / ".hermes/cache/scratch/paretoquant-dual-host",
        )
    )
    config_path = root / "qwen32-config.json"
    metadata_path = root / "qwen32-metadata.json"
    if not config_path.exists() or not metadata_path.exists():
        pytest.skip("optional parent-supplied Qwen32 metadata-only fixtures not available")
    config = json.loads(config_path.read_bytes())
    metadata = json.loads(metadata_path.read_bytes())
    assert metadata["sha"] == "2938092373e5f97b95538884112085364c2da315"
    file_sizes = {
        item["rfilename"]: item["size"]
        for item in metadata["siblings"]
        if item["rfilename"].endswith(".safetensors")
    }
    paths = sorted(root.glob("model-*.safetensors.header.json"))
    assert len(paths) == len(file_sizes) == 4
    assert sum(file_sizes.values()) == 18_431_478_459
    headers = []
    for path in paths:
        raw = path.read_bytes()
        decoded = json.loads(raw)
        # Archived JSON was reformatted: infer payload size from the descriptors,
        # not its text length. This cannot verify the original container length.
        payload_size = max(
            meta["data_offsets"][1] for key, meta in decoded.items() if key != "__metadata__"
        )
        assert payload_size < file_sizes[path.name.removesuffix(".header.json")]
        headers.append(planner().parse_safetensors_header(raw, data_size=payload_size))
    tensors = planner().merge_safetensors_headers(headers)
    plan = planner().plan_qwen2_pipeline(
        tensors,
        model_type=config["model_type"],
        num_hidden_layers=config["num_hidden_layers"],
        split=[28, 36],
        tie_word_embeddings=config["tie_word_embeddings"],
    )
    assert plan["tensor_count"] == len(tensors) == 1671
    assert plan["unique_tensor_bytes"] == 18_431_289_344
    assert plan["replicated_bytes_per_rank"] == 875_898_880
    assert [rank["selected_bytes"] for rank in plan["ranks"]] == [8_556_382_208, 10_750_806_016]
    assert plan["ranks"][0]["layer_indices"] == list(range(36, 64))
    assert plan["ranks"][1]["layer_indices"] == list(range(36))
