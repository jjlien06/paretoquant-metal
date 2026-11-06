"""CPU-only regression tests. Run with PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s spikes/033-qwen38-mixed-3-4-real -p test_allocation_build.py -v."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import allocate_real as ar
import build_mixed as bm

HERE = Path(__file__).resolve().parent


def save_tensors(path, tensors):
    """Write valid little-endian safetensors without any MLX dependency."""
    header: dict[str, dict] = {"__metadata__": {"format": "mlx"}}
    payload, cursor = [], 0
    for name, (dtype, shape, data) in sorted(tensors.items()):
        assert len(data) == bm.DTYPE_BYTES[dtype] * __import__("math").prod(shape)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [cursor, cursor + len(data)]}
        payload.append(data)
        cursor += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("xb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for data in payload:
            f.write(data)


def source(root, bits, layers=3, large=False):
    root.mkdir()
    tensors = {}
    for i in range(layers):
        module = f"{ar.PFX}{i}.mlp.down_proj"
        # Valid packed rows: group_size=64 and one row (3bit=24, 4bit=32 bytes).
        tensors[module + ".weight"] = ("U32", [1, bits * 2], bytes([bits + i]) * (bits * 8))
        for part in ("scales", "biases"):
            tensors[module + "." + part] = ("F16", [1, 1], bytes([bits, i]))
        tensors[f"{ar.PFX}{i}.input_layernorm.weight"] = ("F16", [1], bytes([bits, 11]))
    for module in (ar.EMBED[:-1], ar.HEAD[:-1]):
        tensors[module + ".weight"] = ("U32", [1, bits * 2], bytes([bits]) * (bits * 8))
        for part in ("scales", "biases"):
            tensors[module + "." + part] = ("F16", [1, 1], bytes([bits, 9]))
    tensors[ar.NORM + "weight"] = ("F16", [1], b"\x00\x3c")
    tensors["vision_tower.test"] = ("U8", [2], b"vv")
    if large:
        size = bm.CHUNK + 12
        tensors[ar.PFX + "0.linear_attn.conv1d.weight"] = ("U8", [size, 1, 1], b"x" * size)
    save_tensors(root / "model.safetensors", tensors)
    config = {"model_type": "qwen3_5", "text_config": {"num_hidden_layers": layers, "hidden_size": 64},
              "vision_config": {"ignored": True},
              "quantization": {"bits": bits, "group_size": 64, "mode": "affine"}}
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text('{"version":"1.0"}')
    (root / "tokenizer_config.json").write_text('{"chat_template":"test"}')
    (root / "tokenizer_assets").mkdir()
    (root / "tokenizer_assets" / "vocab.txt").write_text("nested tokenizer asset")
    return bm.scan_checkpoint(root)


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=HERE, prefix=".cpu-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.q4 = source(self.root / "q4", 4)
        self.q3 = source(self.root / "q3", 3)
        self.inventory = {"q4": {"tensors": self.q4}, "q3": {"tensors": self.q3}}
        self.sensitivity = {"layers": [{"layer": i, "relative_mse": loss, "extra": True}
                                        for i, loss in enumerate((0.8, 0.1, 0.4))]}

    def allocate(self, **kw):
        return ar.allocate_inventory(self.inventory, self.sensitivity, expected_layers=3,
                                     targets_gib=(16 / ar.GIB,), **kw)

    def test_exact_plan_matches_exhaustive_search_and_speedup(self):
        result = self.allocate()
        plan = result["plans"][0]
        viable = []
        for flags in itertools.product((0, 1), repeat=3):
            saving = sum(8 * flag for flag in flags)
            if saving >= 16:
                viable.append((sum(row["relative_mse"] * flag for row, flag in zip(self.sensitivity["layers"], flags)), flags))
        loss, flags = min(viable)
        self.assertEqual(plan["q3_layers"], [i for i, flag in enumerate(flags) if flag])
        self.assertAlmostEqual(plan["activation_relative_mse_loss"], loss)
        self.assertTrue(plan["exact"])
        self.assertEqual(plan["actual_saved_bytes"], 16)
        self.assertEqual(plan["q3_layer_bytes"], 56)
        self.assertEqual(plan["predicted_decode_bytes"], plan["text_stored_bytes"] - result["embed_bytes"])
        self.assertEqual(plan["predicted_bytes_speedup"], result["uniform_q4_decode_bytes"] / plan["predicted_decode_bytes"])

    def test_missing_duplicate_nonfinite_and_negative_sensitivity(self):
        for rows in (self.sensitivity["layers"][:-1], self.sensitivity["layers"] + self.sensitivity["layers"][:1],
                     [{"layer": i, "relative_mse": float("nan")} for i in range(3)],
                     [{"layer": i, "relative_mse": -1} for i in range(3)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                ar.allocate_inventory(self.inventory, {"layers": rows}, expected_layers=3)

    def test_overflow_is_real_and_never_approximated(self):
        with self.assertRaises(ar.FrontierOverflow):
            ar.allocate_inventory(self.inventory, self.sensitivity, max_states=1,
                                  expected_layers=3, targets_gib=(0,))

    def test_infeasible_target_explicit(self):
        result = ar.allocate_inventory(self.inventory, self.sensitivity, expected_layers=3)
        self.assertEqual(len(result["plans"]), 4)
        self.assertTrue(all("error" in p for p in result["plans"]))

    def test_cli_64_units_and_exclusive_output(self):
        q4 = source(self.root / "cli4", 4, layers=64)
        q3 = source(self.root / "cli3", 3, layers=64)
        # Synthetic legitimate tiny headers imply all four GiB targets infeasible.
        inv = self.root / "inventory.json"
        inv.write_text(json.dumps({"q4": {"tensors": q4}, "q3": {"tensors": q3}}))
        sens = self.root / "sensitivity.json"
        sens.write_text(json.dumps({"layers": [{"layer": i, "relative_mse": i / 1000} for i in range(64)]}))
        out = self.root / "allocation.json"
        command = [sys.executable, str(HERE / "allocate_real.py"), "--inventory", str(inv),
                   "--sensitivity", str(sens), "--output", str(out)]
        run = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(out.read_text())["n_units"], 64)
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_build_selection_and_overrides(self):
        plan = self.allocate()
        output = self.root / "mixed"
        report = bm.build(self.root / "q4", self.root / "q3", plan, 0, output)
        self.assertTrue(report["all_tensor_hashes_verified"])
        self.assertEqual(report["actual_saved_bytes"], 16)
        actual = bm.scan_checkpoint(output)
        self.assertNotIn("vision_tower.test", actual)
        self.assertEqual(len(list(output.glob("layer-*.safetensors"))), 3)
        self.assertTrue(all((output / f"{group}.safetensors").exists() for group in ("embed", "head", "norm")))
        self.assertFalse((output / "BUILD_INCOMPLETE.json").exists())
        config = json.loads((output / "config.json").read_text())
        self.assertNotIn("vision_config", config)
        for name, info in actual.items():
            selected = ar.layer_of(name) in (1, 2) and name.rsplit(".", 1)[0] in ar.decoder_modules(self.q4)
            expected = self.q3[name] if selected else self.q4[name]
            self.assertEqual(bm.hash_range(info), bm.hash_range(expected), name)
            self.assertEqual(info["shape"], expected["shape"])
            self.assertEqual(info["dtype"], expected["dtype"])
            self.assertEqual(report["tensors"][name]["source"], "q3" if selected else "q4")
        for module in ar.decoder_modules(self.q4):
            self.assertEqual(config["quantization"][module]["bits"], 3 if ar.layer_of(module) in (1, 2) else 4)
        for module in (ar.EMBED[:-1], ar.HEAD[:-1]):
            self.assertEqual(config["quantization"][module]["bits"], 4)
        self.assertEqual((output / "tokenizer_assets/vocab.txt").read_bytes(),
                         (self.root / "q4/tokenizer_assets/vocab.txt").read_bytes())
        before = (output / "mixed-manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            bm.build(self.root / "q4", self.root / "q3", plan, 0, output)
        self.assertEqual((output / "mixed-manifest.json").read_bytes(), before)

    def test_stale_plan_fails_before_output_creation(self):
        plan = self.allocate()
        plan["plans"][0]["actual_saved_bytes"] += 1
        output = self.root / "bad-plan"
        with self.assertRaisesRegex(ValueError, "actual source headers"):
            bm.build(self.root / "q4", self.root / "q3", plan, 0, output)
        self.assertFalse(output.exists())

    def test_build_cli(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps(self.allocate()))
        out = self.root / "cli-output"
        run = subprocess.run([sys.executable, str(HERE / "build_mixed.py"), "--q4", str(self.root / "q4"),
                              "--q3", str(self.root / "q3"), "--plan", str(plan), "--plan-index", "0",
                              "--output", str(out)], capture_output=True, text=True,
                             env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(json.loads((out / "mixed-manifest.json").read_text())["complete"])

    def test_loader_override_contract_without_importing_mlx(self):
        loader = ar.ROOT / ".venv/lib/python3.13/site-packages/mlx_lm/utils.py"
        if not loader.exists():
            self.skipTest("local installed mlx_lm source unavailable")
        import ast
        tree = ast.parse(loader.read_text())
        predicates = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "class_predicate"]
        self.assertTrue(any('config[\'quantization\'][p]' in ast.unparse(node) for node in predicates))
        self.assertNotIn("mlx.core", sys.modules)

    def test_safetensors_reference_cpu_reader(self):
        from safetensors import safe_open
        output = self.root / "reference-output"
        bm.build(self.root / "q4", self.root / "q3", self.allocate(), 0, output)
        for shard in output.glob("*.safetensors"):
            parsed = bm.read_header(shard)
            with safe_open(shard, framework="numpy") as handle:
                self.assertEqual(set(handle.keys()), set(parsed))
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    self.assertEqual(list(tensor.shape), parsed[name]["shape"])
                    self.assertEqual(hashlib.sha256(tensor.tobytes()).hexdigest(), bm.hash_range(parsed[name]))

    def test_hash_verification_failure_keeps_incomplete_marker(self):
        output = self.root / "corrupt-output"
        with patch.object(bm, "hash_range", return_value="not-the-source-hash"):
            with self.assertRaisesRegex(ValueError, "hash/length mismatch"):
                bm.build(self.root / "q4", self.root / "q3", self.allocate(), 0, output)
        self.assertTrue((output / "BUILD_INCOMPLETE.json").exists())
        self.assertFalse((output / "mixed-manifest.json").exists())

    def test_output_nested_in_source_rejected(self):
        output = self.root / "q4" / "nested"
        with self.assertRaisesRegex(ValueError, "nested inside"):
            bm.build(self.root / "q4", self.root / "q3", self.allocate(), 0, output)
        self.assertFalse(output.exists())

    def test_malformed_header_rejected(self):
        path = self.root / "truncated.safetensors"
        path.write_bytes(struct.pack("<Q", 100) + b"{}")
        with self.assertRaises(ValueError):
            bm.read_header(path)

    def test_chunk_bound_on_large_tensor(self):
        source(self.root / "large4", 4, large=True)
        source(self.root / "large3", 3, large=True)
        reads = []
        original = Path.open
        class BoundedReader:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                self.handle.__enter__()
                return self
            def __exit__(self, *args):
                return self.handle.__exit__(*args)
            def read(self, size=-1):
                reads.append(size)
                if size < 0 or size > bm.CHUNK:
                    raise AssertionError(f"Unbounded read: {size}")
                return self.handle.read(size)
            def __getattr__(self, attr):
                return getattr(self.handle, attr)
        def bounded(path, mode="r", *args, **kwargs):
            handle = original(path, mode, *args, **kwargs)
            return BoundedReader(handle) if mode == "rb" else handle
        with patch.object(Path, "open", bounded):
            report = bm.build(self.root / "large4", self.root / "large3",
                              {"plans": [{"q3_layers": [1]}], "n_units": 3}, 0, self.root / "large-out")
        self.assertTrue(report["all_tensor_hashes_verified"])
        self.assertIn(bm.CHUNK, reads)
        self.assertLessEqual(max(reads), bm.CHUNK)


if __name__ == "__main__":
    unittest.main()
