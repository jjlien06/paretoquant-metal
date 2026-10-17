"""CPU-only standard-library tests; no MLX, NumPy, or checkpoint imports."""

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).parent


def module():
    spec = importlib.util.spec_from_file_location("screen", HERE / "screen.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.assets = Path(self.temp.name)

    def fixture(self):
        tensors = {}
        for projection, n, k in [
            ("gate_proj", 27648, 5120),
            ("up_proj", 27648, 5120),
            ("down_proj", 5120, 27648),
        ]:
            for kind in ("weight", "scales", "biases"):
                name = f"model.layers.0.mlp.{projection}.{kind}"
                filename = f"{projection}.{kind}.npy"
                data = filename.encode()
                (self.assets / filename).write_bytes(data)
                tensors[name] = dict(
                    file=filename,
                    original_name=name,
                    shape=[n, k // (8 if kind == "weight" else 64)],
                    dtype="<u4" if kind == "weight" else "<f2",
                    raw_sha256=hashlib.sha256(data).hexdigest(),
                    npy_sha256=hashlib.sha256(data).hexdigest(),
                )
        manifest = dict(
            configuration=dict(bits=4, group_size=64),
            revision="a" * 40,
            source_config_sha256="b" * 64,
            tensors=tensors,
        )
        self.save(manifest)
        return manifest

    def save(self, manifest):
        (self.assets / "manifest.json").write_text(json.dumps(manifest))

    def test_rejects_untrusted_paths_metadata_hashes_and_sizes(self):
        screen = module()
        mutations = [
            lambda m: m["tensors"].pop(next(iter(m["tensors"]))),
            lambda m: m["tensors"].update(extra={}),
            lambda m: m["configuration"].update(bits=3),
            lambda m: m.update(revision="main"),
            lambda m: m.update(source_config_sha256="B" * 64),
        ]
        for field, value in [
            ("file", "../gate_proj.weight.npy"),
            ("file", "/gate_proj.weight.npy"),
            ("file", "gate_proj.weight.bin"),
            ("dtype", "uint32"),
            ("shape", [1, 1]),
            ("original_name", "other"),
            ("raw_sha256", "A" * 64),
            ("npy_sha256", "c" * 64),
        ]:
            mutations.append(
                lambda m, f=field, v=value: m["tensors"][next(iter(m["tensors"]))].update({f: v})
            )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                manifest = self.fixture()
                mutate(manifest)
                self.save(manifest)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    screen.validate_assets(self.assets)
        manifest = self.fixture()
        tensor = self.assets / "gate_proj.weight.npy"
        tensor.unlink()
        tensor.symlink_to(self.assets / "up_proj.weight.npy")
        manifest["tensors"][next(iter(manifest["tensors"]))]["npy_sha256"] = hashlib.sha256(
            tensor.read_bytes()
        ).hexdigest()
        self.save(manifest)
        with self.assertRaises(ValueError):
            screen.validate_assets(self.assets)
        tensor.unlink()
        self.fixture()
        with tensor.open("wb") as handle:
            handle.truncate(256 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            screen.validate_assets(self.assets)
        self.fixture()
        for filename in ("gate_proj.weight.npy", "up_proj.weight.npy"):
            with (self.assets / filename).open("wb") as handle:
                handle.truncate(151 * 1024 * 1024)
        with self.assertRaisesRegex(ValueError, "total"):
            screen.validate_assets(self.assets)

    def test_load_rechecks_actual_arrays_and_raw_hash_before_gpu(self):
        manifest = self.fixture()
        loaded = []

        def load(path, *, allow_pickle):
            self.assertFalse(allow_pickle)
            loaded.append(path)
            name = next(n for n, v in manifest["tensors"].items() if v["file"] == path.name)
            info = manifest["tensors"][name]
            return SimpleNamespace(
                shape=tuple(info["shape"]),
                dtype=SimpleNamespace(str=info["dtype"]),
                tobytes=lambda order: path.read_bytes(),
            )

        np = SimpleNamespace(load=load)
        self.assertEqual(len(module().load_arrays(self.assets, manifest, np)), 9)
        self.assertEqual(len(loaded), 9)
        info = manifest["tensors"][next(iter(manifest["tensors"]))]
        info["raw_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "raw"):
            module().load_arrays(self.assets, manifest, np)
        info["raw_sha256"] = hashlib.sha256((self.assets / info["file"]).read_bytes()).hexdigest()
        original = load
        for field, bad in [("shape", (1, 1)), ("dtype", SimpleNamespace(str="<f4"))]:

            def malformed(path, *, allow_pickle, field=field, bad=bad):
                array = original(path, allow_pickle=allow_pickle)
                setattr(array, field, bad)
                return array

            with self.assertRaisesRegex(ValueError, "actual"):
                module().load_arrays(self.assets, manifest, SimpleNamespace(load=malformed))
        (self.assets / info["file"]).write_bytes(b"changed after validation")
        with self.assertRaisesRegex(ValueError, "file hash"):
            module().load_arrays(self.assets, manifest, np)

    def test_parity_is_finite_tolerance_gated_and_retains_negative_metrics(self):
        screen = module()
        self.assertEqual((screen.RTOL, screen.ATOL), (0.01, 0.02))
        good = screen.parity([0, 2], [0.019, 2.039])
        self.assertTrue(good["passed"])
        self.assertAlmostEqual(good["max_abs_error"], 0.039)
        self.assertGreater(good["relative_mse"], 0)
        bad = screen.parity([0, 2], [0.021, 2])
        self.assertFalse(bad["passed"])
        self.assertTrue(bad["finite"])
        self.assertFalse(screen.parity([1], [float("nan")])["passed"])
        self.assertFalse(screen.parity([float("inf")], [float("inf")])["finite"])
        self.assertEqual(screen.parity([0], [0])["relative_mse"], 0)
        with self.assertRaises(ValueError):
            screen.parity([1], [1, 2])

    def test_three_synthetic_rows_are_independent_seeded_and_unit_rms(self):
        screen = module()
        rows = screen.synthetic_rows(0)
        self.assertEqual(rows, screen.synthetic_rows(0))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1], screen.synthetic_rows(1)[0])
        self.assertNotEqual(rows[0], rows[1])
        for row in rows:
            self.assertEqual(len(row), 5120)
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in row) / 5120), 1, places=14)

    def test_screen_excludes_bad_parity_and_selects_only_fresh_validation(self):
        screen = module()
        functions = {
            "native_stock": lambda x: [x],
            "compiled_stock": lambda x: [x],
            "fused_rpg1": lambda x: [x],
            "fused_rpg2": lambda x: [x],
            "fused_rpg4": lambda x: [x + 1],
            "fused_rpg8": lambda x: [x],
        }
        calls = []

        def benchmark(functions, *, warmup, repeats):
            calls.append(tuple(functions))
            for fn in functions.values():
                fn()
            self.assertEqual((warmup, repeats), (3, 24))
            validation = len(calls) > 3
            return {
                name: {
                    "samples_ms": [
                        (
                            0.95
                            if name == "fused_rpg8"
                            else 0.8
                            if name == "fused_rpg2"
                            else 1.2
                            if validation and name == "fused_rpg1"
                            else 0.7
                            if name == "fused_rpg1"
                            else 1.0
                        )
                    ]
                    * repeats
                }
                for name in functions
            }

        result = screen.screen_functions(
            functions, [1, 2, 3], lambda x: x, benchmark, repeats=24, warmup=3
        )
        self.assertFalse(result["parity"]["fused_rpg4"][0]["passed"])
        self.assertTrue(all("fused_rpg4" not in names for names in calls))
        self.assertEqual(len(calls), 6)
        self.assertNotEqual(calls[0], calls[1])
        self.assertEqual(result["winners"], ["fused_rpg2"])
        self.assertEqual(len(result["timings"]["validation"]), 3)
        self.assertEqual(result["summary"]["validation"]["fused_rpg2"]["sample_count"], 72)
        self.assertEqual(result["summary"]["validation"]["compiled_stock"]["median_ms"], 1)
        functions["compiled_stock"] = lambda x: [1.025]
        functions["native_stock"] = lambda x: [1]
        functions["fused_rpg1"] = lambda x: [0.98]
        result = screen.screen_functions(
            functions, [1, 2, 3], lambda x: x, benchmark, repeats=24, warmup=3
        )
        self.assertFalse(result["parity"]["fused_rpg1"][0]["vs_compiled_stock"]["passed"])
        self.assertFalse(result["parity"]["fused_rpg1"][0]["passed"])
        functions["compiled_stock"] = lambda x: [x + 1]
        calls.clear()
        result = screen.screen_functions(
            functions, [1, 2, 3], lambda x: x, benchmark, repeats=24, warmup=3
        )
        self.assertEqual(result["status"], "control_parity_failed")
        self.assertEqual(calls, [])

    def test_whole_mlp_uses_same_down_and_compiles_stock_and_each_fused_path(self):
        events, compiled = [], []
        weights = {name: name for name in module().EXPECTED}

        def down(activation, *arrays, **kwargs):
            events.append(("down", arrays, kwargs))
            return activation * 10

        def compile(fn):
            compiled.append(fn)
            return fn

        def stock(x, gate, up, **kwargs):
            events.append(("stock", gate, up, kwargs))
            return x * 2

        def fused(x, gate, up, **kwargs):
            events.append(("fused", gate, up, kwargs))
            return x * 2

        mx = SimpleNamespace(compile=compile, quantized_matmul=down)
        kernels = SimpleNamespace(
            stock_gate_up=stock, compiled_stock_gate_up=stock, compiled_fused_gate_up=fused
        )
        functions = module().build_functions(mx, kernels, weights)
        self.assertEqual(set(functions), {"gate_up", "mlp"})
        self.assertEqual(len(compiled), 5)  # whole stock plus whole fused rpg1/2/4/8
        for fn in functions["mlp"].values():
            self.assertEqual(fn(3), 60)
        down_calls = [event for event in events if event[0] == "down"]
        self.assertEqual(len(down_calls), 6)
        for _, arrays, kwargs in down_calls:
            self.assertEqual(
                arrays,
                tuple(
                    f"model.layers.0.mlp.down_proj.{kind}"
                    for kind in ("weight", "scales", "biases")
                ),
            )
            self.assertEqual(kwargs, dict(bits=4, group_size=64, transpose=True))
        self.assertEqual({e[3]["rows_per_group"] for e in events if e[0] == "fused"}, {1, 2, 4, 8})

    def test_gpu_orchestration_restores_device_cache_and_keeps_gate_failures_out_of_mlp(self):
        screen = module()
        events, timed = [], []

        class Array(list):
            def reshape(self, *shape):
                return self

            def tolist(self):
                return list(self)

        def array(value, dtype=None):
            return Array(value)

        mx = SimpleNamespace(
            gpu="gpu",
            float16="f16",
            array=array,
            eval=lambda *x: None,
            default_device=lambda: "cpu",
            set_default_device=lambda v: events.append(("device", v)),
            set_cache_limit=lambda v: events.append(("cache", v)) or 99,
            reset_peak_memory=lambda: None,
            get_peak_memory=lambda: 123,
        )
        functions = {
            "native_stock": lambda x: Array([1]),
            "compiled_stock": lambda x: Array([1]),
            "fused_rpg1": lambda x: Array([1]),
            "fused_rpg4": lambda x: Array([9]),
        }

        def benchmark(functions, **kwargs):
            timed.append(tuple(functions))
            return {name: dict(samples_ms=[1] * kwargs["repeats"]) for name in functions}

        args = SimpleNamespace(seed=0, warmup=3, repeats=24)
        arrays = {name: [1] for name in screen.EXPECTED}
        with patch.object(
            screen, "build_functions", return_value={"gate_up": functions, "mlp": functions}
        ):
            result = screen.gpu_screen(mx, None, benchmark, arrays, args)
        self.assertEqual(result["peak_mlx_bytes"], 123)
        self.assertEqual(result["input_shape"], [1, 5120])
        self.assertTrue(all("fused_rpg4" not in names for names in timed))
        self.assertEqual(result["mlp"]["excluded_gate_up_failures"], ["fused_rpg4"])
        self.assertEqual(
            events,
            [("device", "gpu"), ("cache", 256 * 1024 * 1024), ("cache", 99), ("device", "cpu")],
        )
        events.clear()
        with patch.object(screen, "build_functions", side_effect=RuntimeError("CPU test error")):
            with self.assertRaisesRegex(RuntimeError, "CPU test error"):
                screen.gpu_screen(mx, None, benchmark, arrays, args)
        self.assertEqual(events[-2:], [("cache", 99), ("device", "cpu")])

    def test_cli_existing_output_rejects_before_imports_and_errors_keep_exclusive_evidence(self):
        screen = module()
        self.fixture()
        output = self.assets / "evidence.json"
        argv = ["--assets", str(self.assets), "--output", str(output)]
        output.write_text("untouched")
        with patch.dict(sys.modules, {"numpy": None, "mlx": None, "mlx.core": None}):
            with self.assertRaises(FileExistsError):
                screen.main(argv)
        self.assertEqual(output.read_text(), "untouched")
        output.unlink()
        output.symlink_to(self.assets / "missing.json")
        with self.assertRaises(FileExistsError):
            screen.main(argv)
        output.unlink()
        with patch.dict(sys.modules, {"numpy": None, "mlx": None, "mlx.core": None}):
            with self.assertRaises(ImportError):
                screen.main(argv)
        evidence = json.loads(output.read_text())
        self.assertEqual(evidence["status"], "error")
        self.assertNotIn("timings", evidence)
        self.assertEqual(
            evidence["manifest_sha256"],
            hashlib.sha256((self.assets / "manifest.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(evidence["arguments"]["repeats"], 24)
        self.assertEqual(evidence["arguments"]["warmup"], 3)
        self.assertEqual(evidence["arguments"]["seed"], 0)

    def test_source_hashes_cover_installed_kernels_wrappers_and_harness(self):
        screen = module()
        package = self.assets / "package"
        (package / "kernels").mkdir(parents=True)
        for filename in (
            "metal.py",
            "benchmark.py",
            "kernels/gate_up.metal",
            "kernels/gate_up_packed.metal",
        ):
            (package / filename).write_text(filename)
        hashes = screen.source_hashes(package)
        self.assertEqual(
            set(hashes),
            {
                "metal.py",
                "benchmark.py",
                "kernels/gate_up.metal",
                "kernels/gate_up_packed.metal",
                "screen.py",
            },
        )
        self.assertEqual(hashes["metal.py"], hashlib.sha256(b"metal.py").hexdigest())
        self.assertEqual(
            hashes["screen.py"], hashlib.sha256((HERE / "screen.py").read_bytes()).hexdigest()
        )
        import ast

        tree = ast.parse((HERE / "screen.py").read_text())
        runner = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_gpu"
        )
        imports = [n for n in ast.walk(runner) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertTrue(
            any(isinstance(n, ast.Import) and n.names[0].name == "mlx.core" for n in imports)
        )
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(
                    any(n.name.startswith(("mlx", "numpy", "paretoquant")) for n in node.names)
                )
            elif isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith(("mlx", "numpy", "paretoquant")))

    def test_accepts_exact_nine_tensor_manifest_without_gpu_imports(self):
        self.fixture()
        self.assertEqual(len(module().validate_assets(self.assets)["tensors"]), 9)


if __name__ == "__main__":
    unittest.main()
