"""CPU-only synthetic fixtures: these tests are not model evidence."""

import importlib.util
import types
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("draft_screen.py")
if PATH.exists():
    spec = importlib.util.spec_from_file_location("draft_screen", PATH)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
else:
    screen = types.SimpleNamespace()


class AcceptanceTests(unittest.TestCase):
    def test_matching_prefix_emits_bonus_capped_at_eos(self):
        self.assertTrue(callable(getattr(screen, "acceptance", None)), "missing acceptance utility")
        self.assertEqual(screen.acceptance([1, 9, 3], [1, 2, 3], {3}), (1, 2))
        self.assertEqual(screen.acceptance([1, 2], [1, 2, 3], {3}), (2, 3))
        self.assertEqual(screen.acceptance([3, 7], [3, 7], {3}), (1, 1))
        self.assertEqual(screen.acceptance([9], [3], {3}), (0, 1))
        self.assertEqual(screen.acceptance([1, 3], [1, 3], {3}), (2, 2))

    def test_trace_requires_bounded_regular_json_exact_ids_prompt_and_terminal_eos(self):
        self.assertTrue(
            callable(getattr(screen, "read_reference", None)), "missing trace validation"
        )
        import json
        import tempfile

        with tempfile.TemporaryDirectory(dir=Path.home() / ".hermes/cache/scratch") as root:
            path = Path(root) / "trace.json"
            valid = {"prompt": "raw", "requests": [{"token_ids": [1, 3], "finish_reason": "stop"}]}
            path.write_text(json.dumps(valid))
            self.assertEqual(screen.read_reference(path, None, 4, {3})[1:], ("raw", [1, 3]))
            for bad in ([True, 3], [-1, 3], [4, 3], [], [3, 1, 3], [1, 2]):
                valid["requests"][0]["token_ids"] = bad
                path.write_text(json.dumps(valid))
                with self.assertRaises(ValueError):
                    screen.read_reference(path, None, 4, {3})
            valid["requests"][0]["token_ids"] = [1, 3]
            valid["requests"][0]["finish_reason"] = "length"
            path.write_text(json.dumps(valid))
            with self.assertRaises(ValueError):
                screen.read_reference(path, None, 4, {3})
            valid["requests"][0]["finish_reason"] = "stop"
            path.write_text(json.dumps(valid))
            with self.assertRaises(ValueError):
                screen.read_reference(path, "different", 4, {3})
            del valid["prompt"]
            path.write_text(json.dumps(valid))
            with self.assertRaises(ValueError):
                screen.read_reference(path, None, 4, {3})
            self.assertEqual(screen.read_reference(path, "raw", 4, {3})[1], "raw")
            link = Path(root) / "link"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                screen.read_reference(link, "raw", 4, {3})
            path.write_bytes(b" " * (1024 * 1024 + 1))
            with self.assertRaises(ValueError):
                screen.read_reference(path, "raw", 4, {3})
            with self.assertRaises(ValueError):
                screen.read_reference(Path(root), "raw", 4, {3})

    def test_prepare_binds_pinned_config_receipt_reference_before_runtime(self):
        self.assertTrue(callable(getattr(screen, "prepare", None)), "missing CPU preparation")
        import hashlib
        import json
        import tempfile

        archive = Path(__file__).parents[2] / "results/m2pro-m4-dual-host-phased-32b-v1/qwen32"
        with tempfile.TemporaryDirectory(dir=Path.home() / ".hermes/cache/scratch") as root:
            root = Path(root)
            target, draft = root / "target", root / "draft"
            target.mkdir()
            draft.mkdir()
            config = (archive / "metadata/config.json").read_bytes()
            (target / "config.json").write_bytes(config)
            reference = root / "reference.json"
            reference.write_bytes((archive / "rank0.record.json").read_bytes())
            receipt = {
                "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
                "revision": "2938092373e5f97b95538884112085364c2da315",
                "config_sha256": hashlib.sha256(config).hexdigest(),
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            }
            (target / "target-receipt.json").write_text(json.dumps(receipt))
            (draft / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2",
                        "hidden_size": 896,
                        "intermediate_size": 4864,
                        "num_hidden_layers": 24,
                        "vocab_size": 151936,
                    }
                )
            )
            args = [
                "--draft-model",
                str(draft),
                "--target-tokenizer",
                str(target),
                "--reference-json",
                str(reference),
                "--output",
                str(root / "out.json"),
            ]
            prepared = screen.prepare(args)
            self.assertEqual(prepared["tokens"][-1], 151645)
            self.assertEqual(prepared["args"].block_sizes, [1, 2, 4, 8])
            self.assertEqual(prepared["prompt"], "What is 2 plus 2? Explain briefly.")
            for key, value in [
                ("revision", "wrong"),
                ("config_sha256", "0" * 64),
                ("reference_sha256", "0" * 64),
            ]:
                broken = dict(receipt, **{key: value})
                (target / "target-receipt.json").write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    screen.prepare(args)
            (target / "target-receipt.json").write_text(json.dumps(receipt))
            (target / "config.json").write_bytes(config + b" ")
            with self.assertRaises(ValueError):
                screen.prepare(args)
            (target / "config.json").write_bytes(config)
            (root / "out.json").write_text("already exists")
            with self.assertRaises(ValueError):
                screen.prepare(args)
            (root / "out.json").unlink()
            for tail in (
                ["--block-sizes", "0"],
                ["--block-sizes", "17"],
                ["--block-sizes", "1", "1"],
                ["--max-seconds", "0"],
            ):
                with self.assertRaises(ValueError):
                    screen.prepare(args + tail)
            (draft / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2",
                        "hidden_size": 5120,
                        "intermediate_size": 27648,
                        "num_hidden_layers": 64,
                        "vocab_size": 152064,
                    }
                )
            )
            with self.assertRaises(ValueError):
                screen.prepare(args)

    def test_archived_template_and_exact_id_mapping_not_repaired(self):
        self.assertTrue(
            callable(getattr(screen, "encode_prompt", None)), "missing authoritative prompt path"
        )

        class Tokenizer:
            bos_token = "<B>"

            def apply_chat_template(self, messages, **options):
                self.messages, self.options = messages, options
                return "<B>raw"

            def encode(self, text, *, add_special_tokens):
                self.encoded = (text, add_special_tokens)
                return [1, 2]

        tokenizer = Tokenizer()
        record = {"formatted_prompt": "<B>raw", "requests": [{"prompt_tokens": 2}]}
        self.assertEqual(screen.encode_prompt(record, "raw", tokenizer), ("<B>raw", [1, 2]))
        self.assertEqual(tokenizer.encoded, ("<B>raw", False))
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "raw"}])
        with self.assertRaises(ValueError):
            screen.encode_prompt(dict(record, formatted_prompt="different"), "raw", tokenizer)
        with self.assertRaises(ValueError):
            screen.encode_prompt(
                {"formatted_prompt": "<B>raw", "requests": [{"prompt_tokens": 3}]}, "raw", tokenizer
            )
        target, draft = (
            {0: "zero", 1: "one", 2: "two", 3: "EOS"},
            {0: "zero", 1: "one", 2: "two", 3: "EOS"},
        )
        screen.compatible([1, 3], target, draft, 5, 4)
        with self.assertRaises(ValueError):
            screen.compatible([1], target, dict(draft, **{}) | {1: "different"}, 5, 4)
        with self.assertRaises(ValueError):
            screen.compatible([4], target, draft, 5, 4)
        with self.assertRaises(ValueError):
            screen.compatible([True], target, draft, 5, 4)

    def test_rounds_teacher_force_only_archived_emitted_prefixes(self):
        self.assertTrue(callable(getattr(screen, "screen_trace", None)), "missing trace rounds")
        calls = []

        def propose(prefix, count):
            calls.append((list(prefix), count))
            return ([1, 9] if len(prefix) == 1 else [3])[:count], 0.125

        result = screen.screen_trace([7], [1, 2, 3], 2, {3}, propose)
        self.assertEqual(calls, [([7], 2), ([7, 1, 2], 1)])
        self.assertEqual(result["round_count"], 2)
        self.assertEqual(result["accepted_tokens"], 2)
        self.assertEqual(result["proposed_tokens"], 3)
        self.assertEqual(result["emitted_tokens"], 3)
        self.assertEqual(result["rounds"][0]["proposal"], [1, 9])
        self.assertEqual(result["rounds"][0]["reference_window"], [1, 2, 3])
        self.assertEqual(result["rounds"][0]["accepted_prefix"], [1])
        self.assertEqual(result["draft_seconds"], 0.25)
        self.assertEqual(result["rounds"][1]["emitted"], 1)
        with self.assertRaises(ValueError):
            screen.screen_trace([7], [3], 2, {3}, lambda p, k: ([], 0.1))

    def test_greedy_adapter_synchronizes_fresh_caches_and_stops_at_eos(self):
        self.assertTrue(callable(getattr(screen, "greedy", None)), "missing real runtime adapter")
        import time

        events, caches = [], []

        class Selected:
            def __init__(self, token):
                self.token, self.ready = token, False

            def item(self):
                self.assert_ready()
                return self.token

            def assert_ready(self):
                if not self.ready:
                    raise AssertionError("unsynchronized sample")

        class Scores:
            def __init__(self, token):
                self.token = token

            def __getitem__(self, index):
                return self.token

        class MX:
            int32 = "int32"

            def synchronize(self):
                events.append("sync")

            def array(self, value, dtype):
                return value

            def argmax(self, token, axis):
                return Selected(token)

            def eval(self, selected):
                selected.ready = True
                events.append("eval")

        def cache(model):
            fresh = []
            caches.append(fresh)
            return fresh

        def model(ids, cache):
            cache.append(ids)
            return Scores(2 if len(cache) == 1 else 3)

        for _ in range(2):
            proposal, elapsed = screen.greedy(
                MX(), model, cache, [7, 1], 8, {3}, time.monotonic() + 2
            )
            self.assertEqual(proposal, [2, 3])
            self.assertGreaterEqual(elapsed, 0)
        self.assertIsNot(caches[0], caches[1])
        self.assertEqual(caches[0], [[[7, 1]], [[2]]])
        self.assertEqual(events, ["sync", "eval", "eval", "sync"] * 2)
        with self.assertRaises(TimeoutError):
            screen.greedy(MX(), model, cache, [7], 1, {3}, time.monotonic() - 1)

    def test_inventory_hashes_regular_local_files_and_refuses_links_or_oversize(self):
        self.assertTrue(callable(getattr(screen, "inventory", None)), "missing source binding")
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory(dir=Path.home() / ".hermes/cache/scratch") as root:
            root = Path(root)
            (root / "model.safetensors").write_bytes(b"synthetic CPU fixture")
            found = screen.inventory(root, limit=128)
            self.assertEqual(
                found["model.safetensors"]["sha256"],
                hashlib.sha256(b"synthetic CPU fixture").hexdigest(),
            )
            self.assertEqual(found["model.safetensors"]["bytes"], 21)
            with self.assertRaises(ValueError):
                screen.inventory(root, limit=4)
            (root / "link.json").symlink_to(root / "model.safetensors")
            with self.assertRaises(ValueError):
                screen.inventory(root, limit=128)

    def test_runtime_loads_only_draft_and_reports_auditable_rounds(self):
        self.assertTrue(callable(getattr(screen, "run", None)), "missing integrated runtime")
        import sys
        import tempfile
        from unittest.mock import patch

        events = []

        class Tokenizer:
            bos_token = None
            eos_token_ids = {3}

            def apply_chat_template(self, messages, **kwargs):
                return "rendered"

            def encode(self, text, **kwargs):
                return [0]

            def get_vocab(self):
                return {"prompt": 0, "one": 1, "two": 2, "EOS": 3}

        class Parameter:
            shape, dtype, nbytes = (2,), "fixture", 8

        class Model:
            def parameters(self):
                return {"weight": Parameter()}

        def load(path, **kwargs):
            events.append(("draft_load", path, kwargs))
            return Model(), Tokenizer()

        def load_tokenizer(path, **kwargs):
            events.append(("target_tokenizer", path, kwargs))
            return Tokenizer()

        mlx, lm = types.ModuleType("mlx"), types.ModuleType("mlx_lm")
        core = types.ModuleType("mlx.core")
        core.metal = types.SimpleNamespace(
            device_info=lambda: {"device_name": "CPU fixture NOT evidence"}
        )
        for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
            setattr(core, name, lambda: 8)
        core.default_device = lambda: "CPU fixture"
        core.synchronize = lambda: None
        utils = types.ModuleType("mlx.utils")
        utils.tree_flatten = lambda p: list(p.items())
        lm.load = load
        lm_utils = types.ModuleType("mlx_lm.utils")
        lm_utils.load_tokenizer = load_tokenizer
        cache = types.ModuleType("mlx_lm.models.cache")
        cache.make_prompt_cache = lambda model: []
        modules = {
            "mlx": mlx,
            "mlx.core": core,
            "mlx.utils": utils,
            "mlx_lm": lm,
            "mlx_lm.utils": lm_utils,
            "mlx_lm.models": types.ModuleType("mlx_lm.models"),
            "mlx_lm.models.cache": cache,
        }
        with tempfile.TemporaryDirectory(dir=Path.home() / ".hermes/cache/scratch") as root:
            root = Path(root)
            draft, target = root / "draft", root / "target"
            draft.mkdir()
            target.mkdir()
            for directory in (draft, target):
                (directory / "config.json").write_text("{}")
            (draft / "model.safetensors").write_bytes(b"CPU fixture")
            args = types.SimpleNamespace(
                draft_model=draft, target_tokenizer=target, block_sizes=[1, 2], max_seconds=2
            )
            prepared = dict(
                args=args,
                config={"vocab_size": 5},
                draft_config={"vocab_size": 4},
                record={"formatted_prompt": "rendered", "requests": [{"prompt_tokens": 1}]},
                prompt="raw",
                tokens=[1, 2, 3],
                eos={3},
                reference_sha256="fixture",
                receipt_sha256="fixture",
            )

            def fake_greedy(mx, model, maker, prefix, count, eos, deadline):
                return [1, 2, 3][len(prefix) - 1 : len(prefix) - 1 + count], 0.01

            with (
                patch.dict(sys.modules, modules),
                patch("importlib.metadata.version", return_value="CPU fixture"),
                patch.object(screen, "greedy", side_effect=fake_greedy),
            ):
                report = screen.run(prepared)
            self.assertEqual(len([e for e in events if e[0] == "draft_load"]), 1)
            self.assertEqual(events[0][0], "target_tokenizer")
            loaded = [e for e in events if e[0] == "draft_load"][0]
            self.assertEqual(Path(loaded[1]), draft)
            self.assertFalse(loaded[2]["trust_remote_code"])
            self.assertTrue(loaded[2]["tokenizer_config"]["local_files_only"])
            self.assertFalse(report["actual_target_verification"])
            self.assertEqual(report["diagnostic"], "archived-target-trace acceptance")
            self.assertEqual(len(report["screens"]), 2)
            self.assertEqual(report["draft_loaded_parameter_bytes"], 8)
            self.assertIn("draft_screen.py", report["source_hashes"])
            self.assertIn("model.safetensors", report["draft_source_inventory"])
            self.assertTrue(report["prefix_recomputed_each_round"])

    def test_cli_help_import_and_invalid_input_never_import_model_libraries(self):
        import subprocess
        import sys

        guard = """
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in {'mlx', 'mlx_lm', 'numpy'}:
   raise AssertionError('forbidden heavy import: '+name)
sys.meta_path.insert(0, Guard())
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        help_result = subprocess.run(
            [sys.executable, "-B", "-c", guard, str(PATH), "--help"], capture_output=True, text=True
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--draft-model", help_result.stdout)
        bad = subprocess.run(
            [sys.executable, "-B", "-c", guard, str(PATH)], capture_output=True, text=True
        )
        self.assertNotEqual(bad.returncode, 0)
        self.assertNotIn("forbidden heavy import", bad.stderr)
        imported = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                guard.replace("run_name='__main__'", "run_name='cpu_import'"),
                str(PATH),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)

    def test_main_writes_new_exclusive_json_never_overwrites(self):
        self.assertTrue(callable(getattr(screen, "main", None)), "missing exclusive CLI writer")
        import json
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(dir=Path.home() / ".hermes/cache/scratch") as root:
            output = Path(root) / "out.json"
            prepared = {"args": types.SimpleNamespace(output=output)}
            with (
                patch.object(screen, "prepare", return_value=prepared),
                patch.object(
                    screen, "run", return_value={"diagnostic": "CPU fixture NOT evidence"}
                ),
            ):
                screen.main([])
                self.assertEqual(
                    json.loads(output.read_text()), {"diagnostic": "CPU fixture NOT evidence"}
                )
                with self.assertRaises(FileExistsError):
                    screen.main([])
                self.assertEqual(
                    json.loads(output.read_text()), {"diagnostic": "CPU fixture NOT evidence"}
                )


if __name__ == "__main__":
    unittest.main()
