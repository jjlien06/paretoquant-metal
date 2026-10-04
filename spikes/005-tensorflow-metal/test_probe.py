"""End-to-end acceptance checks for the isolated real-framework probe."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("probe.py")


class ProbePublication(unittest.TestCase):
    def test_report_created_after_preflight_is_preserved(self):
        import builtins
        from contextlib import redirect_stderr, redirect_stdout
        import io
        import runpy
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            output = Path(tmp) / "runtime.json"
            competing = b'{"completed_run": "competing evidence"}\n'
            original_import = builtins.__import__

            def block_tensorflow(name, *args, **kwargs):
                if name == "tensorflow":
                    self.assertFalse(output.exists(), "competing run must publish after preflight")
                    output.write_bytes(competing)
                    raise ImportError("intentional CPU-only publication test")
                return original_import(name, *args, **kwargs)

            stderr = io.StringIO()
            argv = [str(SCRIPT), "--runtime-only", "--output", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch("importlib.metadata.version", return_value="unit-test-fixture"),
                patch("builtins.__import__", side_effect=block_tensorflow),
                redirect_stdout(io.StringIO()),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as error,
            ):
                runpy.run_path(str(SCRIPT), run_name="__main__")
            self.assertEqual(output.read_bytes(), competing)
            self.assertEqual(error.exception.code, 2)
            self.assertIn("File exists", stderr.getvalue())


class ProbeAcceptance(unittest.TestCase):
    def test_runtime_has_real_strict_gpu_execution(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            output = Path(tmp) / "runtime.json"
            run = subprocess.run([sys.executable, str(SCRIPT), "--runtime-only", "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "runtime_ready")
            self.assertIn("GPU:0", result["runtime"]["tensor_device"])
            self.assertFalse(result["runtime"]["soft_device_placement"])
            self.assertEqual(result["runtime"]["values"], [[11.0, 0.0], [25.0, 0.0]])

    def test_invalid_token_count_is_blocked_not_runtime_ready(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            output = Path(tmp) / "invalid.json"
            run = subprocess.run([sys.executable, str(SCRIPT), "--new-tokens", "0", "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["error"], "--new-tokens must be positive")
            self.assertNotIn("loaded_weights", result)

    @unittest.skipUnless(os.environ.get("RUN_QWEN_GENERATION") == "1", "Explicitly opt into model allocation")
    def test_native_local_checkpoint_generates_exactly_eight_tokens(self):
        output = Path(os.environ["QWEN_PROBE_OUTPUT"])
        command = [sys.executable, str(SCRIPT), "--output", str(output)]
        if os.environ.get("QWEN_CPU_CACHE") == "1":
            command.append("--cpu-cache-updates")
        run = subprocess.run(command, capture_output=True, text=True)
        output.with_suffix(".log").write_text(run.stdout + run.stderr)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result = json.loads(output.read_text())
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(result["generation"]["new_token_ids"]), 8)
        self.assertEqual(result["generation"]["new_token_count"], 8)
        self.assertTrue(result["generation"]["completion"].strip())
        self.assertTrue(result["generation"]["finite_forward_logits"])
        self.assertIn("GPU:0", result["generation"]["forward_logits_device"])
        self.assertIn("GPU:0", result["generation"]["embedding_device"])
        self.assertEqual(result["generation"]["parameter_count"], 494032768)
        self.assertTrue(result["generation"]["all_weights_verified"])


if __name__ == "__main__":
    unittest.main()
