"""CPU-only integration tests for owned process supervision."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import remote_actor as actor


class OwnedTests(unittest.TestCase):
    def test_deadline_reaps_term_ignoring_child_preserving_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            began = time.monotonic()
            result = actor.run_owned(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    (
                        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        "print('partial',flush=True); time.sleep(30)"
                    ),
                ],
                job,
                deadline=0.3,
            )
            self.assertEqual(result["reason"], "deadline")
            self.assertTrue(result["cleanup"]["child_reaped"])
            self.assertTrue(result["cleanup"]["group_gone"])
            self.assertLess(time.monotonic() - began, 3)
            self.assertEqual((job / "stdout").read_text(), "partial\n")
            with self.assertRaises(ProcessLookupError):
                os.kill(result["child_pid"], 0)


class AdmissionTests(unittest.TestCase):
    def test_source_inspection_is_stdlib_only(self):
        root = Path(actor.__file__).parent
        hashes = actor.source_hashes(root)
        self.assertEqual(set(hashes), set(actor.SOURCES))
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_cli_admits_only_explicit_scripts_and_env(self):
        root = Path(actor.__file__).parent
        with self.assertRaises(ValueError):
            actor.validate_job({"entrypoint": "other.py"}, root)
        with self.assertRaises(ValueError):
            actor.validate_job(
                {"entrypoint": "probe.py", "argv": [], "env": {"PYTHONPATH": "/tmp"}}, root
            )
        job = actor.validate_job(
            {
                "entrypoint": "probe.py",
                "argv": ["--mode", "generate"],
                "env": {},
                "nonce": "a" * 32,
                "deadline": 1,
            },
            root,
        )
        self.assertEqual(job["entrypoint"], "probe.py")

    def test_redaction_and_exclusive_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            token = "private-session-value"
            result = actor.run_owned(
                [sys.executable, "-c", 'import os; print(os.environ["SECRET"])'],
                job,
                deadline=2,
                env={**os.environ, "SECRET": token},
                secrets=(token,),
            )
            self.assertEqual(result["reason"], "completed")
            self.assertNotIn(token, (job / "stdout").read_text())
            self.assertEqual(job.stat().st_mode & 0o777, 0o700)
            self.assertEqual((job / "stdout").stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                actor.run_owned([sys.executable, "-c", "pass"], job, deadline=1)

    def test_cli_inspection_has_no_mlx_import(self):
        proc = subprocess.run(
            [
                sys.executable,
                actor.__file__,
                "--root",
                str(Path(actor.__file__).parent),
                "--inspect",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("source_sha256", json.loads(proc.stdout))

    def test_signal_and_disconnection_leave_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            for name in actor.SOURCES:
                (root / name).write_text("")
            (root / "probe.py").write_text(
                'import time\nprint("started", flush=True)\ntime.sleep(30)\n'
            )
            for disconnect in (False, True):
                nonce = ("b" if disconnect else "c") * 32
                job = {
                    "entrypoint": "probe.py",
                    "argv": [],
                    "env": {},
                    "nonce": nonce,
                    "deadline": 0.4,
                }
                proc = subprocess.Popen(
                    [sys.executable, actor.__file__, "--root", str(root), "--jobs", tmp],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                proc.stdin.write(json.dumps(job).encode())
                proc.stdin.close()
                until = time.monotonic() + 2
                while not (Path(tmp) / nonce / "pids.json").exists() and time.monotonic() < until:
                    time.sleep(0.01)
                if disconnect:
                    proc.stdout.close()
                    proc.stderr.close()
                else:
                    proc.send_signal(signal.SIGHUP)
                proc.wait(timeout=4)
                receipt = json.loads((Path(tmp) / nonce / "result.json").read_text())
                self.assertEqual(receipt["reason"], "deadline" if disconnect else "signal")
                self.assertTrue(receipt["cleanup"]["group_gone"])
                if not disconnect:
                    proc.stdout.close()
                    proc.stderr.close()

    def test_failure_receipt_survives_callback_exception(self):
        with tempfile.TemporaryDirectory() as tmp:

            def fail(child):
                raise subprocess.TimeoutExpired("private-secret", 1)

            result = actor.run_owned(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                Path(tmp) / "job",
                deadline=1,
                tick=fail,
            )
            self.assertEqual(result["error_type"], "TimeoutExpired")
            self.assertTrue(result["cleanup"]["group_gone"])
            self.assertNotIn("private-secret", (Path(tmp) / "job" / "result.json").read_text())

    def test_nonzero_child_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = actor.run_owned(
                [sys.executable, "-c", "raise SystemExit(9)"], Path(tmp) / "job", deadline=1
            )
            self.assertEqual(result["reason"], "child_failure")
            self.assertEqual(result["returncode"], 9)

    def test_child_sees_durable_pid_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            code = (
                "import json,os; from pathlib import Path; "
                'p=Path(os.environ["JOB"])/"pids.json"; assert p.exists(); '
                'assert json.loads(p.read_text())["child_pid"]==os.getpid()'
            )
            for index in range(4):
                destination = job.with_name(f"job{index}")
                result = actor.run_owned(
                    [sys.executable, "-c", code],
                    destination,
                    deadline=2,
                    env={**os.environ, "JOB": str(destination)},
                )
                self.assertEqual(result["reason"], "completed")

    def test_nonce_status_and_stop_are_file_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            nonce = "d" * 32
            job = Path(tmp) / nonce
            job.mkdir(mode=0o700)
            actor.write_json(job / "pids.json", {"nonce": nonce, "child_pid": 123})
            (job / "stdout").write_text('{"ready":true,"address":["127.0.0.1",1234]}\n')
            status = actor.job_status(Path(tmp), nonce)
            self.assertEqual(status["ready"]["address"][1], 1234)
            actor.request_stop(Path(tmp), nonce)
            self.assertTrue((job / "stop").exists())
            with self.assertRaises(ValueError):
                actor.job_status(Path(tmp), "../other")

    def test_owned_hostfile_is_written_before_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = actor.run_owned(
                [
                    sys.executable,
                    "-c",
                    'import os,json; assert len(json.load(open(os.environ["MLX_HOSTFILE"])))==2',
                ],
                Path(tmp) / "job",
                deadline=2,
                hostfile=[["127.0.0.1:1234"], ["127.0.0.1:1235"]],
            )
            self.assertEqual(result["reason"], "completed")

    def test_source_snapshot_is_executed_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "src"
            root.mkdir()
            for name in actor.SOURCES:
                (root / name).write_text(
                    'print("snapshot", flush=True)\n' if name == "probe.py" else ""
                )
            hashes = actor.source_hashes(root)
            result = actor.run_owned(
                [sys.executable, "-u", str(root / "probe.py")],
                Path(tmp) / "job",
                deadline=2,
                source_root=root,
                expected_hashes=hashes,
            )
            self.assertEqual(result["reason"], "completed")
            self.assertEqual(
                (Path(tmp) / "job" / "sources" / "probe.py").read_bytes(),
                (root / "probe.py").read_bytes(),
            )
            hashes["probe.py"] = "0" * 64
            result = actor.run_owned(
                [sys.executable, "-u", str(root / "probe.py")],
                Path(tmp) / "bad",
                deadline=2,
                source_root=root,
                expected_hashes=hashes,
            )
            self.assertEqual(result["reason"], "exception")
            self.assertIsNone(result["child_pid"])

    def test_inspection_bounds_source_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in actor.SOURCES:
                (Path(tmp) / name).write_text("")
            with (Path(tmp) / "probe.py").open("wb") as handle:
                handle.truncate(3 * 1024**2)
            with self.assertRaises(ValueError):
                actor.source_hashes(tmp)

    def test_unread_output_pipes_cannot_disable_cli_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            for name in actor.SOURCES:
                (root / name).write_text("")
            (root / "probe.py").write_text(
                "import os,time\nfor i in range(100):\n"
                ' os.write(1,b"x"*65536)\n os.write(2,b"y"*65536)\ntime.sleep(30)\n'
            )
            nonce = "a" * 32
            proc = subprocess.Popen(
                [sys.executable, actor.__file__, "--root", str(root), "--jobs", tmp],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            proc.stdin.write(
                json.dumps(
                    {
                        "entrypoint": "probe.py",
                        "argv": [],
                        "env": {},
                        "nonce": nonce,
                        "deadline": 0.3,
                    }
                ).encode()
            )
            proc.stdin.close()
            proc.wait(timeout=4)
            proc.stdout.close()
            proc.stderr.close()
            result = json.loads((Path(tmp) / nonce / "result.json").read_text())
            self.assertEqual(result["reason"], "deadline")
            self.assertTrue(result["cleanup"]["group_gone"])
            self.assertGreater((Path(tmp) / nonce / "stdout").stat().st_size, 65536)

    def test_invalid_job_persists_redacted_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    actor.__file__,
                    "--root",
                    str(Path(actor.__file__).parent),
                    "--jobs",
                    tmp,
                ],
                input='{"entrypoint":"arbitrary.py","secret":"private-value"}',
                capture_output=True,
                text=True,
                timeout=3,
            )
            self.assertEqual(proc.returncode, 1)
            receipts = list(Path(tmp).glob("*/result.json"))
            self.assertEqual(len(receipts), 1)
            result = json.loads(receipts[0].read_text())
            self.assertEqual(result["reason"], "admission_failure")
            self.assertNotIn("private-value", receipts[0].read_text() + proc.stdout + proc.stderr)

    def test_job_input_wait_is_bounded(self):
        reader, writer = os.pipe()
        try:
            with self.assertRaises(TimeoutError):
                actor.read_job(reader, timeout=0.05)
            os.write(writer, b'{"entrypoint":"probe.py"}\n')
            self.assertEqual(actor.read_job(reader, timeout=0.1)["entrypoint"], "probe.py")
        finally:
            os.close(reader)
            os.close(writer)

    def test_controller_death_does_not_disable_actor_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            for name in actor.SOURCES:
                (root / name).write_text("")
            (root / "probe.py").write_text(
                'import time\nprint("owned",flush=True)\ntime.sleep(30)\n'
            )
            nonce = "9" * 32
            code = (
                "import json,os,signal,subprocess,sys,time; from pathlib import Path; "
                'p=subprocess.Popen([sys.executable,sys.argv[1],"--root",sys.argv[2],"--jobs",sys.argv[3]],'
                "stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,"
                "stderr=subprocess.DEVNULL,start_new_session=True); "
                'p.stdin.write(json.dumps({"entrypoint":"probe.py","argv":[],"env":{},'
                '"nonce":"9"*32,"deadline":.3}).encode()); '
                "p.stdin.close(); time.sleep(.1); os.kill(os.getpid(),signal.SIGKILL)"
            )
            parent = subprocess.run(
                [sys.executable, "-c", code, actor.__file__, str(root), tmp], timeout=3
            )
            self.assertEqual(parent.returncode, -signal.SIGKILL)
            receipt = Path(tmp) / nonce / "result.json"
            until = time.monotonic() + 3
            while not receipt.exists() and time.monotonic() < until:
                time.sleep(0.02)
            result = json.loads(receipt.read_text())
            self.assertEqual(result["reason"], "deadline")
            self.assertTrue(result["cleanup"]["child_reaped"])
            self.assertTrue(result["cleanup"]["group_gone"])

    def test_reaped_replacement_leader_absent_never_signals_surviving_group(self):
        from unittest.mock import patch

        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait(timeout=2)
        signals = []

        def unrelated_group(pid, sig):
            self.assertEqual(pid, child.pid)
            if sig:
                signals.append(sig)

        with (
            patch.object(actor.os, "killpg", side_effect=unrelated_group),
            patch.object(actor.os, "kill", side_effect=ProcessLookupError),
            patch.object(actor.time, "sleep"),
            patch.object(actor.time, "monotonic", side_effect=range(20)),
        ):
            result = actor.terminate(child)
        self.assertEqual(signals, [])
        self.assertTrue(result["child_reaped"])
        self.assertFalse(result["group_gone"])
        self.assertFalse(result["pid_reused"])
        self.assertTrue(result["identity_unknown"])
        self.assertTrue(result["pending_os_gpu_termination"])
        self.assertTrue(result["gpu_driver_state"])

    def test_owned_leader_reaped_between_group_signals_never_signals_again(self):
        from unittest.mock import patch

        reader, writer = os.pipe()
        child = subprocess.Popen(
            [sys.executable, "-c", "import os,sys; os.read(int(sys.argv[1]),1)", str(reader)],
            pass_fds=(reader,),
            start_new_session=True,
        )
        os.close(reader)
        poll = child.poll
        signals, signals_after_reap = [], []

        def group_signal(pid, sig):
            self.assertEqual(pid, child.pid)
            if sig:
                signals.append(sig)
                if child.returncode is not None:
                    signals_after_reap.append(sig)
                if sig == signal.SIGTERM:
                    os.write(writer, b"1")  # leader exits after the ownership check

        def owner_poll():
            if not signals:
                return poll()
            until = time.monotonic() + 2
            while poll() is None and time.monotonic() < until:
                time.sleep(0.01)
            self.assertIsNotNone(child.returncode)
            return child.returncode

        try:
            with (
                patch.object(child, "poll", side_effect=owner_poll),
                patch.object(actor.os, "killpg", side_effect=group_signal),
                patch.object(actor.os, "kill", side_effect=ProcessLookupError),
            ):
                result = actor.terminate(child)
            self.assertEqual(signals_after_reap, [])
            self.assertEqual(signals, [signal.SIGTERM])
            self.assertTrue(result["child_reaped"])
            self.assertFalse(result["group_gone"])
            self.assertTrue(result["identity_unknown"])
            self.assertTrue(result["pending_os_gpu_termination"])
        finally:
            os.close(writer)
            if poll() is None:
                child.kill()
            child.wait(timeout=2)

    def test_reaped_pid_reuse_never_signals_replacement(self):
        from unittest.mock import patch

        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait(timeout=2)
        child.pid = os.getpid()  # emulate a replacement identity after reap

        def forbidden_signal(pid, sig):
            if sig:
                raise AssertionError("Reaped numeric PID is no longer an owned leader")

        with patch.object(actor.os, "killpg", side_effect=forbidden_signal):
            result = actor.terminate(child)
        self.assertTrue(result["child_reaped"])
        self.assertFalse(result["group_gone"])
        self.assertTrue(result["pid_reused"])
        self.assertTrue(result["identity_unknown"])
        self.assertTrue(result["pending_os_gpu_termination"])


if __name__ == "__main__":
    unittest.main()
