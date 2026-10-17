"""CPU-only strict action-audit tests; toy tokens are explicit, not GPU evidence."""

import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("mtp_audit.py")


def load_audit():
    if not MODULE.exists():
        raise AssertionError("strict generation auditor is not implemented")
    spec = importlib.util.spec_from_file_location("mtp_audit", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    request = dict(
        id="toy-mtp", repeat=1, method="mtp", max_tokens=5,
        token_ids=[32, 6951, 369, 264, 13094], finish_reason="length",
        verification_rounds=2, accepted_drafts=2, teacher_forwards=2,
        verifier_mode="native", local_rank=0,
        wall_seconds=2.0, first_token_seconds=0.5,
        wall_tokens_per_second=2.5, post_first_tokens_per_second=4 / 1.5,
    )
    ops = [("reset", []), ("forward", [248045, 846]), ("forward", [198]),
           ("snapshot", []), ("verify", [32, 6951]),
           ("snapshot", []), ("verify", [369, 264]), ("finish", [])]
    actions = [dict(phase="rank_action", id=request["id"], repeat=1, rank=0,
                    request="a" * 32, sequence=i, op=op, tokens=tokens, ok=True)
               for i, (op, tokens) in enumerate(ops)]
    return request, actions, [248045, 846, 198]


class GenerationAuditTests(unittest.TestCase):
    def test_review_mutation_zero_rounds_five_outputs(self):
        request, actions, prompt = fixture()
        request.update(verification_rounds=0, teacher_forwards=0, accepted_drafts=0)
        with self.assertRaises(ValueError):
            load_audit().validate_mtp_actions(request, actions, prompt, expected_mode="native")

    def test_review_mutation_changed_accepted_counter(self):
        request, actions, prompt = fixture()
        request["accepted_drafts"] = 1
        with self.assertRaises(ValueError):
            load_audit().validate_mtp_actions(request, actions, prompt, expected_mode="native")

    def test_teacher_counter(self):
        request, actions, prompt = fixture()
        request["teacher_forwards"] = 3
        with self.assertRaises(ValueError):
            load_audit().validate_mtp_actions(request, actions, prompt, expected_mode="native")

    def test_hidden_prefill_and_terminal_forward_are_distinct(self):
        request, actions, prompt = fixture()
        actions[1]["op"] = "prefill_hidden"
        audit = load_audit()
        audit.validate_mtp_actions(request, actions, prompt, expected_mode="native",
                                   hidden_only_prefill=True)
        with self.assertRaises(ValueError):
            audit.validate_mtp_actions(request, actions, prompt, expected_mode="native")
        actions[2]["op"] = "prefill_hidden"
        with self.assertRaises(ValueError):
            audit.validate_mtp_actions(request, actions, prompt, expected_mode="native",
                                       hidden_only_prefill=True)

    def test_long_prompt_chunk256(self):
        request, actions, _ = fixture()
        prompt = list(range(514))
        chunks = [prompt[:256], prompt[256:512], prompt[512:513], prompt[513:]]
        actions = [actions[0]] + [dict(actions[1], tokens=c) for c in chunks] + actions[3:]
        for i, event in enumerate(actions):
            event["sequence"] = i
        audit = load_audit()
        audit.validate_mtp_actions(request, actions, prompt, expected_mode="native")
        actions[1]["tokens"].append(actions[2]["tokens"].pop(0))
        with self.assertRaises(ValueError):
            audit.validate_mtp_actions(request, actions, prompt, expected_mode="native")

    def test_rank1_slot_qmv_mode(self):
        request, actions, prompt = fixture()
        request.update(local_rank=1, verifier_mode="slot-qmv")
        for event in actions:
            event["rank"] = 1
        load_audit().validate_mtp_actions(request, actions, prompt, expected_mode="slot-qmv")

    def test_all_accepted_bonus_at_cap(self):
        request, actions, prompt = fixture()
        before = copy.deepcopy((request, actions, prompt))
        result = load_audit().validate_mtp_actions(
            request, actions, prompt, expected_mode="native")
        self.assertEqual(result["verification_rounds"], 2)
        self.assertEqual(result["accepted_drafts"], 2)
        self.assertEqual(result["output_tokens"], 5)
        self.assertEqual((request, actions, prompt), before)


def scenario(tokens, generation, *, cap=None, accepted=0, rounds=0):
    request, actions, prompt = fixture()
    request.update(token_ids=tokens, max_tokens=cap or len(tokens),
                   verification_rounds=rounds, teacher_forwards=rounds,
                   accepted_drafts=accepted,
                   finish_reason="stop" if tokens[-1] in (248044, 248046) else "length",
                   wall_tokens_per_second=len(tokens) / 2,
                   post_first_tokens_per_second=(
                       (len(tokens) - 1) / 1.5 if len(tokens) > 1 else None))
    actions = actions[:3] + [dict(actions[0], op=op, tokens=payload)
                             for op, payload in generation] + [actions[-1]]
    for i, event in enumerate(actions):
        event["sequence"] = i
    return request, actions, prompt


TERMINAL_SCENARIOS = {
    "accepted_cap": ([32, 6951], [("snapshot", []), ("verify", [32, 6951]),
                                   ("commit_prefix", [1])], dict(accepted=1, rounds=1)),
    "accepted_eos": ([32, 248046], [("snapshot", []), ("verify", [32, 248046]),
                                     ("commit_prefix", [1])], dict(cap=5, accepted=1, rounds=1)),
    "rejected_correction": ([32, 369], [("snapshot", []), ("verify", [32, 6951]),
                                        ("commit_prefix", [1])], dict(rounds=1)),
    "bonus_eos": ([32, 6951, 248046], [("snapshot", []), ("verify", [32, 6951])],
                  dict(cap=5, accepted=1, rounds=1)),
    "initial_cap": ([32], [], {}),
    "initial_eos": ([248044], [], dict(cap=5)),
}


def scenario_test(name, remove_commit=False):
    def test(self):
        tokens, generation, kwargs = TERMINAL_SCENARIOS[name]
        request, actions, prompt = scenario(
            copy.deepcopy(tokens), copy.deepcopy(generation), **kwargs)
        if remove_commit:
            actions = [e for e in actions if e["op"] != "commit_prefix"]
            for i, event in enumerate(actions):
                event["sequence"] = i
            with self.assertRaisesRegex(ValueError, "expected commit_prefix"):
                load_audit().validate_mtp_actions(request, actions, prompt, expected_mode="native")
        else:
            result = load_audit().validate_mtp_actions(
                request, actions, prompt, expected_mode="native")
            self.assertEqual(result["output_tokens"], len(tokens))
    return test


for name in TERMINAL_SCENARIOS:
    setattr(GenerationAuditTests, "test_valid_" + name, scenario_test(name))
for name in ("accepted_cap", "accepted_eos", "rejected_correction"):
    setattr(GenerationAuditTests, "test_reject_missing_commit_" + name, scenario_test(name, True))


# One named unittest per adversarial case, so the reported count is exact.
NEGATIVE_CASES = {
    "wrong_mode": lambda r, a, p: r.update(verifier_mode="slot-qmv"),
    "missing_mode": lambda r, a, p: r.pop("verifier_mode"),
    "wrong_method": lambda r, a, p: r.update(method="native"),
    "wrong_id": lambda r, a, p: a[0].update(id="different"),
    "missing_id": lambda r, a, p: a[0].pop("id"),
    "wrong_repeat": lambda r, a, p: a[0].update(repeat=2),
    "bool_repeat": lambda r, a, p: a[0].update(repeat=True),
    "wrong_rank": lambda r, a, p: a[0].update(rank=1),
    "wrong_phase": lambda r, a, p: a[0].update(phase="token_done"),
    "failed_ack": lambda r, a, p: a[0].update(ok=False),
    "missing_ack": lambda r, a, p: a[0].pop("ok"),
    "changed_nonce": lambda r, a, p: a[1].update(request="b" * 32),
    "missing_nonce": lambda r, a, p: a[0].pop("request"),
    "malformed_nonce": lambda r, a, p: a[0].update(request="x"),
    "sequence_gap": lambda r, a, p: a[1].update(sequence=2),
    "bool_sequence": lambda r, a, p: a[0].update(sequence=False),
    "bool_token": lambda r, a, p: r["token_ids"].__setitem__(0, True),
    "negative_draft": lambda r, a, p: a[4]["tokens"].__setitem__(1, -1),
    "float_payload": lambda r, a, p: a[1]["tokens"].__setitem__(0, 248045.0),
    "missing_tokens": lambda r, a, p: a[4].pop("tokens"),
    "empty_output": lambda r, a, p: r.update(token_ids=[]),
    "empty_prompt": lambda r, a, p: p.clear(),
    "too_many_outputs": lambda r, a, p: r.update(max_tokens=4),
    "early_eos": lambda r, a, p: r["token_ids"].__setitem__(2, 248046),
    "length_before_cap": lambda r, a, p: r.update(max_tokens=6),
    "stop_without_eos": lambda r, a, p: r.update(finish_reason="stop"),
    "unknown_finish": lambda r, a, p: r.update(finish_reason="abort"),
    "zero_wall": lambda r, a, p: r.update(wall_seconds=0),
    "nan_wall": lambda r, a, p: r.update(wall_seconds=float("nan")),
    "derived_infinite_tps": lambda r, a, p: r.update(
        wall_seconds=1e-320, first_token_seconds=1e-321,
        wall_tokens_per_second=float("inf"), post_first_tokens_per_second=float("inf")),
    "first_equal_wall": lambda r, a, p: r.update(first_token_seconds=2),
    "wrong_wall_tps": lambda r, a, p: r.update(wall_tokens_per_second=3),
    "wrong_post_first_tps": lambda r, a, p: r.update(post_first_tokens_per_second=3),
    "bool_counter": lambda r, a, p: r.update(accepted_drafts=True),
    "missing_reset": lambda r, a, p: a.pop(0),
    "extra_reset": lambda r, a, p: a.insert(3, copy.deepcopy(a[0])),
    "wrong_prompt_forward": lambda r, a, p: a[1]["tokens"].reverse(),
    "late_prompt_forward": lambda r, a, p: a.insert(5, copy.deepcopy(a[1])),
    "missing_snapshot": lambda r, a, p: a.pop(3),
    "review_mutation_changed_verify_current": lambda r, a, p: a[6]["tokens"].__setitem__(0, 32),
    "wrong_verify_width": lambda r, a, p: a[4]["tokens"].append(369),
    "missing_finish": lambda r, a, p: a.pop(),
    "early_finish": lambda r, a, p: a.insert(3, copy.deepcopy(a[-1])),
    "after_finish": lambda r, a, p: a.append(copy.deepcopy(a[-1])),
    "restore_not_prefix_mode": lambda r, a, p: a[3].update(op="restore"),
    "unexpected_commit": lambda r, a, p: a[5].update(op="commit_prefix", tokens=[1]),
    "no_generation_five_outputs": lambda r, a, p: a.__setitem__(slice(3, -1), []),
}


def negative_test(mutate):
    def test(self):
        request, actions, prompt = fixture()
        mutate(request, actions, prompt)
        with self.assertRaises(ValueError):
            load_audit().validate_mtp_actions(request, actions, prompt, expected_mode="native")
    return test


for name, mutate in NEGATIVE_CASES.items():
    setattr(GenerationAuditTests, "test_reject_" + name, negative_test(mutate))


def toy_trial(path):
    """Explicitly synthetic CPU schema fixture; never used as physical evidence."""
    path.mkdir()
    sources = path / "sources"
    sources.mkdir()
    request, actions, prompt = fixture()
    row = dict(id=request["id"], case_id="toy", method="mtp", prompt="toy",
               max_tokens=5, warmup=False, repeat=0)
    schedule_bytes = json.dumps(dict(schema="qwen38-workload-schedule-v1", requests=[row])).encode()
    (sources / "benchmark_schedule.json").write_bytes(schedule_bytes)
    registry = {"benchmark_schedule.json": hashlib.sha256(schedule_bytes).hexdigest()}
    (path / "source_sha256.json").write_text(json.dumps(registry))
    (path / "result.json").write_text(json.dumps(dict(status="success", verified_request_count=1)))
    for rank, mode in enumerate(("native", "slot-qmv")):
        r = dict(request, repeat=0, local_rank=rank, verifier_mode=mode,
                 case_id="toy", warmup=False, sample_repeat=0, eval_group_size=32,
                 prompt_token_ids=prompt, prompt_tokens=len(prompt), text="toy",
                 prompt_sha256=hashlib.sha256(b"toy").hexdigest(),
                 local_rank_wall_seconds=2.1, local_peak_mlx_bytes=1024)
        binding = {k: r[k] for k in ("id", "case_id", "method", "warmup", "max_tokens",
                                     "repeat", "sample_repeat", "eval_group_size", "prompt_sha256")}
        active = [dict(e, **binding, rank=rank) for e in actions]
        events = [dict(phase="process_start", rank=rank, source_sha256=registry),
                  dict(binding, phase="request_before", rank=rank, prompt_token_ids=prompt),
                  *active, dict(binding, phase="request_done", rank=rank, request_result=r)]
        stdout = json.dumps(dict(rank=rank, verifier_mode=mode, prefix_rollback=True,
                                 benchmark_schedule_sha256=registry["benchmark_schedule.json"],
                                 requests=[r]))
        stderr = "".join(json.dumps(e) + "\n" for e in events)
        nonce = str(rank) * 32
        rc = dict(nonce=nonce, source_sha256=registry, stdout=stdout, stderr=stderr,
                  result=dict(reason="completed", returncode=0, log_evidence_complete=True,
                              log_cap_exceeded=False, nonce=nonce))
        (path / f"rank{rank}.receipt.json").write_text(json.dumps(rc))


class TrialAuditTests(unittest.TestCase):
    def test_cli_exports_exclusively_and_preserves_raw(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            path, output = Path(tmp) / "trial", Path(tmp) / "audit.json"
            toy_trial(path)
            args = [sys.executable, str(MODULE), str(path), "--output", str(output)]
            first = subprocess.run(args, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue(output.exists(), "CLI must create a separate audit receipt")
            self.assertEqual(json.loads(first.stdout), json.loads(output.read_text()))
            before = output.read_bytes()
            second = subprocess.run(args, capture_output=True, text=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), before)

    def test_cli_refuses_output_inside_trial(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            path = Path(tmp) / "trial"
            toy_trial(path)
            output = path / "new-audit.json"
            result = subprocess.run(
                [sys.executable, str(MODULE), str(path), "--output", str(output)],
                capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_rank_output_disagreement_is_rejected_after_individual_validity(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            path = Path(tmp) / "trial"
            toy_trial(path)
            file = path / "rank1.receipt.json"
            receipt = json.loads(file.read_text())
            rank = json.loads(receipt["stdout"])
            rank["requests"][0]["token_ids"][-1] = 795  # unlogged bonus, still valid locally
            events = [json.loads(line) for line in receipt["stderr"].splitlines()]
            events[-1]["request_result"] = rank["requests"][0]
            receipt.update(stdout=json.dumps(rank),
                           stderr="".join(json.dumps(e) + "\n" for e in events))
            file.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "rank output/counters disagree"):
                load_audit().audit_trial(path)

    def test_rank_action_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            path = Path(tmp) / "trial"
            toy_trial(path)
            file = path / "rank1.receipt.json"
            receipt = json.loads(file.read_text())
            events = [json.loads(line) for line in receipt["stderr"].splitlines()]
            for event in events:
                if event["phase"] == "rank_action":
                    event["request"] = "b" * 32
            receipt["stderr"] = "".join(json.dumps(e) + "\n" for e in events)
            file.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "actual rank actions disagree"):
                load_audit().audit_trial(path)

    def test_incremental_result_copy_mismatch(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            path = Path(tmp) / "trial"
            toy_trial(path)
            file = path / "rank0.receipt.json"
            receipt = json.loads(file.read_text())
            events = [json.loads(line) for line in receipt["stderr"].splitlines()]
            events[-1]["request_result"]["accepted_drafts"] = 1
            receipt["stderr"] = "".join(json.dumps(e) + "\n" for e in events)
            file.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "incremental/final result mismatch"):
                load_audit().audit_trial(path)

    def test_source_tamper_and_duplicate_json_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            path = Path(tmp) / "trial"
            toy_trial(path)
            file = path / "sources/benchmark_schedule.json"
            file.write_bytes(file.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "captured source SHA mismatch"):
                load_audit().audit_trial(path)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_audit()._json('{"status":"failed", "status":"success"}')

    def test_read_only_trial_and_exact_receipt_hashes(self):
        audit = load_audit()
        self.assertTrue(hasattr(audit, "audit_trial"), "posthoc audit_trial is missing")
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            path = Path(tmp) / "trial"
            toy_trial(path)
            before = {str(p.relative_to(path)): p.read_bytes()
                      for p in path.rglob("*") if p.is_file()}
            report = audit.audit_trial(path)
            self.assertEqual(report["verified_request_count"], 1)
            self.assertEqual(report["verified_rank_request_count"], 2)
            self.assertEqual(report["raw_sha256"]["rank0.receipt.json"],
                             hashlib.sha256(before["rank0.receipt.json"]).hexdigest())
            self.assertEqual(before, {str(p.relative_to(path)): p.read_bytes()
                                      for p in path.rglob("*") if p.is_file()})


class ReviewMutationProofTests(unittest.TestCase):
    def check_mutant(self, old, new, test_name):
        global MODULE
        original = MODULE
        source = original.read_text()
        self.assertEqual(source.count(old), 1)
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            MODULE = Path(tmp) / "mtp_audit.py"
            MODULE.write_text(source.replace(old, new))
            try:
                suite = unittest.TestSuite([GenerationAuditTests(test_name)])
                result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
                self.assertEqual(result.testsRun, 1)
                self.assertEqual(len(result.errors), 0)
                self.assertEqual(len(result.failures), 1, "adversarial test must kill this mutant")
            finally:
                MODULE = original

    def test_kills_zero_round_counter_guard_removal(self):
        self.check_mutant(
            'if type(request.get(field)) is not int or request[field] != observed:',
            'if False:', 'test_review_mutation_zero_rounds_five_outputs')

    def test_kills_accepted_counter_guard_removal(self):
        self.check_mutant(
            'if type(request.get(field)) is not int or request[field] != observed:',
            'if field != "accepted_drafts" and ('
            'type(request.get(field)) is not int or request[field] != observed):',
            'test_review_mutation_changed_accepted_counter')

    def test_kills_current_emitted_token_guard_removal(self):
        self.check_mutant('if len(pair) != 2 or pair[0] != tokens[cursor]:',
                          'if len(pair) != 2:',
                          'test_reject_review_mutation_changed_verify_current')


@unittest.skipUnless(os.environ.get("PARETOQUANT_AUDIT_COMPLETED") == "1",
                     "opt-in archived evidence test (only batches 00..03)")
class CompletedEvidenceTests(unittest.TestCase):
    def test_completed_batches_00_through_03_read_only(self):
        base = MODULE.parents[2] / "results/qwen38-broad-v1"
        audit = load_audit()
        reports = []
        for batch in ("00", "01", "02", "03"):  # never discover/poll running batches
            path = base / f"batch-{batch}-v1"
            files = sorted(p for p in path.rglob("*") if p.is_file())
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            report = audit.audit_trial(path)
            self.assertEqual(report["verified_request_count"], 14)
            self.assertEqual(report["verified_mtp_request_count"], 7)
            self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in files})
            self.assertEqual(files, sorted(p for p in path.rglob("*") if p.is_file()))
            reports.append({k: report[k] for k in ("trialdir", "verified_request_count",
                            "actual_rank_action_counts", "raw_sha256")})
        self.assertEqual(sum(r["verified_request_count"] for r in reports), 56)
        print(json.dumps({"completed_read_only_audits": reports}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
