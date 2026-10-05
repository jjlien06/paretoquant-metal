"""Local-only controller contracts. No SSH or model execution."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import supervise


def tensor_fixture():
    """Complete tiny synthetic unittest receipts; never model execution evidence."""
    replicated = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    suffixes = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    inventories = {
        rank: replicated | {f"model.layers.{1 - rank}.{suffix}" for suffix in suffixes}
        for rank in (0, 1)
    }
    ranks = []
    for rank, names in inventories.items():
        ranks.append(
            {
                "rank": rank,
                "world_size": 2,
                "layer_start": 1 - rank,
                "layer_end": 2 - rank,
                "prompt": "fixture prompt",
                "formatted_prompt": "fixture formatted prompt",
                "execution": "phased",
                "parameter_bytes": 4 * len(names),
                "parameter_layout": [
                    {"name": name, "shape": [1], "dtype": "float32", "bytes": 4}
                    for name in sorted(names)
                ],
                "requests": [{"token_ids": [7, 8], "prompt_tokens": 3}],
                "weight_loading": {
                    "all_retained_tensors_loaded": True,
                    "tensor_count": len(names),
                    "parameter_bytes": 4 * len(names),
                    "tensor_sha256": {
                        name: hashlib.sha256(name.encode()).hexdigest() for name in names
                    },
                },
            }
        )
    server = {
        "completed": True,
        "rank": 0,
        "selected_tensor_count": len(inventories[0]),
        "selected_bytes": 4 * len(inventories[0]),
    }
    receipts = [
        {"reason": "completed", "cleanup": {"child_reaped": True, "group_gone": True}}
        for _ in range(3)
    ]
    return ranks, server, receipts, inventories


def metadata_fixture(root):
    """Synthetic metadata index for controller tests, not checkpoint evidence."""
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen2", "num_hidden_layers": 2}))
    _, _, _, inventory = tensor_fixture()
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {name: "model.safetensors" for name in inventory[0] | inventory[1]}}
        )
    )


def metadata_args(metadata, output):
    return supervise.parser().parse_args(
        [
            "--remote-root",
            "/tmp/root",
            "--remote-source",
            "/tmp/model",
            "--local-metadata",
            str(metadata),
            "--output",
            str(output),
            "--split",
            "1",
            "1",
            "--ssh-key",
            "/tmp/key",
            "--reserve-bytes",
            "0",
        ]
    )


class TensorEvidenceTests(unittest.TestCase):
    def verify(self, ranks, server, receipts, inventory):
        return supervise.verify_trial(ranks, server, receipts, [], expected_inventory=inventory)

    def test_complete_synthetic_receipts_pass_in_either_rank_order(self):
        ranks, server, receipts, inventory = tensor_fixture()
        for records in (ranks, list(reversed(ranks))):
            self.assertTrue(
                self.verify(records, server, receipts, inventory)["tensor_completeness"]
            )

    def test_exact_admitted_names_required_despite_matching_counts(self):
        for mutation in ("wrong", "missing", "extra"):
            with self.subTest(mutation=mutation):
                ranks, server, receipts, inventory = tensor_fixture()
                loading = ranks[1]["weight_loading"]
                hashes = loading["tensor_sha256"]
                name = "model.layers.0.self_attn.q_proj.weight"
                if mutation != "extra":
                    hashes.pop(name)
                if mutation != "missing":
                    hashes["not_a_parameter"] = "a" * 64
                loading["tensor_count"] = len(hashes)
                with self.assertRaisesRegex(ValueError, "inventory"):
                    self.verify(ranks, server, receipts, inventory)

    def test_every_digest_must_be_hex_sha256(self):
        for malformed in ("not-a-sha256", "a" * 63, "a" * 65, "g" * 64, "A" * 64, None, 7):
            for rank in (0, 1):
                with self.subTest(digest=malformed, rank=rank):
                    ranks, server, receipts, inventory = tensor_fixture()
                    name = f"model.layers.{1 - rank}.mlp.up_proj.weight"
                    ranks[rank]["weight_loading"]["tensor_sha256"][name] = malformed
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        self.verify(ranks, server, receipts, inventory)

    def test_replicated_tensor_digests_must_agree(self):
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            with self.subTest(name=name):
                ranks, server, receipts, inventory = tensor_fixture()
                ranks[1]["weight_loading"]["tensor_sha256"][name] = "b" * 64
                with self.assertRaisesRegex(ValueError, "Replicated"):
                    self.verify(ranks, server, receipts, inventory)

    def test_parameter_layout_must_cover_inventory_with_consistent_bytes(self):
        ranks, server, receipts, inventory = tensor_fixture()
        cases = []
        missing = copy.deepcopy(ranks)
        missing[1]["parameter_layout"].pop()
        cases.append(missing)
        duplicate = copy.deepcopy(ranks)
        duplicate[1]["parameter_layout"][0] = duplicate[1]["parameter_layout"][1]
        cases.append(duplicate)
        wrong = copy.deepcopy(ranks)
        wrong[1]["parameter_layout"][0]["name"] = "not_a_parameter"
        cases.append(wrong)
        for field in ("parameter_bytes", "loading_bytes", "layout_bytes"):
            changed = copy.deepcopy(ranks)
            if field == "loading_bytes":
                changed[1]["weight_loading"]["parameter_bytes"] += 4
            elif field == "layout_bytes":
                changed[1]["parameter_layout"][0]["bytes"] = -4
            else:
                changed[1][field] += 4
            cases.append(changed)
        for records in cases:
            with self.subTest(records=records):
                with self.assertRaisesRegex(ValueError, "layout|bytes"):
                    self.verify(records, server, receipts, inventory)
        wrong_server = {**server, "selected_bytes": server["selected_bytes"] + 4}
        with self.assertRaisesRegex(ValueError, "bytes"):
            self.verify(ranks, wrong_server, receipts, inventory)

    def test_rank_ranges_must_match_admitted_global_layer_names(self):
        for rank in (0, 1):
            records, server, receipts, inventory = tensor_fixture()
            records[rank]["layer_start"] = 9
            with self.assertRaisesRegex(ValueError, "range"):
                self.verify(records, server, receipts, inventory)

    def test_prompt_execution_and_prompt_token_counts_must_agree(self):
        for field, value in (
            ("prompt", "other prompt"),
            ("formatted_prompt", "other formatting"),
            ("formatted_prompt", None),
            ("execution", "upstream"),
            ("prompt_tokens", 4),
        ):
            with self.subTest(field=field):
                records, server, receipts, inventory = tensor_fixture()
                if field == "prompt_tokens":
                    records[1]["requests"][0][field] = value
                else:
                    records[1][field] = value
                with self.assertRaisesRegex(ValueError, "prompt|execution"):
                    self.verify(records, server, receipts, inventory)

    def test_missing_admitted_inventory_fails_closed(self):
        ranks, server, receipts, _ = tensor_fixture()
        with self.assertRaisesRegex(ValueError, "inventory"):
            supervise.verify_trial(ranks, server, receipts, [])


class MetadataInventoryTests(unittest.TestCase):
    def test_changed_admitted_metadata_fails_closed(self):
        self.assertTrue(callable(getattr(supervise, "verify_metadata", None)))
        for changed_file in ("config.json", "model.safetensors.index.json"):
            with self.subTest(changed_file=changed_file), tempfile.TemporaryDirectory() as tmp:
                metadata = Path(tmp) / "metadata"
                metadata_fixture(metadata)
                identity = {
                    name: hashlib.sha256((metadata / name).read_bytes()).hexdigest()
                    for name in ("config.json", "model.safetensors.index.json")
                }
                supervise.verify_metadata(metadata, identity)
                path = metadata / changed_file
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "metadata changed"):
                    supervise.verify_metadata(metadata, identity)

    def test_missing_index_fails_before_any_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "metadata"
            metadata_fixture(metadata)
            (metadata / "model.safetensors.index.json").unlink()
            calls = []

            def forbidden_transport(args, actor_args):
                calls.append(actor_args)
                raise AssertionError("Transport forbidden before metadata admission")

            result = supervise.run(
                metadata_args(metadata, Path(tmp) / "output"), transport=forbidden_transport
            )
            self.assertEqual(result["error_stage"], "metadata_admission")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(calls, [])

    def test_index_must_be_nonempty_weight_map_object(self):
        _, _, _, inventory = tensor_fixture()
        for weight_map in (list(inventory[0] | inventory[1]), None, {}):
            with self.subTest(weight_map=weight_map), tempfile.TemporaryDirectory() as tmp:
                metadata = Path(tmp) / "metadata"
                metadata_fixture(metadata)
                (metadata / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": weight_map})
                )
                result = supervise.run(
                    metadata_args(metadata, Path(tmp) / "output"),
                    transport=lambda *_: (_ for _ in ()).throw(AssertionError("No transport")),
                )
                self.assertEqual(result["error_type"], "ValueError")
                self.assertEqual(result["error_stage"], "metadata_admission")

    def test_incomplete_or_contradictory_metadata_inventory_rejected(self):
        _, _, _, inventory = tensor_fixture()
        names = inventory[0] | inventory[1]
        config = {"model_type": "qwen2", "num_hidden_layers": 2}
        cases = [
            (config, set(), [1, 1]),
            (config, names - {n for n in names if n.startswith("model.layers.0.")}, [1, 1]),
            (config, names - {"model.norm.weight"}, [1, 1]),
            (config, names - {"lm_head.weight"}, [1, 1]),
            (config, names | {"not_a_parameter"}, [1, 1]),
            (config, names | {"model.layers.2.mlp.up_proj.weight"}, [1, 1]),
            (config, names | {"model.layers.00.mlp.up_proj.weight"}, [1, 1]),
            ({**config, "tie_word_embeddings": True}, names, [1, 1]),
            ({**config, "tie_word_embeddings": "false"}, names, [1, 1]),
            ({**config, "model_type": "other"}, names, [1, 1]),
            (config, names, [1, 2]),
            (config, names, [2]),
        ]
        for admitted_config, admitted_names, split in cases:
            with self.subTest(config=admitted_config, split=split, names=sorted(admitted_names)):
                with self.assertRaises(ValueError):
                    supervise.expected_inventory(admitted_config, admitted_names, split)

    def test_trusted_index_partitions_global_names_with_reverse_rank_ownership(self):
        self.assertTrue(callable(getattr(supervise, "expected_inventory", None)))
        _, _, _, inventory = tensor_fixture()
        names = inventory[0] | inventory[1]
        config = {"model_type": "qwen2", "num_hidden_layers": 2}
        self.assertEqual(supervise.expected_inventory(config, names, [1, 1]), inventory)
        tied_names = names - {"lm_head.weight"}
        config["tie_word_embeddings"] = True
        self.assertEqual(
            supervise.expected_inventory(config, tied_names, [1, 1]),
            {rank: selected - {"lm_head.weight"} for rank, selected in inventory.items()},
        )


class ContractTests(unittest.TestCase):
    def test_delayed_peer_refusal_still_verifies_closed_listener(self):
        import errno
        from unittest.mock import patch

        class DelayedClosedSocket:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def settimeout(self, seconds):
                self.timeout = seconds

            def connect(self, address):
                if self.timeout < 1.0:
                    raise TimeoutError("Peer refusal arrived after one second")
                raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")

        with patch.object(supervise.socket, "socket", return_value=DelayedClosedSocket()):
            self.assertTrue(supervise.listener_closed("169.254.169.190", 50744))

    def test_defaults_and_safe_ssh_quoting(self):
        args = supervise.parser().parse_args(
            [
                "--remote-root",
                "/tmp/spike with spaces",
                "--remote-source",
                "/tmp/model",
                "--local-metadata",
                "/tmp/meta",
                "--output",
                "/tmp/new",
                "--split",
                "28",
                "36",
                "--ssh-key",
                "/tmp/key",
            ]
        )
        self.assertEqual(
            (args.max_tokens, args.repeats, args.deadline, args.execution), (16, 1, 180, "phased")
        )
        argv = supervise.ssh_argv(args, ["--inspect"])
        self.assertEqual(argv[0], "ssh")
        self.assertIn("IdentitiesOnly=yes", argv)
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertIn("'/tmp/spike with spaces/remote_actor.py'", argv[-1])
        with self.assertRaises(ValueError):
            args.ssh_key = str(Path(__file__).resolve().parent / "key")
            supervise.validate_args(args)

    def test_success_requires_complete_matching_receipts(self):
        with self.assertRaises(ValueError):
            supervise.verify_trial([], {}, [], [])
        with self.assertRaises(ValueError):
            supervise.verify_trial(
                [{"rank": 0, "world_size": 2, "requests": []}], {"completed": True}, [], []
            )

    def test_local_controller_deadline_preserves_failure_and_reaps(self):
        import sys
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            args = supervise.parser().parse_args(
                [
                    "--remote-root",
                    "/tmp/root",
                    "--remote-source",
                    "/tmp/model",
                    "--local-metadata",
                    "/tmp/meta",
                    "--output",
                    str(Path(tmp) / "new"),
                    "--split",
                    "1",
                    "1",
                    "--ssh-key",
                    "/tmp/key",
                    "--deadline",
                    ".2",
                    "--reserve-bytes",
                    "0",
                ]
            )
            controller = supervise.Controller(args)
            child = controller.launch(
                "fixture",
                [sys.executable, "-c", 'import time; print("partial",flush=True); time.sleep(30)'],
            )
            with self.assertRaises(TimeoutError):
                while child.poll() is None:
                    controller.check()
                    time.sleep(0.02)
            cleanup = controller.close()
            self.assertTrue(cleanup["fixture"]["child_reaped"])
            self.assertTrue(cleanup["fixture"]["group_gone"])
            self.assertIn("partial", (Path(args.output) / "fixture.stdout").read_text())

    def test_peer_mismatch_stops_before_launch(self):
        with self.assertRaises(ValueError):
            supervise.verify_sources(
                {"probe.py": "a" * 64}, {"source_sha256": {"probe.py": "b" * 64}}
            )
        supervise.verify_sources({"probe.py": "a" * 64}, {"source_sha256": {"probe.py": "a" * 64}})

    def test_durable_source_mismatch_receipt_without_network(self):
        import json
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "metadata"
            metadata_fixture(metadata)
            args = supervise.parser().parse_args(
                [
                    "--remote-root",
                    "/tmp/root",
                    "--remote-source",
                    "/tmp/model",
                    "--local-metadata",
                    str(metadata),
                    "--output",
                    str(Path(tmp) / "new"),
                    "--split",
                    "1",
                    "1",
                    "--ssh-key",
                    "/tmp/key",
                    "--reserve-bytes",
                    "0",
                ]
            )

            def local_only(args, actor_args):
                return [sys.executable, "-c", "print('{\"source_sha256\": {}}')"]

            result = supervise.run(args, transport=local_only)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "ValueError")
            identity = {
                name: hashlib.sha256((metadata / name).read_bytes()).hexdigest()
                for name in ("config.json", "model.safetensors.index.json")
            }
            self.assertEqual(result.get("metadata_sha256"), identity)
            startup = json.loads((Path(args.output) / "startup.json").read_text())
            self.assertEqual(startup.get("metadata_sha256"), identity)
            self.assertEqual(result["error_stage"], "source_admission")
            self.assertTrue((Path(args.output) / "source_sha256.json").exists())
            self.assertEqual(
                json.loads((Path(args.output) / "result.json").read_text())["status"], "failed"
            )

    def test_readiness_deadline_is_failed_and_token_never_recorded(self):
        import json
        import sys
        import tempfile
        from unittest.mock import patch

        from remote_actor import source_hashes

        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "metadata"
            metadata_fixture(metadata)
            args = supervise.parser().parse_args(
                [
                    "--remote-root",
                    "/tmp/root",
                    "--remote-source",
                    "/tmp/model",
                    "--local-metadata",
                    str(metadata),
                    "--output",
                    str(Path(tmp) / "new"),
                    "--split",
                    "1",
                    "1",
                    "--ssh-key",
                    "/tmp/key",
                    "--deadline",
                    ".4",
                    "--reserve-bytes",
                    "0",
                    "--local-address",
                    "127.0.0.1",
                    "--remote-address",
                    "127.0.0.1",
                ]
            )
            peer = {"source_sha256": source_hashes(supervise.ROOT), "available_port": 12345}

            def local_only(args, actor_args):
                if "--inspect" in actor_args:
                    return [sys.executable, "-c", f"print({json.dumps(json.dumps(peer))})"]
                if "--status" in actor_args or "--stop" in actor_args:
                    return [sys.executable, "-c", 'print("{}")']
                return [
                    sys.executable,
                    "-c",
                    (
                        "import sys,time; sys.stdin.read(); "
                        'print("fixture waiting",flush=True); time.sleep(30)'
                    ),
                ]

            token = "f" * 64
            with (
                patch.object(supervise, "free_port", return_value=12346),
                patch.object(supervise, "listener_closed", return_value=True),
                patch.object(
                    supervise.secrets,
                    "token_hex",
                    side_effect=lambda n: token if n == 32 else "e" * 32,
                ),
            ):
                result = supervise.run(args, transport=local_only)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "TimeoutError")
            self.assertEqual(result["error_stage"], "server_readiness")
            self.assertFalse(result["services_closed_verified"])
            self.assertTrue(result["transport_cleanup"]["server"]["group_gone"])
            for path in Path(args.output).rglob("*"):
                if path.is_file():
                    self.assertNotIn(token, path.read_text(errors="replace"))

    def test_completed_control_calls_do_not_accumulate_live_children(self):
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            args = supervise.parser().parse_args(
                [
                    "--remote-root",
                    "/tmp/root",
                    "--remote-source",
                    "/tmp/model",
                    "--local-metadata",
                    "/tmp/meta",
                    "--output",
                    str(Path(tmp) / "new"),
                    "--split",
                    "1",
                    "1",
                    "--ssh-key",
                    "/tmp/key",
                    "--reserve-bytes",
                    "0",
                ]
            )
            controller = supervise.Controller(args)
            try:
                self.assertEqual(controller.call([sys.executable, "-c", 'print("{}")']), {})
                self.assertEqual(controller.children, {})
                self.assertEqual(controller.files, [])
            finally:
                controller.close()
