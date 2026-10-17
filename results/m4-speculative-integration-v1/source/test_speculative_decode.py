"""Stdlib-only vertical TDD; symbolic KV fixtures, never performance claims."""

import importlib.util
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = pathlib.Path(__file__).parent


def load_module():
    path = HERE / "speculative_decode.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("speculative_decode", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sut = load_module()


class AcceptanceTests(unittest.TestCase):
    def test_longest_prefix_and_teacher_bonus(self):
        self.assertIsNotNone(sut, "speculative decoder component is missing")
        for draft, teacher, expected in [
            ([2, 3, 4], [2, 3, 5, 6], ([2, 3, 5], 2)),
            ([2, 3], [2, 3, 4], ([2, 3, 4], 2)),
            ([], [7], ([7], 0)),
        ]:
            self.assertEqual(sut.accept_tokens(draft, teacher, set(), 10), expected)

    def test_confirmed_eos_excludes_bonus_and_counts_only_emitted_prefix(self):
        self.assertEqual(sut.accept_tokens([2, 9, 4], [2, 9, 4, 5], {9}, 8), ([2, 9], 2))
        self.assertEqual(sut.accept_tokens([9], [3, 4], {9}, 8), ([3], 0))
        self.assertEqual(sut.accept_tokens([2, 3], [2, 3, 4], set(), 1), ([2], 1))

    def test_invalid_verification_shape_or_budget_is_rejected(self):
        for proposals, teacher, remaining in [
            ([1], [1], 2),
            ([], [], 1),
            ([], [1], True),
            ([], [1], 0),
            ([True], [1, 2], 2),
            ([], [-1], 1),
        ]:
            with (
                self.subTest(proposals=proposals, teacher=teacher, remaining=remaining),
                self.assertRaises(ValueError),
            ):
                sut.accept_tokens(proposals, teacher, set(), remaining)


class PacketTests(unittest.TestCase):
    def test_fixed_width_roundtrip_rejects_corruption_before_use(self):
        self.assertTrue(hasattr(sut, "packet"), "packet contract is missing")
        packed = sut.packet([2, 3], 4, 7, 11, 10)
        self.assertEqual(packed, [7, 11, 2, 2, 3, -1, -1, -1])
        self.assertEqual(sut.validate_packet(packed, 4, 7, 11, 10), [2, 3])
        corrupt = [
            packed[:-1],
            [8] + packed[1:],
            [7, 12] + packed[2:],
            [7, 11, True] + packed[3:],
            [7, 11, 6] + packed[3:],
            [7, 11, 2, 2, 10, -1, -1, -1],
            [7, 11, 2, 2, 3, 0, -1, -1],
            [7, 11, 2, 2.0, 3, -1, -1, -1],
        ]
        for value in corrupt:
            with self.subTest(value=value), self.assertRaises(ValueError):
                sut.validate_packet(value, 4, 7, 11, 10)
        self.assertEqual(len(sut.packet([1] * 5, 4, 0, 0, 10)), 8)
        for bad in [True, 3, 0, 4.0]:
            with self.assertRaises(ValueError):
                sut.packet([], bad, 0, 0, 10)


class KV:
    """Pinned KVCache behavior: trim changes offset, not stored tail."""

    def __init__(self, tokens=()):
        self.tail = list(tokens)
        self.offset = len(tokens)
        self.trim_calls = []

    @property
    def state(self):
        return self.tail

    def is_trimmable(self):
        return True

    def trim(self, count):
        self.trim_calls.append(count)
        self.offset -= count
        return count

    def consume(self, tokens):
        for token in tokens:
            if self.offset < len(self.tail):
                self.tail[self.offset] = token
            else:
                self.tail.append(token)
            self.offset += 1


class CacheTests(unittest.TestCase):
    def test_trim_all_layers_and_overwrite_stale_rejected_tail(self):
        self.assertTrue(hasattr(sut, "trim_to"), "cache rollback is missing")
        caches = [KV([1, 2, 8, 9]), KV([1, 2, 8, 9])]
        sut.trim_to(caches, 2)
        for state in caches:
            self.assertEqual(state.trim_calls, [2])
            self.assertEqual(state.tail, [1, 2, 8, 9])
            state.consume([3, 4])
            self.assertEqual(state.tail, [1, 2, 3, 4])
        self.assertEqual(sut.cache_offset(caches), 4)
        for bad in [[], [KV(), KV([1])]]:
            with self.assertRaises(ValueError):
                sut.cache_offset(bad)
        for property_name, value in [
            ("is_trimmable", lambda: False),
            ("trim", None),
            ("offset", True),
        ]:
            bad = KV()
            setattr(bad, property_name, value)
            with self.assertRaises(ValueError):
                sut.cache_offset([bad])
        with self.assertRaises(ValueError):
            sut.trim_to(caches, 5)


class Tokenizer:
    bos_token = "<bos>"
    bos_token_id = 0
    eos_token_ids = [9]

    def __init__(self):
        self.vocab = {str(i): i for i in range(10)}
        self.encode_calls = []

    def get_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append((text, add_special_tokens))
        return [0, 1] if add_special_tokens else [1]

    def decode(self, tokens, skip_special_tokens=True):
        return ",".join(map(str, tokens))


class SemanticsTests(unittest.TestCase):
    def test_requires_identical_shared_ids_and_special_token_semantics(self):
        self.assertTrue(hasattr(sut, "validate_tokenizers"), "ID semantics admission is missing")
        teacher, draft = Tokenizer(), Tokenizer()
        sut.validate_tokenizers(teacher, draft, 10, 10)
        for mutate in [
            lambda t: t.vocab.update({"0": 1, "1": 0}),
            lambda t: t.vocab.pop("3"),
            lambda t: setattr(t, "eos_token_ids", [8]),
            lambda t: setattr(t, "bos_token_id", 1),
            lambda t: t.vocab.update({"alias": 3}),
            lambda t: t.vocab.update({"3": True}),
        ]:
            draft = Tokenizer()
            mutate(draft)
            with self.assertRaises(ValueError):
                sut.validate_tokenizers(teacher, draft, 10, 10)
        with self.assertRaises(ValueError):
            sut.validate_tokenizers(teacher, None, 10, 10)
        with self.assertRaises(ValueError):
            sut.validate_tokenizers(teacher, Tokenizer(), 10, 9)

    def test_embedding_padding_and_different_model_vocab_sizes_do_not_change_shared_ids(self):
        self.assertEqual(
            sut.validate_tokenizers(Tokenizer(), Tokenizer(), 12, 10), frozenset(range(10))
        )
        runtime = Runtime()
        teacher, draft = SymbolicModel(lambda p: 1), SymbolicModel(lambda p: 1)
        teacher.args.vocab_size = 12
        with runtime.injected(), self.assertRaises(ValueError):
            sut.speculative_generate(
                teacher,
                Tokenizer(),
                [11],
                draft_model=draft,
                draft_tokenizer=Tokenizer(),
                max_tokens=1,
            )
        self.assertEqual(runtime.teacher_calls, [])


# Explicit integer-prediction doubles: these are not model logits or timings.


class Array:
    def __init__(self, data):
        self.data = data
        self.shape = (
            (len(data), len(data[0])) if data and isinstance(data[0], list) else (len(data),)
        )
        self.ndim = len(self.shape)

    def reshape(self, *shape):
        return Array([x for row in self.data for x in row] if self.ndim == 2 else self.data)

    def astype(self, dtype):
        return self

    def tolist(self):
        return self.data


class Predictions:
    def __init__(self, ids):
        self.ids = ids
        self.shape = (1, len(ids), 10)

    def __getitem__(self, key):
        return Predictions([self.ids[-1]])


class CPU:
    int32 = "int32"
    cpu = "cpu"

    @staticmethod
    def array(data, dtype=None):
        return Array(data)

    @staticmethod
    def eval(*values):
        pass

    @staticmethod
    def argmax(predictions, axis=-1):
        return Array([predictions.ids])


class SymbolicCore:
    def __init__(self, owner, rank=0, size=1):
        self.owner = owner
        self.pipeline_rank, self.pipeline_size = rank, size
        self.start_idx = 38 if size == 2 and rank == 0 else 0
        self.end_idx = 38 if size == 2 and rank == 1 else owner.args.num_hidden_layers
        self.pipeline_layers = [object() for _ in range(self.end_idx - self.start_idx)]

    def __call__(self, inputs, cache):
        return self.owner.core(inputs, cache)


class SymbolicModel:
    model_type = "qwen2"

    def __init__(self, rule):
        self.args = SimpleNamespace(
            vocab_size=10, max_position_embeddings=4096, num_hidden_layers=4
        )
        self.rule = rule
        self.calls = []
        self.cache = None
        self.model = SymbolicCore(self)

    def core(self, inputs, cache):
        self.calls.append(("core", list(inputs.data[0]), cache[0].offset))
        for state in cache:
            state.consume(inputs.data[0])
        return inputs

    def __call__(self, inputs, cache):
        self.calls.append(("head", list(inputs.data[0]), cache[0].offset))
        ids = []
        for token in inputs.data[0]:
            for state in cache:
                state.consume([token])
            ids.append(self.rule(cache[0].tail[: cache[0].offset]))
        return Predictions(ids)


def phased_helpers():
    source = HERE.parent / "007-dual-host-pipeline" / "phased_decode.py"
    spec = importlib.util.spec_from_file_location("phased_helpers_only", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # stdlib-only top level; never call its MLX entry points
    return module


class Runtime:
    def __init__(self):
        self.teacher_calls = []
        self.cpu = CPU()
        self.phase = phased_helpers()
        self.phase.phased_forward = self.forward

    def make_cache(self, model):
        model.cache = [KV() for _ in model.model.pipeline_layers]
        return model.cache

    def forward(self, model, inputs, cache, **options):
        self.teacher_calls.append((list(inputs.data[0]), dict(options), cache[0].offset))
        if not options.get("logits", True):
            model.core(inputs, cache)
            return None
        return model(inputs, cache)

    def injected(self):
        return patch.object(
            sut, "_load_runtime", return_value=(self.cpu, self.make_cache, self.phase), create=True
        )


class GenerationTests(unittest.TestCase):
    def test_single_host_rejection_and_full_acceptance_restore_exact_live_prefix(self):
        self.assertTrue(hasattr(sut, "speculative_generate"), "generation entry is missing")
        runtime = Runtime()
        teacher = SymbolicModel(lambda prefix: (prefix[-1] + 1) % 9)
        # First round rejects the second proposal; later rounds fully accept.
        draft = SymbolicModel(lambda prefix: 8 if len(prefix) == 5 else (prefix[-1] + 1) % 9)
        tokenizer, draft_tokenizer = Tokenizer(), Tokenizer()
        events = []
        prompt = [0, 1, 2, 3]
        with runtime.injected():
            result = sut.speculative_generate(
                teacher,
                tokenizer,
                prompt,
                draft_model=draft,
                draft_tokenizer=draft_tokenizer,
                max_tokens=8,
                draft_tokens=4,
                prefill_step_size=2,
                progress=events.append,
            )
        self.assertEqual(result["token_ids"], [4, 5, 6, 7, 8, 0, 1, 2])
        expected_cached = prompt + result["token_ids"][:-1]
        for model in (teacher, draft):
            for state in model.cache:
                self.assertEqual(state.offset, len(expected_cached))
                self.assertEqual(state.tail[: state.offset], expected_cached)
        self.assertEqual(runtime.teacher_calls[0][0], [0, 1])
        self.assertEqual(runtime.teacher_calls[1][0], [2])
        verify = runtime.teacher_calls[2:]
        self.assertTrue(all(call[1]["all_logits"] for call in verify))
        self.assertTrue(all(-1 not in call[0] for call in runtime.teacher_calls))
        self.assertEqual(result["target_forward_calls"], len(runtime.teacher_calls))
        self.assertEqual(result["draft_calls"], len(draft.calls))
        self.assertIsNone(result["native_generation_tps"])
        self.assertIsNotNone(result["first_token_seconds"])
        self.assertGreater(result["wall_seconds"], 0)
        rounds = [e for e in events if e["phase"] == "speculative_round_done"]
        self.assertEqual(rounds[0]["accepted"], 1)
        self.assertTrue(any(e["accepted"] == 4 for e in rounds))
        self.assertEqual(result["acceptance_stats"]["rounds"], len(rounds))


class DistributedTests(unittest.TestCase):
    def test_two_ranks_fixed_packets_and_round_order_match_single_host(self):
        import concurrent.futures
        import queue
        import threading

        class Bus:
            def __init__(self):
                self.queues = {
                    (kind, src): queue.Queue() for kind in ("packet", "hidden") for src in (0, 1)
                }
                self.events = []
                self.lock = threading.Lock()

            def record(self, event):
                with self.lock:
                    self.events.append(event)

        class Transport:
            def __init__(self, bus, rank, draft):
                self.bus, self.rank, self.draft = bus, rank, draft

            def send(self, data, peer, **options):
                self.assert_cpu(options)
                values = data.tolist()
                self.bus.record(
                    (
                        "packet",
                        self.rank,
                        values,
                        self.draft.cache[0].offset if self.draft else None,
                    )
                )
                self.bus.queues["packet", self.rank].put(data)
                return data

            def recv(self, shape, dtype, peer, **options):
                self.assert_cpu(options)
                data = self.bus.queues["packet", peer].get(timeout=3)
                if data.shape != shape:
                    raise AssertionError("wire shape differs")
                return data

            @staticmethod
            def assert_cpu(options):
                if options.get("stream") != "cpu":
                    raise AssertionError("packets must use CPU stream")

        class PipelineRuntime(Runtime):
            def __init__(self, bus, rank, draft):
                super().__init__()
                self.bus, self.rank = bus, rank
                self.cpu.distributed = Transport(bus, rank, draft)

            def forward(self, model, inputs, cache, **options):
                if self.rank == 1:
                    self.teacher_calls.append(
                        (list(inputs.data[0]), dict(options), cache[0].offset)
                    )
                    model.core(inputs, cache)
                    self.bus.record(("hidden", 1, list(inputs.data[0])))
                    self.bus.queues["hidden", 1].put(inputs.data[0])
                    return None
                hidden = self.bus.queues["hidden", 1].get(timeout=3)
                if hidden != inputs.data[0]:
                    raise AssertionError("teacher ranks used different verification inputs")
                return super().forward(model, inputs, cache, **options)

        bus = Bus()
        draft = SymbolicModel(lambda p: 8 if len(p) == 5 else (p[-1] + 1) % 9)
        teachers = [SymbolicModel(lambda p: (p[-1] + 1) % 9) for _ in range(2)]
        for rank, teacher in enumerate(teachers):
            teacher.args.num_hidden_layers = 64
            teacher.model = SymbolicCore(teacher, rank, 2)
        runtimes = [PipelineRuntime(bus, rank, draft if rank else None) for rank in (0, 1)]
        # Thread-local runtime selection keeps the test boundary explicit.
        local = threading.local()

        def run(rank):
            local.runtime = runtimes[rank]
            group = SimpleNamespace(rank=lambda: rank, size=lambda: 2)
            return sut.speculative_generate(
                teachers[rank],
                Tokenizer(),
                [0, 1, 2, 3],
                draft_model=draft if rank else None,
                draft_tokenizer=Tokenizer(),
                group=group,
                max_tokens=8,
                draft_tokens=4,
                prefill_step_size=2,
            )

        with patch.object(
            sut,
            "_load_runtime",
            side_effect=lambda: (local.runtime.cpu, local.runtime.make_cache, local.runtime.phase),
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run, rank) for rank in (0, 1)]
                results = [f.result(timeout=6) for f in futures]
        self.assertEqual(results[0]["token_ids"], [4, 5, 6, 7, 8, 0, 1, 2])
        self.assertEqual(results[0]["token_ids"], results[1]["token_ids"])
        self.assertEqual(results[0]["draft_calls"], 0)
        self.assertEqual(results[1]["draft_calls"], len(draft.calls))
        for model in [*teachers, draft]:
            for state in model.cache:
                self.assertEqual(state.tail[: state.offset], [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1])
        packets = [event for event in bus.events if event[0] == "packet"]
        self.assertTrue(all(len(event[2]) == 8 for event in packets))
        for proposal, answer in zip(packets[::2], packets[1::2]):
            self.assertEqual((proposal[1], answer[1]), (1, 0))
            self.assertEqual(proposal[2][:2], answer[2][:2])
            self.assertEqual(answer[2][2], proposal[2][2] + 1)
            self.assertEqual(proposal[3], proposal[2][1] + proposal[2][2])
            start, end = bus.events.index(proposal), bus.events.index(answer)
            self.assertEqual(
                [e[0] for e in bus.events[start : end + 1]], ["packet", "hidden", "packet"]
            )


class AdmissionTests(unittest.TestCase):
    def test_raw_context_limit_cannot_be_normalized_into_valid_integer(self):
        runtime = Runtime()
        teacher, draft = SymbolicModel(lambda p: 1), SymbolicModel(lambda p: 1)
        teacher.args.max_position_embeddings = 32
        for bad in [33.0, 32769, float("inf")]:
            with self.subTest(context_limit=bad), runtime.injected(), self.assertRaises(ValueError):
                sut.speculative_generate(
                    teacher,
                    Tokenizer(),
                    [0],
                    draft_model=draft,
                    draft_tokenizer=Tokenizer(),
                    max_tokens=1,
                    context_limit=bad,
                )
        self.assertEqual(runtime.teacher_calls, [])


class GrowthTests(unittest.TestCase):
    def test_wrong_forward_cache_growth_is_rejected_not_hidden_by_trim(self):
        class ExtraTarget(Runtime):
            def forward(self, model, inputs, cache, **options):
                result = super().forward(model, inputs, cache, **options)
                for state in cache:
                    state.consume([7])
                return result

        class ExtraDraft(SymbolicModel):
            def __call__(self, inputs, cache):
                result = super().__call__(inputs, cache)
                for state in cache:
                    state.consume([7])
                return result

        for runtime, draft in [
            (ExtraTarget(), SymbolicModel(lambda p: 8)),
            (Runtime(), ExtraDraft(lambda p: 8)),
        ]:
            teacher = SymbolicModel(lambda p: 1)
            with (
                self.subTest(runtime=type(runtime).__name__),
                runtime.injected(),
                self.assertRaises(ValueError),
            ):
                sut.speculative_generate(
                    teacher,
                    Tokenizer(),
                    [0],
                    draft_model=draft,
                    draft_tokenizer=Tokenizer(),
                    max_tokens=2,
                    draft_tokens=1,
                )


class WireAdmissionTests(unittest.TestCase):
    def test_short_or_eos_interior_proposal_packet_never_reaches_teacher(self):
        for proposals in [[], [1], [9, 1]]:
            runtime = Runtime()
            values = sut.packet(proposals, 4, 0, 0, 10)
            runtime.cpu.distributed = SimpleNamespace(
                recv=lambda *a, **k: Array(values), send=lambda data, *a, **k: data
            )
            group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
            teacher = SymbolicModel(lambda p: 1)
            teacher.args.num_hidden_layers = 64
            teacher.model = SymbolicCore(teacher, 0, 2)
            with (
                self.subTest(proposals=proposals),
                runtime.injected(),
                self.assertRaises(ValueError),
            ):
                sut.speculative_generate(
                    teacher,
                    Tokenizer(),
                    [0],
                    draft_tokenizer=Tokenizer(),
                    group=group,
                    max_tokens=5,
                    draft_tokens=4,
                )
            self.assertEqual(runtime.teacher_calls, [])


class SemanticDataPlaneTests(unittest.TestCase):
    def test_padding_predictions_fail_closed_without_replacing_teacher_greedy_id(self):
        for bad_teacher in (False, True):
            runtime = Runtime()
            teacher = SymbolicModel(lambda p: 8 if bad_teacher else 1)
            draft = SymbolicModel(lambda p: 1 if bad_teacher else 8)
            tokenizer, draft_tokenizer = Tokenizer(), Tokenizer()
            tokenizer.vocab.pop("8")
            draft_tokenizer.vocab.pop("8")
            events = []
            with (
                self.subTest(bad_teacher=bad_teacher),
                runtime.injected(),
                self.assertRaises(ValueError),
            ):
                sut.speculative_generate(
                    teacher,
                    tokenizer,
                    [0],
                    draft_model=draft,
                    draft_tokenizer=draft_tokenizer,
                    max_tokens=1 if bad_teacher else 2,
                    draft_tokens=1,
                    progress=events.append,
                )
            self.assertFalse(any(e["phase"] == "token_done" for e in events))


class BoundaryMatrixTests(unittest.TestCase):
    def test_symbolic_greedy_eos_budget_rejection_cache_matrix(self):
        cases = 0
        for k in (1, 2, 4, 8):
            for budget in range(1, 13):
                for mismatch in range(k + 1):
                    for eos_at in {None, 0, 1, k - 1, k, k + 1}:
                        with self.subTest(k=k, budget=budget, mismatch=mismatch, eos_at=eos_at):
                            runtime = Runtime()
                            prompt = [0, 1, 2]

                            def rule(prefix):
                                index = len(prefix) - len(prompt)
                                return 9 if index == eos_at else (index + 3) % 9

                            def draft_rule(prefix):
                                teacher_id = rule(prefix)
                                return (
                                    (teacher_id + 1) % 9
                                    if len(prefix) - len(prompt) == mismatch
                                    else teacher_id
                                )

                            teacher, draft = SymbolicModel(rule), SymbolicModel(draft_rule)
                            expected = []
                            for _ in range(budget):
                                expected.append(rule(prompt + expected))
                                if expected[-1] == 9:
                                    break
                            with runtime.injected():
                                result = sut.speculative_generate(
                                    teacher,
                                    Tokenizer(),
                                    prompt,
                                    draft_model=draft,
                                    draft_tokenizer=Tokenizer(),
                                    max_tokens=budget,
                                    draft_tokens=k,
                                )
                            self.assertEqual(result["token_ids"], expected)
                            self.assertEqual(
                                result["finish_reason"], "stop" if expected[-1] == 9 else "length"
                            )
                            live = prompt + expected[:-1]
                            for model in (teacher, draft):
                                for state in model.cache:
                                    self.assertEqual(state.offset, len(live))
                                    self.assertEqual(state.tail[: state.offset], live)
                            self.assertEqual(result["draft_calls"], len(draft.calls))
                            self.assertEqual(
                                result["target_forward_calls"], len(runtime.teacher_calls)
                            )
                            self.assertLessEqual(
                                result["acceptance_stats"]["accepted"],
                                result["acceptance_stats"]["proposed"],
                            )
                            cases += 1
        print("symbolic boundary matrix cases:", cases)

    def test_bos_encoding_and_all_admission_options_use_original_helpers(self):
        for prompt, add_special in [("hello", True), ("<bos>hello", False)]:
            runtime, tokenizer = Runtime(), Tokenizer()
            with runtime.injected():
                sut.speculative_generate(
                    SymbolicModel(lambda p: 1),
                    tokenizer,
                    prompt,
                    draft_model=SymbolicModel(lambda p: 1),
                    draft_tokenizer=Tokenizer(),
                    max_tokens=1,
                )
            self.assertEqual(tokenizer.encode_calls, [(prompt, add_special)])
        bad_options = [{"draft_tokens": v} for v in (True, 0, 3, 4.0, 16)]
        bad_options += [{"max_tokens": v} for v in (True, 0, 513, 1.0)]
        bad_options += [{"eval_group_size": v} for v in (True, 3, 8.0)]
        bad_options += [{"prefill_step_size": v} for v in (True, 0, 257)]
        bad_options += [{"context_limit": v} for v in (True, 0, 32769)]
        for options in bad_options:
            runtime = Runtime()
            with self.subTest(options=options), runtime.injected(), self.assertRaises(ValueError):
                sut.speculative_generate(
                    SymbolicModel(lambda p: 1),
                    Tokenizer(),
                    [0],
                    draft_model=SymbolicModel(lambda p: 1),
                    draft_tokenizer=Tokenizer(),
                    **options,
                )
            self.assertEqual(runtime.teacher_calls, [])
        for prompt in ([], [True], [-1], [10], [1.0]):
            runtime = Runtime()
            with self.subTest(prompt=prompt), runtime.injected(), self.assertRaises(ValueError):
                sut.speculative_generate(
                    SymbolicModel(lambda p: 1),
                    Tokenizer(),
                    prompt,
                    draft_model=SymbolicModel(lambda p: 1),
                    draft_tokenizer=Tokenizer(),
                )
            self.assertEqual(runtime.teacher_calls, [])

    def test_no_gpu_or_model_module_was_imported(self):
        import builtins

        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "mlx" or name.startswith(("mlx.", "mlx_lm")):
                raise AssertionError("CPU fixtures attempted a GPU/model import")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            component = load_module()
            runtime = Runtime()
            with patch.object(
                component,
                "_load_runtime",
                return_value=(runtime.cpu, runtime.make_cache, runtime.phase),
            ):
                component.speculative_generate(
                    SymbolicModel(lambda p: 1),
                    Tokenizer(),
                    [0],
                    draft_model=SymbolicModel(lambda p: 1),
                    draft_tokenizer=Tokenizer(),
                    max_tokens=1,
                )


if __name__ == "__main__":
    unittest.main()
