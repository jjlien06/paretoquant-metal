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
    def test_verification_mode_receipt_round_events_and_forward_counters(self):
        for options in ({}, {"verification_mode": "batch"}, {"verification_mode": "sequential"}):
            with self.subTest(options=options):
                runtime, events = Runtime(), []
                with runtime.injected():
                    result = sut.speculative_generate(
                        SymbolicModel(lambda p: (p[-1] + 1) % 9),
                        Tokenizer(),
                        [0, 1, 2],
                        draft_model=SymbolicModel(lambda p: (p[-1] + 1) % 9),
                        draft_tokenizer=Tokenizer(),
                        max_tokens=7,
                        draft_tokens=4,
                        progress=events.append,
                        **options,
                    )
                mode = options.get("verification_mode", "batch")
                self.assertEqual(result.get("verification_mode"), mode)
                before = [e for e in events if e["phase"] == "speculative_round_before"]
                done = [e for e in events if e["phase"] == "speculative_round_done"]
                self.assertTrue(all(e.get("verification_mode") == mode for e in before + done))
                self.assertEqual([e["actual_width"] for e in before], [5, 2])
                self.assertEqual(result["target_verification_calls"], 2)
                self.assertEqual(result["target_forward_calls"], 8 if mode == "sequential" else 3)
                verify = runtime.teacher_calls[1:]
                self.assertEqual(
                    [len(c[0]) for c in verify], [1] * 7 if mode == "sequential" else [5, 2]
                )
                self.assertTrue(all(c[1]["eval_group_size"] == 8 for c in verify))
                self.assertTrue(all(c[1]["progress"] == events.append for c in verify))

    def test_sequential_verification_uses_one_token_forwards_and_live_prefix(self):
        runtime = Runtime()

        def rule(p):
            return (p[-1] + 1 + sum(p[:-1]) % 2) % 9

        teacher = SymbolicModel(rule)
        draft = SymbolicModel(lambda p: 8 if len(p) == 5 else rule(p))
        events = []
        prompt = [0, 1, 2, 3]
        with runtime.injected():
            result = sut.speculative_generate(
                teacher,
                Tokenizer(),
                prompt,
                draft_model=draft,
                draft_tokenizer=Tokenizer(),
                max_tokens=8,
                draft_tokens=4,
                prefill_step_size=2,
                progress=events.append,
                verification_mode="sequential",
            )
        verify = runtime.teacher_calls[2:]
        self.assertTrue(all(len(call[0]) == 1 for call in verify), verify)
        self.assertTrue(all(call[1]["all_logits"] and call[1]["logits"] for call in verify))
        self.assertEqual(result["token_ids"], [5, 6, 8, 1, 3, 4, 6, 8])
        live = prompt + result["token_ids"][:-1]
        for model in (teacher, draft):
            for state in model.cache:
                self.assertEqual(state.offset, len(live))
                self.assertEqual(state.tail[: state.offset], live)
        rounds = [e for e in events if e["phase"] == "speculative_round_done"]
        self.assertEqual(rounds[0]["accepted"], 1)
        self.assertTrue(any(e["accepted"] == 4 for e in rounds))
        self.assertEqual(len(verify), sum(e["actual_width"] for e in rounds))
        self.assertEqual(result["target_forward_calls"], len(runtime.teacher_calls))
        self.assertEqual(result["target_verification_calls"], len(rounds))
        self.assertGreater(len(verify), len(rounds))

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


class VerificationDiagnosticTests(unittest.TestCase):
    def test_sequential_eos_is_emitted_only_after_full_valid_verification(self):
        runtime, events = Runtime(), []

        def observe(event):
            events.append((event, len(runtime.teacher_calls)))

        teacher, draft = SymbolicModel(lambda p: 9), SymbolicModel(lambda p: 1)
        with runtime.injected():
            result = sut.speculative_generate(
                teacher,
                Tokenizer(),
                [0],
                draft_model=draft,
                draft_tokenizer=Tokenizer(),
                max_tokens=5,
                verification_mode="sequential",
                progress=observe,
            )
        self.assertEqual(result["token_ids"], [9])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["target_forward_calls"], 5)
        self.assertEqual(result["target_verification_calls"], 1)
        self.assertEqual([c[0] for c in runtime.teacher_calls], [[0], [1], [1], [1], [1]])
        self.assertEqual([n for e, n in events if e["phase"] == "token_done"], [5])
        for model in (teacher, draft):
            for state in model.cache:
                self.assertEqual(state.offset, 1)
                self.assertEqual(state.tail[: state.offset], [0])

    def test_sequential_validates_every_shape_and_full_packet_before_eos(self):
        class BadStep(Runtime):
            def __init__(self, failure, step):
                super().__init__()
                self.failure, self.step = failure, step

            def forward(self, model, inputs, cache, **options):
                scores = super().forward(model, inputs, cache, **options)
                if len(self.teacher_calls) == self.step:
                    if isinstance(self.failure, tuple):
                        scores.shape = self.failure
                    else:
                        scores.ids = [self.failure]
                return scores

        for failure in ((1, 2, 10), (1, 1, 9), (2, 1, 10), 10, -1, True, 2.0):
            for step in (1, 3, 5):
                with self.subTest(failure=failure, step=step):
                    runtime, events = BadStep(failure, step), []
                    teacher, draft = SymbolicModel(lambda p: 9), SymbolicModel(lambda p: 1)
                    with runtime.injected(), self.assertRaises(ValueError):
                        sut.speculative_generate(
                            teacher,
                            Tokenizer(),
                            [0],
                            draft_model=draft,
                            draft_tokenizer=Tokenizer(),
                            max_tokens=5,
                            draft_tokens=4,
                            verification_mode="sequential",
                            progress=events.append,
                        )
                    # Shape is checked immediately; packet IDs only after all forwards.
                    self.assertEqual(
                        len(runtime.teacher_calls), step if isinstance(failure, tuple) else 5
                    )
                    self.assertFalse(
                        any(e["phase"] in ("token_done", "speculative_round_done") for e in events)
                    )
                    self.assertTrue(all(not s.trim_calls for s in teacher.cache))

    def test_sequential_checks_growth_each_step_not_only_total_width(self):
        class BalancedWrongGrowth(Runtime):
            def forward(self, model, inputs, cache, **options):
                scores = super().forward(model, inputs, cache, **options)
                for state in cache:
                    if len(self.teacher_calls) == 1:
                        state.consume([7])
                    elif len(self.teacher_calls) == 2:
                        state.trim(1)
                return scores

        runtime, events = BalancedWrongGrowth(), []
        with runtime.injected(), self.assertRaisesRegex(ValueError, "Teacher forward cache growth"):
            sut.speculative_generate(
                SymbolicModel(lambda p: 1),
                Tokenizer(),
                [0],
                draft_model=SymbolicModel(lambda p: 1),
                draft_tokenizer=Tokenizer(),
                max_tokens=5,
                verification_mode="sequential",
                progress=events.append,
            )
        self.assertEqual(len(runtime.teacher_calls), 1)
        self.assertFalse(any(e["phase"] == "token_done" for e in events))

    def test_exact_numeric_fixture_golden_vectors_and_live_cache(self):
        class Scores:
            def __init__(self, rows):
                self.rows = rows
                self.shape = (1, len(rows), 10)

            def __getitem__(self, key):
                return Scores([self.rows[-1]])

        class NumericCPU(CPU):
            @staticmethod
            def argmax(scores, axis=-1):
                if axis != -1:
                    raise AssertionError("fixture requires an unmasked vocabulary argmax")
                return Array([[max(range(len(row)), key=row.__getitem__) for row in scores.rows]])

        class NumericModel(SymbolicModel):
            def __call__(self, inputs, cache):
                ids = super().__call__(inputs, cache).ids
                # Exact dyadic score fixture; tie goes to the first vocabulary ID.
                return Scores([[-abs(i - token) * 0.25 for i in range(10)] for token in ids])

        self.assertEqual(
            NumericCPU.argmax(
                Scores([[0.0, 0.5, 0.5, -0.25, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])
            ).tolist(),
            [[1]],
        )
        for mode in ("batch", "sequential"):
            with self.subTest(mode=mode):
                runtime, events = Runtime(), []
                runtime.cpu = NumericCPU()

                def rule(p):
                    return (p[-1] + 1 + sum(p[:-1]) % 2) % 9

                teacher = NumericModel(rule)
                draft = NumericModel(lambda p: 8 if len(p) == 5 else rule(p))
                with runtime.injected():
                    result = sut.speculative_generate(
                        teacher,
                        Tokenizer(),
                        [0, 1, 2, 3],
                        draft_model=draft,
                        draft_tokenizer=Tokenizer(),
                        max_tokens=8,
                        progress=events.append,
                        verification_mode=mode,
                    )
                self.assertEqual(result["token_ids"], [5, 6, 8, 1, 3, 4, 6, 8])
                for model in (teacher, draft):
                    for state in model.cache:
                        self.assertEqual(state.offset, 11)
                        self.assertEqual(
                            state.tail[: state.offset], [0, 1, 2, 3, 5, 6, 8, 1, 3, 4, 6]
                        )
                self.assertIsNone(result["native_generation_tps"])


class DistributedTests(unittest.TestCase):
    def test_two_ranks_fixed_packets_and_round_order_match_single_host(self):
        self.check_two_ranks()

    def test_two_ranks_sequential_packets_follow_all_width_one_steps(self):
        self.check_two_ranks("sequential")

    def check_two_ranks(self, verification_mode="batch"):
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
                verification_mode=verification_mode,
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
        for rank, result in enumerate(results):
            self.assertEqual(result["verification_mode"], verification_mode)
            self.assertEqual(result["target_forward_calls"], len(runtimes[rank].teacher_calls))
            self.assertEqual(
                result["target_verification_calls"], result["acceptance_stats"]["rounds"]
            )
        self.assertEqual(
            [(c[0], c[2]) for c in runtimes[0].teacher_calls],
            [(c[0], c[2]) for c in runtimes[1].teacher_calls],
        )
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
            steps = proposal[2][2] + 1 if verification_mode == "sequential" else 1
            self.assertEqual(
                [e[0] for e in bus.events[start : end + 1]],
                ["packet", *(["hidden"] * steps), "packet"],
            )
            hidden = [e[2] for e in bus.events[start : end + 1] if e[0] == "hidden"]
            if verification_mode == "sequential":
                self.assertTrue(all(len(tokens) == 1 for tokens in hidden))
                self.assertEqual([t[0] for t in hidden[1:]], proposal[2][3 : 3 + proposal[2][2]])
        self.assertEqual(len(packets), 2 * results[0]["target_verification_calls"])


class AdmissionTests(unittest.TestCase):
    def test_invalid_mode_before_cache_or_import(self):
        class StringSubclass(str):
            pass

        for mode in (
            True,
            False,
            None,
            1,
            [],
            "",
            "serial",
            "Batch",
            StringSubclass("batch"),
            StringSubclass("sequential"),
        ):
            with (
                self.subTest(mode=mode),
                patch.object(sut, "_load_runtime") as load,
                self.assertRaisesRegex(ValueError, "verification_mode"),
            ):
                sut.speculative_generate(None, None, None, verification_mode=mode)
            load.assert_not_called()

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
        self.check_padding_predictions()

    def test_sequential_padding_predictions_preserve_unmasked_teacher_greedy_id(self):
        self.check_padding_predictions("sequential")

    def check_padding_predictions(self, verification_mode="batch"):
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
                    verification_mode=verification_mode,
                )
            self.assertFalse(any(e["phase"] == "token_done" for e in events))


class BoundaryMatrixTests(unittest.TestCase):
    def test_symbolic_greedy_eos_budget_rejection_cache_matrix(self):
        self.check_boundary_matrix()

    def test_sequential_prefix_sensitive_eos_budget_rejection_cache_matrix(self):
        self.check_boundary_matrix("sequential", prefix_sensitive=True)

    def check_boundary_matrix(self, verification_mode=None, prefix_sensitive=False):
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
                                history = sum(prefix) if prefix_sensitive else 0
                                return 9 if index == eos_at else (index + 3 + history) % 9

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
                                    **(
                                        {"verification_mode": verification_mode}
                                        if verification_mode
                                        else {}
                                    ),
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
                            if verification_mode == "sequential":
                                verification = [c for c in runtime.teacher_calls if c[1]["logits"]]
                                self.assertTrue(all(len(c[0]) == 1 for c in verification))
                            self.assertEqual(
                                result["target_verification_calls"],
                                result["acceptance_stats"]["rounds"],
                            )
                            self.assertLessEqual(
                                result["acceptance_stats"]["accepted"],
                                result["acceptance_stats"]["proposed"],
                            )
                            cases += 1
        self.assertEqual(cases, 1284)
        print("symbolic boundary matrix cases:", cases, verification_mode or "default batch")

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
            if name in ("mlx", "numpy") or name.startswith(("mlx.", "mlx_lm", "numpy.")):
                raise AssertionError("CPU fixtures attempted a GPU/model import")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            component = load_module()
            runtime = Runtime()
            for mode in ("batch", "sequential"):
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
                        max_tokens=3,
                        verification_mode=mode,
                    )


if __name__ == "__main__":
    unittest.main()
