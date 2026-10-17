"""Additional independent stdlib CPU probes; frozen sources remain unchanged."""
import builtins
import concurrent.futures
import hashlib
import importlib.util
import json
import pathlib
import queue
import random
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

ROOT = pathlib.Path('/Users/jeremylien/.hermes/cache/scratch/paretoquant-dual-host/acceleration-32b')
FROZEN = ROOT / 'speculative-review-1791219843769933000'
TEST = ROOT / 'speculative-independent-review-mirror/008-speculative-decode/test_speculative_decode.py'
sys.dont_write_bytecode = True
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'mlx' or name.startswith(('mlx.', 'mlx_lm')):
        raise AssertionError('GPU/model import forbidden')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('frozen_cpu_doubles', TEST)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
sut = t.sut
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN.iterdir() if p.is_file()}

# Prefix-sensitive random differential tests, including all retained cache prefixes.
rng = random.Random(921732)
fuzz_cases = 0
for case in range(400):
    salt, dsalt = rng.randrange(100000), rng.randrange(100000)
    prompt = [rng.randrange(9) for _ in range(rng.randrange(1, 20))]
    budget, k = rng.randrange(1, 33), rng.choice((1, 2, 4, 8))
    def rule(prefix, seed=salt):
        v = seed
        for token in prefix:
            v = (v * 31 + token + 1) % 1000003
        return v % 10
    def drule(prefix, seed=dsalt):
        return rule(prefix) if (sum(prefix) + seed) % 3 else (rule(prefix) + 1) % 10
    expected = []
    for _ in range(budget):
        expected.append(rule(prompt + expected))
        if expected[-1] == 9:
            break
    runtime = t.Runtime()
    teacher, draft = t.SymbolicModel(rule), t.SymbolicModel(drule)
    events = []
    with runtime.injected():
        out = sut.speculative_generate(teacher, t.Tokenizer(), prompt,
            draft_model=draft, draft_tokenizer=t.Tokenizer(), max_tokens=budget,
            draft_tokens=k, prefill_step_size=rng.randrange(1, 10), progress=events.append)
    assert out['token_ids'] == expected, (case, out, expected)
    assert out['finish_reason'] == ('stop' if expected[-1] == 9 else 'length')
    live = prompt + expected[:-1]
    for model in (teacher, draft):
        assert all(state.offset == len(live) and state.tail[:state.offset] == live for state in model.cache)
    rounds = [e for e in events if e['phase'] == 'speculative_round_done']
    assert out['acceptance_stats'] == {
        'rounds': len(rounds), 'proposed': sum(e['proposed'] for e in rounds),
        'accepted': sum(e['accepted'] for e in rounds)}
    assert sum(e['emitted'] for e in rounds) == len(expected)
    assert out['target_forward_calls'] == len(runtime.teacher_calls)
    assert out['target_verification_calls'] == len(rounds)
    assert out['draft_calls'] == len(draft.calls)
    assert out['native_generation_tps'] is None
    assert 0 <= out['first_token_seconds'] <= out['wall_seconds']
    fuzz_cases += 1

# Stronger wire schedule: every send blocks until the peer has received it.
class Bus:
    def __init__(self):
        self.q = {(kind, rank): queue.Queue() for kind in ('packet', 'hidden') for rank in (0, 1)}
        self.events = []
        self.lock = threading.Lock()
    def send(self, kind, rank, data):
        ack = threading.Event()
        with self.lock:
            self.events.append((kind, rank, data))
        self.q[kind, rank].put((data, ack))
        if not ack.wait(3):
            raise AssertionError('blocking send deadline exceeded')
        return data
    def recv(self, kind, peer):
        data, ack = self.q[kind, peer].get(timeout=3)
        ack.set()
        return data
class Transport:
    def __init__(self, bus, rank):
        self.bus, self.rank = bus, rank
    def send(self, data, peer, **kwargs):
        assert peer == 1 - self.rank and kwargs['stream'] == 'cpu'
        return self.bus.send('packet', self.rank, data)
    def recv(self, shape, dtype, peer, **kwargs):
        assert peer == 1 - self.rank and kwargs['stream'] == 'cpu' and dtype == 'int32'
        data = self.bus.recv('packet', peer)
        assert data.shape == shape
        return data
class PipeRuntime(t.Runtime):
    def __init__(self, bus, rank):
        super().__init__()
        self.bus, self.rank = bus, rank
        self.cpu.distributed = Transport(bus, rank)
    def forward(self, model, inputs, cache, **options):
        if self.rank == 1:
            self.teacher_calls.append((list(inputs.data[0]), dict(options), cache[0].offset))
            model.core(inputs, cache)
            self.bus.send('hidden', 1, list(inputs.data[0]))
            return None
        assert self.bus.recv('hidden', 1) == inputs.data[0]
        return super().forward(model, inputs, cache, **options)
wire_cases = 0
for k in (1, 2, 4, 8):
    for budget in (1, 2, k + 1, k + 3):
        for eos_at in (None, 0, k):
            bus = Bus()
            prompt = [0, 1, 2]
            def rule(prefix):
                index = len(prefix) - len(prompt)
                return 9 if index == eos_at else (sum(prefix) + len(prefix)) % 9
            def drule(prefix):
                return (rule(prefix) + 1) % 9 if len(prefix) % 4 == 0 else rule(prefix)
            teachers = [t.SymbolicModel(rule), t.SymbolicModel(rule)]
            draft = t.SymbolicModel(drule)
            for rank, teacher in enumerate(teachers):
                teacher.args.num_hidden_layers = 64
                teacher.model = t.SymbolicCore(teacher, rank, 2)
            runtimes = [PipeRuntime(bus, rank) for rank in (0, 1)]
            local = threading.local()
            def run(rank):
                runtime = runtimes[rank]
                local.runtime = runtime
                return sut.speculative_generate(teachers[rank], t.Tokenizer(), prompt,
                    draft_model=draft if rank else None, draft_tokenizer=t.Tokenizer(),
                    group=SimpleNamespace(rank=lambda: rank, size=lambda: 2),
                    max_tokens=budget, draft_tokens=k, prefill_step_size=1)
            with patch.object(sut, '_load_runtime', side_effect=lambda: (
                    local.runtime.cpu, local.runtime.make_cache, local.runtime.phase)):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(run, rank) for rank in (0, 1)]
                    results = [f.result(timeout=8) for f in futures]
            expected = []
            for _ in range(budget):
                expected.append(rule(prompt + expected))
                if expected[-1] == 9:
                    break
            assert results[0]['token_ids'] == results[1]['token_ids'] == expected
            assert results[0]['acceptance_stats'] == results[1]['acceptance_stats']
            assert results[0]['draft_calls'] == 0
            live = prompt + expected[:-1]
            for model in (*teachers, draft):
                assert all(c.offset == len(live) and c.tail[:c.offset] == live for c in model.cache)
            packets = [(rank, data.tolist()) for kind, rank, data in bus.events if kind == 'packet']
            assert len(packets) == 2 * results[0]['target_verification_calls']
            for index in range(0, len(packets), 2):
                proposal, answer = packets[index:index + 2]
                assert (proposal[0], answer[0]) == (1, 0)
                assert len(proposal[1]) == len(answer[1]) == k + 4
                assert proposal[1][:2] == answer[1][:2]
                assert answer[1][2] == proposal[1][2] + 1
            wire_cases += 1

# Invalid context/prompt rejected before allocating caches or performing work.
allocation_cases = 0
for options in ({'context_limit': 2, 'max_tokens': 1}, {'context_limit': 32.0, 'max_tokens': 1},
                {'context_limit': 32769}, {'max_tokens': True}, {'prefill_step_size': 0}):
    runtime = t.Runtime()
    teacher, draft = t.SymbolicModel(lambda p: 1), t.SymbolicModel(lambda p: 1)
    allocations = []
    def make_cache(model):
        allocations.append(model)
        return runtime.make_cache(model)
    try:
        with patch.object(sut, '_load_runtime', return_value=(runtime.cpu, make_cache, runtime.phase)):
            sut.speculative_generate(teacher, t.Tokenizer(), [0, 1, 2], draft_model=draft,
                draft_tokenizer=t.Tokenizer(), **options)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid admission accepted: ' + repr(options))
    assert not allocations and not runtime.teacher_calls
    allocation_cases += 1

# First-token timing must include synchronous draft catch-up, not just teacher verification.
runtime = t.Runtime()
clock = [100.0]
class TimedDraft(t.SymbolicModel):
    def core(self, inputs, cache):
        value = super().core(inputs, cache)
        clock[0] += 7.0
        return value
teacher, draft = t.SymbolicModel(lambda p: 1), TimedDraft(lambda p: 1)
with runtime.injected(), patch.object(sut.time, 'perf_counter', side_effect=lambda: clock[0]):
    out = sut.speculative_generate(teacher, t.Tokenizer(), [0], draft_model=draft,
        draft_tokenizer=t.Tokenizer(), max_tokens=1)
assert out['first_token_seconds'] == out['wall_seconds'] == 7.0
assert out['draft_calls'] == out['target_forward_calls'] == 1
assert out['acceptance_stats'] == {'rounds': 1, 'proposed': 0, 'accepted': 0}

after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN.iterdir() if p.is_file()}
assert before == after
assert not any(name == 'mlx' or name.startswith(('mlx.', 'mlx_lm')) for name in sys.modules)
report = {'passed': True, 'prefix_sensitive_fuzz_cases': fuzz_cases,
    'blocking_two_rank_wire_cases': wire_cases, 'preallocation_admission_cases': allocation_cases,
    'synchronous_catchup_timing_cases': 1, 'sources_unchanged': True, 'gpu_imports': []}
(ROOT / 'speculative-independent-probe-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
