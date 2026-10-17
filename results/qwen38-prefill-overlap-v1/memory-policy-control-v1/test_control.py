"""CPU-only vertical tests; no MLX or physical execution."""
import importlib.util
import pathlib
import sys
import types
import unittest

HERE = pathlib.Path(__file__).resolve().parent

def load(name):
    path = HERE / (name + '.py')
    if not path.exists():
        return types.SimpleNamespace()
    spec = importlib.util.spec_from_file_location('control_' + name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

f = load('fixture')

class Bounds(unittest.TestCase):
    def test_exact_modes_and_bounded_configuration(self):
        self.assertTrue(hasattr(f, 'validate_config'), 'missing validated bounded configuration')
        self.assertEqual(f.validate_config('recommended')['total_bytes'], 64 * 1024**2)
        self.assertEqual(f.validate_config('unwired')['active_cap_bytes'], 192 * 1024**2)
        for kwargs in ({'mode': True}, {'mode': 'zero'}, {'mode': 'unwired', 'total_bytes': True},
                       {'mode': 'unwired', 'total_bytes': 64 * 1024**2 + 4},
                       {'mode': 'unwired', 'deadline': True}, {'mode': 'unwired', 'deadline': 31},
                       {'mode': 'unwired', 'deadline': float('nan')},
                       {'mode': 'unwired', 'chunk_bytes': 2 * 1024**2},
                       {'mode': 'unwired', 'chunk_bytes': 4},
                       {'mode': 'unwired', 'total_bytes': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                f.validate_config(**kwargs)

    def test_host_payload_budget_includes_retained_input_lifetimes(self):
        config = f.validate_config('recommended')
        self.assertEqual(config['host_payload_cap_bytes'], 136*1024**2)
        self.assertLessEqual(config['iterations'], 128)
        self.assertLessEqual(config['active_cap_bytes'], 192*1024**2)

VM = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
      'Pages free: 100.\nPages active: 200.\nPages inactive: 300.\n'
      'Pages wired down: 400.\nPages occupied by compressor: 5.\n'
      'Swapins: 0.\nSwapouts: 0.\n')

def sample():
    return dict(vm_stat=VM, pressure='1\n', mem_total='17179869184\n', rss_kib='1024\n')

class Samples(unittest.TestCase):
    def test_strict_sampler_requires_pressure_one_and_vm_counters(self):
        self.assertTrue(hasattr(f, 'parse_sample'), 'missing fail-closed sampler parser')
        parsed = f.parse_sample(sample())
        self.assertEqual(parsed['pressure'], 1)
        self.assertEqual(parsed['vm']['Pages wired down'], 400)
        self.assertEqual(parsed['page_size'], 16384)
        self.assertEqual(parsed['rss_bytes'], 1048576)
        mutations = [{'pressure': x} for x in (True, 1, '2\n', '4\n', '1', '1\nextra')]
        mutations += [{'vm_stat': x} for x in ('', VM + 'Pages free: 2.\n', VM.replace('400.', '-1.'),
                                            VM.replace('Swapouts: 0.\n', ''), VM + 'garbage\n')]
        mutations += [{'rss_kib': '-1\n'}, {'mem_total': True}, {'vm_stat': 'x' * 65537}]
        for mutation in mutations:
            with self.subTest(mutation=repr(mutation)[:80]), self.assertRaises(ValueError):
                f.parse_sample(dict(sample(), **mutation))

    def test_vm_field_population_is_bounded_before_retention(self):
        extra = ''.join('Counter '+chr(65+i//26)+chr(65+i%26)+': 1.\n' for i in range(65))
        with self.assertRaises(ValueError):
            f.parse_sample(dict(sample(),vm_stat=VM+extra))

class Token:
    def __init__(self, data):
        self.data = data

class Fake:
    def __init__(self):
        import weakref
        self.arrays = weakref.WeakSet()
        self.limits = dict(wired=37, memory=999, cache=42)
        self.calls = []
        self.peak = 0
    def describe(self):
        return dict(device=dict(max_recommended_working_set_size=12713115648),
                    versions=dict(mlx='FAKE_CPU_ONLY', python=sys.version))
    def _set(self, key, value):
        self.calls.append((key, value))
        old = self.limits[key]
        self.limits[key] = value
        return old
    def set_wired_limit(self, value): return self._set('wired', value)
    def set_memory_limit(self, value): return self._set('memory', value)
    def set_cache_limit(self, value): return self._set('cache', value)
    def memory(self):
        active = sum(len(a.data) for a in self.arrays)
        self.peak = max(active, self.peak)
        return dict(active=active, cache=0, peak=self.peak)
    def from_u32_bytes(self, data):
        a = Token(data)
        self.arrays.add(a)
        return a
    def evaluate(self, a): self.memory()
    def to_u32_bytes(self, a): return a.data
    def synchronize(self): self.calls.append(('sync',))
    def clear_cache(self): self.calls.append(('clear',))

class Runs(unittest.TestCase):
    def test_small_roundtrip_records_actual_policy_and_restores(self):
        self.assertTrue(hasattr(f, 'run_control'), 'missing injected bounded control')
        for mode in ('recommended', 'unwired'):
            with self.subTest(mode=mode):
                b = Fake()
                r = f.run_control(mode, backend=b, sampler=sample, total_bytes=64, chunk_bytes=16)
                self.assertTrue(r['ok'], r)
                self.assertEqual(r['classification'], 'CONTROL_NOT_MODEL_GATE')
                target = 12713115648 if mode == 'recommended' else 0
                self.assertEqual(r['policy']['requested_bytes'], target)
                self.assertEqual(r['policy']['previous_bytes'], 37)
                self.assertEqual(r['policy']['readback_bytes'], target)
                self.assertEqual(r['policy']['restored_readback_bytes'], 37)
                self.assertEqual(b.limits, dict(wired=37, memory=999, cache=42))
                self.assertEqual(r['retained_bytes'], 64)
                self.assertEqual(len(r['arrays']), 4)
                self.assertEqual([s['stage'] for s in r['samples']], ['before'] + ['during']*4 + ['after'])
                for item in r['arrays']:
                    self.assertEqual(item['expected'], item['observed'])
                self.assertEqual(b.memory()['active'], 0)
                self.assertEqual(r['memory'][-1]['stage'], 'after_cleanup')
                self.assertEqual(r['memory'][-1]['active'], 0)

    def test_retained_payload_is_revalidated_after_last_allocation(self):
        class CorruptRetained(Fake):
            def from_u32_bytes(self, data):
                for old in self.arrays:
                    old.data = b'X' * len(old.data)
                return super().from_u32_bytes(data)
        b = CorruptRetained()
        r = f.run_control('unwired', backend=b, sampler=sample, total_bytes=32, chunk_bytes=16)
        self.assertFalse(r['ok'], 'retained earlier payload corruption escaped')
        self.assertEqual(b.limits['wired'], 37)
        self.assertEqual(b.memory()['active'], 0)

    def test_bool_policy_readback_is_not_integer_zero(self):
        class BoolReadback(Fake):
            def set_wired_limit(self, value):
                old = super().set_wired_limit(value)
                return False if old == 0 else old
        r = f.run_control('unwired', backend=BoolReadback(), sampler=sample, total_bytes=16, chunk_bytes=16)
        self.assertFalse(r['ok'], 'False incorrectly accepted as wired readback 0')

    def test_cleanup_residual_active_memory_fails_closed(self):
        class Leaky(Fake):
            def from_u32_bytes(self, data):
                self.leak = super().from_u32_bytes(data)
                return self.leak
        b = Leaky()
        r = f.run_control('unwired', backend=b, sampler=sample, total_bytes=16, chunk_bytes=16)
        self.assertFalse(r['ok'], 'nonzero active memory escaped cleanup verification')
        self.assertEqual(b.limits['wired'], 37)

class PhysicalBoundary(unittest.TestCase):
    def test_physical_cli_has_no_payload_size_escape(self):
        self.assertTrue(hasattr(f, 'parse_args'), 'missing fixed physical CLI boundary')
        args = f.parse_args(['--mode', 'unwired', '--nonce', 'a'*32, '--host-role', 'local'])
        self.assertEqual(args.mode, 'unwired')
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            f.parse_args(['--mode', 'unwired', '--nonce', 'a'*32, '--host-role', 'local', '--total-bytes', '16'])
        with self.assertRaises(ValueError):
            f.parse_args(['--mode', 'unwired', '--nonce', 'not-a-uuid', '--host-role', 'local'])

    def test_adapter_enforces_strict_allocator_and_reports_device(self):
        self.assertTrue(hasattr(f, 'make_backend'), 'missing injectable MLX adapter')
        class Mx:
            def set_memory_limit(self, value, *, relaxed=True):
                self.received = (value, relaxed)
                return 123
            def device_info(self): return {'max_recommended_working_set_size': 456}
        mx = Mx()
        backend = f.make_backend(mx, None, {'mlx': 'FAKE'})
        self.assertEqual(backend.set_memory_limit(192*1024**2), 123)
        self.assertEqual(mx.received, (192*1024**2, False))
        self.assertEqual(backend.describe()['device']['max_recommended_working_set_size'], 456)

    def test_entrypoint_uses_fixed_payload_and_injected_runtime_only(self):
        self.assertTrue(hasattr(f, 'main'), 'missing physical entrypoint')
        import contextlib
        import io
        import json
        seen = {}
        def runner(mode, **kwargs):
            seen.update(kwargs)
            return dict(ok=True, classification='CONTROL_NOT_MODEL_GATE')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = f.main(['--mode', 'unwired', '--nonce', 'a'*32, '--host-role', 'local'],
                          runtime_factory=lambda: (Fake(), sample), runner=runner)
        self.assertEqual(code, 0)
        self.assertEqual(seen['total_bytes'], 64*1024**2)
        self.assertEqual(seen['chunk_bytes'], 1024**2)
        self.assertLessEqual(seen['deadline'], 30)
        self.assertEqual(json.loads(output.getvalue())['binding']['nonce'], 'a'*32)

class Harness(unittest.TestCase):
    def test_stage_exact_three_sources_against_external_expected_hashes(self):
        import hashlib
        import tempfile
        h = load('harness')
        self.assertTrue(hasattr(h, 'stage'), 'missing source-bound staging helper')
        original = HERE.parents[2] / 'spikes' / '025-qwen38-overlapped-prefill'
        expected = {n: hashlib.sha256((original/n).read_bytes()).hexdigest()
                    for n in ('remote_actor.py', 'pressure_guard.py')}
        fixture_hash = hashlib.sha256((HERE/'fixture.py').read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = pathlib.Path(tmp)/'sources'
            hashes = h.stage(root, original_root=original, expected_original_hashes=expected,
                             expected_fixture_sha256=fixture_hash)
            self.assertEqual(set(hashes), {'fixture.py', 'remote_actor.py', 'pressure_guard.py'})
            self.assertEqual({p.name for p in root.iterdir()}, set(hashes))
            for name in expected:
                self.assertEqual((root/name).read_bytes(), (original/name).read_bytes())
            with self.assertRaises(ValueError):
                h.stage(pathlib.Path(tmp)/'bad', original_root=original,
                        expected_original_hashes=dict(expected, **{'pressure_guard.py': '0'*64}),
                        expected_fixture_sha256=fixture_hash)

    def test_owned_call_scopes_sources_and_always_requires_pressure(self):
        import tempfile
        h = load('harness')
        self.assertTrue(hasattr(h, '_invoke_owned'), 'missing guarded original run_owned seam')
        calls = []
        original_sources = ('ORIGINAL',)
        actor = types.SimpleNamespace(SOURCES=original_sources)
        def owned(argv, job, **kwargs):
            calls.append((argv, job, kwargs, actor.SOURCES))
            return {'reason': 'TEST_ONLY'}
        actor.run_owned = owned
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = pathlib.Path(tmp)
            result = h._invoke_owned(actor, root, root/'job', mode='unwired', host_role='remote',
                                     nonce='a'*32, expected_hashes={n: '0'*64 for n in h.SOURCES}, deadline=45)
        self.assertEqual(result['reason'], 'TEST_ONLY')
        self.assertIs(actor.SOURCES, original_sources)
        argv, job, kwargs, sources = calls[0]
        self.assertEqual(sources, h.SOURCES)
        self.assertTrue(kwargs['require_normal_pressure'])
        self.assertEqual(kwargs['host_role'], 'remote')
        self.assertEqual(kwargs['actor_role'], 'rank1')
        self.assertEqual(kwargs['nonce'], 'a'*32)
        self.assertEqual(kwargs['deadline'], 45)
        self.assertEqual(kwargs['source_root'], root)
        self.assertEqual(kwargs['expected_hashes'], {n:'0'*64 for n in h.SOURCES})
        self.assertIn(str(root/'fixture.py'), argv)
        actor.run_owned = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('TEST'))
        with self.assertRaises(RuntimeError):
            h._invoke_owned(actor, root, root/'job', mode='recommended', host_role='local',
                            nonce='b'*32, expected_hashes={n:'0'*64 for n in h.SOURCES}, deadline=45)
        self.assertIs(actor.SOURCES, original_sources)

    def test_admission_binds_external_originals_hashes_nonce_role_and_deadline(self):
        import hashlib
        import tempfile
        h = load('harness')
        self.assertTrue(hasattr(h, '_admit'), 'missing external source and identity admission')
        original = HERE.parents[2] / 'spikes' / '025-qwen38-overlapped-prefill'
        hashes = {n: hashlib.sha256((original/n).read_bytes()).hexdigest() for n in h.ORIGINALS}
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = pathlib.Path(tmp)/'sources'
            sources = h.stage(root, original_root=original, expected_original_hashes=hashes,
                              expected_fixture_sha256=hashlib.sha256((HERE/'fixture.py').read_bytes()).hexdigest())
            admitted = dict(schema='wired-policy-control-admission-v1', nonce='a'*32, mode='unwired',
                            host_role='local', source_sha256=sources, original_sha256=hashes,
                            harness_sha256=hashlib.sha256((HERE/'harness.py').read_bytes()).hexdigest())
            kwargs = dict(mode='unwired', host_role='local', nonce='a'*32, deadline=45)
            payloads = h._admit(root, admitted, **kwargs)
            self.assertEqual(set(payloads), set(h.SOURCES))
            for mutation in ({'nonce':'b'*32}, {'host_role':'remote'}, {'mode':'recommended'},
                             {'harness_sha256':'0'*64}, {'original_sha256':dict(hashes, **{'pressure_guard.py':'0'*64})}):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    h._admit(root, dict(admitted, **mutation), **kwargs)
            for deadline in (True, 46, float('nan'), 0):
                with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                    h._admit(root, admitted, **dict(kwargs, deadline=deadline))
            (root/'fixture.py').write_bytes(b'changed')
            with self.assertRaises(ValueError): h._admit(root, admitted, **kwargs)

    def test_execute_loads_byte_exact_originals_and_restores_import_scope_on_error(self):
        import hashlib
        import tempfile
        from unittest.mock import patch
        h = load('harness')
        self.assertTrue(hasattr(h, 'execute'), 'missing parent physical executor')
        original = HERE.parents[2] / 'spikes' / '025-qwen38-overlapped-prefill'
        hashes = {n:hashlib.sha256((original/n).read_bytes()).hexdigest() for n in h.ORIGINALS}
        before = sys.modules.get('pressure_guard')
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = pathlib.Path(tmp)/'sources'
            sources = h.stage(root, original_root=original, expected_original_hashes=hashes,
                              expected_fixture_sha256=hashlib.sha256((HERE/'fixture.py').read_bytes()).hexdigest())
            admitted = dict(schema='wired-policy-control-admission-v1', nonce='a'*32, mode='unwired',
                            host_role='local', source_sha256=sources, original_sha256=hashes,
                            harness_sha256=hashlib.sha256((HERE/'harness.py').read_bytes()).hexdigest())
            def do_not_launch(actor, *args, **kwargs):
                self.assertEqual(actor.run_owned.__code__.co_filename, str(root/'remote_actor.py'))
                self.assertEqual(sys.modules['pressure_guard'].__file__, str(root/'pressure_guard.py'))
                self.assertEqual(actor.SOURCES[0], 'cpu_only_runner.py')
                raise RuntimeError('TEST_STOP_BEFORE_ANY_PHYSICAL_EXECUTION')
            with patch.object(h, '_invoke_owned', do_not_launch), self.assertRaises(RuntimeError):
                h.execute(root, pathlib.Path(tmp)/'job', mode='unwired', host_role='local',
                          nonce='a'*32, admission=admitted)
        self.assertIs(sys.modules.get('pressure_guard'), before)

    def test_collection_uses_original_pressure_verifier_and_independent_records(self):
        import json
        import tempfile
        import copy
        h = load('harness')
        self.assertTrue(hasattr(h, '_collect'), 'missing readback and original pressure verification')
        # Synthetic TEST-only receipt constructed through the real guard codec.
        guard = types.ModuleType('TEST_guard')
        original = HERE.parents[2]/'spikes'/'025-qwen38-overlapped-prefill'/'pressure_guard.py'
        exec(compile(original.read_bytes(), str(original), 'exec'), guard.__dict__)
        binding = dict(host_role='local', actor_role='rank0', nonce='a'*32, actor_pid=11, child_pid=22)
        now = [100.01]
        pg = guard.PressureGuard(True, sampler=lambda:1, clock=lambda:now[0], began=100.)
        for stage, when in [('pre_spawn',100.01),('owned_child',100.15),('post_child',100.4)]:
            now[0] = when
            pg.sample(stage, force=True)
        result = dict(binding, reason='completed', require_normal_pressure=True, actor_started_monotonic=100.,
                      execution=dict(child_start_seconds=.05,gate_release_seconds=.06,child_end_seconds=.3),
                      elapsed=.45, returncode=0, log_evidence_complete=True,
                      cleanup=dict(child_reaped=True,group_gone=True,pending_os_gpu_termination=False),
                      pressure=pg.receipt(binding=binding))
        startup = {k:v for k,v in result.items() if k in ('host_role','actor_role','nonce','actor_pid','actor_started_monotonic','require_normal_pressure')}
        pids = dict(binding, actor_started_monotonic=100., child_start_seconds=.05)
        report = f.run_control('unwired', backend=Fake(), sampler=sample, total_bytes=256, chunk_bytes=4)
        report['config'] = f.validate_config('unwired')
        report['retained_bytes'] = 64*1024**2
        report['binding'] = dict(binding, mode='unwired')
        report['binding'].pop('actor_pid')
        for array in report['arrays']: array['bytes'] = 1024**2
        admitted = dict(nonce='a'*32,host_role='local',mode='unwired',source_sha256={n:'0'*64 for n in h.SOURCES})
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            jobs = pathlib.Path(tmp)
            def write():
                for name, obj in [('startup.json',startup),('pids.json',pids),('result.json',result),
                                  ('source_sha256.json',admitted['source_sha256']),('stdout',report)]:
                    (jobs/name).write_text(json.dumps(obj))
            write()
            self.assertTrue(h._collect(jobs, result, guard, admission=admitted)['ok'])
            result['pressure']['samples'][1]['level'] = 2
            write()
            with self.assertRaises(ValueError): h._collect(jobs, result, guard, admission=admitted)
            result['pressure']['samples'][1]['level'] = 1
            pids['child_pid'] = 99
            write()
            with self.assertRaises(ValueError): h._collect(jobs, result, guard, admission=admitted)
            pids['child_pid'] = 22
            report['binding']['nonce'] = 'b'*32
            write()
            with self.assertRaises(ValueError): h._collect(jobs, result, guard, admission=admitted)

    def test_parent_cli_requires_external_admission_before_executor(self):
        import tempfile
        import json
        import io
        import contextlib
        from unittest.mock import patch
        h = load('harness')
        self.assertTrue(hasattr(h, 'main'), 'missing admitted parent CLI')
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = pathlib.Path(tmp)
            argv = ['--root',str(root),'--jobs',str(root/'newjob'),'--mode','unwired',
                    '--host-role','local','--nonce','a'*32]
            called = []
            with patch.object(h, 'execute', lambda *a, **kw: called.append(kw) or {'ok':True}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError): h.main(argv)
                self.assertEqual(called, [])
                (root/'admission.json').write_text(json.dumps({'TEST_EXTERNAL_ADMISSION':True}))
                self.assertEqual(h.main(argv), 0)
            self.assertEqual(called[0]['admission'], {'TEST_EXTERNAL_ADMISSION':True})
            self.assertEqual(called[0]['deadline'], 45)

class ErrorRestoration(unittest.TestCase):
    def test_evaluation_failure_restores_and_frees(self):
        class Broken(Fake):
            def evaluate(self, a): raise RuntimeError('TEST_EVAL')
        b = Broken()
        r = f.run_control('recommended', backend=b, sampler=sample, total_bytes=16, chunk_bytes=16)
        self.assertFalse(r['ok'])
        self.assertEqual(b.limits, dict(wired=37,memory=999,cache=42))
        self.assertEqual(b.memory()['active'], 0)
        self.assertTrue(any(c['step']=='clear_cache' and c['ok'] for c in r['cleanup']))

    def test_roundtrip_error_restores_previous_nonzero_limit(self):
        class Broken(Fake):
            def to_u32_bytes(self, a): return b'changed-payload!!'
        b = Broken()
        r = f.run_control('unwired', backend=b, sampler=sample, total_bytes=16, chunk_bytes=16)
        self.assertFalse(r['ok'])
        self.assertEqual(r['policy']['restored_readback_bytes'], 37)
        self.assertEqual(b.memory()['active'], 0)

    def test_pressure_two_during_run_restores_policy(self):
        b = Fake()
        calls = [0]
        def sampler():
            calls[0] += 1
            return dict(sample(),pressure='2\n' if calls[0]==2 else '1\n')
        r = f.run_control('recommended', backend=b, sampler=sampler, total_bytes=32, chunk_bytes=16)
        self.assertFalse(r['ok'])
        self.assertEqual(b.limits, dict(wired=37,memory=999,cache=42))
        self.assertEqual(b.memory()['active'], 0)

    def test_sampler_is_required_and_abnormal_before_prevents_setters(self):
        b = Fake()
        with self.assertRaises(ValueError):
            f.run_control('unwired',backend=b,sampler=None,total_bytes=16,chunk_bytes=16)
        r = f.run_control('unwired',backend=b,sampler=lambda:dict(sample(),pressure='2\n'),total_bytes=16,chunk_bytes=16)
        self.assertFalse(r['ok'])
        self.assertFalse(any(c[0] in ('wired','memory','cache') for c in b.calls))

    def test_oversize_active_counter_fails_and_restores(self):
        class OverCap(Fake):
            def memory(self):
                m = super().memory()
                if m['active']: m['active'] = 192*1024**2 + 1
                return m
        b = OverCap()
        r = f.run_control('unwired',backend=b,sampler=sample,total_bytes=16,chunk_bytes=16)
        self.assertFalse(r['ok'])
        self.assertEqual(b.limits['wired'],37)
        self.assertEqual(b.memory()['active'],0)

    def test_deadline_stops_before_next_allocation_and_restores(self):
        b = Fake()
        now = [0.]
        def sampler():
            if b.arrays: now[0] = 31.
            return sample()
        r = f.run_control('unwired',backend=b,sampler=sampler,clock=lambda:now[0],total_bytes=32,chunk_bytes=16)
        self.assertFalse(r['ok'])
        self.assertEqual(len(r['arrays']),1)
        self.assertEqual(b.limits['wired'],37)

    def test_cleanup_sync_exception_does_not_skip_restoration(self):
        class BrokenSync(Fake):
            def synchronize(self): raise RuntimeError('TEST_SYNC')
        b = BrokenSync()
        r = f.run_control('recommended',backend=b,sampler=sample,total_bytes=16,chunk_bytes=16)
        self.assertFalse(r['ok'])
        self.assertEqual(b.limits,dict(wired=37,memory=999,cache=42))
        self.assertEqual(b.memory()['active'],0)
        self.assertTrue(any(c['step']=='restore_wired' and c['ok'] for c in r['cleanup']))

class CpuBoundary(unittest.TestCase):
    def test_only_physical_runtime_can_import_mlx_or_subprocess(self):
        import ast
        tree = ast.parse((HERE/'fixture.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'physical_runtime':
                continue
            for item in ast.walk(node):
                names = ([a.name for a in item.names] if isinstance(item,ast.Import)
                         else [item.module or ''] if isinstance(item,ast.ImportFrom) else [])
                self.assertFalse(any(n.split('.')[0] in ('mlx','subprocess','socket','signal') for n in names))
        runner = load('run_cpu_tests')
        self.assertTrue(hasattr(runner, 'install_guards'), 'missing executable guarded CPU receipt runner')

if __name__ == '__main__':
    unittest.main(verbosity=2)
