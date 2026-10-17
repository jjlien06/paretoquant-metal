"""Guarded v2 verification; original v1 evidence remains untouched."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import unittest

HERE = Path(__file__).resolve().parent

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError('Missing exact CPU verification module loader')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def install_guards():
    return load(HERE.parent / 'memory-policy-control-v1/run_cpu_tests.py',
                'base_cpu_guard').install_guards()

def main():
    import time
    run_id = 'cpu-' + str(time.time_ns())
    sys.dont_write_bytecode = True
    base = HERE.parent / 'memory-policy-control-v1'
    attempts = install_guards()
    module = load(HERE / 'test_control.py', 'control_v2_tests')
    tests = unittest.defaultTestLoader.loadTestsFromModule(module)
    with (HERE / (run_id + '-output.txt')).open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    passed = result.wasSuccessful() and not attempts
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    original = HERE.parents[2] / 'spikes/025-qwen38-overlapped-prefill'
    proof = dict(schema='wired-policy-control-v2-cpu-verification', passed=passed,
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        forbidden_attempts=attempts, hardware_executed=False,
        source_sha256={n: sha(HERE/n) for n in ['fixture.py', 'harness.py', 'test_control.py', 'run_cpu_tests.py']},
        guard_sha256=sha(base / 'run_cpu_tests.py'),
        original_sha256={n: sha(original/n) for n in ['remote_actor.py','pressure_guard.py']},
        correction='Use installed MLX single-argument advisory API, not unsupported relaxed=False. Structural64MiB retention and observed192MiB cap are not a strict allocator ceiling. Pressure1 watchdog unchanged.',
        red='api-red.txt and parent terminal regression show the original TypeError; v1 preserved.',
        limits='CPU only; physical review and bounded execution remain pending.')
    proof['output_log'] = run_id + '-output.txt'
    proof['receipt'] = run_id + '-verification.json'
    with (HERE / proof['receipt']).open('x') as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps(proof, indent=2))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
