#!/usr/bin/env python3
"""Run only injected CPU tests, prohibit physical side effects, emit a receipt."""
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent


def install_guards():
    """Process-local guards; intentionally never removed in this CPU process."""
    import signal
    import socket
    import subprocess
    attempts = []
    def deny(label):
        def blocked(*args, **kwargs):
            attempts.append(label)
            raise AssertionError('CPU_ONLY forbidden operation: ' + label)
        return blocked
    original_import = builtins.__import__
    forbidden = {'mlx', 'mlx_lm', 'numpy', 'torch', 'transformers'}
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in forbidden:
            return deny('import:' + name)()
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    for module, names in ((subprocess, ('Popen','run','call','check_call','check_output')),
                          (signal, ('signal','raise_signal','pthread_kill')),
                          (os, ('kill','killpg','system','fork','posix_spawn','posix_spawnp')),
                          # Preserve socket's class for stdlib SSL subclass imports;
                          # the audit hook rejects socket.__new__ before creation.
                          (socket, ('create_connection','getaddrinfo'))):
        for name in names:
            if hasattr(module, name): setattr(module, name, deny(module.__name__ + '.' + name))
    def audit(event, args):
        if (event.startswith(('subprocess.', 'socket.', 'os.exec', 'os.spawn'))
                or event in ('os.system','os.fork','os.kill','os.killpg','signal.pthread_kill')):
            deny('audit:' + event)()
    sys.addaudithook(audit)
    return attempts


def main():
    sys.dont_write_bytecode = True
    began = time.monotonic()
    attempts = install_guards()
    spec = importlib.util.spec_from_file_location('test_control', HERE/'test_control.py')
    assert spec and spec.loader
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    def ids(suite):
        result = []
        for test in suite:
            result.extend(ids(test) if isinstance(test, unittest.TestSuite) else [test.id()])
        return result
    test_ids = ids(suite)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    original = HERE.parents[2]/'spikes'/'025-qwen38-overlapped-prefill'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {name:sha(HERE/name) for name in ('fixture.py','harness.py','test_control.py','run_cpu_tests.py')}
    originals = {name:sha(original/name) for name in ('remote_actor.py','pressure_guard.py')}
    tdd = []
    for number in range(1,19):
        red = HERE/('tdd-%02d-red.txt' % number)
        final = HERE/('tdd-%02d-green-final.txt' % number)
        green = final if final.exists() else HERE/('tdd-%02d-green.txt' % number)
        tdd.append(dict(cycle=number, red=red.name, red_sha256=sha(red),
                        red_failed='FAILED (failures=1)' in red.read_text(),
                        green=green.name, green_sha256=sha(green),
                        green_passed=green.read_text().rstrip().endswith('OK')))
    ok = result.wasSuccessful() and not attempts and result.testsRun == len(test_ids)
    receipt = dict(schema='wired-policy-control-cpu-receipt-v1', classification='CONTROL_NOT_MODEL_GATE',
                   scope='CPU_ONLY_TESTS_NOT_PHYSICAL_APPROVAL', ok=ok, tests_run=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors), test_ids=test_ids,
                   forbidden_attempts=attempts, physical_executed=False, mlx_imported=False,
                   source_sha256=hashes, original_sha256=originals,
                   child_source_registry=['fixture.py','remote_actor.py','pressure_guard.py'],
                   original_sources_are_external_admission=False,
                   tdd_command='PYTHONDONTWRITEBYTECODE=1 python3 -B test_control.py',
                   cpu_command='PYTHONDONTWRITEBYTECODE=1 python3 -B run_cpu_tests.py',
                   tdd_cycles=tdd, supplemental_regression_tests=7,
                   order_per_host=['recommended','unwired','unwired','recommended'],
                   host_order=['local_M2','remote_M4'], elapsed_seconds=time.monotonic()-began,
                   python=sys.version, executable=sys.executable,
                   limits=dict(retained_bytes=64*1024**2,active_cap_bytes=192*1024**2,
                               child_deadline_seconds=30,actor_deadline_seconds=45),
                   pending=['independent_read_only_review','parent_admission','parent_physical_runs'])
    (HERE/'cpu-receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(cpu_receipt='cpu-receipt.json', ok=ok, tests_run=result.testsRun,
                         forbidden_attempts=attempts)), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
