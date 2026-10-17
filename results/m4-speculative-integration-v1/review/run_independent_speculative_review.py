"""Frozen-source review harness: stdlib CPU tests, no source modifications."""
import builtins
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import unittest

ROOT = pathlib.Path('/Users/jeremylien/.hermes/cache/scratch/paretoquant-dual-host/acceleration-32b')
FROZEN = ROOT / 'speculative-review-1791219843769933000'
MIRROR = ROOT / 'speculative-independent-review-mirror'
SPIKE = MIRROR / '008-speculative-decode'
PIPELINE = MIRROR / '007-dual-host-pipeline'
SPIKE.mkdir(parents=True, exist_ok=True)
PIPELINE.mkdir(parents=True, exist_ok=True)

def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FROZEN.iterdir()) if p.is_file()}

before = hashes()
admission = json.loads((FROZEN / 'admission.json').read_text())
assert all(before[name] == expected for name, expected in admission['files'].items())
for name in ('speculative_decode.py', 'test_speculative_decode.py'):
    shutil.copyfile(FROZEN / name, SPIKE / name)
shutil.copyfile(FROZEN / 'phased_decode.py', PIPELINE / 'phased_decode.py')
assert hashlib.sha256((PIPELINE / 'phased_decode.py').read_bytes()).hexdigest() == before['phased_decode.py']
for name in ('speculative_decode.py', 'test_speculative_decode.py'):
    assert hashlib.sha256((SPIKE / name).read_bytes()).hexdigest() == before[name]

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'mlx' or name.startswith(('mlx.', 'mlx_lm')):
        raise AssertionError('GPU/model import forbidden during independent CPU review: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('independent_cpu_tests', SPIKE / 'test_speculative_decode.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)

class CountingResult(unittest.TextTestResult):
    subtest_count = 0
    def addSubTest(self, test, subtest, err):
        self.subtest_count += 1
        super().addSubTest(test, subtest, err)

suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
result = unittest.TextTestRunner(verbosity=2, resultclass=CountingResult).run(suite)
after = hashes()
assert before == after, 'Frozen sources changed during review'
assert not any(name == 'mlx' or name.startswith(('mlx.', 'mlx_lm')) for name in sys.modules)
report = {
    'tests_run': result.testsRun,
    'subtests_observed': result.subtest_count,
    'failures': len(result.failures),
    'errors': len(result.errors),
    'skipped': len(result.skipped),
    'successful': result.wasSuccessful(),
    'source_hashes_before': before,
    'source_hashes_after': after,
    'admission_hashes_match': True,
    'sources_unchanged': before == after,
    'gpu_imports': [],
    'scope': 'Frozen component and frozen stdlib CPU symbolic tests; no MLX/GPU, network, SSH, installs, credentials, live source edits or Git writes.'
}
(ROOT / 'speculative-independent-cpu-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if not result.wasSuccessful():
    sys.exit(1)
