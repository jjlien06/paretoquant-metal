"""Parent-owned eight-case 64MiB allocation control; never loads a model."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path('/Users/jeremylien/Projects/paretoquant-metal')
OUT = ROOT / 'results/qwen38-prefill-overlap-v1'
FIXTURE = OUT / 'memory-policy-control-v2'
SCRATCH = Path('/Users/jeremylien/.hermes/cache/scratch')
EXPECTED = {
    'fixture.py': '4ad969f59d3b4bf9e1b0ff8076490474dc6c91dd1e96ba9509bd49c6f5a2ced8',
    'remote_actor.py': '3bacf37ba85de04fecc7dee67abccead5f7aa805bc4a4e45fbb4bf90302bf81c',
    'pressure_guard.py': '17402e9c8c043b31f42325f1e255484385ef6bd38b9e99a5a1f44c203456fb20',
    'harness.py': '0a409ca1a8e77dd554447d565db4de40c91aa4045e425fa7fe24ba6ed678f88f',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw)


def store(path, value):
    write(path, json.dumps(value, indent=2, allow_nan=False).encode())


def route_interface(text):
    match = re.search(r'^\s*interface:\s*(\S+)', text, re.M)
    if match is None:
        raise ValueError('missing route interface')
    return match[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[a-f0-9]{64}', args.review_sha256)
    lock = (SCRATCH / 'paretoquant-physical-owner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    review_path = OUT / 'memory-policy-control-independent-review-v2.json'
    review_raw = review_path.read_bytes()
    assert sha(review_raw) == args.review_sha256
    review = json.loads(review_raw)
    assert review['passed'] is True
    bindings = review.get('sourcehashes', review.get('source_sha256', {}))
    assert all(h in json.dumps(bindings) for n, h in EXPECTED.items() if n in {'fixture.py', 'harness.py'})
    payloads = {}
    for name, wanted in EXPECTED.items():
        path = FIXTURE / name if name in {'fixture.py', 'harness.py'} else ROOT / 'spikes/025-qwen38-overlapped-prefill' / name
        assert path.is_file() and not path.is_symlink()
        payloads[name] = path.read_bytes()
        assert sha(payloads[name]) == wanted
    profile = json.loads((SCRATCH / 'overlap-physical-profile-flat-1792265796214463000.json').read_text())
    ssh = ['ssh', '-T', '-i', profile['ssh_key'], '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=8', '-o', 'HostKeyAlias=' + profile['ssh_host_key_alias'], profile['ssh_target']]
    local_python = str(ROOT / '.venv/bin/python')
    remote_python = '/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python'

    def remote(script, data, timeout=25):
        proc = subprocess.run(ssh + [shlex.join([remote_python, '-B', '-c', script])], input=json.dumps(data), text=True, capture_output=True, timeout=timeout)
        if proc.returncode or len(proc.stdout.encode()) > 8 * 1024**2:
            raise RuntimeError('bounded remote control failed')
        return json.loads(proc.stdout)

    def health():
        local = int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], capture_output=True, text=True, check=True, timeout=3).stdout)
        route = subprocess.run(['route', '-n', 'get', profile['remote_address']], capture_output=True, text=True, check=True, timeout=5).stdout
        assert local == 1 and route_interface(route) == 'bridge0'
        values = remote("import json,subprocess,sys,re; x=json.load(sys.stdin); c=lambda a:subprocess.run(a,capture_output=True,text=True,check=True,timeout=5).stdout; route=c(['route','-n','get',x['peer']]); print(json.dumps({'pressure':int(c(['sysctl','-n','kern.memorystatus_vm_pressure_level'])),'interface':re.search(r'interface:\\s*(\\S+)',route)[1]}))", {'peer': profile['local_address']})
        assert values == {'pressure': 1, 'interface': 'bridge0'}
        return {'local_pressure': 1, 'remote_pressure': 1, 'both_route_interfaces': 'bridge0', 'timestamp': time.time()}

    initial_health = health()
    trial = OUT / ('memory-policy-physical-v2-' + str(time.time_ns()))
    trial.mkdir(mode=0o700)
    (trial / 'admissions').mkdir(mode=0o700)
    local_sources = trial / 'local-sources'
    local_sources.mkdir(mode=0o700)
    for name in ('fixture.py', 'remote_actor.py', 'pressure_guard.py'):
        write(local_sources / name, payloads[name])
        assert sha((local_sources / name).read_bytes()) == EXPECTED[name]
    write(trial / 'harness.py', payloads['harness.py'])
    remote_root = SCRATCH / trial.name
    stage_script = r'''import base64,json,sys,hashlib
from pathlib import Path
x=json.load(sys.stdin);r=Path(x['root']);assert r.parent==Path('/Users/jeremylien/.hermes/cache/scratch') and r.name.startswith('memory-policy-physical-v2-');r.mkdir(mode=0o700);s=r/'sources';s.mkdir(mode=0o700);(r/'admissions').mkdir(mode=0o700)
for n,b64 in x['payloads'].items():
 assert n in {'fixture.py','harness.py','remote_actor.py','pressure_guard.py'};b=base64.b64decode(b64,validate=True);assert hashlib.sha256(b).hexdigest()==x['expected'][n];p=r/n if n=='harness.py' else s/n
 with p.open('xb') as f:f.write(b)
print(json.dumps({'staged':True}))
'''
    remote(stage_script, {'root': str(remote_root), 'payloads': {n: base64.b64encode(b).decode() for n, b in payloads.items()}, 'expected': EXPECTED})
    verify_script = "import sys,json,hashlib;from pathlib import Path;x=json.load(sys.stdin);r=Path(x['root']);observed={n:hashlib.sha256((r/n if n=='harness.py' else r/'sources'/n).read_bytes()).hexdigest() for n in x['expected']};assert observed==x['expected'];print(json.dumps(observed))"
    assert remote(verify_script, {'root': str(remote_root), 'expected': EXPECTED}) == EXPECTED
    order = ['recommended', 'unwired', 'unwired', 'recommended']
    plan = [{'host_role': host, 'mode': mode, 'index': i} for host in ['local', 'remote'] for i, mode in enumerate(order)]
    store(trial / 'parent-plan.json', {'classification': 'CONTROL_NOT_MODEL_GATE', 'review_sha256': args.review_sha256, 'controller_sha256': sha(Path(__file__).read_bytes()), 'source_sha256': EXPECTED, 'initial_health': initial_health, 'plan': plan, 'remote_root': str(remote_root), 'model_requests': 0, 'limits': '64MiB structural retention;192MiB observed checkpoint cap and advisory runtime limit; pressure1 mandatory; not a strict allocator/RSS bound'})
    results = []
    try:
        for case in plan:
            health()
            nonce = uuid.uuid4().hex
            case_id = case['host_role'] + '-' + str(case['index']) + '-' + case['mode']
            admission = dict(schema='wired-policy-control-admission-v1', nonce=nonce, mode=case['mode'], host_role=case['host_role'], source_sha256={n: EXPECTED[n] for n in ('fixture.py', 'remote_actor.py', 'pressure_guard.py')}, original_sha256={n: EXPECTED[n] for n in ('remote_actor.py', 'pressure_guard.py')}, harness_sha256=EXPECTED['harness.py'])
            admission_path = trial / 'admissions' / (case_id + '.json')
            store(admission_path, admission)
            if case['host_role'] == 'local':
                job = trial / case_id
                argv = [local_python, '-B', str(trial / 'harness.py'), '--root', str(local_sources), '--jobs', str(job), '--mode', case['mode'], '--host-role', 'local', '--nonce', nonce, '--admission', str(admission_path)]
            else:
                job = remote_root / case_id
                remote_admission = remote_root / 'admissions' / (case_id + '.json')
                admission_readback = remote("import json,sys;from pathlib import Path;x=json.load(sys.stdin);p=Path(x['path']);assert p.parent.name=='admissions' and p.parent.parent.parent==Path('/Users/jeremylien/.hermes/cache/scratch');b=json.dumps(x['admission'],sort_keys=True).encode();f=p.open('xb');f.write(b);f.close();print(json.dumps({'readback':json.loads(p.read_text())}))", {'path': str(remote_admission), 'admission': admission})
                assert admission_readback == {'readback': admission}
                argv = ssh + [shlex.join([remote_python, '-B', str(remote_root / 'harness.py'), '--root', str(remote_root / 'sources'), '--jobs', str(job), '--mode', case['mode'], '--host-role', 'remote', '--nonce', nonce, '--admission', str(remote_admission)])]
            completed = subprocess.run(argv, capture_output=True, timeout=75)
            write(trial / (case_id + '.stdout'), completed.stdout)
            write(trial / (case_id + '.stderr'), completed.stderr)
            if case['host_role'] == 'remote':
                fetched = remote(r'''import json,sys,base64,hashlib
from pathlib import Path
x=json.load(sys.stdin);p=Path(x['job']);assert p.parent.parent==Path('/Users/jeremylien/.hermes/cache/scratch');out={}
for n in ['startup.json','pids.json','result.json','source_sha256.json','stdout','stderr','sources/fixture.py','sources/remote_actor.py','sources/pressure_guard.py']:
 f=p/n
 if f.is_file() and not f.is_symlink():
  b=f.read_bytes();assert len(b)<=2*1024**2;out[n]={'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}
print(json.dumps(out))
''', {'job': str(job)})
                archive = trial / case_id
                archive.mkdir(mode=0o700)
                for name, record in fetched.items():
                    assert name in {'startup.json','pids.json','result.json','source_sha256.json','stdout','stderr','sources/fixture.py','sources/remote_actor.py','sources/pressure_guard.py'}
                    data = base64.b64decode(record['data'], validate=True)
                    assert sha(data) == record['sha256']
                    target = archive / name
                    target.parent.mkdir(mode=0o700, exist_ok=True)
                    write(target, data)
            assert completed.returncode == 0, 'control failed; retained raw evidence'
            value = json.loads(completed.stdout)
            assert value['ok'] is True and value['classification'] == 'CONTROL_NOT_MODEL_GATE'
            control = value['control']
            expected_device = 'Apple M2 Pro' if case['host_role'] == 'local' else 'Apple M4'
            assert control['backend']['device']['device_name'] == expected_device
            assert control['backend']['memory_limit_semantics'] == 'advisory-with-observed-cap'
            assert 64 * 1024**2 <= control['memory'][-2]['active'] <= 192 * 1024**2
            before, last, after = control['samples'][0], control['samples'][-2], control['samples'][-1]
            summary = {**case, 'nonce': nonce, 'ok': True, 'device': expected_device, 'retained_bytes': control['retained_bytes'], 'policy': control['policy'], 'elapsed_seconds': control['elapsed_seconds'], 'before_os_wired_bytes': before['vm']['Pages wired down'] * before['page_size'], 'last_os_wired_bytes': last['vm']['Pages wired down'] * last['page_size'], 'after_os_wired_bytes': after['vm']['Pages wired down'] * after['page_size'], 'peak_mlx_bytes': max(v['peak'] for v in control['memory']), 'actor_cleanup': value['actor']['cleanup'], 'result_sha256': sha(completed.stdout)}
            results.append(summary)
            store(trial / (case_id + '-summary.json'), summary)
            print(json.dumps({'completed': len(results), 'planned': len(plan), 'host': case['host_role'], 'mode': case['mode'], 'ok': True}), flush=True)
        final_health = health()
        assert len(results) == len(plan) == 8
        store(trial / 'summary.json', {'classification': 'CONTROL_NOT_MODEL_GATE', 'all_planned_controls_passed': True, 'controls': results, 'final_health': final_health, 'model_admission': False, 'pressure_cause_established': False, 'performance_accepted': False})
        print(json.dumps({'status': 'all_eight_bounded_controls_passed', 'trial_path': str(trial), 'model_admission': False}), flush=True)
        return 0
    except Exception as exc:
        store(trial / 'failed.json', {'classification': 'CONTROL_NOT_MODEL_GATE', 'error_type': type(exc).__name__, 'controls_completed': len(results), 'controls': results, 'next': 'Inspect exact retained case receipts; do not automatically retry or infer cleanup on a transport failure.'})
        print(json.dumps({'status': 'bounded_control_failed', 'error_type': type(exc).__name__, 'trial_path': str(trial), 'controls_completed': len(results)}), flush=True)
        return 1


if __name__ == '__main__':
    os.environ['TMPDIR'] = str(SCRATCH)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    raise SystemExit(main())
