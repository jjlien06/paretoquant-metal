"""Pinned-key SSH, exclusive staging, bounded actors and exact readback."""
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

LOCAL = Path(__file__).parent
REPO = Path('/Users/jeremylien/Projects/paretoquant-metal')
PYTHON = '/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python'
SSH = ['/usr/bin/ssh', '-6', '-i', '/Users/jeremylien/.ssh/paretoquant_mini_ed25519',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'HostKeyAlias=fe80::56:c507:c02a:a630%bridge0', '-o', 'ConnectTimeout=5',
       'jeremylien@fe80::431:67b7:34f0:62c9%bridge0']
GUARDED = '/Users/jeremylien/Projects/paretoquant-dual-host/spikes/007-dual-host-pipeline-guarded-1a7a2d85b186'


def remote(code, *, data=None, timeout=30):
    result = subprocess.run(SSH + [shlex.join([PYTHON, '-u', '-c', code])],
        input=data, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'SSH exit {result.returncode}: {result.stderr}; stdout: {result.stdout}')
    return result.stdout


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def deploy():
    root = '/Users/jeremylien/Projects/paretoquant-dual-host/spikes/009-sequential-' + str(time.time_ns())
    sources = {n: REPO / 'spikes' / '009-speculative-32b' / n for n in ['speculative_decode.py', 'test_speculative_gpu.py', 'draft_screen.py']}
    sources.update({n: REPO / 'spikes' / '007-dual-host-pipeline' / n for n in ['phased_decode.py', 'remote_actor.py']})
    sources.update({n: LOCAL / n for n in ['checkpoint_modes.py', 'tiny_modes.py']})
    data = {n: p.read_bytes() for n, p in sources.items()}
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}
    assert hashes['speculative_decode.py'] == '25dd62cb80b4d9bcabe78d9a396615f8a658936799908cc0b10b153f2b65b856'
    assert hashes['phased_decode.py'] == 'b4857df43ce1f312bef4ec19e0afcc5c3f4e3d73af09f1ff75c49f4dd73b7d8f'
    payload = {'root': root, 'files': {n: base64.b64encode(b).decode() for n, b in data.items()}}
    code = "import sys,json,base64,hashlib; from pathlib import Path; p=json.load(sys.stdin); root=Path(p['root']); root.mkdir(mode=0o700);\nfor n,b in p['files'].items():\n assert '/' not in n; raw=base64.b64decode(b);\n with (root/n).open('xb') as f: f.write(raw)\nprint(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in p['files']}))"
    readback = json.loads(remote(code, data=json.dumps(payload)))
    assert readback == hashes
    guard_code = f"import json,hashlib; from pathlib import Path; root=Path({GUARDED!r}); print(json.dumps({{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}}))"
    guard = json.loads(remote(guard_code))
    record = {'root': root, 'files': hashes, 'readback_verified': True,
              'guarded_root': GUARDED, 'guarded_source_sha256_before': guard,
              'scope': 'Isolated sequential diagnostic; original full32B runtime untouched',
              'transport': 'Thunderbolt bridge0 IPv6; strict prior host key alias'}
    exclusive(LOCAL / 'deployment.json', record)
    print(json.dumps(record, indent=2))


def run(label):
    record = json.loads((LOCAL / 'deployment.json').read_text())
    root = record['root']
    hashes = record['files']
    if label in ['tiny-batch', 'tiny-sequential']:
        argv = [PYTHON, '-u', root + '/tiny_modes.py', label.removeprefix('tiny-')]
        deadline = 120
        outputs = [label + '.json']
    elif label == 'checkpoint-modes':
        for prior in ['tiny-batch', 'tiny-sequential']:
            assert json.loads((LOCAL / (prior + '.json')).read_text())['passed'] is True
        argv = [PYTHON, '-u', root + '/checkpoint_modes.py']
        deadline = 180
        outputs = ['checkpoint-modes.json']
    else:
        raise ValueError(label)
    code = f"""import sys,json,hashlib,os,subprocess; from pathlib import Path
root=Path({root!r}); expected={hashes!r}
assert {{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected}}==expected
pressure=subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True).strip()
assert pressure=='1', ('memory pressure',pressure)
s=os.statvfs('/'); assert s.f_bavail*s.f_frsize > 2*1024**3
sys.path.insert(0,str(root)); import remote_actor
result=remote_actor.run_owned({argv!r},root/('job-'+{label!r}),deadline={deadline},reserve_bytes=2*1024**3)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['reason']=='completed' and result['cleanup']['group_gone'] and result['cleanup']['child_reaped'] else 1)
"""
    stdout = remote(code, timeout=deadline + 80)
    actor = json.loads(stdout)
    exclusive(LOCAL / (label + '-actor.json'), actor)
    collect = f"""import json,base64,hashlib; from pathlib import Path
root=Path({root!r}); names={outputs!r}+['job-'+{label!r}+'/'+n for n in ['startup.json','pids.json','result.json','stdout','stderr']]
files={{n:(root/n).read_bytes() for n in names}}
print(json.dumps({{n:{{'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}} for n,b in files.items()}}))
"""
    fetched = json.loads(remote(collect))
    for name, value in fetched.items():
        data = base64.b64decode(value['base64'])
        assert hashlib.sha256(data).hexdigest() == value['sha256']
        target = LOCAL / name
        target.parent.mkdir(exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    report = json.loads((LOCAL / outputs[0]).read_text())
    assert report['sources']['speculative_decode.py'] == hashes['speculative_decode.py']
    assert report['sources']['phased_decode.py'] == hashes['phased_decode.py']
    code = f"import json,hashlib; from pathlib import Path; root=Path({GUARDED!r}); print(json.dumps({{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}}))"
    guarded_after = json.loads(remote(code))
    assert guarded_after == record['guarded_source_sha256_before']
    verification = {'fetched_sha256': {n: v['sha256'] for n, v in fetched.items()},
        'guarded_source_sha256_after': guarded_after, 'guarded_unchanged': True,
        'report_sources_bound': True, 'actor_cleanup': actor['cleanup']}
    exclusive(LOCAL / (label + '-readback.json'), verification)
    print(json.dumps({'label': label, 'passed': report['passed'], 'cases': report['case_count'],
        'peak_mlx_bytes': report['peak_mlx_bytes'], 'actor': actor,
        'local_report': str(LOCAL / outputs[0]), 'guarded_unchanged': True}, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'deploy':
        deploy()
    else:
        run(sys.argv[1])
