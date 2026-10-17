"""Parent-owned, single-run physical pilot. Not performance acceptance."""
import fcntl
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path('/Users/jeremylien/Projects/paretoquant-metal')
OUT = ROOT / 'results/qwen38-prefill-overlap-v1'
SCRATCH = Path('/Users/jeremylien/.hermes/cache/scratch')
sys.path.insert(0, str(ROOT))
os.environ['TMPDIR'] = str(SCRATCH)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from benchmarks.qwen38_overlap import supervisor025 as supervisor


def hashed(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def run(argv, timeout=8):
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout).stdout


def interface(route):
    match = re.search(r'^\s*interface:\s*(\S+)', route, re.M)
    if match is None:
        raise ValueError('Missing admitted route interface')
    return match[1]


def project_workers(text):
    found = []
    for line in text.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        if re.search(r'/(?:probe|remote_actor|supervise)\.py(?:\s|$)', fields[1]):
            found.append(int(fields[0]))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=[f'batch-{i:02d}' for i in range(6)])
    phase = parser.parse_args().phase
    pilot = OUT / 'physical-pilot-noncritical-v3-1792272408272316000'
    pilot_trust = OUT / (pilot.name + '-external-trust.json')
    pilot_trust_sha = '875aa7174d5f3654b3415d784f99354959a9ba0fcc605dfc8a1f267deeb3e40d'
    assert hashed(pilot_trust) == pilot_trust_sha
    lock = (SCRATCH / 'paretoquant-physical-owner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repair = json.loads((OUT / 'parent-noncritical-freeze.json').read_text())
    deployment = json.loads((OUT / 'parent-noncritical-deployment.json').read_text())
    profile_path = Path(deployment['private_profile_path'])
    profile = json.loads(profile_path.read_text())
    manifest_path, digest = Path(repair['freeze_path']), repair['freeze_sha256']
    assert deployment['manifest_sha256'] == digest
    manifest = supervisor.read_freeze(manifest_path, digest)
    assert supervisor.expected_pressure_policy(manifest) == 'noncritical'
    assert manifest['memory_acceptance']['sha256'] == 'a0341821118cfd83d45f0f8979152aaffae3f296b372f8dc29b58aacc35d409f'
    review_digest = os.environ['PARETOQUANT_COMBINED_REVIEW_SHA256']
    assert re.fullmatch('[a-f0-9]{64}', review_digest)
    reviewed = [('noncritical-combined-independent-review.json', review_digest)]
    report = json.loads((OUT / reviewed[0][0]).read_text())
    assert hashed(OUT / reviewed[0][0]) == review_digest and report['passed'] is True
    def hash_leaves(value):
        if type(value) is str and re.fullmatch('[a-f0-9]{64}', value):
            return {value}
        if type(value) is dict:
            return set().union(*(hash_leaves(v) for v in value.values()))
        if type(value) is list:
            return set().union(*(hash_leaves(v) for v in value))
        return set()
    approved_hashes = hash_leaves(report.get('source_sha256', report.get('sourcehashes', {})))
    critical = [ROOT / 'spikes/025-qwen38-overlapped-prefill' / n for n in
        ['pressure_guard.py','remote_actor.py','supervise.py','probe.py','mtp_runtime.py',
         'overlap_prefill.py','workload_evidence.py','stream_weights.py']]
    critical += [ROOT / 'benchmarks/qwen38_overlap' / n for n in
        ['cli.py','supervisor025.py','integration025.py']]
    for file in critical:
        assert hashed(file) in approved_hashes, str(file)
    for relative, expected in json.loads((OUT / 'acceptance.json').read_text())['source_preservation_sha256'].items():
        assert hashed(ROOT / relative) == expected, relative
    for phase_name, item in manifest['phases'].items():
        for name, expected in item['sources'].items():
            assert hashed(manifest_path.parent / item['source_dir'] / name) == expected
        assert Path(profile['remote_root_by_phase'][phase_name]).parents[1] == Path('/Users/jeremylien/Projects/paretoquant-dual-host')
    assert Path(profile['sidecar']).exists()
    trial = OUT / ('physical-suite-noncritical-v1-' + phase + '-' + str(time.time_ns()))
    admission_path = OUT / (trial.name + '-parent-admission.json')
    authorization = None

    def guard(plan):
        nonlocal authorization
        assert plan['manifest_sha256'] == digest and plan['phase'] == phase
        argv = plan['public_argv']
        assert '--require-normal-pressure' not in argv
        assert argv[argv.index('--pressure-policy') + 1] == 'noncritical'
        pressure = int(run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level']))
        assert pressure in (1, 2)
        assert interface(run(['route', '-n', 'get', profile['remote_address']])) == 'bridge0'
        assert not project_workers(run(['ps', '-axo', 'pid=,args=']))
        assert os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize >= 8 * 1024**3
        ssh = ['ssh', '-T', '-i', profile['ssh_key'], '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=8', '-o', 'HostKeyAlias=' + profile['ssh_host_key_alias'], profile['ssh_target']]
        remote = r'''import json,sys,subprocess,hashlib,os,re,importlib.metadata
from pathlib import Path
x=json.loads(sys.stdin.buffer.read(512*1024))
def call(a):return subprocess.run(a,capture_output=True,text=True,check=True,timeout=5).stdout
pressure=int(call(['sysctl','-n','kern.memorystatus_vm_pressure_level']));assert pressure in (1,2)
route=call(['route','-n','get',x['peer']]);assert re.search(r'^\s*interface:\s*(\S+)',route,re.M)[1]=='bridge0'
workers=[]
for line in call(['ps','-axo','pid=,args=']).splitlines():
 f=line.strip().split(None,1)
 if len(f)==2 and re.search(r'/(?:probe|remote_actor|supervise)\.py(?:\s|$)',f[1]):workers.append(int(f[0]))
assert not workers
for phase,rootname in x['roots'].items():
 d=Path(rootname);assert d.parents[1]==Path('/Users/jeremylien/Projects/paretoquant-dual-host');assert (d.parents[1]/'.venv/bin/python').is_file()
 assert {p.name for p in d.iterdir()}==set(x['sources'][phase])
 for name,h in x['sources'][phase].items():
  p=d/name;assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==h
for name in [x['weights'],x['small']]:assert Path(name).is_dir()
s=os.statvfs(x['weights']);assert s.f_bavail*s.f_frsize>=8*1024**3
print(json.dumps({'pressure':pressure,'route_interface':'bridge0','conflicting_project_workers':workers,'source_files_verified':sum(len(v) for v in x['sources'].values()),'mlx_version':importlib.metadata.version('mlx'),'mlx_lm_version':importlib.metadata.version('mlx-lm')}))
'''
        payload = {'peer': profile['local_address'], 'roots': {phase: profile['remote_root_by_phase'][phase]}, 'sources': {phase: manifest['phases'][phase]['sources']}, 'weights': profile['remote_source'], 'small': profile['small_model_remote']}
        command = ssh + [shlex.join(['/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python', '-B', '-c', remote])]
        result = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise RuntimeError('remote physical prerequisites failed')
        remote_health = json.loads(result.stdout)
        authorization = {
            'schema': 'overlap-parent-physical-attempt-authorization-v2',
            'pressure_policy': 'noncritical',
            'allowed_pressure_levels': [1, 2],
            'abort_pressure_levels': [4],
            'memory_acceptance_sha256': manifest['memory_acceptance']['sha256'],
            'artifact_kind': 'PARENT_REVIEWED_PHYSICAL_APPROVAL',
            'approved': True, 'phase': phase, 'manifest_sha256': digest,
            'review_receipt_sha256': reviewed[-1][1],
            'review_receipts': dict(reviewed),
            'controller_sha256': hashed(__file__),
            'timestamp': time.time(), 'trial_path': str(trial),
            'local_pressure': pressure, 'local_route_interface': 'bridge0',
            'remote_health': remote_health,
            'local_mlx_version': importlib.metadata.version('mlx'),
            'local_mlx_lm_version': importlib.metadata.version('mlx-lm'),
            'scope': 'Authorize one bounded batch: real small/full gates precede fourteen requests; complete real pilot and byte budget independently re-audited. Not numerical or performance acceptance.',
            'performance_accepted': False,
        }
        exclusive_json(admission_path, authorization)
        return authorization

    result = supervisor.guarded_run(manifest_path, digest, phase, trial, execute=True, profile_path=profile_path, parent_guard=guard, pilot_admission=(pilot / 'raw', pilot_trust, pilot_trust_sha))
    summary = {'schema': 'overlap-parent-batch-execution-v1', 'phase': phase, 'timestamp': time.time(), 'trial_path': str(trial), 'admission_path': str(admission_path), 'guarded_run': result, 'performance_accepted': False}
    raw_result = trial / 'raw/result.json'
    if raw_result.is_file():
        raw = json.loads(raw_result.read_text())
        summary['controller_result'] = {k: raw.get(k) for k in ['status', 'error_type', 'error_stage', 'owned_actor_cleanup_verified', 'services_closed_verified', 'pending_cleanup', 'elapsed_seconds']}
    exclusive_json(OUT / (trial.name + '-execution.json'), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if result.get('returncode') == 0 else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status': 'parent_physical_attempt_failed', 'error_type': type(exc).__name__}), flush=True)
        raise
