#!/usr/bin/env python3
"""Parent-only source-bound CONTROL_NOT_MODEL_GATE harness; inert on import."""
import hashlib
import os
import re
from pathlib import Path

SOURCES = ('fixture.py', 'remote_actor.py', 'pressure_guard.py')
ORIGINALS = ('remote_actor.py', 'pressure_guard.py')
ORDER_PER_HOST = ('recommended', 'unwired', 'unwired', 'recommended')
HOST_ORDER = ('local', 'remote')  # parent: M2, then M4; no concurrent runs
SOURCE_CAP = 2 * 1024**2


def source_bytes(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('require a regular non-symlink source')
    with path.open('rb') as handle:
        data = handle.read(SOURCE_CAP + 1)
    if len(data) > SOURCE_CAP:
        raise ValueError('source exceeds bounded inspection')
    return data


def valid_hashes(hashes, names):
    if (type(hashes) is not dict or set(hashes) != set(names)
            or any(type(v) is not str or re.fullmatch('[a-f0-9]{64}', v) is None
                   for v in hashes.values())):
        raise ValueError('require exact external expected source hashes')
    return dict(hashes)


def stage(root, *, original_root, expected_original_hashes, expected_fixture_sha256):
    """Copy only the reviewed fixture + byte-exact original actor/guard. No run."""
    expected = valid_hashes(expected_original_hashes, ORIGINALS)
    expected.update(valid_hashes({'fixture.py': expected_fixture_sha256}, ('fixture.py',)))
    original_root = Path(original_root).resolve(strict=True)
    payloads = {n: source_bytes(original_root/n) for n in ORIGINALS}
    payloads['fixture.py'] = source_bytes(Path(__file__).with_name('fixture.py'))
    observed = {n: hashlib.sha256(data).hexdigest() for n, data in payloads.items()}
    if observed != expected:
        raise ValueError('staging bytes differ from external admission')
    root = Path(root)
    root.mkdir(mode=0o700)  # new only; never overwrite a parent staging tree
    for name in SOURCES:
        with (root/name).open('xb') as handle:
            handle.write(payloads[name])
        (root/name).chmod(0o600)
    if any(source_bytes(root/n) != payloads[n] for n in SOURCES):
        raise ValueError('staged readback differs')
    return observed


def _invoke_owned(actor, root, jobs, *, mode, host_role, nonce, expected_hashes, deadline):
    """Only executor seam: parent holds exclusive synchronous actor ownership."""
    import sys
    saved = actor.SOURCES
    actor.SOURCES = SOURCES
    try:
        env = {k:v for k,v in os.environ.items() if k in {'PATH', 'HOME', 'TMPDIR', 'LANG'}}
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return actor.run_owned(
            [sys.executable, '-B', '-u', str(root/'fixture.py'), '--mode', mode,
             '--nonce', nonce, '--host-role', host_role], jobs,
            deadline=deadline, env=env, nonce=nonce, source_root=root,
            expected_hashes=expected_hashes, require_normal_pressure=True,
            host_role=host_role, actor_role='rank0' if host_role == 'local' else 'rank1')
    finally:
        actor.SOURCES = saved


def _admit(root, admission, *, mode, host_role, nonce, deadline):
    import math
    if (type(mode) is not str or mode not in ORDER_PER_HOST
            or type(host_role) is not str or host_role not in HOST_ORDER
            or type(nonce) is not str or re.fullmatch('[a-f0-9]{32}', nonce) is None
            or type(deadline) not in (int, float) or not math.isfinite(deadline) or not 0 < deadline <= 45):
        raise ValueError('invalid literal trial identity/deadline')
    keys = {'schema', 'nonce', 'mode', 'host_role', 'source_sha256', 'original_sha256', 'harness_sha256'}
    if (type(admission) is not dict or set(admission) != keys
            or admission['schema'] != 'wired-policy-control-admission-v1'
            or any(type(admission[k]) is not str or admission[k] != v
                   for k,v in dict(nonce=nonce, mode=mode, host_role=host_role).items())):
        raise ValueError('missing external trial admission or identity mismatch')
    expected = valid_hashes(admission['source_sha256'], SOURCES)
    originals = valid_hashes(admission['original_sha256'], ORIGINALS)
    harness_hash = valid_hashes({'harness.py':admission['harness_sha256']}, ('harness.py',))['harness.py']
    if (any(expected[n] != originals[n] for n in ORIGINALS)
            or hashlib.sha256(source_bytes(__file__)).hexdigest() != harness_hash):
        raise ValueError('original guard/actor or harness differs from external admission')
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('require regular source root')
    payloads = {n:source_bytes(root/n) for n in SOURCES}
    if {n:hashlib.sha256(data).hexdigest() for n,data in payloads.items()} != expected:
        raise ValueError('source bytes differ from external admission')
    return payloads


def execute(root, jobs, *, mode, host_role, nonce, admission, deadline=45):
    """PHYSICAL parent entry. CPU tests replace _invoke_owned, never run_owned."""
    import sys
    import types
    payloads = _admit(root, admission, mode=mode, host_role=host_role, nonce=nonce, deadline=deadline)
    root = Path(root).resolve(strict=True)
    jobs = Path(jobs)
    if jobs.exists() or jobs.is_symlink() or not jobs.parent.is_dir():
        raise ValueError('jobs must be a new private trial directory in an existing parent')
    jobs = jobs.parent.resolve(strict=True) / jobs.name
    previous_guard = sys.modules.get('pressure_guard')
    try:
        guard = types.ModuleType('pressure_guard')
        guard.__file__ = str(root/'pressure_guard.py')
        exec(compile(payloads['pressure_guard.py'], guard.__file__, 'exec'), guard.__dict__)
        sys.modules['pressure_guard'] = guard
        actor = types.ModuleType('wired_control_original_actor')
        actor.__file__ = str(root/'remote_actor.py')
        exec(compile(payloads['remote_actor.py'], actor.__file__, 'exec'), actor.__dict__)
        result = _invoke_owned(actor, root, jobs, mode=mode, host_role=host_role, nonce=nonce,
                               expected_hashes=dict(admission['source_sha256']), deadline=deadline)
        return _collect(jobs, result, guard, admission=admission)
    finally:
        if previous_guard is None:
            sys.modules.pop('pressure_guard', None)
        else:
            sys.modules['pressure_guard'] = previous_guard


def _collect(jobs, returned_result, guard, *, admission):
    """Read exact actor targets; original pressure verifier decides coverage."""
    import json
    jobs = Path(jobs)
    read = lambda name: json.loads(source_bytes(jobs/name))
    result = read('result.json')
    if result != returned_result or read('source_sha256.json') != admission['source_sha256']:
        raise ValueError('actor result/source readback mismatch')
    if (result.get('reason') != 'completed' or type(result.get('returncode')) is not int
            or result['returncode'] != 0 or result.get('log_evidence_complete') is not True
            or result.get('cleanup', {}).get('child_reaped') is not True
            or result.get('cleanup', {}).get('group_gone') is not True
            or result.get('cleanup', {}).get('pending_os_gpu_termination') is not False
            or result.get('cleanup', {}).get('identity_unknown', False) is not False
            or result.get('cleanup', {}).get('pid_reused', False) is not False):
        raise ValueError('actor incomplete/failed/cleanup pending; inspect retained job receipts')
    actor_role = 'rank0' if admission['host_role'] == 'local' else 'rank1'
    status = dict(nonce=admission['nonce'], startup=read('startup.json'), pids=read('pids.json'), result=result)
    binding = guard.expected_pressure_binding(status, nonce=admission['nonce'],
                                             host_role=admission['host_role'], actor_role=actor_role)
    control = read('stdout')
    wanted = {k:v for k,v in binding.items() if k != 'actor_pid'}
    wanted['mode'] = admission['mode']
    if (type(control) is not dict or control.get('binding') != wanted
            or control.get('classification') != 'CONTROL_NOT_MODEL_GATE' or control.get('ok') is not True
            or control.get('errors') != [] or control.get('retained_bytes') != 64*1024**2):
        raise ValueError('child control outcome/identity mismatch')
    config = control.get('config', {})
    if (config.get('mode') != admission['mode'] or config.get('total_bytes') != 64*1024**2
            or config.get('chunk_bytes') != 1024**2 or config.get('active_cap_bytes') != 192*1024**2
            or config.get('iterations') != 64 or type(config.get('deadline')) not in (int,float)
            or not 0 < config['deadline'] <= 30):
        raise ValueError('physical bounds changed')
    policy = control.get('policy', {})
    target = (control.get('backend', {}).get('device', {}).get('max_recommended_working_set_size')
              if admission['mode'] == 'recommended' else 0)
    if (type(target) is not int or any(type(policy.get(k)) is not int
            for k in ('requested_bytes', 'previous_bytes', 'readback_bytes', 'restored_readback_bytes'))
            or policy['requested_bytes'] != target or policy['readback_bytes'] != target
            or policy['restored_readback_bytes'] != policy['previous_bytes']):
        raise ValueError('wired policy/readback/restoration mismatch')
    arrays = control.get('arrays', [])
    if (type(arrays) is not list or len(arrays) != 64
            or any(a.get('index') != i or a.get('bytes') != 1024**2
                   or a.get('retained_verified') is not True or a.get('expected') != a.get('observed')
                   for i,a in enumerate(arrays))):
        raise ValueError('missing complete retained roundtrip evidence')
    memory = control.get('memory', [])
    samples = control.get('samples', [])
    if (len(memory) != 66 or len(samples) != 66
            or memory[0].get('stage') != 'before' or memory[-1].get('stage') != 'after_cleanup'
            or any(type(m.get(k)) is not int or not 0 <= m[k] <= 192*1024**2
                   for m in memory for k in ('active','cache','peak'))
            or memory[-1]['active'] != 0 or memory[-1]['cache'] != 0
            or [s.get('stage') for s in samples] != ['before'] + ['during']*64 + ['after']
            or any(type(s.get('pressure')) is not int or s['pressure'] != 1 for s in samples)):
        raise ValueError('memory/pressure boundary evidence incomplete or out of bounds')
    return dict(ok=True, classification='CONTROL_NOT_MODEL_GATE', binding=binding,
                source_sha256=dict(admission['source_sha256']), actor=result, control=control)


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--jobs', required=True, help='new per-trial directory, not an existing jobs root')
    parser.add_argument('--mode', required=True, choices=('recommended','unwired'))
    parser.add_argument('--host-role', required=True, choices=HOST_ORDER)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--admission', help='parent-created JSON, default ROOT/admission.json')
    parser.add_argument('--deadline', type=float, default=45)
    args = parser.parse_args(argv)
    root = Path(args.root)
    admission = json.loads(source_bytes(Path(args.admission) if args.admission else root/'admission.json'))
    result = execute(root, args.jobs, mode=args.mode, host_role=args.host_role, nonce=args.nonce,
                     admission=admission, deadline=args.deadline)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    import json
    import sys
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps(dict(ok=False, classification='CONTROL_NOT_MODEL_GATE',
                              error_type=type(exc).__name__)), flush=True)
        sys.exit(1)
