"""Re-test the 40Gb/s bridge and immutable grouped decoder, bounded and owned."""
import base64
import errno
import hashlib
import importlib.util
import json
import os
import select
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
PREVIOUS = Path('/Users/jeremylien/.hermes/cache/scratch/paretoquant-dual-host')
FROZEN = PREVIOUS / 'acceleration-32b/grouped-review-1791187162875738000'
PYTHON = '/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python'
HOST_ALIAS = 'fe80::56:c507:c02a:a630%bridge0'
TARGET = 'jeremylien@fe80::431:67b7:34f0:62c9%bridge0'
SSH = ['/usr/bin/ssh', '-6', '-T', '-i', '/Users/jeremylien/.ssh/paretoquant_mini_ed25519',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'HostKeyAlias=' + HOST_ALIAS, '-o', 'ConnectTimeout=5',
       '-o', 'ServerAliveInterval=2', '-o', 'ServerAliveCountMax=2', TARGET]
LOCAL_IP = '169.254.248.125'
REMOTE_IP = '169.254.169.190'
sys.path.insert(0, str(FROZEN))
import remote_actor
import supervise


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def remote(code, data=None, timeout=20):
    result = subprocess.run(SSH + [shlex.join([PYTHON, '-u', '-c', code])],
        input=data, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'SSH failure {result.returncode}: {result.stderr}')
    return result.stdout


def resources():
    pressure = int(subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True))
    stat = os.statvfs('/')
    peer = json.loads(remote("import os,json,subprocess; s=os.statvfs('/'); print(json.dumps({'pressure':int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True)),'free_bytes':s.f_bavail*s.f_frsize,'swap':subprocess.check_output(['sysctl','vm.swapusage'],text=True)}))"))
    result = {'local': {'pressure': pressure, 'free_bytes': stat.f_bavail * stat.f_frsize,
                       'swap': subprocess.check_output(['sysctl', 'vm.swapusage'], text=True)}, 'remote': peer}
    return result


def verify_frozen():
    manifest = json.loads((FROZEN / 'manifest.json').read_text())['files']
    assert all(hashlib.sha256((FROZEN / n).read_bytes()).hexdigest() == h for n, h in manifest.items())
    assert manifest['remote_actor.py'] == '907d6c759a78be53b5cae220d707d2d323b2aa98dbc5421067bcc46aff8ad025'
    assert manifest['phased_decode.py'] == 'a75cc6777f052135211ce9f386528059d494f69ca3e264f48d920157ba71308f'
    return manifest


def bandwidth():
    verify_frozen()
    route = subprocess.check_output(['/sbin/route', '-n', 'get', REMOTE_IP], text=True)
    assert 'interface: bridge0' in route
    before = resources()
    exclusive(BASE / 'bandwidth-preflight.json', {'resources': before, 'route': route})
    root = '/Users/jeremylien/Projects/paretoquant-dual-host/spikes/cable40-' + str(time.time_ns())
    files = {'tcp_smoke.py': (BASE / 'tcp_smoke.py').read_bytes(), 'remote_actor.py': (FROZEN / 'remote_actor.py').read_bytes()}
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    payload = {'root': root, 'files': {n: base64.b64encode(b).decode() for n, b in files.items()}}
    code = "import sys,json,base64,hashlib; from pathlib import Path; p=json.load(sys.stdin); root=Path(p['root']); root.mkdir(mode=0o700);\nfor n,b in p['files'].items():\n assert '/' not in n\n with (root/n).open('xb') as f:f.write(base64.b64decode(b))\nprint(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in p['files']}))"
    assert json.loads(remote(code, json.dumps(payload))) == hashes
    exclusive(BASE / 'bandwidth-deployment.json', {'root': root, 'sources': hashes, 'readback_verified': True})
    code = f"""import sys,json,hashlib; from pathlib import Path
root=Path({root!r}); expected={hashes!r}
assert {{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected}}==expected
sys.path.insert(0,str(root)); import remote_actor
result=remote_actor.run_owned([{PYTHON!r},'-u',str(root/'tcp_smoke.py'),'serve',{REMOTE_IP!r},{LOCAL_IP!r},str(root/'server.json')],root/'job-server',deadline=90,reserve_bytes=2*1024**3,stream=True)
print(json.dumps({{'actor':result}}),flush=True)
raise SystemExit(0 if result['reason']=='completed' and result['cleanup']['group_gone'] else 1)
"""
    with (BASE / 'server-transport.stderr').open('xb') as stderr:
        process = subprocess.Popen(SSH + [shlex.join([PYTHON, '-u', '-c', code])],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr, start_new_session=True)
        port = None
        normal = False
        try:
            assert select.select([process.stdout], [], [], 15)[0], 'Server readiness timeout'
            line = process.stdout.readline()
            ready = json.loads(line)
            assert ready['ready'] is True and ready['address'][0] == REMOTE_IP and ready['peer'] == LOCAL_IP
            port = ready['address'][1]
            exclusive(BASE / 'bandwidth-ready.json', ready)
            client = remote_actor.run_owned([sys.executable, '-u', str(BASE / 'tcp_smoke.py'), 'client',
                REMOTE_IP, str(port), LOCAL_IP, str(BASE / 'client.json')], BASE / 'job-client',
                deadline=60, reserve_bytes=2*1024**3, stream=True)
            exclusive(BASE / 'client-actor.json', client)
            assert client['reason'] == 'completed' and client['cleanup']['group_gone']
            remaining = process.communicate(timeout=20)[0]
            assert process.returncode == 0
            with (BASE / 'server-transport.stdout').open('xb') as stream:
                stream.write(line + remaining)
            normal = True
        finally:
            if not normal:
                remote(f"from pathlib import Path; p=Path({root!r})/'job-server'/'stop'; p.touch(exist_ok=True)")
                process.wait(timeout=110)
            cleanup = remote_actor.terminate(process)
            exclusive(BASE / 'server-transport-cleanup.json', cleanup)
            assert cleanup['child_reaped'] and cleanup['group_gone']
    collect = f"""import json,base64,hashlib; from pathlib import Path
root=Path({root!r}); names=['server.json']+['job-server/'+n for n in ['startup.json','pids.json','result.json','stdout','stderr']]
print(json.dumps({{n:{{'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest(),'base64':base64.b64encode((root/n).read_bytes()).decode()}} for n in names}}))
"""
    fetched = json.loads(remote(collect))
    for name, value in fetched.items():
        data = base64.b64decode(value['base64'])
        assert hashlib.sha256(data).hexdigest() == value['sha256']
        target = BASE / 'remote-evidence' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    server_actor = json.loads((BASE / 'remote-evidence/job-server/result.json').read_text())
    assert server_actor['reason'] == 'completed' and server_actor['cleanup']['group_gone'] and server_actor['cleanup']['child_reaped']
    closed = supervise.listener_closed(REMOTE_IP, port)
    assert closed
    report = json.loads((BASE / 'client.json').read_text())
    server = json.loads((BASE / 'remote-evidence/server.json').read_text())
    assert report['source_sha256'] == server['source_sha256'] == hashes['tcp_smoke.py']
    assert server['request_count'] == len(server['rows']) == len(report['rows']) == 10
    for a, b in zip(report['rows'], server['rows']):
        assert a['bytes'] == b['receipt']['bytes'] and a['direction'] == b['request']['direction']
        if a['digest_verified']:
            assert a['sha256'] == b['receipt']['sha256']
    exclusive(BASE / 'bandwidth-verification.json', {'listener_closed': closed,
        'server_actor': server_actor, 'client_actor': client, 'source_hashes': hashes,
        'fetched_sha256': {n: v['sha256'] for n, v in fetched.items()}, 'route': route,
        'all_byte_counts_and_untimed_digests_match': True})
    print(json.dumps({'bandwidth': report['summary'], 'listener_closed': closed, 'actors_clean': True}, indent=2))


def model(label):
    hashes = verify_frozen()
    if label == 'small':
        old = PREVIOUS / 'small-phased-supervised-v1'
        repeat, deadline = 1, 180
    elif label == 'qwen32':
        old = PREVIOUS / 'acceleration-32b/qwen32-group8-v1'
        repeat, deadline = 4, 240
        assert json.loads((BASE / 'small/result.json').read_text())['status'] == 'success'
        assert json.loads((BASE / 'small-parent-verification.json').read_text())['historical_tokens_equal']
    else:
        raise ValueError(label)
    before = resources()
    exclusive(BASE / (label + '-preflight.json'), before)
    assert before['local']['pressure'] == before['remote']['pressure'] == 1
    assert min(before['local']['free_bytes'], before['remote']['free_bytes']) > 4*1024**3
    previous = json.loads((old / 'startup.json').read_text())['args']
    previous.update(output=str(BASE / label), repeats=repeat, deadline=deadline,
                    eval_group_size=8, reserve_bytes=4*1024**3, ssh_target=TARGET)
    args = supervise.parser().parse_args([item for key,value in previous.items()
        for item in ['--'+key.replace('_','-'), *([str(x) for x in value] if isinstance(value,list) else [str(value)])]])
    assert args.local_address == LOCAL_IP and args.remote_address == REMOTE_IP
    exclusive(BASE / (label + '-admission.json'), {'args': vars(args), 'approved_source_sha256': hashes,
        'ssh_host_key_alias': HOST_ALIAS, 'scope': 'Unchanged approved grouped decoder; no speculation'})
    def transport(config, actor_args):
        argv = supervise.ssh_argv(config, actor_args)
        return [argv[0], '-o', 'HostKeyAlias=' + HOST_ALIAS, *argv[1:]]
    result = supervise.run(args, transport=transport)
    print(json.dumps({'label': label, 'controller_result': result}, indent=2), flush=True)
    assert result['status'] == 'success'
    previous_receipt = json.loads((old / 'rank0.receipt.json').read_text())
    original = [json.loads(line) for line in previous_receipt['stdout'].splitlines() if line.strip()][-1]
    reference_tokens = original['requests'][0]['token_ids']
    expected_tensors = original['weight_loading']['tensor_sha256']
    ranks = []
    for rank in [0, 1]:
        receipt = json.loads((BASE / label / f'rank{rank}.receipt.json').read_text())
        record = [json.loads(line) for line in receipt['stdout'].splitlines() if line.strip()][-1]
        assert len(record['requests']) == repeat
        assert all(request['token_ids'] == reference_tokens for request in record['requests'])
        prior = json.loads((old / f'rank{rank}.receipt.json').read_text())
        prior_record = [json.loads(line) for line in prior['stdout'].splitlines() if line.strip()][-1]
        assert record['weight_loading']['tensor_sha256'] == prior_record['weight_loading']['tensor_sha256']
        ranks.append(record)
        exclusive(BASE / label / f'rank{rank}.record.json', record)
    assert json.loads((BASE / label / 'source_sha256.json').read_text()) == {n: hashes[n] for n in remote_actor.SOURCES}
    verification = {'historical_tokens_equal': True, 'all_rank_tensor_hashes_equal_historical': True,
        'request_count_per_rank': repeat, 'token_count_including_EOS': len(reference_tokens),
        'approved_sources_unchanged': True, 'actors_and_listeners_clean': True,
        'after_resources': resources()}
    exclusive(BASE / (label + '-parent-verification.json'), verification)
    print(json.dumps({'label': label, 'verified': verification, 'rank0_requests': ranks[0]['requests']}, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'bandwidth':
        bandwidth()
    else:
        model(sys.argv[1])
