"""Bounded real-GPU diagnostic with immutable dependencies and exact readback."""
import base64
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

LOCAL = Path(__file__).parent
OLD = Path('/Users/jeremylien/.hermes/cache/scratch/paretoquant-dual-host/acceleration-32b/sequential-physical-v1/control.py')
spec = importlib.util.spec_from_file_location('previous_transport', OLD)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
ARCHIVE = Path('/Users/jeremylien/Projects/paretoquant-metal/results/m4-sequential-verification-v1/source')


def exclusive(name, value):
    with (LOCAL / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def deploy():
    root = '/Users/jeremylien/Projects/paretoquant-dual-host/spikes/010-optrace-' + str(time.time_ns())
    sources = {n: ARCHIVE / n for n in ['draft_screen.py', 'phased_decode.py', 'remote_actor.py']}
    sources['operation_trace.py'] = LOCAL / 'operation_trace.py'
    data = {n: p.read_bytes() for n, p in sources.items()}
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}
    assert hashes['phased_decode.py'] == 'b4857df43ce1f312bef4ec19e0afcc5c3f4e3d73af09f1ff75c49f4dd73b7d8f'
    payload = {'root': root, 'files': {n: base64.b64encode(b).decode() for n, b in data.items()}}
    readback = json.loads(transport.remote("""import sys,json,base64,hashlib; from pathlib import Path
p=json.load(sys.stdin); root=Path(p['root']); root.mkdir(mode=0o700)
for n,b in p['files'].items():
 assert '/' not in n; raw=base64.b64decode(b)
 with (root/n).open('xb') as f: f.write(raw)
print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in p['files']}))
""", data=json.dumps(payload)))
    assert readback == hashes
    guarded = json.loads(transport.remote(f"import json,hashlib; from pathlib import Path; r=Path({transport.GUARDED!r}); print(json.dumps({{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('*.py')}}))"))
    exclusive('deployment.json', {'root':root, 'source_sha256': hashes, 'readback_verified':True,
        'guarded_before':guarded, 'scope':'Actual checkpoint read-only operation localization, not acceleration'})
    print(json.dumps({'deployed':True,'sources':len(hashes),'root':root}))


def run():
    deployment = json.loads((LOCAL/'deployment.json').read_text())
    root = deployment['root']
    hashes = deployment['source_sha256']
    argv = [transport.PYTHON, '-u', root + '/operation_trace.py']
    code = f"""import sys,json,hashlib,os,subprocess; from pathlib import Path
root=Path({root!r}); expected={hashes!r}
assert {{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected}}==expected
pressure=subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True).strip()
assert pressure=='1', ('memory pressure',pressure)
s=os.statvfs('/'); assert s.f_bavail*s.f_frsize > 2*1024**3
preflight={{'pressure':pressure,'free_disk_bytes':s.f_bavail*s.f_frsize,'swap':subprocess.check_output(['sysctl','vm.swapusage'],text=True)}}
sys.path.insert(0,str(root)); import remote_actor
actor=remote_actor.run_owned({argv!r},root/'job-trace',deadline=150,reserve_bytes=2*1024**3)
print(json.dumps({{'actor':actor,'preflight':preflight}}),flush=True)
"""
    invocation = json.loads(transport.remote(code, timeout=240))
    exclusive('trace-actor.json', invocation)
    fetched = json.loads(transport.remote(f"""import json,base64,hashlib; from pathlib import Path
root=Path({root!r}); names=['operation-trace.json']+['job-trace/'+n for n in ['startup.json','pids.json','result.json','stdout','stderr']]+list({hashes!r})
files={{n:(root/n).read_bytes() for n in names if (root/n).is_file()}}
print(json.dumps({{n:{{'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}} for n,b in files.items()}}))
"""))
    for n,value in fetched.items():
        b = base64.b64decode(value['base64'])
        assert hashlib.sha256(b).hexdigest() == value['sha256']
        destination = LOCAL / 'readback' / n
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(b)
    guarded = json.loads(transport.remote(f"import json,hashlib; from pathlib import Path; r=Path({transport.GUARDED!r}); print(json.dumps({{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('*.py')}}))"))
    assert guarded == deployment['guarded_before']
    assert all(fetched[n]['sha256'] == digest for n,digest in hashes.items())
    actor = invocation['actor']
    assert actor == json.loads((LOCAL/'readback/job-trace/result.json').read_text())
    cleanup = actor['cleanup']
    exclusive('trace-readback.json', {'fetched_sha256':{n:v['sha256'] for n,v in fetched.items()},
        'guarded_unchanged':True, 'executed_source_exact':True, 'actor_cleanup':cleanup})
    print(json.dumps({'actor':actor,'guarded_unchanged':True},indent=2))
    assert actor['reason'] == 'completed' and cleanup['child_reaped'] and cleanup['group_gone']
    report = json.loads((LOCAL/'readback/operation-trace.json').read_text())
    assert report['case_count'] == len(report['cases']) == 4
    assert report['observer_exact_all_cases']
    for n,digest in report['source_sha256'].items():
        assert hashes[n] == digest
    print(json.dumps({'cases':4,'exact_observer_witnesses':True,'peak_mlx_bytes':report['peak_mlx_bytes'],
        'first_differences':[{'prompt':c['prompt_index'],'width':c['verification_width'],
            'nonexact':[r['operation'] for r in c['operations'] if not r['exact']][:5],
            'out_of_tolerance':[r['operation'] for r in c['operations'] if not r['close']][:5]}
            for c in report['cases']]},indent=2))


if __name__ == '__main__':
    {'deploy': deploy, 'run':run}[sys.argv[1]]()
