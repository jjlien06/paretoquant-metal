"""Exclusive frozen candidate deployment and bounded actual operator gate."""
import base64
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

LOCAL=Path(__file__).parent
TRANSPORT=Path('/Users/jeremylien/.hermes/cache/scratch/paretoquant-dual-host/acceleration-32b/sequential-physical-v1/control.py')
spec=importlib.util.spec_from_file_location('owned_transport',TRANSPORT)
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
CANDIDATE=Path('/Users/jeremylien/Projects/paretoquant-metal/spikes/010-width-stable-verification')
ACTOR=Path('/Users/jeremylien/Projects/paretoquant-metal/results/m4-sequential-verification-v1/source/remote_actor.py')
CORE={'slot_qmv.py':'bf0a643bd1dc3018a3890631862145625f4b48c12ca43a8910e9f1eb45cf7bfa',
    'slot_qmv_header.metal':'936aa5fefd86bb53bf3ae8ba66a512145f9d04e22350097dad7c4abfabd3c7aa',
    'slot_qmv_body.metal':'03f9ac57ad8fb1ab3d9424a2d677477e8cb81d5ca7527680a34b9dd18b6525df'}


def exclusive(name,data):
    with (LOCAL/name).open('x') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)


def guarded():
    return json.loads(transport.remote(f"import json,hashlib;from pathlib import Path;r=Path({transport.GUARDED!r});print(json.dumps({{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('*.py')}}))"))


def deploy():
    manifest=[]
    for line in (CANDIDATE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        p=Path(name)
        assert p.parent==CANDIDATE and hashlib.sha256(p.read_bytes()).hexdigest()==digest
        manifest.append({'name':p.name,'sha256':digest})
    assert len(manifest)==10 and len({r['name'] for r in manifest})==10
    sources={n:CANDIDATE/n for n in CORE}
    sources['LICENSE']=CANDIDATE/'LICENSE'
    sources['remote_actor.py']=ACTOR
    sources['operator_gate.py']=LOCAL/'operator_gate.py'
    data={n:p.read_bytes() for n,p in sources.items()}
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in data.items()}
    assert all(hashes[n]==digest for n,digest in CORE.items())
    root='/Users/jeremylien/Projects/paretoquant-dual-host/spikes/010-slot-operator-'+str(time.time_ns())
    payload={'root':root,'files':{n:base64.b64encode(b).decode() for n,b in data.items()}}
    readback=json.loads(transport.remote("""import sys,json,hashlib,base64;from pathlib import Path
p=json.load(sys.stdin);r=Path(p['root']);r.mkdir(mode=0o700)
for n,b in p['files'].items():
 assert '/'not in n
 with (r/n).open('xb')as f:f.write(base64.b64decode(b))
print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in p['files']}))
""",data=json.dumps(payload)))
    assert readback==hashes
    exclusive('deployment.json',{'root':root,'source_sha256':hashes,'candidate_manifest':manifest,
        'readback_verified':True,'guarded_before':guarded(),'scope':'Independent array operator gate; no decoder/library modification'})
    print(json.dumps({'deployed':True,'sources':len(hashes),'root':root}))


def run():
    deployment=json.loads((LOCAL/'deployment.json').read_text())
    root=deployment['root']; hashes=deployment['source_sha256']
    argv=[transport.PYTHON,'-u',root+'/operator_gate.py']
    invocation=json.loads(transport.remote(f"""import sys,json,hashlib,os,subprocess;from pathlib import Path
r=Path({root!r});expected={hashes!r}
assert {{n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in expected}}==expected
pressure=subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True).strip()
assert pressure=='1',('memory pressure',pressure)
s=os.statvfs('/');assert s.f_bavail*s.f_frsize>2*1024**3
preflight={{'pressure':pressure,'disk_free_bytes':s.f_bavail*s.f_frsize,'swap':subprocess.check_output(['sysctl','vm.swapusage'],text=True)}}
sys.path.insert(0,str(r));import remote_actor
actor=remote_actor.run_owned({argv!r},r/'job-operator',deadline=150,reserve_bytes=2*1024**3)
print(json.dumps({{'actor':actor,'preflight':preflight}}),flush=True)
""",timeout=240))
    exclusive('operator-actor.json',invocation)
    fetched=json.loads(transport.remote(f"""import json,hashlib,base64;from pathlib import Path
r=Path({root!r});names=['operator-gate.json','case-progress.jsonl']+['job-operator/'+n for n in ['startup.json','pids.json','result.json','stdout','stderr']]+list({hashes!r})
files={{n:(r/n).read_bytes()for n in names if(r/n).is_file()}}
print(json.dumps({{n:{{'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}}for n,b in files.items()}}))
"""))
    for n,value in fetched.items():
        b=base64.b64decode(value['base64']);assert hashlib.sha256(b).hexdigest()==value['sha256']
        p=LOCAL/'readback'/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as stream:stream.write(b)
    assert all(fetched[n]['sha256']==digest for n,digest in hashes.items())
    assert guarded()==deployment['guarded_before']
    actor=invocation['actor']
    assert actor==json.loads((LOCAL/'readback/job-operator/result.json').read_text())
    exclusive('operator-readback.json',{'executed_source_exact':True,'guarded_unchanged':True,
        'actor_cleanup':actor['cleanup'],'fetched_sha256':{n:v['sha256']for n,v in fetched.items()}})
    report=json.loads((LOCAL/'readback/operator-gate.json').read_text()) if'operator-gate.json'in fetched else None
    if report is not None:
        assert all(hashes[n]==digest for n,digest in report['source_sha256'].items())
    print(json.dumps({'actor':actor,'guarded_unchanged':True,'report':None if report is None else{
        'passed':report['passed'],'cases':report['case_count'],'expected':report['expected_case_count'],
        'exact':report['exact_case_count'],'close':report['close_case_count'],'peak_mlx_bytes':report['peak_mlx_bytes'],
        'fatal_traceback':report['fatal_traceback']}},indent=2))
    assert actor['cleanup']['child_reaped'] and actor['cleanup']['group_gone']
    assert actor['reason']=='completed' and actor['returncode']==0 and report is not None and report['passed']


if __name__=='__main__':
    {'deploy':deploy,'run':run}[sys.argv[1]]()
