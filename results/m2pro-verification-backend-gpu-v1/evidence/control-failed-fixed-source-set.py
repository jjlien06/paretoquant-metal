import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
import remote_actor
pressure=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True));assert pressure==1
s=os.statvfs('/');assert s.f_bavail*s.f_frsize>4*1024**3
hashes=json.loads((ROOT/'deployment.json').read_text())['source_sha256']
actor=remote_actor.run_owned([sys.executable,'-u',str(ROOT/'operator_gate.py')],ROOT/'job-adapter',deadline=150,reserve_bytes=4*1024**3,source_root=ROOT,expected_hashes=hashes)
with (ROOT/'actor.json').open('x') as f:json.dump({'actor':actor,'preflight':{'pressure':pressure,'free_disk_bytes':s.f_bavail*s.f_frsize}},f,indent=2)
print(json.dumps(actor,indent=2))
assert actor['reason']=='completed' and actor['returncode']==0 and actor['cleanup']['child_reaped'] and actor['cleanup']['group_gone']
report=json.loads((ROOT/'adapter-gate.json').read_text());assert report['passed']
print(json.dumps({'passed':report['passed'],'cases':report['case_count'],'device':report['device'],'peak_mlx_bytes':report['peak_mlx_bytes']},indent=2))
