"""Exclusive two-Mac spike deployment with exact remote source readback."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
SSH=['ssh','-i','/Users/jeremylien/.ssh/paretoquant_mini_ed25519',
     '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=6',
     '-o','HostKeyAlias=fe80::56:c507:c02a:a630%bridge0',
     'jeremylien@fe80::14b2:4360:a750:94e5%bridge0']
REMOTE=r'''
from pathlib import Path
import sys,json,hashlib,subprocess,os
p=json.load(sys.stdin)
name=p['name'];assert name and '/' not in name and '\\' not in name
root=Path('/Users/jeremylien/Projects/paretoquant-dual-host/spikes')
version=p['version'];assert version in {'v1','v2','v3','v4','v5'}
dest=root/(name+'-20261016-'+version);dest.mkdir(parents=True,exist_ok=False)
for n,s in p['files'].items():
 assert Path(n).name==n and Path(n).suffix in {'.py','.json','.metal'}
 with (dest/n).open('x') as f:f.write(s)
actual={n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in p['files']}
assert actual==p['hashes']
env=dict(os.environ);env['TMPDIR']=str(Path.home()/'.hermes/cache/scratch')
r=subprocess.run(['/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python','-m','pytest','-q',str(dest)],env=env,capture_output=True,text=True,timeout=135)
print(json.dumps({'source_sha256':actual,'remote_root':str(dest),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}));sys.exit(r.returncode)
'''


def main():
    name=sys.argv[1]
    version=sys.argv[2] if len(sys.argv)>2 else 'v1'
    assert version in {'v1','v2','v3','v4','v5'}
    assert Path(name).name==name
    source=(ROOT/'spikes'/name).resolve(strict=True)
    assert source.parent==ROOT/'spikes'
    files={p.name:p.read_text() for p in source.iterdir() if p.suffix in {'.py','.json','.metal'} and p.is_file()}
    hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()}
    result=subprocess.run([*SSH,'/usr/bin/python3 -c '+shlex.quote(REMOTE)],
                          input=json.dumps({'name':name,'version':version,'files':files,'hashes':hashes}),
                          capture_output=True,text=True,timeout=160)
    if not result.stdout.strip():raise RuntimeError(result.stderr)
    receipt=json.loads(result.stdout)
    output=Path(__file__).with_name(name+'-deployment-m4-tests'+('' if version=='v1' else '-'+version)+'.json')
    with output.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))
    assert receipt['source_sha256']==hashes
    if result.returncode:raise RuntimeError(f"Physical tests failed: {result.returncode}")

if __name__=='__main__':main()
