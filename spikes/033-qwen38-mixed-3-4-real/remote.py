"""Pinned SSH transport for spike033; no GPU imports locally."""
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
SPIKE=ROOT/'spikes/033-qwen38-mixed-3-4-real'
RESULT=ROOT/'results/qwen38-mixed-3-4-real-v1'
REMOTE='/Users/jeremylien/Projects/paretoquant-dual-host/spikes/033-qwen38-mixed-3-4-real'
PY='/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=fe80::56:c507:c02a:a630%bridge0','-i',str(pathlib.Path.home()/'.ssh/paretoquant_mini_ed25519'),'jeremylien@fe80::1456:cf50:88f2:90d9%bridge0']
def run(command,timeout=60):
    r=subprocess.run(SSH+[command],text=True,capture_output=True,timeout=timeout)
    print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
    return r
if __name__=='__main__':
    import shlex
    if sys.argv[1]=='deploy':
        run('mkdir -p '+shlex.quote(REMOTE)).check_returncode()
        hashes={}
        for p in sorted(SPIKE.glob('*.py')):
            data=p.read_bytes(); target=REMOTE+'/'+p.name
            r=subprocess.run(SSH+["test ! -e "+shlex.quote(target)+" && /bin/cat > "+shlex.quote(target)],input=data,capture_output=True)
            if r.returncode: raise RuntimeError((p.name,r.stderr))
            h=hashlib.sha256(data).hexdigest();hashes[p.name]=h
            r=run('shasum -a 256 '+shlex.quote(target));assert h in r.stdout
        RESULT.mkdir(parents=True,exist_ok=True)
        with (RESULT/'deployed-source-hashes.json').open('x') as f:json.dump(hashes,f,indent=2)
    elif sys.argv[1]=='exec':
        r=run(sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 600);sys.exit(r.returncode)
    elif sys.argv[1]=='get':
        rel=sys.argv[2];target=RESULT/rel;target.parent.mkdir(parents=True,exist_ok=True)
        r=subprocess.run(SSH+['/bin/cat '+shlex.quote(REMOTE+'/'+rel)],capture_output=True,timeout=60);r.check_returncode()
        with target.open('xb') as f:f.write(r.stdout)
        h=hashlib.sha256(r.stdout).hexdigest();r=run('shasum -a 256 '+shlex.quote(REMOTE+'/'+rel));assert h in r.stdout
        print(str(target))
