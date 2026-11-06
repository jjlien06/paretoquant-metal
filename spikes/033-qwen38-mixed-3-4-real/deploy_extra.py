"""Deploy additional source/data exclusively and record exact remote hashes."""
import hashlib,json,os,pathlib,shlex,subprocess,sys
from remote import SSH,REMOTE,ROOT,RESULT,SPIKE
if __name__=='__main__':
    receipt={}
    for rel in sys.argv[1:]:
        p=SPIKE/rel;data=p.read_bytes();target=REMOTE+'/'+rel
        parent=str(pathlib.PurePosixPath(target).parent)
        r=subprocess.run(SSH+['mkdir -p '+shlex.quote(parent)+' && test ! -e '+shlex.quote(target)+' && /bin/cat > '+shlex.quote(target)],input=data,capture_output=True,timeout=60);r.check_returncode()
        r=subprocess.run(SSH+['shasum -a 256 '+shlex.quote(target)],capture_output=True,text=True,timeout=30);r.check_returncode();h=hashlib.sha256(data).hexdigest();assert h in r.stdout;receipt[rel]=h
    RESULT.mkdir(parents=True,exist_ok=True)
    name='deployment-'+hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()[:16]+'.json'
    with (RESULT/name).open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))
