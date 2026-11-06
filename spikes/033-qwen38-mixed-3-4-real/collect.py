"""Read back exact remote receipts and verify transfer SHA256."""
import hashlib,json,pathlib,shlex,subprocess
from remote import SSH,REMOTE,RESULT
FILES=['q3-hf-metadata.json','q3-download-receipt.json','download-guard.jsonl','download.log','small-guard.jsonl','small.log','small-guard-r2.jsonl','small-r2.log','small-parity-r2.json']
if __name__=='__main__':
    out=RESULT/'remote-evidence';out.mkdir(exist_ok=False);rows={}
    for name in FILES:
        remote=REMOTE+'/'+name
        r=subprocess.run(SSH+['/bin/cat '+shlex.quote(remote)],capture_output=True,timeout=60);r.check_returncode()
        data=r.stdout;h=hashlib.sha256(data).hexdigest()
        r=subprocess.run(SSH+['shasum -a 256 '+shlex.quote(remote)],capture_output=True,text=True,timeout=30);r.check_returncode();assert h in r.stdout
        with (out/name).open('xb') as f:f.write(data)
        rows[name]={'bytes':len(data),'sha256':h,'remote_sha256_equal':True}
    with (RESULT/'evidence-transfer.json').open('x') as f:json.dump(rows,f,indent=2)
    print(json.dumps(rows,indent=2))
