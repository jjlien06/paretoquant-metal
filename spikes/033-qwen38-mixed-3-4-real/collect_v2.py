"""Collect stage receipts without transferring checkpoints; verify exact bytes."""
import hashlib,json,pathlib,shlex,subprocess
from remote import SSH,REMOTE,RESULT
if __name__=='__main__':
    files=['small-v2-guard.jsonl','small-v2.log','calibration-v2-guard.jsonl','calibration-v2.log','sensitivity-v2.json.partial.jsonl','input-audit-guard.jsonl','input-audit.log','finish-stages-v2.log','final-assurance-v2.json','assurance-hf-q3.json','assurance-hf-q4.json','assurance-v2.log']
    for name in ['q4','mixed05','mixed15','q3']:
        files += ['eval-'+name+'-v2.json','eval-'+name+'-v2.log','eval-'+name+'-v2-guard.jsonl']
    for name in ['mixed05','mixed15']:
        files += ['build-'+name+'-v2.log','build-'+name+'-v2-guard.jsonl','checkpoint-'+name+'/mixed-manifest.json','checkpoint-'+name+'/config.json','checkpoint-'+name+'/model.safetensors.index.json']
    out=RESULT/'continuation-evidence';out.mkdir(exist_ok=False);rows={}
    for name in files:
        target=REMOTE+'/'+name
        r=subprocess.run(SSH+['/bin/cat '+shlex.quote(target)],capture_output=True,timeout=90);r.check_returncode()
        data=r.stdout;h=hashlib.sha256(data).hexdigest()
        r=subprocess.run(SSH+['shasum -a 256 '+shlex.quote(target)],capture_output=True,text=True,timeout=30);r.check_returncode();assert h in r.stdout
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(data)
        rows[name]={'sha256':h,'bytes':len(data),'remote_sha256_equal':True}
    with (RESULT/'continuation-transfer.json').open('x') as f:json.dump(rows,f,indent=2)
    print(json.dumps({'files':len(rows),'all_transfers_hash_equal':True}))
