"""Fetch pinned q3, streamed to disk; verify LFS sizes and SHA256."""
import hashlib,json,pathlib,shutil,time,urllib.request
from guard import preflight
ROOT=pathlib.Path(__file__).resolve().parent
REPO='lukaskremla/Qwen3.8-27B-3bit-MLX';REV='b3503b31b36125107e12a2b57fbe690c37bfc053'
def getjson(url):
    with urllib.request.urlopen(url,timeout=60) as r:return json.load(r)
def dump(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2)
if __name__=='__main__':
    preflight();api=getjson(f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true');assert api['sha']==REV
    out=ROOT/'q3-source';out.mkdir(exist_ok=False)
    dump(ROOT/'q3-hf-metadata.json',api)
    files=[x for x in api['siblings'] if x['rfilename'].endswith(('.safetensors','.json','.jinja','.txt','.model'))]
    total=sum(x['size'] for x in files)
    free=shutil.disk_usage(ROOT).free
    print(json.dumps({'download_bytes':total,'free_bytes':free,'reserve_bytes':8*2**30}),flush=True)
    assert free>total+8*2**30
    rows=[]
    for info in files:
        name=info['rfilename'];assert '..' not in pathlib.Path(name).parts
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        expected=info.get('lfs',{}).get('sha256')
        if name.endswith('.safetensors'):assert expected and info['lfs']['size']==info['size']
        h=hashlib.sha256();n=0;url=f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'
        with urllib.request.urlopen(url,timeout=120) as src,p.with_suffix(p.suffix+'.partial').open('xb') as dst:
            while True:
                data=src.read(4*2**20)
                if not data:break
                n+=len(data);h.update(data);dst.write(data)
        assert n==info['size'],(name,n,info['size'])
        if expected:assert h.hexdigest()==expected,(name,h.hexdigest(),expected)
        p.with_suffix(p.suffix+'.partial').rename(p)
        row={'file':name,'bytes':n,'sha256':h.hexdigest(),'expected_lfs_sha256':expected,'verified':True};rows.append(row)
        print(json.dumps(row),flush=True)
    dump(ROOT/'q3-download-receipt.json',{'repo':REPO,'revision':REV,'bytes':total,'files':rows,'verified':True})
