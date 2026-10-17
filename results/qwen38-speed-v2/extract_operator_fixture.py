"""Extract actual packed checkpoint matrices, with trusted payload hash checks.

Read-only on the mini. Writes one new local operator fixture, never a checkpoint
replacement. No MLX import or GPU job is used during extraction.
"""
import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'models/qwen38-operator-fixture-v2'
SOURCE = '/Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
PREFIXES = ['language_model.model.layers.0.mlp.' + n for n in ('gate_proj', 'up_proj', 'down_proj')]
PREFIXES += ['language_model.model.layers.0.linear_attn.conv1d']

REMOTE = r'''
import sys,json,struct,os,hashlib
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['source']);index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
keys=sorted(k for k in index if any(k.startswith(prefix+'.') for prefix in p['prefixes']))
assert set(keys)==set(p['expected'])
headers={};entries={};offset=0;payloads=[]
for name in keys:
 file=root/index[name]
 if file.name not in headers:
  with file.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];assert n<16*1024**2
   headers[file.name]=(8+n,json.loads(f.read(n)))
 data_start,header=headers[file.name];entry=header[name];a,b=entry['data_offsets']
 with file.open('rb') as f:f.seek(data_start+a);content=f.read(b-a)
 assert len(content)==b-a and hashlib.sha256(content).hexdigest()==p['expected'][name]
 entries[name]={'dtype':entry['dtype'],'shape':entry['shape'],'data_offsets':[offset,offset+len(content)]}
 offset+=len(content);payloads.append(content)
assert offset<256*1024**2
h=json.dumps(entries,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
sys.stdout.buffer.write(struct.pack('<Q',len(h)));sys.stdout.buffer.write(h)
for data in payloads:sys.stdout.buffer.write(data)
sys.stdout.buffer.flush()
print(json.dumps({'tensor_count':len(entries),'data_bytes':offset,'source':p['source']}),file=sys.stderr)
'''


def main():
    receipt=json.loads((ROOT/'results/qwen38-speed-v2/control-before-v1/rank1.receipt.json').read_text())
    actual=json.loads(receipt['stdout'])['weight_loading']['tensor_sha256']
    expected={k:v for k,v in actual.items() if any(k.startswith(prefix+'.') for prefix in PREFIXES)}
    assert len(expected)==10, sorted(expected)
    OUT.mkdir(exist_ok=False)
    path=OUT/'operator.safetensors'
    ssh=['ssh','-i','/Users/jeremylien/.ssh/paretoquant_mini_ed25519',
         '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=6',
         '-o','HostKeyAlias=fe80::56:c507:c02a:a630%bridge0',
         'jeremylien@fe80::14b2:4360:a750:94e5%bridge0',
         '/usr/bin/python3 -c '+shlex.quote(REMOTE)]
    with path.open('xb') as destination:
        result=subprocess.run(ssh,input=json.dumps({'source':SOURCE,'prefixes':PREFIXES,'expected':expected}).encode(),
                              stdout=destination,stderr=subprocess.PIPE,timeout=80)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));start=8+n
        assert set(header)==set(expected)
        for name,entry in header.items():
            a,b=entry['data_offsets'];f.seek(start+a);data=f.read(b-a)
            assert len(data)==b-a and hashlib.sha256(data).hexdigest()==expected[name],name
    summary={'source':SOURCE,'fixture':str(path),'tensor_sha256':expected,
             'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'remote_extraction':json.loads(result.stderr),
             'purpose':'Actual operator fixture only; not complete-model generation evidence'}
    with (OUT/'manifest.json').open('x') as out:json.dump(summary,out,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
