"""Parent-only bounded read-only checkpoint/link admission preparation."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import argparse
import time

ROOT = Path('/Users/jeremylien/Projects/paretoquant-metal')
OUT = ROOT / 'results/qwen38-prefill-overlap-v1'
SMALL = Path('/Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/small-control')
META = ROOT / 'models/qwen3.8-27b-4bit-tokenizer-only'
HEAD = ROOT / 'models/qwen3.8-27b-mtp-4bit-b643c01b6d3b094e325edb6ebd832e16c486c575'
TRUSTED = Path('/Users/jeremylien/Projects/local-agent-dual-mac/results/mlx-vision-admission/readback-admission.json')
parser = argparse.ArgumentParser()
parser.add_argument('--private-link', required=True)
link = json.loads(Path(parser.parse_args().private_link).read_text())
assert link['host_key_verified_using_existing_alias'] is True
receipt = {'schema': 'overlap-readonly-checkpoint-preflight-v2', 'started_at': time.time(),
           'status': 'failed', 'model_imported': False, 'gpu_execution': False,
           'hardware_execution_approved': False, 'local': {}, 'remote': None}

def pressure():
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                             capture_output=True, text=True, check=True, timeout=3).stdout)

def hashed(path):
    before = path.stat()
    h = hashlib.sha256()
    count = 0
    with path.open('rb') as f:
        while True:
            block = f.read(8 * 1024**2)
            if not block:
                break
            h.update(block)
            count += len(block)
            if count % (512 * 1024**2) == 0 and pressure() != 1:
                raise RuntimeError('pressure prerequisite lost during read')
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError('file changed during read')
    return {'bytes': count, 'sha256': h.hexdigest()}

try:
    if pressure() != 1:
        raise RuntimeError('local pressure must be normal before readback')
    admission_raw = TRUSTED.read_bytes()
    admission = json.loads(admission_raw)
    receipt['trusted_checkpoint_admission_sha256'] = hashlib.sha256(admission_raw).hexdigest()
    receipt['local']['head'] = hashed(HEAD / 'model.safetensors')
    if receipt['local']['head'] != {'bytes': 238934137, 'sha256': '76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6'}:
        raise ValueError('trusted MTP head mismatch')
    receipt['local']['metadata'] = {}
    for name, expected in admission['metadata_sha256'].items():
        value = hashed(META / name)
        if value['sha256'] != expected:
            raise ValueError('target metadata mismatch: ' + name)
        receipt['local']['metadata'][name] = value
    small_config = hashed(SMALL / 'config.json')
    small_index = hashed(SMALL / 'model.safetensors.index.json')
    if small_config['sha256'] != 'ba7770da23eae5ebd6827571f086e331956b33f4442a9e876fb4aa10969a6772':
        raise ValueError('small config mismatch')
    if small_index['sha256'] != '6e48f2fa5d6f033a6d77bf833abfa9698ca24d1715ecea4c67447bcfaee44650':
        raise ValueError('small index mismatch')
    receipt['local']['small_weight'] = hashed(SMALL / 'model.safetensors')
    receipt['local']['small_config'] = small_config
    receipt['local']['small_index'] = small_index
    addresses = {'--local-address': link['local_address'], '--remote-address': link['remote_address']}
    route = subprocess.run(['route', '-n', 'get', addresses['--remote-address']],
                           capture_output=True, text=True, check=True, timeout=5).stdout
    match = re.search(r'^\s*interface:\s*(\S+)', route, re.M)
    if not match or match[1] != 'bridge0':
        raise ValueError('local route is not bridge0')
    receipt['local']['route_interface'] = match[1]
    payload = {'checkpoint_dir': admission['checkpoint_dir'], 'shards': admission['shards'],
               'metadata': admission['metadata_sha256'], 'small_dir': '/Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462',
               'small_weight': receipt['local']['small_weight'], 'small_config': small_config,
               'small_index': small_index, 'peer': addresses['--local-address']}
    remote = r'''import hashlib,json,signal,subprocess,sys,time,re
from pathlib import Path
signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError('read deadline')))
signal.alarm(160)
x=json.loads(sys.stdin.buffer.read(2*1024**2));r={'shards':[],'metadata':{},'pressure_samples':[]}
def pressure():
 p=int(subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True,check=True,timeout=3).stdout);r['pressure_samples'].append({'timestamp':time.time(),'level':p})
 if p!=1:raise ValueError('remote pressure not normal')
def hashed(p):
 if p.is_symlink():raise ValueError('symlink rejected')
 b=p.stat();h=hashlib.sha256();n=0
 with p.open('rb') as f:
  while True:
   block=f.read(8*1024**2)
   if not block:break
   h.update(block);n+=len(block)
   if n%(512*1024**2)==0:pressure()
 a=p.stat()
 if (b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns)!=(a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns):raise ValueError('file changed during read')
 return {'bytes':n,'sha256':h.hexdigest()}
pressure();root=Path(x['checkpoint_dir'])
for e in x['shards']:
 v=hashed(root/e['name'])
 if v['bytes']!=e['bytes'] or v['sha256']!=e['sha256']:raise ValueError('publisher shard mismatch')
 r['shards'].append({'file':e['name'],**v})
for name,h in x['metadata'].items():
 v=hashed(root/name)
 if v['sha256']!=h:raise ValueError('target metadata mismatch')
 r['metadata'][name]=v
s=Path(x['small_dir'])
for name,expected in [('model.safetensors',x['small_weight']),('config.json',x['small_config']),('model.safetensors.index.json',x['small_index'])]:
 v=hashed(s/name)
 if v!=expected:raise ValueError('small local/remote byte mismatch')
r['small_weight_matches_local']=True
route=subprocess.run(['route','-n','get',x['peer']],capture_output=True,text=True,check=True,timeout=5).stdout
m=re.search(r'^\s*interface:\s*(\S+)',route,re.M)
if not m or m[1]!='bridge0':raise ValueError('remote route not bridge0')
r['route_interface']=m[1];pressure();r['disk']=subprocess.run(['df','-h','/System/Volumes/Data'],capture_output=True,text=True,check=True,timeout=5).stdout
print(json.dumps(r))
'''
    profile = json.loads((ROOT / 'results/qwen38-speed-v2/fast-profile-v2.json').read_text())['parameters']
    call = subprocess.run(['ssh', '-T', '-i', profile['ssh_key'], '-o', 'BatchMode=yes',
                           '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
                           '-o', 'ConnectTimeout=8', '-o', 'HostKeyAlias=' + profile['ssh_host_key_alias'],
                           link['ssh_target'], shlex.join([
                               '/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python',
                               '-B', '-c', remote])], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=180)
    if call.returncode != 0:
        raise RuntimeError('remote read-only verification failed: ' + call.stderr[-1200:])
    if len(call.stdout.encode()) > 128 * 1024:
        raise ValueError('readback bound')
    receipt['remote'] = json.loads(call.stdout)
    receipt['local']['final_pressure'] = pressure()
    if receipt['local']['final_pressure'] != 1:
        raise RuntimeError('local pressure prerequisite lost')
    receipt['status'] = 'checkpoint-bytes-and-private-bridge-routes-verified-readonly'
    receipt['limits'] = ['Local/remote small bytes agree; no new publisher-origin attestation for small weights.',
                         'Target whole shards match previously pinned publisher SHA256; MTP head matches pinned runtime constant.',
                         'No model/device import, loaded memory fit, source deployment or physical numerical admission.',
                         'Pressure samples are dated read-only observations; refresh before model loading.']
except Exception as exc:
    receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    receipt['finished_at'] = time.time()
    path = OUT / ('readonly-preflight-' + str(time.time_ns()) + '.json')
    with path.open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps({'status': receipt['status'], 'error': receipt.get('error'),
                      'remote_shards_verified': len((receipt['remote'] or {}).get('shards', [])),
                      'MTP_head_verified': receipt['local'].get('head'),
                      'local_pressure': receipt['local'].get('final_pressure'),
                      'artifact': str(path)}, indent=2))
sys.exit(0 if receipt['status'].endswith('verified-readonly') else 1)
