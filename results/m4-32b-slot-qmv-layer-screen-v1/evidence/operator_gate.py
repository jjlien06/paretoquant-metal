"""Bounded actual32B weight operator screen, not whole-model acceleration."""
import hashlib,json,statistics,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
import mlx.core as mx
import numpy as np
from slot_qmv import slot_qmv
CHECKPOINT=Path('/Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen32-2938092373e5f97b95538884112085364c2da315')
EXPECTED={'model.layers.0.mlp.gate_proj.weight': '07ed99780ae3a9777a5fc727f746819d516c60fbadf68fcd0aef0a75d5abccd1', 'model.layers.0.mlp.gate_proj.scales': '123a894a12ebc509ba4708052e0f2c52de35715bede58c3c1039c3fb2b735426', 'model.layers.0.mlp.gate_proj.biases': '4b6c82106f1b50cc0773e26fe2ee6215ce826e81618f5175557384e93875cc80', 'model.layers.0.mlp.down_proj.weight': '6cea41e0b63c160870ca201578b31a85a1a1cd356078d784a8382c68f5a1935b', 'model.layers.0.mlp.down_proj.scales': 'a7fdadb28896f9657bf5f44f03e358a4a3c0032303f160b9d28cdd219c5a0bfb', 'model.layers.0.mlp.down_proj.biases': 'a16be96a119c285d69485c6d27f6ecf9963b40b48f83308db74e2ed06ad09e32'}
index_path=CHECKPOINT/'model.safetensors.index.json';config_path=CHECKPOINT/'config.json'
metadata_before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [index_path,config_path]}
index=json.loads(index_path.read_text())['weight_map']
def tensor(name):
 shard=index[name];assert Path(shard).name==shard
 path=CHECKPOINT/shard;assert not path.is_symlink()
 with path.open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];assert 0<size<16*1024**2
  desc=json.loads(f.read(size))[name];lo,hi=desc['data_offsets']
  assert 0<hi-lo<128*1024**2
  f.seek(8+size+lo);payload=f.read(hi-lo)
 assert hashlib.sha256(payload).hexdigest()==EXPECTED[name]
 assert desc['dtype'] in ['U32','F16']
 view=np.frombuffer(payload,dtype={'U32':np.uint32,'F16':np.float16}[desc['dtype']]).reshape(desc['shape'])
 arr=mx.array(view);mx.eval(arr);return arr
previous=mx.default_device();mx.set_default_device(mx.gpu)
old_limit=mx.set_cache_limit(128*1024**2);mx.reset_peak_memory();mx.random.seed(9837)
rows=[]
try:
 for projection in ['gate_proj','down_proj']:
  p='model.layers.0.mlp.'+projection
  weights,scales,biases=[tensor(p+'.'+n) for n in ['weight','scales','biases']]
  k=weights.shape[1]*8;n=weights.shape[0]
  assert k in [5120,27648] and n in [5120,27648] and scales.shape==(n,k//64)
  for width in [1,2,5,9]:
   x=mx.random.normal((1,width,k)).astype(mx.float16);mx.eval(x)
   def stock():
    return mx.concatenate([mx.quantized_matmul(x[:,i:i+1,:],weights,scales,biases,transpose=True,group_size=64,bits=4) for i in range(width)],axis=1)
   def shared():return slot_qmv(x,weights,scales,biases,bits=4,group_size=64)
   a,b=stock(),shared();mx.eval(a,b)
   na,nb=np.array(a),np.array(b)
   exact=bool(np.array_equal(na,nb));finite=bool(np.isfinite(na).all() and np.isfinite(nb).all())
   close=bool(finite and np.allclose(na,nb,atol=.002,rtol=.002))
   samples=[]
   if exact and close:
    for _ in range(2):mx.eval(stock(),shared());mx.synchronize()
    for trial in range(8):
     order=['stock_single_slots','shared_slots'] if trial%2==0 else ['shared_slots','stock_single_slots']
     for name in order:
      fn=stock if name=='stock_single_slots' else shared
      mx.synchronize();started=time.perf_counter_ns()
      for _ in range(3):out=fn();mx.eval(out)
      mx.synchronize();elapsed=(time.perf_counter_ns()-started)/3e9
      samples.append({'trial':trial,'path':name,'seconds_per_evaluated_call':elapsed,'evaluations':3})
   medians={name:statistics.median(s['seconds_per_evaluated_call'] for s in samples if s['path']==name) for name in ['stock_single_slots','shared_slots']} if samples else None
   rows.append({'projection':p,'K':k,'N':n,'width':width,'dtype':str(x.dtype),'bits':4,'group_size':64,'exact':exact,'finite':finite,'close':close,'passed':exact and close,'samples':samples,'medians':medians,'isolated_speed_ratio':medians['stock_single_slots']/medians['shared_slots'] if medians else None})
   print(json.dumps({'projection':projection,'width':width,'exact':exact,'ratio':rows[-1]['isolated_speed_ratio']}),flush=True)
  del weights,scales,biases,x,a,b;mx.clear_cache()
 assert len(rows)==8
 assert metadata_before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [index_path,config_path]}
 report={'passed':all(x['passed'] for x in rows),'case_count':len(rows),'expected_case_count':8,'exact_case_count':sum(x['exact'] for x in rows),'close_case_count':sum(x['close'] for x in rows),'cases':rows,'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['slot_qmv.py','slot_qmv_header.metal','slot_qmv_body.metal',Path(__file__).name]},'tensor_sha256':EXPECTED,'checkpoint_metadata_sha256':metadata_before,'peak_mlx_bytes':mx.get_peak_memory(),'fatal_traceback':None,'scope':'Random F16 activations with unchanged genuine32B layer0 gate/down weights; isolated synchronized graph-build+GPU-call comparison against concatenated stock width-one QMV; not whole32B, native batchedQMM, or inference speedup'}
 with (ROOT/'operator-gate.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
finally:
 mx.set_cache_limit(old_limit);mx.set_default_device(previous)
raise SystemExit(0 if report['passed'] else 1)
