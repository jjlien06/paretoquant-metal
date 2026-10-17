"""Actual floating native-cache attention/future-prefix gate, no timing claim."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache
from stable_attention import stable_causal_attention
previous=mx.default_device();mx.set_default_device(mx.gpu)
old_limit=mx.set_cache_limit(128*1024**2);mx.reset_peak_memory();mx.random.seed(8671)
rows=[];poison_rows=[]
try:
 for hq,hkv,d in [(14,2,64),(40,8,128)]:
  for dtype in [mx.float16,mx.float32]:
   for width in [1,2,5,9]:
    for prefix in [32,256]:
     length=prefix+width
     q=mx.random.normal((1,hq,width,d)).astype(dtype)
     k=mx.random.normal((1,hkv,length,d)).astype(dtype)
     v=mx.random.normal((1,hkv,length,d)).astype(dtype)
     cache=KVCache();k,v=cache.update_and_fetch(k,v);mx.eval(q,k,v,cache.state)
     before=[np.array(t[..., :length,:]) for t in cache.state[:2]]
     mask=None if width==1 else "causal"
     actual=stable_causal_attention(q,k,v,cache=cache,scale=d**-.5,mask=mask)
     expected=mx.concatenate([mx.fast.scaled_dot_product_attention(mx.contiguous(q[...,i:i+1,:]),k[...,:prefix+i+1,:],v[...,:prefix+i+1,:],scale=d**-.5,mask=None) for i in range(width)],axis=-2)
     mx.eval(actual,expected)
     a,b=np.array(actual),np.array(expected)
     exact=bool(np.array_equal(a,b));finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
     immutable=cache.offset==length and all(np.array_equal(old,np.array(new[..., :length,:])) for old,new in zip(before,cache.state[:2]))
     row={'heads':hq,'kv_heads':hkv,'head_dim':d,'dtype':str(dtype),'width':width,'prefix':prefix,'shape':list(a.shape),'exact':exact,'finite':finite,'close':bool(finite and np.allclose(a,b,atol=.002,rtol=.002)),'cache_immutable':immutable}
     row['passed']=row['shape']==[1,hq,width,d] and exact and finite and immutable and row['close'];rows.append(row)
     for i in range(width-1):
      end=prefix+i+1
      pk=mx.concatenate([k[...,:end,:],mx.full((1,hkv,length-end,d),1000,dtype=dtype)],axis=-2)
      pv=mx.concatenate([v[...,:end,:],mx.full((1,hkv,length-end,d),-1000,dtype=dtype)],axis=-2)
      pc=KVCache();pk,pv=pc.update_and_fetch(pk,pv)
      out=stable_causal_attention(q,pk,pv,cache=pc,scale=d**-.5,mask="causal")
      mx.eval(out)
      safe=bool(np.array_equal(a[...,:i+1,:],np.array(out)[...,:i+1,:]))
      poison_rows.append({'heads':hq,'kv_heads':hkv,'head_dim':d,'dtype':str(dtype),'width':width,'prefix':prefix,'poison_after':end,'tested_last_query':i,'unchanged_earlier_outputs':safe,'cache_offset_unchanged':pc.offset==length})
 report={'passed':all(x['passed'] for x in rows) and all(x['unchanged_earlier_outputs'] and x['cache_offset_unchanged'] for x in poison_rows),'case_count':len(rows),'expected_case_count':32,'exact_case_count':sum(x['exact'] for x in rows),'close_case_count':sum(x['close'] for x in rows),'cases':rows,'future_poison_case_count':len(poison_rows),'future_poison_cases':poison_rows,'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['stable_attention.py',Path(__file__).name]},'peak_mlx_bytes':mx.get_peak_memory(),'fatal_traceback':None,'scope':'Random real native floating-cache attention on small and32B GQA geometries; no real32B generation or speed claim'}
 assert len(rows)==32 and len(poison_rows)==104
 with (ROOT/'operator-gate.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps({'passed':report['passed'],'cases':len(rows),'future_poison_cases':len(poison_rows)}),flush=True)
finally:
 mx.set_cache_limit(old_limit);mx.set_default_device(previous)
raise SystemExit(0 if report['passed'] else 1)
