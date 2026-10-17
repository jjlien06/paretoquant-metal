"""Actual Metal compilation and stock width-one QMV differential gate."""
import hashlib
import importlib.metadata
import itertools
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0,str(ROOT))
import mlx.core as mx
import numpy as np
from slot_qmv import slot_qmv

ATOL=RTOL=0.002
GEOMETRIES=[(32,16,32),(256,24,64),(384,24,128),(512,64,64),(896,128,64),(4864,128,64),(5120,128,64)]
DTYPES=[mx.float16,mx.float32]
WIDTHS=[1,2,5,9]
BITS=[3,4]
previous_device=mx.default_device()
mx.set_default_device(mx.gpu)
old_cache_limit=mx.set_cache_limit(128*1024*1024)
mx.reset_peak_memory()
rows=[]
fatal=None
expected=len(GEOMETRIES)*len(DTYPES)*len(BITS)*len(WIDTHS)
try:
    mx.random.seed(7391)
    for (k,n,group),dtype,bits in itertools.product(GEOMETRIES,DTYPES,BITS):
        full=mx.random.normal((n,k),dtype=dtype)*mx.array(.2,dtype=dtype)
        weights,scales,biases=mx.quantize(full,group_size=group,bits=bits)
        mx.eval(weights,scales,biases)
        del full
        for width in WIDTHS:
            row={'k':k,'n':n,'group_size':group,'bits':bits,'width':width,'dtype':str(dtype),'fast':k%512==0}
            try:
                x=mx.random.normal((1,width,k),dtype=dtype)*mx.array(.25,dtype=dtype)
                control=mx.concatenate([mx.quantized_matmul(x[:,i:i+1,:],weights,scales,biases,
                    transpose=True,group_size=group,bits=bits) for i in range(width)],axis=1)
                candidate=slot_qmv(x,weights,scales,biases,bits=bits,group_size=group)
                mx.eval(control,candidate)
                assert control.shape==candidate.shape==(1,width,n)
                a=np.array(candidate).astype(np.float32)
                b=np.array(control).astype(np.float32)
                finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
                delta=np.abs(a-b)
                close=bool(finite and np.allclose(a,b,atol=ATOL,rtol=RTOL))
                row.update({'finite':finite,'exact':bool(np.array_equal(a,b)),
                    'close':close,'max_abs_error':float(delta.max()),
                    'out_of_tolerance':int(np.sum(delta>ATOL+RTOL*np.abs(b))),
                    'shape':list(candidate.shape)})
            except BaseException as exc:
                row.update({'close':False,'exception_type':type(exc).__name__,'exception':str(exc)})
                fatal=traceback.format_exc()
            rows.append(row)
            with (ROOT/'case-progress.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            if fatal is not None:
                print(fatal,file=sys.stderr,flush=True)
                raise RuntimeError('Actual GPU operator gate encountered an execution error')
    assert len(rows)==expected
except BaseException:
    if fatal is None:
        fatal=traceback.format_exc()
    print(fatal,file=sys.stderr,flush=True)
finally:
    passed=fatal is None and len(rows)==expected and all(row['close'] for row in rows)
    report={'scope':'Actual GPU array operator gate only; random quantized fixtures, not actual checkpoint/full32B/speculative speed admission',
        'passed':passed,'case_count':len(rows),'expected_case_count':expected,
        'exact_case_count':sum(row.get('exact',False) for row in rows),
        'close_case_count':sum(row.get('close',False) for row in rows),
        'cases':rows,'fatal_traceback':fatal,'atol':ATOL,'rtol':RTOL,
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['slot_qmv.py','slot_qmv_header.metal','slot_qmv_body.metal',Path(__file__).name]},
        'versions':{n:importlib.metadata.version(n) for n in ['mlx','mlx-metal']},
        'device':mx.device_info(),'peak_mlx_bytes':mx.get_peak_memory()}
    with (ROOT/'operator-gate.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'passed':passed,'cases':len(rows),'expected':expected,'exact':report['exact_case_count'],
        'close':report['close_case_count'],'max_abs_error':max((r.get('max_abs_error',0) for r in rows),default=0),
        'fatal':fatal is not None,'peak_mlx_bytes':report['peak_mlx_bytes']}),flush=True)
    mx.set_cache_limit(old_cache_limit)
    mx.set_default_device(previous_device)
raise SystemExit(0 if passed else 1)
