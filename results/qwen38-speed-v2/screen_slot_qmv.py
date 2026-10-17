"""Actual BF16 packed checkpoint numerical/time diagnostic for isolated018.

Uses fixed random activations, not calibration/quality evidence. Timings are
interleaved warmed operator timings, not full-model throughput.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

ROOT=Path(__file__).resolve().parents[2]
RUNTIME=Path(os.environ.get('PARETOQUANT_SLOT_RUNTIME',str(ROOT/'spikes/018-qwen38-slot-qmv')))
sys.path.insert(0,str(RUNTIME))
from slot_qmv import slot_qmv


def main():
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(128*1024**2)
    mx.random.seed(557)
    fixture=Path(os.environ.get('PARETOQUANT_OPERATOR_FIXTURE',str(ROOT/'models/qwen38-operator-fixture-v2/operator.safetensors')))
    arrays=mx.load(str(fixture))
    rows=[]
    for name in ('gate_proj','up_proj','down_proj'):
        prefix='language_model.model.layers.0.mlp.'+name
        w,s,b=(arrays[prefix+'.'+part] for part in ('weight','scales','biases'))
        k=w.shape[1]*8
        for width in (1,2,3):
            x=mx.random.normal((1,width,k)).astype(mx.bfloat16);mx.eval(x)
            stock=lambda:mx.concatenate([mx.quantized_matmul(mx.contiguous(x[:,i:i+1,:]),w,s,b,transpose=True,group_size=64,bits=4) for i in range(width)],axis=1)
            custom=lambda:slot_qmv(x,w,s,b,bits=4,group_size=64)
            a,c=stock(),custom();mx.eval(a,c)
            diff=mx.abs(a.astype(mx.float32)-c.astype(mx.float32))
            exact=bool(mx.array_equal(a,c).item())
            strict=bool(mx.allclose(a,c,atol=1e-6,rtol=1e-5).item())
            samples={'stock':[],'custom':[]}
            for _ in range(8):mx.eval(stock(),custom())
            for trial in range(24):
                for label,call in ([('stock',stock),('custom',custom)] if trial%2==0 else [('custom',custom),('stock',stock)]):
                    started=time.perf_counter();mx.eval(call());samples[label].append(time.perf_counter()-started)
            row={'projection':name,'logical_shape':[w.shape[0],k],'width':width,'dtype':str(x.dtype),
                 'exact':exact,'strict_pass':strict,'max_abs_error':float(mx.max(diff).item()),
                 'samples_seconds':samples,'median_ms':{label:statistics.median(values)*1000 for label,values in samples.items()}}
            row['operator_speed_ratio']=row['median_ms']['stock']/row['median_ms']['custom']
            rows.append(row)
            print(json.dumps({key:value for key,value in row.items() if key!='samples_seconds'}),flush=True)
    receipt={'scope':'Actual weight operator screen only, no complete-generation/quality claim','rows':rows,
             'source_sha256':{n:hashlib.sha256((RUNTIME/n).read_bytes()).hexdigest() for n in ['slot_qmv.py','slot_qmv_header.metal','slot_qmv_body.metal']},
             'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'device':mx.device_info(),
             'peak_mlx_bytes':mx.get_peak_memory()}
    output=Path(os.environ.get('PARETOQUANT_SLOT_OUTPUT',str(Path(__file__).with_name('018-operator-m2-v1.json'))))
    with output.open('x') as f:json.dump(receipt,f,indent=2)

if __name__=='__main__':main()
