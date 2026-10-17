"""Local identical-input row-split numerical diagnostic (no collectives)."""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn


def probe():
    mx.random.seed(731)
    layer=nn.QuantizedLinear(256,128,bias=False,group_size=32,bits=4)
    layer.apply(lambda a:a.astype(mx.bfloat16) if mx.issubdtype(a.dtype,mx.floating) else a)
    x=mx.random.normal((1,1,256)).astype(mx.bfloat16)
    mx.eval(x,layer.parameters())
    full=layer(x)
    parts=[]; precise=[]
    for r in (0,1):
        xx=mx.contiguous(mx.split(x,2,axis=-1)[r])
        w,s,b=[mx.contiguous(mx.split(a,2,axis=-1)[r]) for a in (layer.weight,layer.scales,layer.biases)]
        kw=dict(transpose=True,group_size=32,bits=4)
        parts.append(mx.quantized_matmul(xx,w,scales=s,biases=b,**kw))
        precise.append(mx.quantized_matmul(xx.astype(mx.float32),w,scales=s.astype(mx.float32),biases=b.astype(mx.float32),**kw))
    variants={'bf16_partials_bf16_sum':parts[0]+parts[1],
              'rounded_partials_f32_sum':(parts[0].astype(mx.float32)+parts[1].astype(mx.float32)).astype(mx.bfloat16),
              'f32_partials_f32_sum_castback':(precise[0]+precise[1]).astype(mx.bfloat16)}
    rows={}
    for name,y in variants.items():
        d=mx.abs(full.astype(mx.float32)-y.astype(mx.float32))
        rows[name]=dict(max_abs=float(mx.max(d).item()),nonexact=int(mx.sum(full!=y).item()),dtype=str(y.dtype))
    rows['partial_rounding_witness']=dict(max_abs=float(mx.max(mx.abs(parts[0].astype(mx.float32)-precise[0])).item()))
    full_f32=mx.quantized_matmul(x.astype(mx.float32),layer.weight,
        scales=layer.scales.astype(mx.float32),biases=layer.biases.astype(mx.float32),
        transpose=True,group_size=32,bits=4)
    f32sum=precise[0]+precise[1]
    rows['f32_full_vs_f32_partials']=dict(max_abs=float(mx.max(mx.abs(full_f32-f32sum)).item()),
        nonexact_after_bf16_cast=int(mx.sum(full_f32.astype(mx.bfloat16)!=f32sum.astype(mx.bfloat16)).item()))
    # Same original BF16 scale/affine-bias values, but BF16 versus F32
    # dequantization arithmetic need not represent the same effective matrix.
    deq_bf16=mx.dequantize(layer.weight,layer.scales,layer.biases,group_size=32,bits=4)
    deq_f32=mx.dequantize(layer.weight,layer.scales.astype(mx.float32),layer.biases.astype(mx.float32),group_size=32,bits=4)
    rows['bf16_vs_f32_dequantization']=dict(max_abs=float(mx.max(mx.abs(deq_bf16.astype(mx.float32)-deq_f32)).item()),
        nonexact=int(mx.sum(deq_bf16.astype(mx.float32)!=deq_f32).item()))
    return rows

if __name__=='__main__':
    result=probe()
    p=Path(__file__).parent/'evidence-v2';p.mkdir(exist_ok=True)
    with (p/'row-repro.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
