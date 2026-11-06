import sys,time,statistics; sys.path.insert(0,'.')
import mlx.core as mx
from slot_qmv_t import slot_qmv_t
N,K=17408,5120
ws=[mx.quantize((mx.random.normal((N,K))*0.02).astype(mx.bfloat16),group_size=64,bits=4) for _ in range(4)]; mx.eval(ws)
fs={}
for W in (1,2,3):
  x=mx.random.normal((1,W,K)).astype(mx.bfloat16); mx.eval(x)
  fs[f'stock{W}']=(lambda x: lambda q: mx.quantized_matmul(x,*q,transpose=True,group_size=64,bits=4))(x)
  fs[f'shared{W}']=(lambda x: lambda q: slot_qmv_t(x,*q,bits=4,group_size=64,rows=4,simds=2,mode=1))(x)
  fs[f'outer{W}']=(lambda x: lambda q: slot_qmv_t(x,*q,bits=4,group_size=64,rows=4,simds=2,mode=2))(x)
t={k:[] for k in fs}
for r in range(15):
  for k,f in fs.items():
    mx.synchronize();t0=time.perf_counter();mx.eval([f(q) for q in ws]);mx.synchronize();t[k].append((time.perf_counter()-t0)/4*1e3)
print({k:round(statistics.median(v),4) for k,v in t.items()})
