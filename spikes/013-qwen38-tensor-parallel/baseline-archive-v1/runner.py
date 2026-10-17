"""One bounded physical two-rank request. Supervisor owns process deadlines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import mlx.core as mx
from tp import inventory,plan_rank,read_chunks,load_rank,skeleton
from execute import cpu_collectives,forward,broadcast_token

# Declared before measurements. TP changes reduction order, NOT exact-token proof.
TOLERANCES={'F32':dict(atol=2e-5,rtol=2e-4),'BF16':dict(atol=0.03,rtol=0.03),'F16':dict(atol=0.01,rtol=0.01)}


def compare(reference,actual,tolerance):
    if reference.shape!=actual.shape or reference.dtype!=actual.dtype:
        return dict(passed=False,reason='shape/dtype mismatch',reference_shape=list(reference.shape),actual_shape=list(actual.shape))
    diff=mx.abs(reference.astype(mx.float32)-actual.astype(mx.float32))
    return dict(passed=bool(mx.allclose(reference,actual,**tolerance).item()),max_abs=float(mx.max(diff).item()))


def compare_caches(full,shard,model,rank,tolerance):
    from mlx_lm.models.cache import KVCache
    args=model.language_model.args; kd=args.linear_num_key_heads*args.linear_key_head_dim
    rows=[]
    for index,(a,b) in enumerate(zip(full,shard)):
        values=[]
        if isinstance(a,KVCache):
            if a.offset!=b.offset: values.append(dict(passed=False,reason='offset'))
            for field in ('keys','values'):
                x=getattr(a,field)[...,:a.offset,:]; y=getattr(b,field)[...,:b.offset,:]
                if args.num_key_value_heads>=2:x=mx.split(x,2,axis=1)[rank]
                values.append(dict(state=field,**compare(x,y,tolerance)))
        else:
            x=a[0]; parts=mx.split(x,[kd,2*kd],axis=-1)
            x=mx.concatenate([mx.split(p,2,axis=-1)[rank] for p in parts],axis=-1)
            values.append(dict(state='conv',**compare(x,b[0],tolerance)))
            x=mx.split(a[1],2,axis=1)[rank]
            values.append(dict(state='ssm',**compare(x,b[1],tolerance)))
            for field in ('left_padding','lengths'):
                x=getattr(a,field);y=getattr(b,field)
                values.append(dict(state=field,passed=(x is None and y is None) if x is None or y is None else bool(mx.array_equal(x,y).item())))
        rows.append(dict(layer=index,states=values,passed=all(v['passed'] for v in values)))
    return rows


def run(args):
    if not all(os.environ.get(k) for k in ('MLX_HOSTFILE','MLX_RANK')): raise ValueError('Explicit distributed environment required')
    if not 1<=args.max_tokens<=4: raise ValueError('Only 1..4 generated tokens admitted')
    prompt=json.loads(args.prompt_tokens)
    if not isinstance(prompt,list) or not prompt or len(prompt)>128 or any(type(x)!=int or x<0 for x in prompt): raise ValueError('Invalid bounded token prompt')
    mx.set_default_device(mx.gpu); mx.set_cache_limit(64*1024**2)
    cfg=json.loads((Path(args.model)/'config.json').read_text()); tensors=inventory(args.model)
    group=mx.distributed.init(strict=True,backend='ring')
    if group.size()!=2: raise ValueError('Exactly two ranks required')
    rank=group.rank(); plans=plan_rank(cfg,tensors,rank)
    def records():
        for p in plans.values():yield from read_chunks(p,max_chunk_bytes=args.chunk_bytes)
    model,loading=load_rank(cfg,tensors,plans,group,records(),budget_bytes=args.budget_bytes,max_chunk_bytes=args.chunk_bytes)
    # Compare config, prompt and every replicated digest before entering the forward.
    agreement=hashlib.sha256(json.dumps(dict(config=cfg,prompt=prompt,replicas=loading['replicated_sha256']),sort_keys=True).encode()).digest()
    fingerprint=mx.array(list(agreement),mx.uint8)
    gathered=mx.distributed.all_gather(fingerprint,group=group,stream=mx.cpu);mx.eval(gathered)
    if not mx.array_equal(gathered[:32],gathered[32:]).item():raise ValueError('Rank prompt/config/replica disagreement')
    full=None
    if args.reference_small:
        total=sum((d['data_offsets'][1]-d['data_offsets'][0]) for d in tensors.values())
        if total>64*1024**2:raise ValueError('Reference is restricted to <=64 MiB fixture')
        full=skeleton(cfg,tensors)
        weights={}
        for path in {d['path'] for d in tensors.values()}:weights.update(mx.load(path))
        full.load_weights(list(weights.items()));mx.eval(full.parameters());del weights
    cache=model.make_cache(); ref_cache=full.make_cache() if full else None
    dtype=next(d['dtype'] for k,d in tensors.items() if k.endswith('.embed_tokens.scales'))
    tolerance=TOLERANCES[dtype];steps=[];tokens=[];completed=set()
    blocks=[prompt[i:min(i+3,len(prompt)-1)] for i in range(0,len(prompt)-1,3)]+[prompt[-1:]]
    with cpu_collectives(group) as calls:
        for ids in blocks:
            x=mx.array([ids],mx.int32);actual=forward(model,x,cache,completed)
            if full:
                expected=full(x,cache=ref_cache);mx.eval(expected,[c.state for c in ref_cache])
                cs=compare_caches(ref_cache,cache,full,rank,tolerance)
                steps.append(dict(logits=compare(expected,actual,tolerance),cache=cs))
        for i in range(args.max_tokens):
            token=broadcast_token(actual,group);tokens.append(token)
            if i+1<args.max_tokens:
                x=mx.array([[token]],mx.int32);actual=forward(model,x,cache,completed)
                if full:
                    expected=full(x,cache=ref_cache);mx.eval(expected,[c.state for c in ref_cache])
                    steps.append(dict(logits=compare(expected,actual,tolerance),cache=compare_caches(ref_cache,cache,full,rank,tolerance)))
    passed=all(s['logits']['passed'] and all(c['passed'] for c in s['cache']) for s in steps)
    from tp import source_attestation
    sources=source_attestation(tensors,max_chunk_bytes=args.chunk_bytes)
    result=dict(schema='qwen35-native-tp-run-v1',rank=rank,world_size=2,tokens=tokens,
        source_tensor_sha256=sources,
        layers_executed=sorted(completed),collective_calls=calls['all_sum'],loading=loading,
        peak_mlx_bytes=mx.get_peak_memory(),numerics=dict(checked=bool(full),passed=passed if full else None,
        tolerance=tolerance,steps=steps,label='tolerant TP reduction-order comparison; NOT exact greedy equivalence'))
    with Path(args.output).open('x') as f:json.dump(result,f)
    print(json.dumps(dict(rank=rank,tokens=tokens,numerics_passed=result['numerics']['passed'],output=args.output)))
    return 0 if not full or passed else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--budget-bytes',type=int,required=True)
    p.add_argument('--chunk-bytes',type=int,default=1024*1024)
    p.add_argument('--prompt-tokens',default='[1,7,3,4,8,9]');p.add_argument('--max-tokens',type=int,default=3)
    p.add_argument('--reference-small',action='store_true');p.add_argument('--output',required=True)
    raise SystemExit(run(p.parse_args()))
