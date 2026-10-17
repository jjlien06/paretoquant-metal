"""Bounded native Qwen3.5 2-way TP; canonical already-sanitized text tensors only."""
import hashlib
import json
import math
import re
import struct
from pathlib import Path

SIZES={'F32':4,'F16':2,'BF16':2,'U32':4}


def inventory(directory):
    result={}
    for path in sorted(Path(directory).glob('model*.safetensors')):
        if path.is_symlink(): raise ValueError('Weight symlink rejected')
        with path.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0]
            if n>16*1024**2: raise ValueError('Header budget exceeded')
            header=json.loads(f.read(n))
        size=path.stat().st_size
        for name,d in header.items():
            if name=='__metadata__': continue
            if name.startswith(('vision_tower.','visual.')): continue
            if not name.startswith('language_model.'): raise ValueError('Unknown tensor root: '+name)
            lo,hi=d['data_offsets']; shape=d['shape']
            if name in result or d['dtype'] not in SIZES or not all(type(x)==int and x>0 for x in shape):
                raise ValueError('Invalid tensor metadata')
            if lo<0 or hi-lo!=math.prod(shape)*SIZES[d['dtype']] or 8+n+hi>size:
                raise ValueError('Invalid storage offsets')
            result[name]={**d,'path':str(path.resolve()),'base':8+n,'source_stat':[size,path.stat().st_mtime_ns]}
    if not result: raise ValueError('No text tensors')
    return result


class Rank:
    def __init__(self,rank): self.r=rank
    def rank(self): return self.r
    def size(self): return 2


def skeleton(config,tensors,group=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3_5 import Model,ModelArgs
    if config.get('model_type')!='qwen3_5' or config.get('model_file'): raise ValueError('Only built-in qwen3_5')
    cfg=config.get('text_config',config)
    if cfg.get('num_experts',0): raise ValueError('MoE not admitted')
    model=Model(ModelArgs.from_dict(config)); model.eval()
    def predicate(path,module):
        if path+'.scales' not in tensors: return False
        if not hasattr(module,'to_quantized'): raise ValueError('Unsupported quantized module')
        dim=module.weight.shape[-1]; w=tensors[path+'.weight']; s=tensors[path+'.scales']
        if w['dtype']!='U32' or s['dtype'] not in {'F16','BF16','F32'}: raise ValueError('Invalid quant storage')
        if dim%s['shape'][-1] or (w['shape'][-1]*32)%dim: raise ValueError('Invalid packing')
        return dict(group_size=dim//s['shape'][-1],bits=w['shape'][-1]*32//dim,mode='affine')
    nn.quantize(model,class_predicate=predicate)
    params=dict(tree_flatten(model.parameters()))
    if params.keys()!=tensors.keys(): raise ValueError('Incomplete architectural tensor names')
    for name,a in params.items():
        d=tensors[name]
        if list(a.shape)!=d['shape']: raise ValueError('Architectural shape mismatch: '+name)
        if (a.dtype==mx.uint32 and d['dtype']!='U32') or (a.dtype!=mx.uint32 and d['dtype'] not in {'F16','BF16','F32'}):
            raise ValueError('Invalid role dtype: '+name)
    del params
    if group is not None:
        if group.size()!=2: raise ValueError('Exactly two ranks required')
        model.shard(group)
    return model


def plan_rank(config,tensors,rank):
    if rank not in (0,1): raise ValueError('Invalid rank')
    cfg=config.get('text_config',config)
    kd=cfg['linear_num_key_heads']*cfg['linear_key_head_dim']
    result={}
    for name,d in tensors.items():
        shape=d['shape']; axis=None; cuts=None
        if '.layers.' in name:
            if re.search(r'\.(mlp\.(gate_proj|up_proj)|self_attn\.q_proj|linear_attn\.in_proj_(z|b|a))\.',name): axis=0
            elif re.search(r'\.(mlp\.down_proj|self_attn\.o_proj|linear_attn\.out_proj)\.',name):
                if name.endswith('.bias'): axis=None
                else: axis=len(shape)-1
            elif re.search(r'\.self_attn\.(k_proj|v_proj)\.',name):
                if cfg['num_key_value_heads']>=2: axis=0
            elif '.linear_attn.conv1d.' in name or '.linear_attn.in_proj_qkv.' in name:
                axis=0; cuts=[kd,2*kd]
            elif name.endswith(('.linear_attn.A_log','.linear_attn.dt_bias')): axis=0
        if axis is None:
            spans=[(0,math.prod(shape)*SIZES[d['dtype']])]; target=shape[:]
        else:
            length=shape[axis]; bounds=[0]+(cuts or [])+[length]
            ranges=[]
            for lo,hi in zip(bounds,bounds[1:]):
                if hi<=lo or (hi-lo)%2: raise ValueError('Nondivisible segment: '+name)
                width=(hi-lo)//2; ranges.append((lo+rank*width,lo+(rank+1)*width))
            item=SIZES[d['dtype']]; inner=math.prod(shape[axis+1:])*item
            outer=math.prod(shape[:axis]); spans=[]
            for row in range(outer):
                spans.extend(((row*length+lo)*inner,(row*length+hi)*inner) for lo,hi in ranges)
            target=shape[:]; target[axis]//=2
        result[name]={**d,'name':name,'shape':target,'source_shape':shape,'spans':spans,'replicated':axis is None,
                      'bytes':sum(b-a for a,b in spans)}
    # Native code is the admission oracle; all random/quantization graphs stay lazy.
    from mlx.utils import tree_flatten
    template=skeleton(config,tensors,Rank(rank))
    expected=dict(tree_flatten(template.parameters()))
    for name,p in result.items():
        if list(expected[name].shape)!=p['shape']: raise ValueError('Native shard disagreement: '+name)
    return result


def load_rank(config,tensors,plans,group,records,*,budget_bytes,max_chunk_bytes):
    import mlx.core as mx
    import numpy as np
    total=sum(p['bytes'] for p in plans.values())
    if type(budget_bytes)!=int or budget_bytes<=0 or total+max(p['bytes'] for p in plans.values())>budget_bytes:
        raise ValueError('Parameter plus assembly scratch budget exceeded')
    if plans!=plan_rank(config,tensors,group.rank()): raise ValueError('Untrusted rank plan')
    model=skeleton(config,tensors,group)
    dtype_map={'F32':np.float32,'F16':np.float16,'BF16':np.uint16,'U32':np.uint32}
    hashes={}; chunk_hashes={}; records=iter(records)
    for name,p in plans.items():
        parts=[]; offset=0; digest=hashlib.sha256(); entries=[]
        while offset<p['bytes']:
            try: record=next(records)
            except StopIteration: raise ValueError('Incomplete coverage') from None
            payload=record['payload']
            if record['name']!=name or record['offset']!=offset or not 0<len(payload)<=max_chunk_bytes or offset+len(payload)>p['bytes']:
                raise ValueError('Chunk coverage mismatch')
            if len(payload)%SIZES[p['dtype']]: raise ValueError('Unaligned chunk')
            h=hashlib.sha256(payload).hexdigest()
            if h!=record['sha256']: raise ValueError('Chunk digest mismatch')
            arr=mx.array(np.frombuffer(payload,dtype=dtype_map[p['dtype']]))
            if p['dtype']=='BF16': arr=arr.view(mx.bfloat16)
            mx.eval(arr); parts.append(arr); digest.update(payload)
            entries.append(dict(offset=offset,bytes=len(payload),sha256=h)); offset+=len(payload)
        arr=mx.concatenate(parts).reshape(p['shape']); mx.eval(arr)
        model.load_weights([(name,arr)],strict=False)
        hashes[name]=digest.hexdigest(); chunk_hashes[name]=entries
        del parts,arr; mx.clear_cache()
    if next(records,None) is not None: raise ValueError('Extra coverage')
    # No evaluation before every full random graph has been replaced.
    mx.eval(model.parameters())
    return model,dict(schema='qwen35-native-tp-v1',rank=group.rank(),world_size=2,
        parameter_bytes=total,assembly_scratch_bytes=max(p['bytes'] for p in plans.values()),
        tensor_sha256=hashes,chunk_receipts=chunk_hashes,
        replicated_sha256={k:hashes[k] for k,p in plans.items() if p['replicated']},
        all_layers=list(range(len(model.layers))))


def source_attestation(tensors,*,max_chunk_bytes):
    """Hash full SOURCE tensors with bounded reads, never transmit unplanned bytes."""
    hashes={}
    for name,d in tensors.items():
        plan={**d,'name':name,'spans':[(0,d['data_offsets'][1]-d['data_offsets'][0])]}
        digest=hashlib.sha256()
        for chunk in read_chunks(plan,max_chunk_bytes=max_chunk_bytes):digest.update(chunk['payload'])
        hashes[name]=digest.hexdigest()
    return hashes


def send_stream(stream,records,*,max_chunk_bytes):
    """Write to an actor-owned authenticated binary channel; no listener spawned."""
    for record in records:
        payload=record['payload']
        if not 0<len(payload)<=max_chunk_bytes:raise ValueError('Payload cap exceeded')
        header=json.dumps({k:v for k,v in record.items() if k!='payload'}|{'bytes':len(payload)}).encode()
        if len(header)>4096:raise ValueError('Header cap exceeded')
        stream.write(struct.pack('<I',len(header)));stream.write(header);stream.write(payload)
    stream.write(struct.pack('<I',0));stream.flush()


def receive_stream(stream,*,max_chunk_bytes):
    def exact(n):
        parts=bytearray()
        while len(parts)<n:
            data=stream.read(n-len(parts))
            if not data:raise ValueError('Truncated frame')
            parts.extend(data)
        return bytes(parts)
    while True:
        size=struct.unpack('<I',exact(4))[0]
        if size==0:return
        if size>4096:raise ValueError('Header cap exceeded')
        header=json.loads(exact(size));size=header.pop('bytes')
        if type(size)!=int or not 0<size<=max_chunk_bytes:raise ValueError('Payload cap exceeded')
        yield {**header,'payload':exact(size)}


def read_chunks(plan,*,max_chunk_bytes):
    if type(max_chunk_bytes)!=int or max_chunk_bytes<4 or max_chunk_bytes%4: raise ValueError('Chunk cap must be positive multiple of four')
    path=Path(plan['path']); st=path.stat()
    if [st.st_size,st.st_mtime_ns]!=plan['source_stat']: raise ValueError('Source changed')
    offset=0; buf=bytearray()
    with path.open('rb') as f:
        for lo,hi in plan['spans']:
            f.seek(plan['base']+plan['data_offsets'][0]+lo)
            while lo<hi:
                take=min(hi-lo,max_chunk_bytes-len(buf)); data=f.read(take)
                if len(data)!=take: raise ValueError('Truncated source')
                buf.extend(data); lo+=take
                if len(buf)==max_chunk_bytes:
                    payload=bytes(buf); yield dict(name=plan['name'],offset=offset,payload=payload,sha256=hashlib.sha256(payload).hexdigest())
                    offset+=len(payload); buf.clear()
        if buf:
            payload=bytes(buf); yield dict(name=plan['name'],offset=offset,payload=payload,sha256=hashlib.sha256(payload).hexdigest())
    st=path.stat()
    if [st.st_size,st.st_mtime_ns]!=plan['source_stat']: raise ValueError('Source changed during read')
