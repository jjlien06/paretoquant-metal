import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model, ModelArgs


class Group:
    def __init__(self, rank): self.r = rank
    def rank(self): return self.r
    def size(self): return 2


def config():
    return dict(model_type='qwen3_5', hidden_size=128, intermediate_size=256,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_value_heads=6,
        linear_num_key_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=32, linear_conv_kernel_dim=4,
        full_attention_interval=2, tie_word_embeddings=False,
        rope_parameters=dict(type='default', partial_rotary_factor=0.5, rope_theta=10000),
        quantization=dict(group_size=32,bits=4))


@pytest.fixture(params=['F32','BF16'])
def checkpoint(tmp_path, request):
    mx.set_default_device(mx.gpu)
    mx.random.seed(731)
    cfg=config()
    full=Model(ModelArgs.from_dict(cfg))
    nn.quantize(full,group_size=32,bits=4)
    if request.param=='BF16':
        full.apply(lambda a: a.astype(mx.bfloat16) if mx.issubdtype(a.dtype,mx.floating) else a)
    full.eval(); mx.eval(full.parameters())
    mx.save_safetensors(str(tmp_path/'model.safetensors'), dict(tree_flatten(full.parameters())))
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    return tmp_path,cfg,full


def test_framed_stream_attests_source_and_exact_rank_coverage(checkpoint):
    import io
    from tp import inventory,plan_rank,read_chunks,source_attestation,send_stream,receive_stream
    path,cfg,_=checkpoint; tensors=inventory(path)
    plans=[plan_rank(cfg,tensors,r) for r in (0,1)]
    att=source_attestation(tensors,max_chunk_bytes=256)
    for name,d in tensors.items():
        with open(d['path'],'rb') as f:
            f.seek(d['base']+d['data_offsets'][0]);raw=f.read(d['data_offsets'][1]-d['data_offsets'][0])
        assert att[name]==hashlib.sha256(raw).hexdigest()
        coverage=np.zeros(len(raw),np.uint8)
        for p in plans:
            for lo,hi in p[name]['spans']:coverage[lo:hi]+=1
        assert np.all(coverage==(2 if plans[0][name]['replicated'] else 1))
    stream=io.BytesIO()
    def records():
        for p in plans[1].values():yield from read_chunks(p,max_chunk_bytes=256)
    send_stream(stream,records(),max_chunk_bytes=256)
    stream.seek(0)
    received=list(receive_stream(stream,max_chunk_bytes=256))
    assert received==list(records())
    with pytest.raises(ValueError,match='cap'):
        stream.seek(0);list(receive_stream(stream,max_chunk_bytes=128))


def test_collective_scope_restores_on_exception():
    from execute import cpu_collectives
    original=mx.distributed.all_sum
    device=mx.default_device()
    with pytest.raises(RuntimeError,match='witness'):
        with cpu_collectives(Group(0)):
            assert mx.distributed.all_sum is not original
            raise RuntimeError('witness')
    assert mx.distributed.all_sum is original
    assert mx.default_device()==device


def test_two_rank_gpu_prefill_decode_every_cache(checkpoint,tmp_path):
    import os,socket,subprocess,sys
    path,cfg,_=checkpoint
    sockets=[]
    for _ in (0,1):
        s=socket.socket(); s.bind(('127.0.0.1',0)); sockets.append(s)
    hosts=[[f'127.0.0.1:{s.getsockname()[1]}'] for s in sockets]
    for s in sockets:s.close()
    hostfile=tmp_path/'ring.json'; hostfile.write_text(json.dumps(hosts))
    processes=[]
    try:
        for rank in (0,1):
            processes.append(subprocess.Popen([sys.executable,str(Path(__file__).with_name('runner.py')),
                '--model',str(path),'--budget-bytes',str(2**20),'--reference-small',
                '--output',str(tmp_path/f'result{rank}.json')],
                env=dict(os.environ,MLX_HOSTFILE=str(hostfile),MLX_RANK=str(rank)),
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
        for p in processes:
            out,err=p.communicate(timeout=45)
            assert p.returncode==0,(out,err)
        reports=[json.loads((tmp_path/f'result{r}.json').read_text()) for r in (0,1)]
        for r in reports:
            assert set(r['source_tensor_sha256'])==set(r['loading']['tensor_sha256'])
            assert r['numerics']['passed'],r['numerics']
            assert r['layers_executed']==list(range(cfg['num_hidden_layers']))
            assert len(r['tokens'])==3
            assert len(r['numerics']['steps'])==5
            assert r['collective_calls']==5*2*cfg['num_hidden_layers']
        assert reports[0]['tokens']==reports[1]['tokens']
        assert reports[0]['loading']['replicated_sha256']==reports[1]['loading']['replicated_sha256']
    finally:
        for p in processes:
            if p.poll() is None:p.kill(); p.wait(timeout=5)


def test_stream_loader_exact_bounded_and_rejects_corruption(checkpoint):
    from tp import inventory, plan_rank, read_chunks, load_rank
    path,cfg,full=checkpoint
    tensors=inventory(path); plans=plan_rank(cfg,tensors,0)
    def records():
        for p in plans.values(): yield from read_chunks(p,max_chunk_bytes=256)
    model,receipt=load_rank(cfg,tensors,plans,Group(0),records(),budget_bytes=2**20,max_chunk_bytes=256)
    native=Model(ModelArgs.from_dict(cfg)); nn.quantize(native,group_size=32,bits=4)
    native.load_weights(list(tree_flatten(full.parameters()))); native.shard(Group(0))
    actual=dict(tree_flatten(model.parameters()))
    for name,a in tree_flatten(native.parameters()):
        assert a.dtype==actual[name].dtype
        assert mx.array_equal(a,actual[name]).item()
    assert receipt['parameter_bytes']==sum(p['bytes'] for p in plans.values())
    assert set(receipt['tensor_sha256'])==set(plans)
    assert set(receipt['replicated_sha256'])=={k for k,p in plans.items() if p['replicated']}
    with pytest.raises(ValueError,match='budget'):
        load_rank(cfg,tensors,plans,Group(0),records(),budget_bytes=1,max_chunk_bytes=256)
    bad=list(records()); bad[0]={**bad[0],'sha256':'0'*64}
    with pytest.raises(ValueError,match='digest'):
        load_rank(cfg,tensors,plans,Group(0),iter(bad),budget_bytes=2**20,max_chunk_bytes=256)
    with pytest.raises(ValueError,match='coverage'):
        load_rank(cfg,tensors,plans,Group(0),iter(bad[1:]),budget_bytes=2**20,max_chunk_bytes=256)


def test_plans_equal_genuinely_upstream_sharded_parameters(checkpoint):
    from tp import inventory, plan_rank, read_chunks
    path,cfg,full=checkpoint
    tensors=inventory(path)
    for rank in (0,1):
        native=Model(ModelArgs.from_dict(cfg)); nn.quantize(native,group_size=32,bits=4)
        native.load_weights(list(tree_flatten(full.parameters())))
        native.shard(Group(rank)); mx.eval(native.parameters())
        plans=plan_rank(cfg,tensors,rank)
        actual=dict(tree_flatten(native.parameters()))
        assert plans.keys()==actual.keys()
        for name,p in plans.items():
            chunks=list(read_chunks(p,max_chunk_bytes=256))
            assert all(len(c['payload'])<=256 for c in chunks)
            raw=b''.join(c['payload'] for c in chunks)
            a=actual[name]
            expected=np.array(a.view(mx.uint16) if a.dtype==mx.bfloat16 else a).tobytes()
            assert raw==expected,name
            assert list(a.shape)==p['shape']
            assert sum(hi-lo for lo,hi in p['spans'])==len(raw)
            assert [c['offset'] for c in chunks]==list(range(0,len(raw),256))
            assert all(c['sha256']==hashlib.sha256(c['payload']).hexdigest() for c in chunks)
