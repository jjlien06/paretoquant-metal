import mlx.core as mx
import pytest
from test_phased_cpu import tiny_model
mx.set_default_device(mx.cpu)


def test_backend_real_packed_hybrid_generation_matches_native_fresh_greedy():
    from mlx_backend import MTPBackend
    from mtp_head import MTPHead,BoundHead
    from speculative import generate
    from cache_state import assert_cache_equal
    model=tiny_model(); head=MTPHead(model.language_model.args)
    bound=BoundHead(head,model.model.embed_tokens,model.language_model.lm_head)
    backend=MTPBackend(model,bound,prefill_step_size=2)
    for prompt in ([1,2,3,4,5],[2,1]):
        cache=model.make_cache()
        for start in range(0,len(prompt)-1,2):
            x=mx.array([prompt[start:min(start+2,len(prompt)-1)]],mx.int32)
            mx.eval(model(x,cache=cache),[c.state for c in cache])
        current=prompt[-1]; expected=[]
        for _ in range(8):
            scores=model(mx.array([[current]],mx.int32),cache=cache)
            mx.eval(scores,[c.state for c in cache]); current=int(mx.argmax(scores[:,-1,:],-1).item())
            expected.append(current)
        result=generate(backend,prompt,max_tokens=8,eos_ids=set())
        assert result['token_ids']==expected
        assert_cache_equal(cache,backend.cache)
        assert backend.mtp_cache.offset==len(prompt)+len(expected)-2


def test_rank_control_enforces_request_sequence_snapshot_restore_and_token_bound():
    from rank_control import Command,Worker
    events=[]
    worker=Worker(lambda op,tokens: events.append((op,tokens)))
    for seq,op,tokens in [(0,'reset',[]),(1,'forward',[1]),(2,'snapshot',[]),
                          (3,'forward',[2,3]),(4,'restore',[]),(5,'forward',[2]),(6,'finish',[])]:
        command=Command('request1',seq,op,tokens)
        ack=worker.handle(command)
        assert ack=={'request':'request1','sequence':seq,'op':op,'rank':1,'ok':True}
    assert [x[0] for x in events]==['reset','forward','snapshot','forward','restore','forward','finish']
    with pytest.raises(ValueError): worker.handle(Command('request1',6,'finish',[]))
    with pytest.raises(ValueError): Command('request2',0,'forward',list(range(257))).validate()
    with pytest.raises(ValueError): Command('request2',0,'forward',[True]).validate()
    with pytest.raises(ValueError): Worker(lambda *x:None).handle(Command('x',0,'restore',[]))


def test_known_failed_m4_width2_gate_blocks_distributed_backend_before_work():
    from types import SimpleNamespace as S
    from mlx_backend import MTPBackend
    core=S(pipeline_rank=0,pipeline_size=2,start_idx=1,end_idx=2,pipeline_layers=[object()])
    model=S(model_type='qwen3_5',model=core,language_model=S(args=S(num_hidden_layers=2)))
    group=S(rank=lambda:0,size=lambda:2)
    with pytest.raises(ValueError,match='M4 width2 numerical gate failed'):
        MTPBackend(model,None,group=group,peer=object())

