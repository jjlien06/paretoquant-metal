"""Explicit layer execution with native operators and scoped CPU all-reduces."""
from contextlib import contextmanager
import threading
import mlx.core as mx
from mlx_lm.models.qwen3_5 import create_attention_mask,create_ssm_mask

_LOCK=threading.Lock()


@contextmanager
def cpu_collectives(group):
    if not _LOCK.acquire(blocking=False): raise RuntimeError('Collective scope already active')
    original=mx.distributed.all_sum
    calls={'all_sum':0}
    def reduce(x,*args,**kwargs):
        mx.eval(x)  # complete GPU producer, not merely a constant CPU input
        kwargs['stream']=mx.cpu; kwargs.setdefault('group',group)
        y=original(x,*args,**kwargs); mx.eval(y)
        calls['all_sum']+=1
        return y
    try:
        mx.distributed.all_sum=reduce
        yield calls
    finally:
        mx.distributed.all_sum=original
        _LOCK.release()


def forward(model,ids,cache,completed=None):
    if len(cache)!=len(model.layers): raise ValueError('Cache/layer coverage mismatch')
    text=model.language_model; core=text.model
    h=core.embed_tokens(ids); mx.eval(h)
    masks={True:None,False:None}
    for linear in (True,False):
        pair=next(((l,c) for l,c in zip(model.layers,cache) if l.is_linear==linear),None)
        if pair:
            masks[linear]=(create_ssm_mask if linear else create_attention_mask)(h,pair[1])
    for i,(layer,c) in enumerate(zip(model.layers,cache)):
        h=layer(h,mask=masks[layer.is_linear],cache=c)
        mx.eval(h,c.state)
        if completed is not None: completed.add(i)
    h=core.norm(h)
    out=core.embed_tokens.as_linear(h) if text.args.tie_word_embeddings else text.lm_head(h)
    mx.eval(out)
    return out


def broadcast_token(scores,group):
    if group.rank()==0:
        token=mx.argmax(scores[:,-1,:],axis=-1).astype(mx.int32); mx.eval(token)
        mx.eval(mx.distributed.send(token,1,group=group,stream=mx.cpu))
    else:
        token=mx.distributed.recv((1,),mx.int32,0,group=group,stream=mx.cpu); mx.eval(token)
    return int(token.item())
