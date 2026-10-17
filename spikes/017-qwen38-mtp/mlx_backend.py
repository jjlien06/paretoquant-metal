"""Reference MLX adapter: width2 verify, full hybrid snapshots, width1 replay.

Single-host CPU fixture is tested. The peer begin/finish transport is an explicit
integration seam, NOT a deployed two-host runtime. No device selection or loading.
Inputs to native standalone head are NORMALIZED final target hidden states.
"""
import uuid
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from cache_state import snapshot,restore
from phased_decode import phased_forward,ownership
from rank_control import Command


class MTPBackend:
    def __init__(self,model,head,*,group=None,peer=None,prefill_step_size=256,eval_group_size=1):
        rank,size=ownership(model,group)
        if size==2:
            raise ValueError('M4 width2 numerical gate failed; distributed MTP blocked pending batch-invariant verifier admission')
        if rank!=0 or (size==2 and peer is None):
            raise ValueError('Require final rank0 and an explicit peer for two ranks')
        if type(prefill_step_size) is not int or not 1<=prefill_step_size<=256:
            raise ValueError('Bounded prefill size required')
        self.model,self.head,self.group,self.peer=model,head,group,peer
        self.prefill_step_size,self.eval_group_size=prefill_step_size,eval_group_size
        self.cache=None; self.mtp_cache=None; self.sequence=0

    def _command(self,op,tokens=()):
        command=Command(self.request,self.sequence,op,list(tokens)); command.validate()
        self.sequence+=1
        if self.peer is not None: self.peer.begin(command)
        if op=='reset': self.cache=self.model.make_cache()
        elif op=='snapshot': self.saved=snapshot(self.cache)
        elif op=='restore': restore(self.cache,self.saved)
        elif op=='forward':
            result=phased_forward(self.model,mx.array([tokens],mx.int32),self.cache,
                                  group=self.group,eval_group_size=self.eval_group_size,
                                  all_logits=True,return_hidden=True)
        if self.peer is not None:
            ack=self.peer.finish()
            expected={'request':command.request,'sequence':command.sequence,'op':op,'rank':1,'ok':True}
            if ack!=expected: raise ValueError('Rank1 acknowledgment mismatch')
        return result if op=='forward' else None

    def fresh(self,prompt):
        if any(t>=self.model.language_model.args.vocab_size for t in prompt):
            raise ValueError('Prompt outside target vocabulary')
        self.request=uuid.uuid4().hex; self.sequence=0
        self._command('reset'); self.mtp_cache=KVCache()
        for start in range(0,len(prompt)-1,self.prefill_step_size):
            end=min(start+self.prefill_step_size,len(prompt)-1)
            _,hidden=self._command('forward',prompt[start:end])
            shifted=mx.array([prompt[start+1:end+1]],mx.int32)
            self.head.forward(hidden,shifted,self.mtp_cache,position=start,logits=False)
        scores,hidden=self._command('forward',[prompt[-1]])
        return int(mx.argmax(scores[:,-1,:],-1).item()),hidden

    def draft(self,hidden,y):
        scores=self.head.forward(hidden,mx.array([[y]],mx.int32),self.mtp_cache,
                                 position=self.mtp_cache.offset)
        return int(mx.argmax(scores[:,-1,:],-1).item())

    def snapshot(self):
        self._command('snapshot')
        return snapshot([self.mtp_cache])

    def restore(self,saved):
        self._command('restore'); restore([self.mtp_cache],saved)

    def verify(self,tokens):
        scores,hidden=self._command('forward',tokens)
        return mx.argmax(scores,-1)[0].tolist(),[hidden[:,i:i+1,:] for i in range(len(tokens))]

    def replay(self,y):
        return self.verify([y])

    def advance_mtp(self,hidden,token):
        self.head.forward(hidden,mx.array([[token]],mx.int32),self.mtp_cache,
                          position=self.mtp_cache.offset,logits=False)

    def finish(self):
        self._command('finish')


class Rank1Target:
    """Pass dispatch to rank_control.Worker; receive loop/transport supplied later."""
    def __init__(self,model,group,*,eval_group_size=1):
        if ownership(model,group)!=(1,2): raise ValueError('Require first-stage rank1')
        self.model,self.group,self.eval_group_size=model,group,eval_group_size
        self.cache=None

    def dispatch(self,op,tokens):
        if op=='reset': self.cache=self.model.make_cache()
        elif op=='snapshot': self.saved=snapshot(self.cache)
        elif op=='restore': restore(self.cache,self.saved)
        elif op=='forward':
            phased_forward(self.model,mx.array([tokens],mx.int32),self.cache,
                           group=self.group,eval_group_size=self.eval_group_size,logits=False)
        elif op=='finish': self.saved=None
