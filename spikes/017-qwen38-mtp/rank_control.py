"""Bounded per-rank control contract; transport intentionally caller-owned.

Send command BEFORE rank0 enters phased_forward; wait for ack AFTER forward.
Rank1 owns only native target cache, never an MTP module or output projection.
Every request starts with reset and monotonically increasing sequence numbers.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    request: str
    sequence: int
    op: str
    tokens: list

    def validate(self):
        if (not isinstance(self.request,str) or not 1<=len(self.request)<=64
                or type(self.sequence) is not int or self.sequence<0
                or self.op not in ('reset','forward','snapshot','restore','finish')
                or type(self.tokens) is not list or len(self.tokens)>256
                or any(type(t) is not int or t<0 for t in self.tokens)
                or (self.op=='forward') != bool(self.tokens)):
            raise ValueError('Invalid bounded rank command')


class Worker:
    def __init__(self,dispatch):
        self.dispatch=dispatch; self.request=None; self.sequence=-1; self.saved=False

    def handle(self,command):
        command.validate()
        if command.op=='reset':
            if command.sequence!=0 or command.request==self.request:
                raise ValueError('Reset requires fresh request and sequence0')
        elif command.request!=self.request or command.sequence!=self.sequence+1:
            raise ValueError('Rank request/sequence mismatch')
        if command.op=='restore' and not self.saved:
            raise ValueError('No rank-local snapshot')
        self.dispatch(command.op,command.tokens)
        self.request=command.request; self.sequence=command.sequence
        if command.op=='reset': self.saved=False
        if command.op=='snapshot': self.saved=True
        if command.op=='finish': self.saved=False
        return {'request':command.request,'sequence':command.sequence,
                'op':command.op,'rank':1,'ok':True}
