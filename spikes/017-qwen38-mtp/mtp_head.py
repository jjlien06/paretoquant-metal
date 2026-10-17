"""Isolated dense Qwen native MTP head. Never attaches parameters to the target.

Equation/concat: primary_sources/vlm_head.py; normalized target-hidden contract:
primary_sources/vlm_engine.py. MTPLX pre_norm defaults are NOT this sidecar contract.
Unprefixed pinned MLX sidecar is already absolute RMSNorm convention. No heuristic.
"""
import copy
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

TARGET_REVISION = '10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
SIDECAR_REVISION = 'b643c01b6d3b094e325edb6ebd832e16c486c575'
SIDECAR_SHA256 = '76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6'
SIDECAR_CONFIG_SHA256 = '16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55'
SIDECAR_BYTES = 238934137
NORM_SUFFIXES = ('norm.weight',)


def convert_norms(weights, *, encoding):
    if encoding not in ('absolute', 'delta'):
        raise ValueError('Norm encoding must be explicit absolute/delta')
    out = {}
    for key, value in weights.items():
        name = key.removeprefix('mtp.')
        if name in out:
            raise ValueError('Duplicate normalized head key')
        if encoding == 'delta' and name.endswith(NORM_SUFFIXES):
            if value.ndim != 1 or not mx.issubdtype(value.dtype, mx.floating):
                raise ValueError('Norm must be floating vector')
            value = value + 1.0
        out[name] = value
    return out


def validate_binding(config, target_text, *, target_revision, sidecar_sha256):
    trusted = json.loads((Path(__file__).parent/'primary_sources/sidecar_config.json').read_text())
    if config != trusted or target_revision != TARGET_REVISION or sidecar_sha256 != SIDECAR_SHA256:
        raise ValueError('Untrusted sidecar/config/target revision binding')
    for key, value in trusted['text_config'].items():
        if key.startswith('mtp_') or key in ('model_type','dtype','layer_types','bos_token_id','eos_token_id','pad_token_id','use_cache','initializer_range','attention_dropout','mamba_ssm_dtype','output_gate_type','attn_output_gate','hidden_act'):
            continue
        if target_text.get(key) != value:
            raise ValueError(f'Target architecture mismatch: {key}')


class MTPHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2*args.hidden_size,args.hidden_size,bias=False)
        self.layers = [DecoderLayer(args,layer_idx=args.full_attention_interval-1)]
        self.norm = nn.RMSNorm(args.hidden_size,eps=args.rms_norm_eps)


class BoundHead:
    """Non-module owner: shared target weights cannot enter head serialization."""
    def __init__(self, module, embedding, output):
        self.module, self.embedding, self.output = module, embedding, output

    def forward(self, hidden, tokens, cache, *, position, logits=True):
        if type(position) is not int or position < 0 or cache.offset != position:
            raise ValueError('MTP position differs from native KV cache offset')
        if tokens.ndim != 2 or tokens.shape[0] != 1 or hidden.shape[:2] != tokens.shape:
            raise ValueError('Require aligned [1,T,H] hidden and [1,T] tokens')
        m = self.module
        mixed = m.fc(mx.concatenate([m.pre_fc_norm_embedding(self.embedding(tokens)),
                                      m.pre_fc_norm_hidden(hidden)],axis=-1))
        h = m.layers[0](mixed,mask=create_attention_mask(mixed,cache),cache=cache)
        out = self.output(m.norm(h)) if logits else None
        mx.eval(h, cache.state, out)
        return out


def load_head_weights(args, path, *, group_size, bits, norm_encoding):
    weights = convert_norms(mx.load(str(path)),encoding=norm_encoding)
    head = MTPHead(args)
    nn.quantize(head,group_size=group_size,bits=bits,mode='affine')
    # strict load validates complete name/shape coverage; enforce architectural dtypes too.
    for name,value in weights.items():
        if name.endswith('.weight') and value.ndim == 2:
            if value.dtype != mx.uint32:
                raise ValueError('Packed affine weight must be uint32')
        elif not mx.issubdtype(value.dtype,mx.floating):
            raise ValueError('Ordinary head/scales/biases must be floating')
    head.load_weights(list(weights.items()),strict=True)
    head.eval()
    mx.eval(head.parameters())
    return head


def load_trusted_head(directory, target, *, target_config, target_revision):
    """Caller supplies an already independently attested exact target checkpoint.

    This checks sidecar bytes/config, not a claim about the target's live weights.
    Only call on rank0 (last layers/output). Large runtime remains parent-gated.
    """
    directory = Path(directory)
    raw = (directory/'config.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SIDECAR_CONFIG_SHA256:
        raise ValueError('Sidecar config digest mismatch')
    path = directory/'model.safetensors'
    if path.stat().st_size != SIDECAR_BYTES:
        raise ValueError('Sidecar size mismatch')
    with path.open('rb') as f:
        digest = hashlib.file_digest(f,'sha256').hexdigest()
    config = json.loads(raw)
    validate_binding(config,target_config.get('text_config',target_config),
                     target_revision=target_revision,sidecar_sha256=digest)
    if target.model.pipeline_rank != 0:
        raise ValueError('MTP head belongs only to rank0')
    args = TextModelArgs.from_dict(copy.deepcopy(config['text_config']))
    head = load_head_weights(args,path,group_size=64,bits=4,norm_encoding='absolute')
    embedding = target.model.embed_tokens
    output = embedding.as_linear if args.tie_word_embeddings else target.language_model.lm_head
    return BoundHead(head,embedding,output)
