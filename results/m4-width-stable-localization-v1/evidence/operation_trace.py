"""Read-only operation localization; no tolerance changes or dispatch admission."""
import hashlib
import importlib.metadata
import json
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import qwen2
from mlx_lm.models.cache import make_prompt_cache
import draft_screen
import phased_decode

CHECKPOINT = Path('/Users/jeremylien/Projects/paretoquant-dual-host/small-model')
REFERENCE = Path('/Users/jeremylien/Projects/paretoquant-dual-host/spikes/009-draft-screen-1791216672479820000/reference.json')
ATOL = RTOL = 0.002


def difference(a, b):
    a, b = np.array(a).astype(np.float32), np.array(b).astype(np.float32)
    assert a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    delta = np.abs(a - b)
    return {'shape': list(a.shape), 'exact': bool(np.array_equal(a, b)),
        'close': bool(np.allclose(a, b, rtol=RTOL, atol=ATOL)),
        'max_abs_error': float(delta.max()),
        'out_of_tolerance': int(np.sum(delta > ATOL + RTOL * np.abs(b)))}


def cache_difference(a, b):
    assert len(a) == len(b)
    rows = []
    for li, (left, right) in enumerate(zip(a, b)):
        assert left.offset == right.offset
        n = left.offset
        for name, x, y in zip(['keys', 'values'], left.state[:2], right.state[:2]):
            rows.append({'layer': li, 'kind': name,
                **difference(x[..., :n, :], y[..., :n, :])})
    return rows


class Recorder:
    def __init__(self, model):
        self.active = False
        self.records = defaultdict(list)
        self.layers = []
        self.names = {}
        self.types = set()
        for i, layer in enumerate(model.model.layers):
            modules = {'block': layer, 'attention': layer.self_attn, 'mlp': layer.mlp,
                'input_norm': layer.input_layernorm, 'post_attention_norm': layer.post_attention_layernorm}
            modules.update({'attention.' + name: getattr(layer.self_attn, name)
                for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']})
            modules.update({'mlp.' + name: getattr(layer.mlp, name)
                for name in ['gate_proj', 'up_proj', 'down_proj']})
            for name, module in modules.items():
                self.names[id(module)] = (i, name)
                self.types.add(type(module))

    def wrapper(self, original):
        recorder = self
        def wrapped(owner, *args, **kwargs):
            entry = recorder.names.get(id(owner))
            if not recorder.active or entry is None:
                return original(owner, *args, **kwargs)
            layer, name = entry
            if name == 'block':
                recorder.layers.append(layer)
            try:
                result = original(owner, *args, **kwargs)
                recorder.records[f'{layer:02d}.{name}'].append(result)
                return result
            finally:
                if name == 'block':
                    assert recorder.layers.pop() == layer
        return wrapped

    def attention_wrapper(self, original):
        recorder = self
        def wrapped(queries, keys, values, *args, **kwargs):
            result = original(queries, keys, values, *args, **kwargs)
            if recorder.active:
                layer = recorder.layers[-1]
                width = queries.shape[-2]
                for name, value in [('queries', queries), ('new_keys', keys[..., -width:, :]),
                    ('new_values', values[..., -width:, :]), ('output', result)]:
                    recorder.records[f'{layer:02d}.sdpa.{name}'].append(value)
            return result
        return wrapped

    def install(self, context):
        for cls in self.types:
            context.enter_context(patch.object(cls, '__call__', self.wrapper(cls.__call__)))
        context.enter_context(patch.object(qwen2, 'scaled_dot_product_attention',
            self.attention_wrapper(qwen2.scaled_dot_product_attention)))


def prefill(model, tokens):
    cache = make_prompt_cache(model)
    for start in range(0, len(tokens) - 1, 32):
        chunk = tokens[start:min(start + 32, len(tokens) - 1)]
        phased_decode.phased_forward(model, mx.array([chunk], dtype=mx.int32), cache,
            logits=False, all_logits=False, eval_group_size=8)
    return cache


def forward_case(model, prompt_ids, forced_inputs, mode, traced):
    cache = prefill(model, prompt_ids)
    recorder = Recorder(model)
    logits = []
    with ExitStack() as context:
        if traced:
            recorder.install(context)
        recorder.active = traced
        groups = [forced_inputs] if mode == 'batch' else [[x] for x in forced_inputs]
        for inputs in groups:
            scores = phased_decode.phased_forward(model, mx.array([inputs], dtype=mx.int32), cache,
                logits=True, all_logits=True, eval_group_size=8)
            mx.eval(scores)
            logits.append(scores)
        recorder.active = False
    joined = mx.concatenate(logits, axis=1)
    mx.eval(joined)
    return cache, joined, recorder.records


previous_device = mx.default_device()
mx.set_default_device(mx.gpu)
old_cache_limit = mx.set_cache_limit(256 * 1024 * 1024)
mx.reset_peak_memory()
try:
    before = draft_screen.inventory(CHECKPOINT)
    assert hashlib.sha256((CHECKPOINT / 'config.json').read_bytes()).hexdigest() == '43a1c2f3b692e8cca5974f4753514e94ee4c969f8853b6d02acc16c7e64805da'
    model, tokenizer = load(str(CHECKPOINT))
    mx.eval(model.parameters())
    prompts = [json.loads(REFERENCE.read_text())['formatted_prompt'],
        tokenizer.apply_chat_template([{'role':'user','content':'Write three short facts about the Moon.'}], tokenize=False, add_generation_prompt=True)]
    cases = []
    for pi, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        control = phased_decode.phased_generate(model, tokenizer, prompt, max_tokens=32,
            eval_group_size=8, prefill_step_size=32)
        assert len(ids) == control['prompt_tokens']
        for width in ([2, 5, 9] if pi == 0 else [5]):
            forced = [ids[-1], *control['token_ids'][:width - 1]]
            assert len(forced) == width
            batch_cache, batch_logits, _ = forward_case(model, ids, forced, 'batch', False)
            serial_cache, serial_logits, _ = forward_case(model, ids, forced, 'serial', False)
            tb_cache, tb_logits, batch_ops = forward_case(model, ids, forced, 'batch', True)
            ts_cache, ts_logits, serial_ops = forward_case(model, ids, forced, 'serial', True)
            batch_witness = cache_difference(batch_cache, tb_cache)
            serial_witness = cache_difference(serial_cache, ts_cache)
            observer_exact = (all(x['exact'] for x in batch_witness + serial_witness)
                and difference(batch_logits, tb_logits)['exact'] and difference(serial_logits, ts_logits)['exact'])
            assert set(batch_ops) == set(serial_ops)
            operations = []
            for name in batch_ops:
                assert len(batch_ops[name]) == 1 and len(serial_ops[name]) == width
                axis = 2 if '.sdpa.' in name else 1
                serial = mx.concatenate(serial_ops[name], axis=axis)
                mx.eval(batch_ops[name][0], serial)
                operations.append({'operation': name, **difference(batch_ops[name][0], serial)})
            natural_cache = cache_difference(batch_cache, serial_cache)
            row = {'prompt_index': pi, 'verification_width': width, 'forced_inputs': forced,
                'observer_exact_same_mode': observer_exact, 'batch_observer_cache': batch_witness,
                'serial_observer_cache': serial_witness, 'natural_teacher_cache': natural_cache,
                'natural_logits': difference(batch_logits, serial_logits), 'operations': operations,
                'batch_argmax': mx.argmax(batch_logits, axis=-1).tolist(),
                'serial_argmax': mx.argmax(serial_logits, axis=-1).tolist()}
            cases.append(row)
            print(json.dumps({'case':len(cases), 'width':width, 'prompt':pi,
                'observer_exact':observer_exact, 'first_nonexact_operations':[r['operation'] for r in operations if not r['exact']][:5],
                'first_out_of_tolerance_operations':[r['operation'] for r in operations if not r['close']][:5],
                'max_cache_abs_error':max(r['max_abs_error'] for r in natural_cache)}), flush=True)
    assert before == draft_screen.inventory(CHECKPOINT)
    report = {'cases': cases, 'case_count':len(cases), 'observer_exact_all_cases': all(c['observer_exact_same_mode'] for c in cases),
        'scope':'Read-only actual checkpoint operation localization with exact same-mode observer witnesses; no dispatch or32B/speedup admission',
        'tolerance':{'rtol':RTOL,'atol':ATOL}, 'checkpoint_inventory':before,
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['phased_decode.py','draft_screen.py',Path(__file__).name]},
        'installed_source_sha256':{'qwen2.py':hashlib.sha256(Path(qwen2.__file__).read_bytes()).hexdigest()},
        'versions':{n:importlib.metadata.version(n) for n in ['mlx','mlx-metal','mlx-lm']},
        'peak_mlx_bytes':mx.get_peak_memory(), 'device':mx.device_info()}
    with (ROOT/'operation-trace.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'cases':len(cases),'observer_exact_all_cases':report['observer_exact_all_cases'],'peak_mlx_bytes':report['peak_mlx_bytes']}),flush=True)
finally:
    mx.set_cache_limit(old_cache_limit)
    mx.set_default_device(previous_device)
