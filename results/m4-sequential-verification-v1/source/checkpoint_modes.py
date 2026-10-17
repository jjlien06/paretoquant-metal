"""Bounded same-checkpoint numerical diagnostic; not an acceleration benchmark."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models import cache as cache_module
import draft_screen
import phased_decode
import speculative_decode

CHECKPOINT = Path('/Users/jeremylien/Projects/paretoquant-dual-host/small-model')
REFERENCE = Path('/Users/jeremylien/Projects/paretoquant-dual-host/spikes/009-draft-screen-1791216672479820000/reference.json')
CONFIG_SHA = '43a1c2f3b692e8cca5974f4753514e94ee4c969f8853b6d02acc16c7e64805da'
ATOL = RTOL = 0.002


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint():
    return {p.name: digest(p) for p in sorted(CHECKPOINT.iterdir()) if p.is_file()}


def compare_cache(control, actual, offset):
    records = []
    if len(control) != len(actual):
        return {'passed': False, 'layer_count_mismatch': True, 'tensors': []}
    for layer, (left, right) in enumerate(zip(control, actual)):
        for index, (a, b) in enumerate(zip(left.state[:2], right.state[:2])):
            a = np.array(a[..., :offset, :]).astype(np.float32)
            b = np.array(b[..., :offset, :]).astype(np.float32)
            same_shape = a.shape == b.shape
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            close = bool(same_shape and finite and np.allclose(a, b, rtol=RTOL, atol=ATOL))
            records.append({'layer': layer, 'tensor': ['key', 'value'][index],
                            'control_shape': list(a.shape), 'actual_shape': list(b.shape),
                            'finite': finite, 'close': close,
                            'max_abs_error': float(np.max(np.abs(a - b))) if same_shape and finite else None})
    return {'passed': len(records) == 2 * len(control) and all(r['close'] for r in records),
            'tensors': records}


before_inventory = draft_screen.inventory(CHECKPOINT)
assert digest(CHECKPOINT / 'config.json') == CONFIG_SHA
before_weights = fingerprint()
previous_device = mx.default_device()
mx.set_default_device(mx.gpu)
old_cache_limit = mx.set_cache_limit(256 * 1024 * 1024)
mx.reset_peak_memory()
created = []
original_make_cache = cache_module.make_prompt_cache


def tracked_cache(owner):
    result = original_make_cache(owner)
    created.append(result)
    return result


try:
    model, tokenizer = load(str(CHECKPOINT))
    mx.eval(model.parameters())
    model_bytes = sum(t.nbytes for _, t in tree_flatten(model.parameters()))
    assert model_bytes == 260563712
    reference = json.loads(REFERENCE.read_text())
    prompts = [reference['formatted_prompt'], tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Write three short facts about the Moon.'}],
        tokenize=False, add_generation_prompt=True)]
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        for k in [1, 4, 8]:
            for verification_mode in ['batch', 'sequential']:
                created.clear()
                events = []
                with patch.object(cache_module, 'make_prompt_cache', tracked_cache):
                    control = phased_decode.phased_generate(model, tokenizer, prompt,
                        max_tokens=32, eval_group_size=8, prefill_step_size=32)
                    trial = speculative_decode.speculative_generate(model, tokenizer, prompt,
                        draft_model=model, draft_tokenizer=tokenizer, max_tokens=32,
                        draft_tokens=k, eval_group_size=8, prefill_step_size=32,
                        verification_mode=verification_mode, progress=events.append)
                assert len(created) == 3
                offset = control['prompt_tokens'] + len(control['token_ids']) - 1
                offsets = [[state.offset for state in cache] for cache in created]
                teacher_cache = compare_cache(created[0], created[1], offset)
                draft_cache = compare_cache(created[0], created[2], offset)
                rounds = [e for e in events if e['phase'] == 'speculative_round_done']
                before_rounds = [e for e in events if e['phase'] == 'speculative_round_before']
                token_events = [e['token_id'] for e in events if e['phase'] == 'token_done']
                stats = trial['acceptance_stats']
                prefill_calls = len(range(0, trial['prompt_tokens'] - 1, 32))
                verification_forwards = (len(rounds) if verification_mode == 'batch'
                    else sum(e['actual_width'] for e in rounds))
                accounting = (len(rounds) == stats['rounds'] == trial['target_verification_calls']
                    and sum(e['accepted'] for e in rounds) == stats['accepted']
                    and sum(e['proposed'] for e in rounds) == stats['proposed']
                    and sum(e['emitted'] for e in rounds) == len(trial['token_ids'])
                    and token_events == trial['token_ids']
                    and trial['target_forward_calls'] == prefill_calls + verification_forwards
                    and trial['verification_mode'] == verification_mode
                    and len(before_rounds) == len(rounds)
                    and all(e['verification_mode'] == verification_mode for e in before_rounds + rounds))
                row = {'prompt_index': prompt_index, 'k': k, 'verification_mode': verification_mode,
                    'control': control, 'trial': trial, 'expected_offset': offset,
                    'cache_offsets': offsets, 'all_cache_offsets_match': all(n == offset for cache in offsets for n in cache),
                    'token_ids_equal': control['token_ids'] == trial['token_ids'],
                    'finish_reason_equal': control['finish_reason'] == trial['finish_reason'],
                    'teacher_live_kv': teacher_cache, 'draft_live_kv': draft_cache,
                    'accounting': accounting, 'rounds': rounds}
                row['passed'] = all([row['token_ids_equal'], row['finish_reason_equal'],
                    row['all_cache_offsets_match'], teacher_cache['passed'], draft_cache['passed'], accounting])
                rows.append(row)
                print(json.dumps({'case': len(rows), 'mode': verification_mode,
                    'prompt': prompt_index, 'k': k, 'passed': row['passed'],
                    'teacher_kv_close': teacher_cache['passed'], 'draft_kv_close': draft_cache['passed']}), flush=True)
    after_inventory = draft_screen.inventory(CHECKPOINT)
    after_weights = fingerprint()
    assert before_inventory == after_inventory and before_weights == after_weights
    report = {'passed': all(r['passed'] for r in rows), 'case_count': len(rows), 'cases': rows,
        'scope': 'Unchanged real mixed3/4bit0.5B self-draft numerical control on M4; no32B or speedup claim',
        'sequential_is_diagnostic': True, 'model_bytes': model_bytes,
        'same_model_self_draft': True, 'prompt_strings': prompts, 'inventory': before_inventory,
        'config_sha256': CONFIG_SHA, 'reference_sha256': digest(REFERENCE),
        'checkpoint_sha256_before': before_weights, 'checkpoint_sha256_after': after_weights,
        'tolerance': {'rtol': RTOL, 'atol': ATOL},
        'sources': {n: digest(ROOT / n) for n in ['speculative_decode.py', 'phased_decode.py', 'draft_screen.py', Path(__file__).name]},
        'versions': {n: importlib.metadata.version(n) for n in ['mlx', 'mlx-metal', 'mlx-lm']},
        'peak_mlx_bytes': mx.get_peak_memory(), 'device': mx.device_info()}
    with (ROOT / 'checkpoint-modes.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'passed': report['passed'], 'case_count': len(rows),
        'by_mode': {mode: {'passed': sum(r['passed'] for r in rows if r['verification_mode'] == mode),
                         'cases': sum(r['verification_mode'] == mode for r in rows)} for mode in ['batch', 'sequential']},
        'peak_mlx_bytes': report['peak_mlx_bytes']}), flush=True)
finally:
    mx.set_cache_limit(old_cache_limit)
    mx.set_default_device(previous_device)
