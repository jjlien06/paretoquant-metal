"""Hold genuine checkpoint Q/K/V equal and isolate SDPA verification width."""
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
from mlx_lm import load
from mlx_lm.models import qwen2
from mlx_lm.models.cache import make_prompt_cache
import draft_screen
import phased_decode

CHECKPOINT = Path('/Users/jeremylien/Projects/paretoquant-dual-host/small-model')
REFERENCE = Path('/Users/jeremylien/Projects/paretoquant-dual-host/spikes/009-draft-screen-1791216672479820000/reference.json')
ATOL = RTOL = 0.002


def difference(a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = np.abs(a - b)
    return {'shape':list(a.shape),'exact':bool(np.array_equal(a,b)),
        'close':bool(np.allclose(a,b,rtol=RTOL,atol=ATOL)),
        'max_abs_error':float(delta.max()),
        'out_of_tolerance':int(np.sum(delta > ATOL+RTOL*np.abs(b)))}


def prefill(model, ids):
    cache = make_prompt_cache(model)
    for start in range(0,len(ids)-1,32):
        phased_decode.phased_forward(model,mx.array([ids[start:min(start+32,len(ids)-1)]],dtype=mx.int32),
            cache,logits=False,all_logits=False,eval_group_size=8)
    return cache


def batch_forward(model, ids, forced, capture):
    cache = prefill(model,ids)
    entries = []
    original = qwen2.scaled_dot_product_attention
    def spy(queries,keys,values,*args,**kwargs):
        result = original(queries,keys,values,*args,**kwargs)
        entries.append((queries,keys,values,args,kwargs,result))
        return result
    with patch.object(qwen2,'scaled_dot_product_attention',spy if capture else original):
        logits = phased_decode.phased_forward(model,mx.array([forced],dtype=mx.int32),cache,
            logits=True,all_logits=True,eval_group_size=8)
        mx.eval(logits)
    return cache,logits,entries


previous_device = mx.default_device()
mx.set_default_device(mx.gpu)
old_cache_limit = mx.set_cache_limit(256*1024*1024)
mx.reset_peak_memory()
try:
    before = draft_screen.inventory(CHECKPOINT)
    assert hashlib.sha256((CHECKPOINT/'config.json').read_bytes()).hexdigest() == '43a1c2f3b692e8cca5974f4753514e94ee4c969f8853b6d02acc16c7e64805da'
    model,tokenizer = load(str(CHECKPOINT))
    mx.eval(model.parameters())
    prompts = [json.loads(REFERENCE.read_text())['formatted_prompt'],
        tokenizer.apply_chat_template([{'role':'user','content':'Write three short facts about the Moon.'}],tokenize=False,add_generation_prompt=True)]
    cases = []
    for pi,prompt in enumerate(prompts):
        control = phased_decode.phased_generate(model,tokenizer,prompt,max_tokens=32,eval_group_size=8,prefill_step_size=32)
        ids = tokenizer.encode(prompt,add_special_tokens=True)
        assert len(ids) == control['prompt_tokens']
        for width in ([2,5,9] if pi == 0 else [5]):
            forced = [ids[-1],*control['token_ids'][:width-1]]
            assert len(forced) == width
            natural_cache,natural_logits,_ = batch_forward(model,ids,forced,False)
            captured_cache,captured_logits,entries = batch_forward(model,ids,forced,True)
            assert len(entries) == len(model.model.layers) == 24
            witnesses = []
            for layer,(a,b) in enumerate(zip(natural_cache,captured_cache)):
                assert a.offset == b.offset
                for name,x,y in zip(['keys','values'],a.state[:2],b.state[:2]):
                    witnesses.append({'layer':layer,'kind':name,**difference(x[...,:a.offset,:],y[...,:b.offset,:])})
            observer_exact = all(r['exact'] for r in witnesses) and difference(natural_logits,captured_logits)['exact']
            operations = []
            for layer,(q,k,v,args,kwargs,output) in enumerate(entries):
                assert not args and kwargs['mask'] == 'causal'
                assert q.shape[-2] == width and k.shape[-2] == v.shape[-2]
                assert not hasattr(kwargs['cache'],'bits')
                mx.eval(q,k,v,output)
                prefix = k.shape[-2]-width
                outputs = []
                for position in range(width):
                    end = prefix+position+1
                    row_kwargs = dict(kwargs)
                    row_kwargs['mask'] = None
                    row = qwen2.scaled_dot_product_attention(mx.contiguous(q[...,position:position+1,:]),
                        k[...,:end,:],v[...,:end,:],**row_kwargs)
                    outputs.append(row)
                serial = mx.concatenate(outputs,axis=2)
                mx.eval(serial)
                operations.append({'operation':f'{layer:02d}.sdpa.equal_inputs',
                    'query_heads':q.shape[1],'kv_heads':k.shape[1],'gqa_factor':q.shape[1]//k.shape[1],
                    'query_width':width,'key_tokens':k.shape[-2],'scale':kwargs['scale'],
                    **difference(output,serial)})
            cases.append({'prompt_index':pi,'verification_width':width,'forced_inputs':forced,
                'observer_exact_same_mode':observer_exact,'observer_cache':witnesses,
                'operations':operations})
            print(json.dumps({'case':len(cases),'prompt':pi,'width':width,'observer_exact':observer_exact,
                'failed_layers':[r['operation'] for r in operations if not r['close']],
                'max_abs_error':max(r['max_abs_error'] for r in operations)}),flush=True)
    assert before == draft_screen.inventory(CHECKPOINT)
    report = {'scope':'Read-only held-equal genuine checkpoint Q/K/V SDPA width isolation; no fix or32B/performance admission',
        'cases':cases,'case_count':len(cases),'observer_exact_all_cases':all(c['observer_exact_same_mode'] for c in cases),
        'tolerance':{'rtol':RTOL,'atol':ATOL},'checkpoint_inventory':before,
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['phased_decode.py','draft_screen.py',Path(__file__).name]},
        'installed_source_sha256':{'qwen2.py':hashlib.sha256(Path(qwen2.__file__).read_bytes()).hexdigest()},
        'versions':{n:importlib.metadata.version(n) for n in ['mlx','mlx-metal','mlx-lm']},
        'peak_mlx_bytes':mx.get_peak_memory(),'device':mx.device_info()}
    with (ROOT/'operation-trace.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'cases':len(cases),'observer_exact_all_cases':report['observer_exact_all_cases'],'peak_mlx_bytes':report['peak_mlx_bytes']}),flush=True)
finally:
    mx.set_cache_limit(old_cache_limit)
    mx.set_default_device(previous_device)
