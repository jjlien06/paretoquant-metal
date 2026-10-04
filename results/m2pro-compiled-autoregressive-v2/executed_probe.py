"""Real, paired autoregressive request checks against native MLX-LM generation."""

import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path
from time import perf_counter_ns

import mlx.core as mx
from mlx_lm import load, stream_generate

from paretoquant.benchmark import environment
from paretoquant.cli import _chat_prompt, _local_model
from paretoquant.decode import FixedDecoder, greedy_token_ids
from paretoquant.manifest import load_manifest, validate_manifest, validate_model_dispatch
from paretoquant.runtime import install_fusion
from paretoquant.statistics import paired_latency_ratio

ROOT = Path('/Users/jeremylien/Projects/paretoquant-metal')


def native(model, tokenizer, prompt, count):
    rows = list(stream_generate(model, tokenizer, prompt=prompt, max_tokens=count))
    return [row.token for row in rows], ''.join(row.text for row in rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--cases', nargs='+', default=['short','medium','long'])
    args=parser.parse_args()
    assert args.repeats >= 2 and args.max_tokens >= 2
    assert not args.output.exists()
    source=_local_model(args.model)
    before=environment()
    manifest=load_manifest(source/'execution_manifest.json')
    dispatch=validate_manifest(source,manifest,before)
    stock, tokenizer=load(str(source))
    fused,_=load(str(source))
    for model in (stock,fused):validate_model_dispatch(model,dispatch)
    installed=install_fusion(fused,dispatch)
    mx.eval(stock.parameters(),fused.parameters())
    calibration=json.loads((ROOT/'src/paretoquant/data/calibration.json').read_text())
    prompts={
        'short':'Explain binary search briefly.',
        'medium':calibration[0]+'\nExplain the boundary cases of binary search.',
        'long':'\n'.join([calibration[1]]*6)+'\nExplain why memory access patterns matter for GPU kernels.',
    }
    cases={}
    for case_name in args.cases:
        text=_chat_prompt(tokenizer,prompts[case_name]); ids=tokenizer.encode(text)
        decoders={n:FixedDecoder(m,capacity=len(ids)+args.max_tokens-1,native_prefill=True) for n,m in [('stock',stock),('fused',fused)]}
        def compiled(name):
            tokens=greedy_token_ids(decoders[name],ids,max_tokens=args.max_tokens,eos_tokens=tokenizer.eos_token_ids)
            emitted=tokens[:-1] if tokens and tokens[-1] in tokenizer.eos_token_ids else tokens
            return emitted,tokenizer.decode(emitted)
        functions={
            'stock_native':lambda:native(stock,tokenizer,text,args.max_tokens),
            'fused_native':lambda:native(fused,tokenizer,text,args.max_tokens),
            'stock_compiled':lambda:compiled('stock'),
            'fused_compiled':lambda:compiled('fused'),
        }
        print(f'Real autoregressive check: {case_name}, {len(ids)} prompt tokens',flush=True)
        cold={}; first={}; samples={name:[] for name in functions}; trial_order=[]
        for name,fn in functions.items():
            mx.synchronize();start=perf_counter_ns();tokens,response=fn();mx.synchronize()
            cold[name]={'loaded_model_first_call_ms':(perf_counter_ns()-start)/1e6,'token_ids':tokens,'response':response}
            first[name]=tokens
        for backend in ('stock','fused'):
            assert first[f'{backend}_native']==first[f'{backend}_compiled'],(case_name,backend,first)
        names=list(functions)
        for trial in range(-2,args.repeats):
            order=names[trial%len(names):]+names[:trial%len(names)]
            if trial>=0:trial_order.append(order)
            for name in order:
                mx.synchronize();start=perf_counter_ns();tokens,response=functions[name]();mx.synchronize()
                elapsed=(perf_counter_ns()-start)/1e6
                assert tokens==first[name],(case_name,name,'generation changed')
                row={'wall_ms':elapsed,'generated_token_count':len(tokens),'token_ids':tokens,'response':response,'stop_reason':'token_limit' if len(tokens)==args.max_tokens else 'eos'}
                if trial>=0:samples[name].append(row)
        timing={name:{'samples':rows,'median_wall_ms':statistics.median(r['wall_ms'] for r in rows),'median_tokens_per_second':statistics.median(r['generated_token_count']*1000/r['wall_ms'] for r in rows)} for name,rows in samples.items()}
        pairs={'compiled_fused_vs_native_fused':('fused_native','fused_compiled'),'compiled_fused_vs_native_stock':('stock_native','fused_compiled'),'compiled_fusion_only':('stock_compiled','fused_compiled'),'compiled_stock_vs_native_stock':('stock_native','stock_compiled')}
        ratios={name:{**paired_latency_ratio([r['wall_ms'] for r in samples[a]],[r['wall_ms'] for r in samples[b]]),'baseline':a,'optimized':b} for name,(a,b) in pairs.items()}
        cases[case_name]={'prompt_text':prompts[case_name],'formatted_prompt':text,'prompt_token_ids':ids,'max_tokens':args.max_tokens,'timing':timing,'ratios':ratios,'first_call':cold,'measured_trial_order':trial_order,'compiled_trace_counts':{n:{'model':d.trace_count,'sample':d.sample_trace_count} for n,d in decoders.items()},'within_backend_generation_identical':True}
        for name,r in ratios.items():print(name,round(r['estimate'],4),round(r['ci_low'],4),round(r['ci_high'],4),flush=True)
        print('tok_s',{n:round(v['median_tokens_per_second'],2) for n,v in timing.items()},flush=True)
        gc.collect();mx.clear_cache()
    code_paths=[Path(__file__),ROOT/'src/paretoquant/decode.py',ROOT/'src/paretoquant/runtime.py',ROOT/'src/paretoquant/metal.py',ROOT/'src/paretoquant/cli.py']
    result={'schema_version':1,'scope':'warmed_single_request_autoregressive_greedy_generation_with_native_prefill_and_text_rendering','environment_before':before,'environment_after':environment(),'source_model_sha256':manifest['model_sha256'],'manifest_sha256':hashlib.sha256((source/'execution_manifest.json').read_bytes()).hexdigest(),'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},'fused_pair_count':len(installed),'cases':cases,'limitations':['Models already loaded; loading/process startup excluded.','Native MLX-LM stream_generate vs explicit greedy compiled decoder; neither produces a serving/batching result.','First-call numbers include graph compilation and are single observations, not paired cold-process benchmarks.','Compiled decoder is reused between warmed requests; CLI constructs a decoder per process.','Every measured request includes native prefill/cache setup, token sampling, EOS handling and final text decoding.','Only greedy token generation is supported by the custom sampled graph; no logits processors or returned log probabilities.','Native prefill scheduling and output rendering implementations differ.','Background swap pressure retained; no user apps were terminated.','Intervals cover sampled trials, not other workloads/devices.']}
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(payload)
    print(args.output.resolve())


if __name__=='__main__':main()
