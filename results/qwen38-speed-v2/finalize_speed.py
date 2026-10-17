"""Final target/sources/numerics/output/cleanup verification and profile publication."""
import importlib.util
import json
from pathlib import Path
import statistics
import shlex

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('trial_evidence',ROOT/'compare_trials.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def verify_exact_tree(value):
    if isinstance(value,dict):
        if 'strict_pass' in value:
            assert value['strict_pass'] is True
            if 'exact' in value:assert value['exact'] is True
        for v in value.values():verify_exact_tree(v)
    elif isinstance(value,list):
        for v in value:verify_exact_tree(v)


def main():
    configurations=[('counting','native-counting-28x36-g32-v1',['021-prefix-full-counting-v2']),
                    ('prose','native-explanation-28x36-g32-v1',['021-prefix-full-prose-v1','021-prefix-full-prose-confirm-v2'])]
    summaries=[]
    for task,native_name,names in configurations:
        _,native,reference_hashes=module.load_trial(native_name)
        ids=native[0]['requests'][0]['token_ids']
        native_warm=[(len(q['token_ids'])-1)/(q['wall_seconds']-q['first_token_seconds']) for q in native[0]['requests'][1:]]
        warm=[];walls=[];samples=[];peak=[0,0]
        for name in names:
            final,ranks,hashes=module.load_trial(name)
            assert hashes==reference_hashes and len(hashes)==1847
            assert final['strict_numerical_admission'] and final['rank_actions_agree']
            startup=json.loads((ROOT/name/'startup.json').read_text())['args']
            assert startup['prefix_rollback'] is True and startup['split']==[28,36]
            for rank,r in enumerate(ranks):
                assert all(q['token_ids']==ids and q['finish_reason']=='stop' for q in r['requests'])
                peak[rank]=max(peak[rank],r['peak_mlx_bytes'])
                for gates in r['verification']['groups'].values():
                    assert gates['small']['passed'] and gates['full']['passed']
                    prefix=gates['full']['prefix_rollback']
                    assert prefix['passed'] and [p['keep'] for p in prefix['prefixes']]==[0,1,2]
                    verify_exact_tree(gates['full'])
            for q in ranks[0]['requests']:
                sample={'trial':name,'repeat':q['repeat'],'events_including_eos':len(q['token_ids']),
                        'post_first_tps':q['post_first_tokens_per_second'],
                        'wall_seconds':q['wall_seconds'],'first_token_seconds':q['first_token_seconds'],
                        'accepted_drafts':q['accepted_drafts'],'verification_rounds':q['verification_rounds'],
                        'teacher_forwards':q['teacher_forwards'],'finish_reason':q['finish_reason']}
                samples.append(sample)
                if q['repeat']>0:warm.append(sample['post_first_tps']);walls.append(sample['wall_seconds'])
        median=statistics.median(warm)
        summaries.append({'task':task,'native_trial':native_name,'native_warm_tps_median':statistics.median(native_warm),
                          'candidate_warm_tps_median':median,'candidate_warm_tps_range':[min(warm),max(warm)],
                          'candidate_warm_count':len(warm),'candidate_warm_wall_median':statistics.median(walls),
                          'decode_ratio':median/statistics.median(native_warm),'warm_minimum_10_verified':min(warm)>=10,
                          'same_native_token_ids':True,'token_ids':ids,'text':native[0]['requests'][0]['text'],
                          'serialized_text_tensors':len(reference_hashes),'target_digests_unchanged':True,
                          'strict_full_gate_and_prefixes_bitexact':True,'peak_mlx_bytes_by_rank':peak,'samples':samples})
    assert all(r['warm_minimum_10_verified'] for r in summaries)
    final={'schema':'paretoquant-qwen38-final-speed-v2','model':'mlx-community/Qwen3.8-27B-4bit',
           'revision':'10c35caafbb80f7dc6a7a432cdd11af10a6d4818','results':summaries,
           'metric':'Post-first generated token events/s including EOS; excludes setup, numerical gates and prefill/first-token latency',
           'scope':'Verified warm >=10 on two tested prompts; NOT universal/cold/whole-request >=10 guarantee',
           'design':'Short separate-process pilot with contemporaneous same-placement native controls; not randomized statistical proof',
           'no_persistent_server':True}
    with (ROOT/'final-verification-v2.json').open('x') as out:json.dump(final,out,indent=2)
    startup=json.loads((ROOT/'021-prefix-full-prose-confirm-v2/startup.json').read_text())['args']
    profile={'schema':'paretoquant-qwen38-fast-profile-v2','status':'Verified bounded warm two-prompt >=10 tps pilot',
             'model':final['model'],'revision':final['revision'],'local_runtime':str(REPO/'spikes/021-qwen38-prefix-rollback'),
             'source_sha256':json.loads((ROOT/'021-prefix-full-prose-confirm-v2/source_sha256.json').read_text()),
             'parameters':startup,'evidence':'final-verification-v2.json','target_dtype':'native BF16; original packed affine Q4/g64 unchanged',
             'sidecar_revision':'b643c01b6d3b094e325edb6ebd832e16c486c575','warnings':[
                 'Validate current bridge addresses and memory; do not kill unrelated apps',
                 'Speculative acceptance and therefore throughput depend on prompt',
                 'Whole-request throughput and cold setup are slower than post-first decoding',
                 'All numerical gates rerun before every benchmark; no bypass flag',
                 'Original source/failed evidence unchanged; no FP16 or requantization adopted']}
    with (ROOT/'fast-profile-v2.json').open('x') as out:json.dump(profile,out,indent=2)
    command=[str(REPO/'.venv/bin/python'),'-u',str(REPO/'spikes/021-qwen38-prefix-rollback/supervise.py')]
    for key,value in startup.items():
        if key in {'output','prompt'} or value is None or value is False or value==[]:continue
        flag='--'+key.replace('_','-')
        if value is True:command.append(flag)
        elif isinstance(value,list):command.extend([flag,*map(str,value)])
        else:command.extend([flag,str(value)])
    script='#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'
    script+='prompt=${1:-'+shlex.quote(startup['prompt'])+'}\n'
    script+='output="$PWD/results/qwen38-live-$(date +%s)-$$"\n'
    script+='sysctl kern.memorystatus_vm_pressure_level vm.swapusage\ndf -h /System/Volumes/Data\n'
    script+=shlex.join(command)+' --output "$output" --prompt "$prompt"\n'
    script+='"'+str(REPO/'.venv/bin/python')+'" -c '+shlex.quote('import json,sys; from pathlib import Path; r=Path(sys.argv[1]); d=json.loads(json.loads((r/"rank0.receipt.json").read_text())["stdout"]); [(print(q["text"]),print("post-first tokens/s:",round(q["post_first_tokens_per_second"],3))) for q in d["requests"]]; print("Evidence:",r)')+' "$output"\n'
    with (ROOT/'run-fast.sh').open('x') as out:out.write(script)
    for row in summaries:print(row['task'],'native',row['native_warm_tps_median'],'candidate',row['candidate_warm_tps_median'],'range',row['candidate_warm_tps_range'],'ratio',row['decode_ratio'])

if __name__=='__main__':main()
