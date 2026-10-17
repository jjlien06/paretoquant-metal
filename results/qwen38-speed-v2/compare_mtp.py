"""Verified full-model 019 versus same-placement native controls."""
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('trial_evidence',ROOT/'compare_trials.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    all_rows=[]
    for task in ('counting','explanation'):
        native_name=f'native-{task}-28x36-g32-v1'
        mtp_name=f'019-full-{task}-28x36-g32-v1'
        native_final,native,native_hashes=module.load_trial(native_name)
        mtp_final,mtp,mtp_hashes=module.load_trial(mtp_name)
        assert native_hashes==mtp_hashes
        assert len(native_hashes)==1847
        assert mtp_final['strict_numerical_admission'] and mtp_final['rank_actions_agree']
        ids=native[0]['requests'][0]['token_ids']
        rows=[]
        for name,ranks in [(native_name,native),(mtp_name,mtp)]:
            for rank in ranks:
                assert len(rank['requests'])==3
                assert all(q['token_ids']==ids for q in rank['requests'])
            requests=ranks[0]['requests']
            metrics=[]
            for q in requests:
                count=len(q['token_ids'])
                metrics.append({'repeat':q['repeat'],'events_including_eos':count,
                                'finish_reason':q['finish_reason'],
                                'wall_seconds':q['wall_seconds'],
                                'first_token_seconds':q['first_token_seconds'],
                                'post_first_tps':(count-1)/(q['wall_seconds']-q['first_token_seconds']),
                                'accepted_drafts':q.get('accepted_drafts'),
                                'verification_rounds':q.get('verification_rounds')})
            row={'trial':name,'requests':metrics,
                 'warm_post_first_tps_median':statistics.median(q['post_first_tps'] for q in metrics[1:]),
                 'warm_wall_seconds_median':statistics.median(q['wall_seconds'] for q in metrics[1:])}
            rows.append(row)
        full=[]
        for rank in mtp:
            for size,gates in rank['verification']['groups'].items():
                for window in gates['full']['windows']:
                    assert window['logits']['exact'] and window['argmax_equal']
                    assert all(part['exact'] for part in window['caches'])
                    full.append({'rank':rank['rank'],'group':size,'inputs':window['inputs'],
                                 'logits_exact':True,'cache_parts_exact':len(window['caches'])})
        all_rows.append({'task':task,'token_ids':ids,'text':mtp[0]['requests'][0]['text'],
                         'same_target_tensor_digests':True,'text_tensor_count':len(native_hashes),
                         'all_six_completions_identical':True,'modes':rows,
                         'full_gate_windows':full,
                         'decode_ratio':rows[1]['warm_post_first_tps_median']/rows[0]['warm_post_first_tps_median'],
                         'ten_tps_on_all_mtp_requests':all(q['post_first_tps']>=10 for q in rows[1]['requests'])})
    result={'schema':'qwen38-native-mtp-controlled-pilot-v1','model':'mlx-community/Qwen3.8-27B-4bit',
            'revision':'10c35caafbb80f7dc6a7a432cdd11af10a6d4818','split':[28,36],'eval_group_size':32,
            'rank_verifier':['native','slot-qmv'],'target_values_and_quantization_unchanged':True,
            'design':'Short separate-process pilot; candidate precedes native controls, not randomized; first request excluded from warm medians',
            'results':all_rows,'scope':'10 tps achieved on counting, NOT a universal/general-prose minimum'}
    with (ROOT/'019-controlled-comparison-v1.json').open('x') as output:json.dump(result,output,indent=2)
    for row in all_rows:print(row['task'],'native',row['modes'][0]['warm_post_first_tps_median'],'mtp',row['modes'][1]['warm_post_first_tps_median'],'ratio',row['decode_ratio'],'all>=10',row['ten_tps_on_all_mtp_requests'])

if __name__=='__main__':main()
