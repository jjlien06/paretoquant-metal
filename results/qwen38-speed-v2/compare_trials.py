"""Compare actual completed trials; never synthesize missing actor results."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def load_trial(name):
    root = ROOT / name
    final = json.loads((root / 'result.json').read_text())
    assert final['status'] == 'success', (name, final)
    assert final['owned_actor_cleanup_verified'] and final['services_closed_verified']
    assert not final['cleanup_errors']
    actors = [json.loads((root / f'rank{rank}.receipt.json').read_text()) for rank in (0, 1)]
    ranks = [json.loads(a['stdout']) for a in actors]
    expected_sources = json.loads((root / 'source_sha256.json').read_text())
    for a in actors:
        assert a['source_sha256'] == expected_sources
        assert a['result']['returncode'] == 0
        assert a['result']['cleanup']['child_reaped'] and a['result']['cleanup']['group_gone']
    assert len(ranks[0]['requests']) == len(ranks[1]['requests'])
    for left, right in zip(ranks[0]['requests'], ranks[1]['requests']):
        assert left['token_ids'] == right['token_ids']
        assert left['finish_reason'] == right['finish_reason']
    digests = {}
    for rank in ranks:
        for name, digest in rank['weight_loading']['tensor_sha256'].items():
            assert name not in digests or digests[name] == digest
            digests[name] = digest
    return final, ranks, digests


def summarize(name):
    final, ranks, digests = load_trial(name)
    rows = []
    for q in ranks[0]['requests']:
        count = len(q['token_ids'])
        rows.append({'repeat': q['repeat'], 'events_including_eos': count,
                     'first_token_seconds': q['first_token_seconds'],
                     'request_wall_seconds': q['wall_seconds'],
                     'post_first_tps': (count - 1) / (q['wall_seconds'] - q['first_token_seconds']),
                     'finish_reason': q['finish_reason']})
    warm = rows[1:]
    return {'trial': name, 'rows': rows,
            'warm_post_first_tps_median': statistics.median(r['post_first_tps'] for r in warm),
            'warm_wall_seconds_median': statistics.median(r['request_wall_seconds'] for r in warm),
            'token_ids': ranks[0]['requests'][0]['token_ids'],
            'text_tensor_count': len(digests),
            'rank_environment_verified': final.get('rank_environment_verified'),
            'sources_and_owned_cleanup_verified': True}, digests


def main():
    baseline, before_hashes = summarize('control-before-v1')
    candidate, after_hashes = summarize('015-full-fast-sync-v1')
    assert before_hashes == after_hashes
    assert baseline['token_ids'] == candidate['token_ids']
    result = {'baseline': baseline, 'candidate': candidate,
              'serialized_text_tensor_digests_unchanged': True,
              'decode_ratio': candidate['warm_post_first_tps_median'] / baseline['warm_post_first_tps_median'],
              'ten_tps_reached': candidate['warm_post_first_tps_median'] >= 10,
              'verdict': 'No useful demonstrated acceleration; risky synchronization flag remains opt-in/off',
              'design': 'Separate process short pilot; no statistical claim; both warm samples counted'}
    dest = ROOT / '015-comparison.json'
    with dest.open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','candidate')}, indent=2))
    print('baseline_tps',baseline['warm_post_first_tps_median'], 'candidate_tps',candidate['warm_post_first_tps_median'])


if __name__ == '__main__':
    main()
