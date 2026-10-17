"""CPU-only receipt analysis. No imports/execution of archived model code.

Run: python3 extract_timeline.py
Writes summary.json beside this script; reads only archived JSON/source bytes.
Only same-rank elapsed_seconds differences are used; never compare host clocks.
"""
import collections
import hashlib
import json
import pathlib
import statistics

BASE = pathlib.Path('/Users/jeremylien/Projects/paretoquant-metal/results/m2pro-m4-dual-host-phased-32b-v1')
OUT = pathlib.Path(__file__).resolve().with_name('summary.json')


def load(rank, size='qwen32'):
    path = BASE / size / f'rank{rank}.receipt.json'
    raw = path.read_bytes()
    receipt = json.loads(raw)
    output = json.loads(receipt['stdout'])
    events = []
    ignored = []
    for line in receipt['stderr'].splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            ignored.append(line)
            continue
        assert isinstance(event, dict) and 'phase' in event
        assert event['rank'] == rank
        events.append(event)
    assert events and all(b['elapsed_seconds'] >= a['elapsed_seconds'] for a, b in zip(events, events[1:]))
    assert receipt['result']['returncode'] == 0
    for name, expected in receipt['source_sha256'].items():
        assert hashlib.sha256((BASE / 'executed-source' / name).read_bytes()).hexdigest() == expected, name
    return receipt, output, events, hashlib.sha256(raw).hexdigest(), len(ignored)


def stats(values):
    return {'count': len(values), 'sum_seconds': sum(values), 'median_seconds': statistics.median(values),
            'min_seconds': min(values), 'max_seconds': max(values)}


def analyze(rank, size='qwen32'):
    receipt, output, events, digest, ignored = load(rank, size)
    by_phase = collections.defaultdict(list)
    for event in events:
        by_phase[event['phase']].append(event)
    def event(phase, i=0):
        return by_phase[phase][i]
    def t(phase, i=0):
        return event(phase, i)['elapsed_seconds']
    def delta(a, b):
        return t(b) - t(a)
    request = output['requests'][0]
    tokens = by_phase['token_done']
    assert len(tokens) == len(request['token_ids'])
    assert [e['token_id'] for e in tokens] == request['token_ids']
    assert [e['token_index'] for e in tokens] == list(range(len(tokens)))
    forwards = []
    stage = None
    pending = {}
    spans = []
    for e in events:
        phase = e['phase']
        if phase in ('prefill_before', 'prefill_last_before', 'decode_before'):
            stage = 'prefill_39' if phase == 'prefill_before' else ('prefill_last_1' if phase == 'prefill_last_before' else 'decode')
        prefix = phase.removesuffix('_before')
        if phase in ('embedding_before', 'layer_before', 'handoff_receive_before', 'handoff_send_before', 'head_before', 'token_receive_before', 'token_send_before'):
            key = (prefix, e.get('layer_index'))
            assert key not in pending, key
            pending[key] = (e, stage)
        elif phase in ('embedding_done', 'layer_done', 'handoff_receive_done', 'handoff_send_done', 'head_done', 'token_done'):
            prefix = phase.removesuffix('_done')
            if phase == 'token_done':
                prefix = 'token_send' if rank == 0 else 'token_receive'
            key = (prefix, e.get('layer_index'))
            if key not in pending:
                continue
            start, span_stage = pending.pop(key)
            spans.append({'kind': prefix, 'stage': span_stage, 'layer_index': e.get('layer_index'),
                          'start_seconds': start['elapsed_seconds'], 'end_seconds': e['elapsed_seconds'],
                          'duration_seconds': e['elapsed_seconds'] - start['elapsed_seconds']})
    assert not pending
    assert len(by_phase['layer_before']) == len(by_phase['layer_done']) == 42 * (26 if rank == 0 else 38) if size == 'qwen32' else True
    grouped = collections.defaultdict(list)
    for span in spans:
        grouped[f"{span['stage']}/{span['kind']}"] .append(span['duration_seconds'])
    first_layer = next(s for s in spans if s['stage'] == 'prefill_39' and s['kind'] == 'layer')
    chunk_layers = [s for s in spans if s['stage'] == 'prefill_39' and s['kind'] == 'layer']
    bottleneck = max((s for s in spans if s['stage'] != 'decode' and s['kind'] not in ('token_receive', 'token_send')), key=lambda s: s['duration_seconds'])
    first_done = tokens[0]['elapsed_seconds']
    last_done = tokens[-1]['elapsed_seconds']
    first_event_delta = first_done - t('prefill_before')
    assert abs(first_event_delta - request['first_token_seconds']) < 0.001
    partition_spans = [s for s in spans if s['end_seconds'] <= first_done]
    partition = collections.defaultdict(float)
    for s in partition_spans:
        label = s['kind']
        if label == 'layer':
            label = 'first_chunk_first_layer' if s is first_layer else f"{s['stage']}_other_layers"
        partition[label] += s['duration_seconds']
    partition['unbracketed_seconds'] = first_event_delta - sum(partition.values())
    assert abs(sum(partition.values()) - first_event_delta) < 1e-9
    memory_phases = ('tensor_loading_done', 'loading_done', 'request_before')
    snapshots = [{k: e[k] for k in ('phase', 'elapsed_seconds', 'active_mlx_bytes', 'peak_mlx_bytes', 'cache_mlx_bytes', 'process_maxrss')}
                 for e in events if e['phase'] in memory_phases]
    for phase in (('layer_before', 'layer_done') if rank == 0 else ('embedding_before', 'embedding_done')):
        e = event(phase)
        snapshots.append({k: e[k] for k in ('phase', 'elapsed_seconds', 'active_mlx_bytes', 'peak_mlx_bytes', 'cache_mlx_bytes', 'process_maxrss')})
    return {
        'receipt_path': str(BASE / size / f'rank{rank}.receipt.json'), 'receipt_sha256': digest,
        'source_hashes_verified': len(receipt['source_sha256']), 'event_count': len(events), 'ignored_non_json_lines': ignored,
        'event_counts': dict(collections.Counter(e['phase'] for e in events)),
        'host': output['device']['device_name'], 'parameter_bytes': output['parameter_bytes'],
        'request': {k: v for k, v in request.items() if k not in ('token_ids', 'text')},
        'generated_token_count': len(tokens),
        'startup': {'process_start_marker_seconds': t('process_start'), 'loading_before_seconds': t('loading_before'),
                    'loading_done_seconds': t('loading_done'), 'loading_span_seconds': delta('loading_before', 'loading_done'),
                    'tensor_loading_span_seconds': delta('tensor_loading_before', 'tensor_loading_done'),
                    'tensor_loading_done_to_loading_done_seconds': delta('tensor_loading_done', 'loading_done'),
                    'request_before_seconds': t('request_before'), 'request_before_to_prefill_before_seconds': delta('request_before', 'prefill_before')},
        'first_token': {'prefill_before_seconds': t('prefill_before'), 'token_done_seconds': first_done,
                        'event_delta_seconds': first_event_delta, 'event_delta_minus_stdout_timer_seconds': first_event_delta - request['first_token_seconds'],
                        'prefill_39_to_prefill_last_seconds': delta('prefill_before', 'prefill_last_before'),
                        'prefill_last_to_token_done_seconds': first_done - t('prefill_last_before'),
                        'disjoint_partition_seconds': dict(partition)},
        'first_chunk_first_layer': first_layer,
        'first_chunk_layers_aggregate': stats([s['duration_seconds'] for s in chunk_layers]),
        'first_chunk_other_layers_aggregate': stats([s['duration_seconds'] for s in chunk_layers[1:]]),
        'largest_nonwait_first_token_span': bottleneck,
        'span_aggregates': {k: stats(v) for k, v in sorted(grouped.items())},
        'memory_snapshots': snapshots,
        'post_first_token': {'event_interval_seconds': last_done - first_done, 'remaining_events': len(tokens) - 1,
                             'event_tokens_per_second': (len(tokens) - 1) / (last_done - first_done),
                             'stdout_wall_minus_ttft_seconds': request['wall_seconds'] - request['first_token_seconds']},
    }


summary = {'clock_rule': 'All durations are same-rank monotonic elapsed_seconds deltas. Host clocks never subtracted.',
           'timer_semantics': 'stdout timer starts after tokenization/cache creation, immediately before prefill_before; TTFT sampled before token_done reporter; request wall sampled before tokenizer.decode/request_done.',
           'ranks': {str(rank): analyze(rank) for rank in (0, 1)}}
assert summary['ranks']['0']['first_chunk_first_layer']['duration_seconds'] > 47
assert summary['ranks']['1']['span_aggregates']['prefill_39/embedding']['max_seconds'] > 7
summary['rank0_first_layer_percent_of_ttft'] = 100 * summary['ranks']['0']['first_chunk_first_layer']['duration_seconds'] / summary['ranks']['0']['request']['first_token_seconds']
summary['supervisor_elapsed_seconds'] = json.loads((BASE / 'qwen32/result.json').read_text())['elapsed_seconds']
summary['small_control'] = {}
for rank in (0, 1):
    _, output, events, _, _ = load(rank, 'small')
    e0 = next(e for e in events if e['phase'] == ('layer_before' if rank == 0 else 'embedding_before'))
    e1 = next(e for e in events if e['phase'] == ('layer_done' if rank == 0 else 'embedding_done'))
    summary['small_control'][str(rank)] = {'ttft_seconds': output['requests'][0]['first_token_seconds'],
                                        'first_compute_span_seconds': e1['elapsed_seconds'] - e0['elapsed_seconds'],
                                        'note': 'Different 0.5B model; not a controlled warm/cold 32B comparison.'}
WARM = pathlib.Path('/Users/jeremylien/.hermes/cache/scratch/paretoquant-dual-host/qwen32-phased-warm-baseline-v1')
summary['warm_baseline'] = {}
for rank in (0, 1):
    original, original_output, _, _, _ = load(rank)
    path = WARM / f'rank{rank}.receipt.json'
    raw = path.read_bytes()
    receipt = json.loads(raw)
    output = json.loads(receipt['stdout'])
    assert receipt['result']['returncode'] == 0
    assert receipt['result']['cleanup']['child_reaped'] and receipt['result']['cleanup']['group_gone']
    assert receipt['source_sha256'] == original['source_sha256']
    assert output['weight_loading']['tensor_sha256'] == original_output['weight_loading']['tensor_sha256']
    events = [json.loads(line) for line in receipt['stderr'].splitlines() if line.startswith('{')]
    assert all(b['elapsed_seconds'] >= a['elapsed_seconds'] for a, b in zip(events, events[1:]))
    assert sum(e['phase'] == 'loading_done' for e in events) == 1
    assert len(output['requests']) == 3
    requests = []
    for request in output['requests']:
        repeat = request['repeat']
        subset = [e for e in events if e.get('repeat') == repeat]
        token_events = [e for e in subset if e['phase'] == 'token_done']
        assert request['token_ids'] == original_output['requests'][0]['token_ids']
        assert [e['token_id'] for e in token_events] == request['token_ids']
        assert len(token_events) == 41 and request['finish_reason'] == 'stop'
        assert sum(e['phase'] == 'layer_done' for e in subset) == 42 * (26 if rank == 0 else 38)
        first_time = token_events[0]['elapsed_seconds']
        pre_first = [e for e in subset if e['elapsed_seconds'] <= first_time]
        def first_span(prefix):
            a = next(e for e in pre_first if e['phase'] == prefix + '_before')
            b = next(e for e in pre_first if e['phase'] == prefix + '_done')
            return b['elapsed_seconds'] - a['elapsed_seconds']
        post = request['wall_seconds'] - request['first_token_seconds']
        metrics = {'repeat': repeat, 'ttft_seconds': request['first_token_seconds'], 'wall_seconds': request['wall_seconds'],
                   'post_first_seconds': post, 'post_first_tps': 40 / post,
                   'first_layer_seconds': first_span('layer')}
        if rank == 0:
            metrics['first_handoff_receive_seconds'] = first_span('handoff_receive')
        else:
            metrics['first_embedding_seconds'] = first_span('embedding')
            received = next(e for e in pre_first if e['phase'] == 'token_receive_before')
            metrics['first_token_receive_wait_seconds'] = first_time - received['elapsed_seconds']
        requests.append(metrics)
    summary['warm_baseline'][str(rank)] = {'receipt_path': str(path), 'receipt_sha256': hashlib.sha256(raw).hexdigest(),
                                         'same_source_and_tensor_hashes': True, 'same_41_tokens_all_3_requests': True,
                                         'single_load_3_fresh_cache_requests': True, 'requests': requests}
summary['warm_baseline_rank0_cold_to_warm_ttft_speedups'] = [
    summary['warm_baseline']['0']['requests'][0]['ttft_seconds'] / r['ttft_seconds']
    for r in summary['warm_baseline']['0']['requests'][1:]]
OUT.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'summary_path': str(OUT), 'rank0_first_layer_percent_of_ttft': summary['rank0_first_layer_percent_of_ttft'],
                  'ranks': {rank: {k: data[k] for k in ('event_count', 'source_hashes_verified', 'startup', 'first_token', 'first_chunk_first_layer', 'first_chunk_other_layers_aggregate', 'post_first_token')} for rank, data in summary['ranks'].items()},
                  'warm_baseline': summary['warm_baseline'],
                  'warm_ttft_speedups': summary['warm_baseline_rank0_cold_to_warm_ttft_speedups']}, indent=2))
