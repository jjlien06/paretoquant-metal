"""Minimal CPU-only smoke: independently certify real-data optimum by cost classes."""
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/qwen38-mixed-two-host-v1/m4-targeted'


def main():
    document = json.loads((OUT / 'allocator-plans.json').read_text())
    sensitivity = json.loads(Path(document['sensitivity_path']).read_text())
    errors = {r['layer']: r['relative_mse'] for r in sensitivity['layers']}
    plan = document['plans'][0]
    metadata = document['m4_targeted']
    selected = plan['q3_layers']
    costs = document['layer_bytes']
    assert document['schema_version'] == 1 and document['n_units'] == 64
    assert len(selected) == len(set(selected)) == plan['n_q3_layers']
    assert set(selected) <= set(range(36))
    assert math.isclose(sum(errors[i] for i in selected), plan['activation_relative_mse_loss'], rel_tol=1e-14)
    assert plan['activation_relative_mse_loss'] <= metadata['original_mixed15_loss_budget']
    savings = sum(costs[str(i)]['q4']-costs[str(i)]['q3'] for i in selected)
    assert savings == plan['actual_saved_bytes'] == plan['saved_bytes_vs_uniform_q4']
    assert document['uniform_q4_text_stored_bytes']-savings == plan['text_stored_bytes']
    assert sum(costs[str(i)]['q3' if i in selected else 'q4'] for i in range(64)) == plan['unit_bytes']
    assert sum(metadata['targeted'][f'rank{r}_stored_bytes'] for r in range(2)) == plan['text_stored_bytes']
    assert metadata['targeted']['rank1_stored_bytes'] == metadata['uniform_q4']['rank1_stored_bytes']
    assert metadata['allocator_calls'][-1]['feasible'] is False
    # Equal-saving layers are interchangeable for the bytes objective. For any
    # count from each class the cheapest-loss prefix is a valid optimal witness.
    groups = {}
    for i in range(36):
        saving = costs[str(i)]['q4']-costs[str(i)]['q3']
        groups.setdefault(saving, []).append(errors[i])
    prefixes = {}
    for saving, losses in groups.items():
        running = [0.0]
        for loss in sorted(losses):
            running.append(running[-1]+loss)
        prefixes[saving] = running
    best_saving, checked = 0, 0
    for counts in itertools.product(*(range(len(p)) for p in prefixes.values())):
        checked += 1
        loss = sum(prefixes[s][n] for s, n in zip(prefixes, counts))
        if loss <= metadata['original_mixed15_loss_budget']:
            best_saving = max(best_saving, sum(s*n for s, n in zip(prefixes, counts)))
    assert savings == best_saving
    data = json.loads((ROOT / 'spikes/033-qwen38-mixed-3-4-real/data/test.json').read_text())
    length = data['sequence_length']
    scored = len(data['tokens'])//length*(length-1)
    assert length == 256 and scored == 6120
    receipt = {'passed': True, 'cpu_only': True, 'independent_cost_class_combinations': checked,
               'exact_optimal_saved_bytes': savings, 'n_q3_layers': len(selected),
               'scored_tokens_for_later_evaluation': scored,
               'build_run': False, 'evaluation_run': False, 'gpu_run': False}
    with (OUT / 'cpu-smoke.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
