"""CPU-only exact M4 (rank 0, layers 0..35) bytes allocation.

Binary-search the byte lattice using ParetoQuant's exact minimum-loss allocator.
This inverts the existing bytes-constrained solver without rounding loss costs:
find the smallest rank-0 payload budget whose minimum loss <= mixed15's loss.
No MLX, checkpoint reads, hardware probes, or remote operations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from allocate_real import (EMBED, HEAD, NORM, allocate_inventory, layer_of,
                           nbytes, text_tensor, allocate, Unit, Option,
                           InfeasibleBudget, FrontierOverflow)


def solve(inventory, sensitivity, reference, max_states=10000):
    # Reuse the original inventory/sensitivity validation and schema/byte accounting.
    result = allocate_inventory(inventory, sensitivity, max_states,
                                targets_gib=(), expected_layers=64)
    references = [p for p in reference['plans'] if p.get('target_savings_gib') == 1.5]
    if len(references) != 1 or 'error' in references[0]:
        raise ValueError('Need exactly one successful original mixed15 (1.5 GiB) plan')
    original = references[0]
    if reference['layer_bytes'] != result['layer_bytes']:
        raise ValueError('Reference layer payloads disagree with inventory')
    if type(original['q3_layers']) is not list or any(type(i) is not int for i in original['q3_layers']):
        raise ValueError('Invalid original mixed15 layer IDs')
    if len(set(original['q3_layers'])) != len(original['q3_layers']) or not set(original['q3_layers']) <= set(range(64)):
        raise ValueError('Invalid original mixed15 layer set')
    errors = {row['layer']: row['relative_mse'] for row in sensitivity['layers']}
    budget = original['activation_relative_mse_loss']
    if not math.isfinite(budget) or budget < 0 or not math.isclose(
            sum(errors[i] for i in sorted(original['q3_layers'])), budget,
            rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('Original mixed15 loss disagrees with sensitivity')
    costs = result['layer_bytes']
    allowed = range(36)
    base = sum(costs[str(i)]['q4'] for i in allowed)
    minimum = sum(costs[str(i)]['q3'] for i in allowed)
    savings = [costs[str(i)]['q4'] - costs[str(i)]['q3'] for i in allowed]
    if any(s <= 0 for s in savings):
        raise ValueError('Expected every q3 demotion to reduce bytes')
    quantum = math.gcd(*savings)
    units = [Unit(f'L{i:02d}', (
        Option('q4', 4, costs[str(i)]['q4'], 0.0, 0.0),
        Option('q3', 3, costs[str(i)]['q3'], 0.0, errors[i]),
    )) for i in allowed]
    calls = []

    def probe(byte_budget):
        try:
            plan = allocate(units, byte_budget, max_states=max_states)
        except InfeasibleBudget:
            calls.append({'budget_bytes': byte_budget, 'feasible': False})
            return None
        feasible = plan.loss <= budget  # Deliberately no epsilon or rounded loss.
        calls.append({'budget_bytes': byte_budget, 'minimum_loss': plan.loss,
                      'feasible': feasible, 'states_considered': plan.states_considered})
        return plan if feasible else None

    # All achievable byte totals are minimum + k*gcd(savings).
    low, high = 0, (base - minimum) // quantum
    best = probe(base)
    if best is None:
        raise ValueError('Even all-q4 is not feasible')
    while low < high:
        middle = (low + high) // 2
        candidate = probe(minimum + middle * quantum)
        if candidate is None:
            low = middle + 1
        else:
            high, best = middle, candidate
    optimum_budget = minimum + low * quantum
    if best.memory_bytes != optimum_budget:
        best = probe(optimum_budget)
    if best is None or best.memory_bytes != optimum_budget:
        raise AssertionError('Byte-lattice optimum was not attained')
    predecessor = optimum_budget - quantum
    if predecessor >= minimum and probe(predecessor) is not None:
        raise AssertionError('A smaller lattice byte budget is feasible')
    chosen = [i for i in allowed if best.choices[f'L{i:02d}'].label == 'q3']
    saved = base - best.memory_bytes
    stored = result['uniform_q4_text_stored_bytes'] - saved
    item = dict(original)
    item.update(target_savings_gib=saved / 2**30, target_saved_bytes=saved,
                budget_bytes=result['unit_bytes_q4'] - saved, exact=best.exact,
                q3_layers=chosen, n_q3_layers=len(chosen),
                unit_bytes=result['unit_bytes_q4'] - saved,
                q3_layer_bytes=sum(costs[str(i)]['q3'] for i in chosen),
                saved_bytes_vs_uniform_q4=saved, actual_saved_bytes=saved,
                text_stored_bytes=stored, predicted_decode_bytes=stored-result['embed_bytes'],
                predicted_bytes_speedup=result['uniform_q4_decode_bytes']/(stored-result['embed_bytes']),
                activation_relative_mse_loss=best.loss,
                states_considered=sum(c.get('states_considered', 0) for c in calls))
    result['plans'] = [item]

    def rank_payloads(selected):
        selected = set(selected)
        totals = [0, 0]
        for name, info in inventory['q4']['tensors'].items():
            if not text_tensor(name):
                continue
            layer = layer_of(name)
            rank = 0 if layer is not None and layer < 36 or name.startswith(EMBED) else 1
            totals[rank] += nbytes(info)
        for i in selected:
            totals[0 if i < 36 else 1] -= costs[str(i)]['q4'] - costs[str(i)]['q3']
        return {'rank0_stored_bytes': totals[0], 'rank0_decode_bytes': totals[0]-result['embed_bytes'],
                'rank1_stored_bytes': totals[1], 'rank1_decode_bytes': totals[1]}

    result['m4_targeted'] = {
        'objective': 'minimize rank0 tensor payload bytes subject to original mixed15 total predicted loss',
        'algorithm': 'exact ParetoQuant allocate + integer byte-lattice binary search; no approximate fallback',
        'rank0_layers': [0, 35], 'rank1_layers': [36, 63],
        'allowed_q3_layers': list(allowed), 'fixed_q4_layers': list(range(36, 64)),
        'rank_assignment': 'rank0 embedding and layers 0..35; rank1 layers 36..63, final norm and lm_head',
        'original_mixed15_loss_budget': budget, 'original_mixed15_q3_layers': original['q3_layers'],
        'loss_budget_remaining': budget-best.loss,
        'byte_quantum': quantum, 'optimal_rank0_quantized_bytes': optimum_budget,
        'next_smaller_rank0_quantized_budget_bytes': predecessor,
        'original_mixed15': rank_payloads(original['q3_layers']),
        'uniform_q4': rank_payloads([]), 'targeted': rank_payloads(chosen),
        'allocator_calls': calls,
        'quality_gate': {'maximum_measured_perplexity_increase_percent': 3.5,
                         'scored_tokens_required': 6120, 'measured': False,
                         'warning': 'Activation MSE is a proxy, not a perplexity guarantee; evaluate before acceptance.'}}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'sensitivity', 'reference-plans', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--max-states', type=int, default=10000)
    args = ap.parse_args()
    try:
        inputs = [args.inventory, args.sensitivity, args.reference_plans]
        result = solve(*(json.loads(p.read_text()) for p in inputs), max_states=args.max_states)
        for key, path in zip(('inventory', 'sensitivity', 'reference_plans'), inputs):
            result[key+'_path'] = str(path.resolve())
            result[key+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
    except (ValueError, KeyError, OSError, FrontierOverflow) as exc:
        ap.exit(2, f'M4 allocation failed (no overwrite or approximation): {exc}\n')
    print(json.dumps({'output': str(args.output), 'q3_layers': result['plans'][0]['q3_layers'],
                      'loss': result['plans'][0]['activation_relative_mse_loss'],
                      'm4_targeted': result['m4_targeted']['targeted']}))


if __name__ == '__main__':
    main()
