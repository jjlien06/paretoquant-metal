"""Run the preserved actual packed GPU fixture in one explicit mode."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import speculative_decode
import test_speculative_gpu

mode = sys.argv[1]
assert mode in ['batch', 'sequential']
original = speculative_decode.speculative_generate
receipts = []

def request(*args, **kwargs):
    kwargs['verification_mode'] = mode
    result = original(*args, **kwargs)
    assert result['verification_mode'] == mode
    receipts.append(result)
    return result

with patch.object(speculative_decode, 'speculative_generate', request):
    report = test_speculative_gpu.run_cases()
assert len(receipts) == report['case_count'] == 80
for row, receipt in zip(report['cases'], receipts):
    assert all(e['verification_mode'] == mode for e in row['rounds'])
    prefill_calls = len(range(0, receipt['prompt_tokens'] - 1, 2))
    forwards = len(row['rounds']) if mode == 'batch' else sum(e['actual_width'] for e in row['rounds'])
    assert receipt['target_forward_calls'] == prefill_calls + forwards
report['verification_mode'] = mode
report['receipts'] = receipts
report['sequential_is_diagnostic'] = True
with (ROOT / ('tiny-' + mode + '.json')).open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps({'passed': report['passed'], 'cases': report['case_count'], 'mode': mode}))
raise SystemExit(0 if report['passed'] else 1)
