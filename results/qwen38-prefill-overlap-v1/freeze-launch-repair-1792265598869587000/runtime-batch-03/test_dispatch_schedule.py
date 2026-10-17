from copy import deepcopy

import pytest
from test_workload_evidence import evidence


def test_controller_dispatches_schedules_without_original_all_repeat_equivalence():
    from supervise import verify_completed_trial

    good = evidence()
    good["args"].benchmark_schedule = True
    result = verify_completed_trial(**good)
    assert result["verified_request_count"] == 2
    bad = deepcopy(good)
    bad["rows"][0]["max_tokens"] = 6
    with pytest.raises(ValueError):
        verify_completed_trial(**bad)
