from copy import deepcopy

import pytest
from test_workload_evidence import evidence


def test_unbound_request_boundary_cannot_hide_from_schedule_verifier():
    from workload_evidence import verify_schedule_trial

    bad = evidence()
    event = deepcopy(next(e for e in bad["phases"][0] if e["phase"] == "request_before"))
    event.pop("repeat")
    bad["phases"][0].append(event)
    with pytest.raises(ValueError, match="Unbound"):
        verify_schedule_trial(**bad)
