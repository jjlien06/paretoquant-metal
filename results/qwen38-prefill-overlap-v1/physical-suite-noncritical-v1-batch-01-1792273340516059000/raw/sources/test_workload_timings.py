"""Timing scopes are observed local wall, never additive GPU attribution."""

from test_workload_evidence import evidence


def test_timing_summary_separates_requests_methods_and_marks_command_intervals():
    from workload_evidence import verify_schedule_trial

    good = evidence()
    result = verify_schedule_trial(**good)
    per_request = result["request_phase_timing_by_rank"][0]
    assert len(per_request) == 2
    assert per_request[0]["id"] == "native" and per_request[0]["method"] == "native"
    assert per_request[1]["method"] == "mtp"
    assert "inter_action_until_verify" in per_request[1]["operations"]
    assert all(
        "non-additive" in row["scope"] for r in per_request for row in r["operations"].values()
    )
