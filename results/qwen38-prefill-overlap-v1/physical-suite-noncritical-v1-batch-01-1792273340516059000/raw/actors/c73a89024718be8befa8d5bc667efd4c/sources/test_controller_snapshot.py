"""Regression: registered supervisor must not be exclusively copied twice."""

from pathlib import Path


def test_controller_source_order_is_unique_and_supervisor_is_last(tmp_path):
    import remote_actor
    import supervise

    assert hasattr(supervise, "controller_source_names"), (
        "Controller source snapshot order is missing"
    )
    names = supervise.controller_source_names()
    assert len(names) == len(set(names))
    assert set(names) == set(remote_actor.SOURCES) | {"supervise.py"}
    assert names[-1] == "supervise.py"
    for name in names:
        source = Path(supervise.__file__).parent / name
        with (tmp_path / name).open("xb") as handle:
            handle.write(source.read_bytes())
