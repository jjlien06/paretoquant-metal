"""TEST-only input trust probes, no model/runtime execution."""

import hashlib
import json

import pytest

from .test_harness import api


def test_external_hash_and_bounded_json(tmp_path):
    h = api()
    assert hasattr(h, "load_bound"), "External hash-bound bounded reader missing"
    path = tmp_path / "manifest.json"
    data = b'{"sources":{"runtime.py":"trusted"}}'
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert h.load_bound(path, digest) == json.loads(data)
    with pytest.raises(ValueError):
        h.load_bound(path, "0" * 64)
    with pytest.raises(ValueError):
        h.load_bound(path, digest, limit=4)
    symlink = tmp_path / "link.json"
    symlink.symlink_to(path)
    with pytest.raises(ValueError):
        h.load_bound(symlink, digest)
    for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{"x":Infinity}'):
        path.write_bytes(data)
        with pytest.raises(ValueError):
            h.load_bound(path, hashlib.sha256(data).hexdigest())
