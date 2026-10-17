import hashlib
from pathlib import Path
import pytest


def test_source_registry_includes_all_runtime_helpers_and_trusted_binding():
    from source_registry import SOURCES,source_hashes,verify_sources
    root=Path(__file__).parent
    expected={p.name for p in root.glob('*.py') if not p.name.startswith('test_')}
    assert expected<=set(SOURCES)
    assert 'primary_sources/sidecar_config.json' in SOURCES
    hashes=source_hashes(root); verify_sources(root,hashes)
    assert set(hashes)==set(SOURCES)
    bad=dict(hashes); bad['speculative.py']='0'*64
    with pytest.raises(ValueError): verify_sources(root,bad)
    missing=dict(hashes); missing.pop('cache_state.py')
    with pytest.raises(ValueError): verify_sources(root,missing)
