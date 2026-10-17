"""Fixed isolated registry; parent actor must snapshot these exact relative paths.

Do not pass this mapping to unmodified 012 run_owned: its fixed SOURCES will omit
helpers. Add each dependency to an ISOLATED actor registry and admission schema.
"""
import hashlib
from pathlib import Path

SOURCES=('source_registry.py','phased_decode.py','mtp_head.py','speculative.py',
         'mlx_backend.py','rank_control.py','cache_state.py','numerical_gate.py',
         'width_gate.py','primary_sources/sidecar_config.json',
         'primary_sources/registry.json','primary_sources/vlm_head.py',
         'primary_sources/vlm_engine.py','primary_sources/mtplx_qwen_head.py',
         'primary_sources/mtplx_patch.py')


def source_hashes(root):
    root=Path(root).resolve(strict=True)
    out={}
    for name in SOURCES:
        path=root/name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('Source escapes registry root')
        out[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def verify_sources(root,expected):
    if set(expected)!=set(SOURCES) or source_hashes(root)!=expected:
        raise ValueError('Executed source hashes differ from exact admission')
