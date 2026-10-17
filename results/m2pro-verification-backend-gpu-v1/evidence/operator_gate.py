"""Actual M2 packed GPU adapter qualification; no real-checkpoint or speed claim."""
import json,sys,hashlib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
import mlx.core as mx
from mlx_lm.models import qwen2
import phased_decode,test_speculative_gpu
from slot_qmv import slot_qmv
from stable_attention import stable_causal_attention
from verification_backend import make_verification_forward
forward=make_verification_forward(phased_decode.phased_forward,mx=mx,qwen2=qwen2,qmm=slot_qmv,attention=stable_causal_attention)
with patch.object(phased_decode,'phased_forward',forward):
 report=test_speculative_gpu.run_cases()
assert report['case_count']==len(report['cases'])==80
report['backend_source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['slot_qmv.py','slot_qmv_header.metal','slot_qmv_body.metal','stable_attention.py','verification_backend.py',Path(__file__).name]}
report['scope']='Actual M2Pro random tiny packed4bitF16 teacher/draft GPU; public opt-in backend, EOS/budget/rejection/rollback/tied/untied/group1/8. Not genuine checkpoint, two-host or32B performance.'
with (ROOT/'adapter-gate.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({'passed':report['passed'],'cases':report['case_count'],'peak_mlx_bytes':report['peak_mlx_bytes']}))
raise SystemExit(0 if report['passed'] else 1)
