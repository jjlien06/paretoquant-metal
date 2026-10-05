"""Diagnostic only: one real 32B checkpoint layer, NOT a 32B inference result."""

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from shard_plan import select_pipeline_rank
from stream_weights import checkpoint_inventory, load_from_tensors, tensor_records

root = Path(sys.argv[1]).resolve(strict=True)
config = json.loads((root / "config.json").read_text())
config["num_hidden_layers"] = 1
all_tensors, provenance = checkpoint_inventory(root)
tensors = {
    key: value for key, value in all_tensors.items()
    if not key.startswith("model.layers.") or key.startswith("model.layers.0.")
}
plan = select_pipeline_rank(
    tensors, model_type="qwen2", num_hidden_layers=1, split=[1], rank=0,
    tie_word_embeddings=False,
)
selected = {key: tensors[key] for key in plan["selected_keys"]}

class Single:
    def rank(self): return 0
    def size(self): return 1

mx.set_default_device(mx.gpu)
mx.set_cache_limit(64 * 1024**2)
mx.set_wired_limit(2 * 1024**3)
model, _ = load_from_tensors(
    config, Single(), [1], selected, tensor_records(selected, provenance),
    budget_bytes=2 * 1024**3,
)
print(json.dumps({"phase":"loaded", "active_mlx_bytes":mx.get_active_memory()}), file=sys.stderr, flush=True)
cache = make_prompt_cache(model)
results = []
for label, inputs in [("prefill", mx.array([[1]*32])), ("cached_decode", mx.array([[2]]))]:
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    assert bool(mx.all(mx.isfinite(logits)).item())
    results.append({"phase":label,"shape":list(logits.shape),"dtype":str(logits.dtype),"finite":True})
    print(json.dumps(results[-1]), file=sys.stderr, flush=True)
print(json.dumps({"full_32b_generation":False,"diagnostic":"one original checkpoint layer plus replicated non-layer tensors","results":results,"peak_mlx_bytes":mx.get_peak_memory()}), flush=True)
