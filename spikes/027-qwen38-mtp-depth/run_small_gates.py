"""Run small-model width-W prefix gates and depth-k end-to-end equivalence.

Usage: .venv/bin/python spikes/027-qwen38-mtp-depth/run_small_gates.py \
         --device cpu|gpu --out results/qwen38-mtp-depth-assessment-v1/small-<device>.json
Only the local 0.8B control model is loaded. Never the 27B.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in three sentences why the sky appears blue during the day.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", nargs="+", default=["native", "singleton-qmm"])
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite receipt")
    import mlx.core as mx

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    import small_harness as h

    t0 = time.time()
    model, tok = h.load_small(h.MODEL_DIR)
    eos = set(tok.eos_token_ids)
    receipt = {
        "scope": "small 0.8B control, single host; replay stand-in drafter; NOT 27B/MTP/physical admission",
        "device": args.device, "device_info": mx.device_info() if args.device == "gpu" else "cpu",
        "host": platform.node(), "mlx": mx.__version__,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(HERE.iterdir()) if p.suffix in (".py", ".metal")},
        "model_dir": h.MODEL_DIR,
        "model_config_sha256": hashlib.sha256(
            (Path(h.MODEL_DIR) / "config.json").read_bytes()).hexdigest(),
        "tolerances": {"atol": h.ATOL, "rtol": h.RTOL},
        "capture_gates": [], "e2e": [],
    }
    modes = list(args.modes)
    if args.device == "gpu" and "slot-qmv" not in modes:
        modes.append("slot-qmv")
    ids = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                   enable_thinking=False) for p in PROMPTS]
    # capture gates on prompt0 with the model's own greedy continuation as forced tokens
    ref0 = h.sequential_greedy(model, ids[0], 8, set())[0]
    for mode in modes:
        for width in (2, 3, 4):
            g = h.capture_gate(model, ids[0], ref0[1:], width, mode)
            receipt["capture_gates"].append(g)
            print(json.dumps({"gate": width, "mode": mode, "strict": g["strict_pass"],
                              "argmax": g["argmax_pass"],
                              "worst_cache_abs": max(r["caches"]["max_abs"] for r in g["prefixes"])}),
                  flush=True)
    for pi, prompt in enumerate(ids):
        reference = h.sequential_greedy(model, prompt, args.max_tokens, eos)
        for mode in modes:
            for depth in (1, 2, 3):
                for rate in (0.0, 0.3, 0.6, 1.0):
                    r = h.e2e(model, prompt, depth=depth, rate=rate, mode=mode,
                              max_tokens=args.max_tokens, eos_ids=eos, seed=pi,
                              reference=reference)
                    r["prompt_index"] = pi
                    receipt["e2e"].append(r)
                    print(json.dumps({k: r[k] for k in ("prompt_index", "mode", "depth",
                                                        "mismatch_rate", "tokens_equal",
                                                        "verification_rounds", "accepted_drafts")}
                                     | {"cache_strict": r["final_cache_vs_sequential"]["strict_pass"],
                                        "cache_abs": r["final_cache_vs_sequential"]["max_abs"]}),
                          flush=True)
    receipt["elapsed_seconds"] = time.time() - t0
    receipt["summary"] = {
        mode: {
            "capture_strict_pass": {g["width"]: g["strict_pass"] for g in receipt["capture_gates"]
                                    if g["mode"] == mode},
            "capture_argmax_pass": {g["width"]: g["argmax_pass"] for g in receipt["capture_gates"]
                                    if g["mode"] == mode},
            "e2e_runs": sum(r["mode"] == mode for r in receipt["e2e"]),
            "e2e_tokens_equal": sum(r["tokens_equal"] for r in receipt["e2e"] if r["mode"] == mode),
            "e2e_final_cache_strict": sum(r["final_cache_vs_sequential"]["strict_pass"]
                                          for r in receipt["e2e"] if r["mode"] == mode),
            "e2e_worst_cache_abs": max(r["final_cache_vs_sequential"]["max_abs"]
                                       for r in receipt["e2e"] if r["mode"] == mode),
        } for mode in modes
    }
    with open(out, "x") as f:
        json.dump(receipt, f, indent=1)
    print(json.dumps(receipt["summary"], indent=1))


if __name__ == "__main__":
    main()
