#!/usr/bin/env python3
"""Opt-in local Qwen2 eager decode diagnostics; run GPU jobs sequentially."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paretoquant import profiling  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="local saved Qwen2 directory")
    parser.add_argument(
        "--output", required=True, type=Path, help="new JSON file; parent must exist"
    )
    parser.add_argument("--prompt", default="Explain binary search briefly.")
    parser.add_argument("--decode-steps", type=int, default=16, help="1..512")
    parser.add_argument(
        "--capture-step", type=int, default=0, help="zero-based step within schedule"
    )
    parser.add_argument("--repeats", type=int, default=10, help="isolated samples, 1..200")
    parser.add_argument("--warmup", type=int, default=2, help="isolated warmups, 0..50")
    parser.add_argument("--full-repeats", type=int, default=3, help="uninstrumented samples, 1..20")
    parser.add_argument("--full-warmup", type=int, default=1, help="uninstrumented warmups, 0..10")
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=4096, help="1..32768; fail, not truncate"
    )
    parser.add_argument(
        "--stock", action="store_true", help="strict admission but no fusion install"
    )
    args = parser.parse_args(argv)
    options = {
        key: getattr(args, key)
        for key in (
            "decode_steps",
            "capture_step",
            "repeats",
            "warmup",
            "full_repeats",
            "full_warmup",
            "max_prompt_tokens",
        )
    }
    try:
        profiling.validate_options(**options)
        report = profiling.profile_saved_decode(
            args.model,
            args.output,
            prompt=args.prompt,
            stock=args.stock,
            **options,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"profiling failed: {error}\n")
    print(
        json.dumps(
            {
                "evidence": str(args.output.absolute()),
                "operation_count": len(report["isolated_diagnostics"]["operations"]),
                "installed_fusion_count": len(report["admission"]["installed_fusion"]),
                "additive_cost_attribution": False,
                "whole_model_speedup_claim": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
