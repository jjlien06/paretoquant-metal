#!/usr/bin/env python3
"""Opt-in local saved-weight Metal retuning (no whole-model speed claim)."""

import argparse
import json
import sys
from pathlib import Path

# Also work from an uninstalled source checkout with its existing MLX environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paretoquant.retune import retune_saved_model  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="local saved affine Qwen2 model")
    parser.add_argument(
        "--output", required=True, type=Path, help="new directory; writes output/model"
    )
    parser.add_argument(
        "--repeats", type=int, default=20, help="samples per method per phase (default 20)"
    )
    parser.add_argument("--warmup", type=int, default=3, help="warmup calls per method (default 3)")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="strict fresh median reduction required for fusion (default 0.05)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="local JSON list of texts; otherwise use shipped calibration data",
    )
    args = parser.parse_args(argv)
    try:
        report = retune_saved_model(
            args.model,
            args.output,
            calibration=args.calibration,
            repeats=args.repeats,
            warmup=args.warmup,
            min_improvement=args.min_improvement,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"retune failed: {error}\n")
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "model": report["output_model"],
                "profile": str(args.output.resolve() / "retune_profile.json"),
                "whole_model_speedup_claim": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
