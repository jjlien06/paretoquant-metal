#!/usr/bin/env python3
"""Sweep an existing measured profile; standard library only, never loads MLX."""

import argparse
import sys
from collections import Counter
from pathlib import Path

# Support running from a source checkout without installing MLX or this package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paretoquant.sweep import sweep_profile, write_sweep  # noqa: E402


def _numbers(tokens, field):
    try:
        return [float(item) for token in tokens for item in token.split(",")]
    except ValueError as exc:
        raise ValueError(f"{field} must be numbers separated by spaces or commas") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path,
                        help="Existing hardware-measured profile.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="Absent or empty directory; nonempty output is never overwritten")
    parser.add_argument("--memory-fractions", nargs="+", default=[".90", ".94", ".97", "1", "1.03"],
                        help="Positive distinct fractions of uniform4 model parameter bytes")
    parser.add_argument("--latency-factors", nargs="+", default=[".9", "1", "1.15"],
                        help="Positive distinct factors of uniform4 stock gate/up latency sum")
    parser.add_argument("--max-states", type=int, default=10000,
                        help="Exact frontier cap; overflow is an error, not infeasibility")
    args = parser.parse_args(argv)
    try:
        if args.output.is_symlink() or (args.output.exists() and (
            not args.output.is_dir() or any(args.output.iterdir())
        )):
            raise FileExistsError(f"Refusing nonempty output: {args.output}")
        fractions = _numbers(args.memory_fractions, "memory_fractions")
        factors = _numbers(args.latency_factors, "latency_factors")
        result = sweep_profile(args.profile, fractions, factors, max_states=args.max_states)
        paths = write_sweep(result, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    counts = Counter(row["status"] for row in result["rows"])
    print(f"{len(result['rows'])} rows: {counts['feasible']} feasible, "
          f"{counts['infeasible']} infeasible, {counts['error']} errors (unknown feasibility)")
    for path in paths:
        print(path.resolve())
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
