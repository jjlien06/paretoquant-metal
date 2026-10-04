#!/usr/bin/env python3
"""Run offline quality evaluation from a checkout without installation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paretoquant.quality import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
