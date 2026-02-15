"""Compatibility runner for rolling routing."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_refactor_skeleton.runner.rolling_runner_adapter import run_refactor_rolling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility rolling entrypoint (legacy default)")
    parser.add_argument("--route", default=os.getenv("QRS_ROLLING_ROUTE", "legacy"), choices=["legacy", "new"])
    args, unknown = parser.parse_known_args()
    args.unknown = [x for x in unknown if x != "--"]
    return args


def main() -> int:
    args = parse_args()
    os.environ["QRS_ROLLING_ROUTE"] = str(args.route)
    print(f"[COMPAT] routing={args.route}")
    return int(run_refactor_rolling(argv=args.unknown))


if __name__ == "__main__":
    raise SystemExit(main())
