"""Compatibility entrypoint for engine routing."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_refactor_skeleton.pipeline.engine_compat import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility engine entrypoint (default legacy route)")
    parser.add_argument(
        "--route",
        default=os.getenv("QRS_PIPELINE_ROUTE", "legacy"),
        choices=["legacy", "new"],
        help="Pipeline route selector",
    )
    args, unknown = parser.parse_known_args()
    args.unknown = [x for x in unknown if x != "--"]
    return args


def main() -> int:
    args = parse_args()
    os.environ["QRS_PIPELINE_ROUTE"] = str(args.route)
    print(f"[COMPAT] routing={args.route}")
    return int(run_pipeline("engine", argv=args.unknown))


if __name__ == "__main__":
    raise SystemExit(main())
