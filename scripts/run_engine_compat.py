"""Compatibility runner skeleton for refactored engine package."""

from __future__ import annotations

import argparse

from quant_refactor.core.config import EngineConfig
from quant_refactor.pipeline.engine_compat import run_engine_compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refactor engine in legacy-compatible mode")
    parser.add_argument("--data-1m-csv", required=True, help="Input 1m OHLCV CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trade-start", default=None, help="Trade window start date")
    parser.add_argument("--trade-end", default=None, help="Trade window end date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EngineConfig(
        data_1m_csv=args.data_1m_csv,
        out_dir=args.out_dir,
        random_seed=args.seed,
        trade_start=args.trade_start,
        trade_end=args.trade_end,
    )
    run_engine_compat(cfg)


if __name__ == "__main__":
    main()
