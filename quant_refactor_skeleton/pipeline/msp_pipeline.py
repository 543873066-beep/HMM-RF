from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

# assembly-only module imports for migration wiring
from quant_refactor_skeleton.data import loaders as data_loaders
from quant_refactor_skeleton.data import resample as data_resample
from quant_refactor_skeleton.features import feature_builder as feature_builder_mod
from quant_refactor_skeleton.hmm import model as hmm_model
from quant_refactor_skeleton.hmm import portrait as hmm_portrait
from quant_refactor_skeleton.hmm import states as hmm_states
from quant_refactor_skeleton.overlay import overlay_builder as overlay_builder_mod
from quant_refactor_skeleton.super_state import super_hmm as super_hmm_mod


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QRS new-route MSP pipeline (skeleton)")
    p.add_argument("--input_csv", type=str, default=r"data/sh000852_5m.csv")
    p.add_argument("--out_dir", type=str, default=r"outputs")
    p.add_argument("--run_id", type=str)
    p.add_argument("--rs_5m", type=int)
    p.add_argument("--rs_30m", type=int)
    p.add_argument("--rs_1d", type=int)
    p.add_argument("--data_end_date", type=str)
    p.add_argument("--trade_start_date", type=str)
    p.add_argument("--trade_end_date", type=str)
    p.add_argument("--enable_backtest", type=str)
    p.add_argument("--input_tf_minutes", type=int)
    p.add_argument("--export_live_pack", type=str)
    p.add_argument("--live_pack_dir", type=str)
    return p


def _normalize_argv(argv: Optional[Sequence[str]]) -> list[str]:
    return list(argv or [])


def run_msp_pipeline(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, _ = parser.parse_known_args(_normalize_argv(argv))

    input_csv = str(args.input_csv or "").strip()
    if not input_csv:
        print("[QRS:new] error: --input_csv is required")
        return 2

    input_path = Path(input_csv)
    if not input_path.exists():
        print(f"[QRS:new] error: input_csv does not exist: {input_path}")
        return 2

    out_dir = Path(str(args.out_dir or "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[QRS:new] pipeline=msp input_csv={input_path}")
    print(f"[QRS:new] pipeline=msp out_dir={out_dir.resolve()}")
    print(f"[QRS:new] pipeline=msp run_id={run_id}")
    print("[QRS:new] skeleton reached: downstream data/features/hmm/rf stages are not wired in N1")
    return 3


__all__ = [
    "data_loaders",
    "data_resample",
    "feature_builder_mod",
    "hmm_model",
    "hmm_portrait",
    "hmm_states",
    "overlay_builder_mod",
    "run_msp_pipeline",
    "super_hmm_mod",
]
