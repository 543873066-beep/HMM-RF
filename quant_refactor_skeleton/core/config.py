from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    input_csv: str = r"data/sh000852_5m.csv"
    out_dir: str = r"outputs"
    run_id: Optional[str] = None
    input_tf_minutes: int = 5
    tf_5m: str = "5min"
    min_count_5m: int = 1
    min_count_30m: int = 6
    ma_fast: int = 20
    ma_slow: int = 60
    vol_short: int = 20
    vol_long: int = 60
    rsi_n: int = 14
    atr_n: int = 14
    adx_n: int = 14
    super_state_export_start_date: Optional[str] = "2022-01-13 13:30:00"
    super_state_export_end_date: Optional[str] = None
    enable_legacy_backfill: bool = False
    disable_legacy_equity_fallback_in_rolling: bool = False


def build_pipeline_config(
    input_csv: Optional[str] = None,
    out_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    input_tf_minutes: Optional[int] = None,
    super_state_export_start_date: Optional[str] = None,
    super_state_export_end_date: Optional[str] = None,
    enable_legacy_backfill: Optional[bool] = None,
    disable_legacy_equity_fallback_in_rolling: Optional[bool] = None,
) -> PipelineConfig:
    cfg = PipelineConfig()
    if input_csv:
        cfg.input_csv = str(input_csv)
    if out_dir:
        cfg.out_dir = str(out_dir)
    if run_id:
        cfg.run_id = str(run_id)
    if input_tf_minutes is not None:
        cfg.input_tf_minutes = int(input_tf_minutes)
    if super_state_export_start_date is not None:
        cfg.super_state_export_start_date = str(super_state_export_start_date)
    if super_state_export_end_date is not None:
        cfg.super_state_export_end_date = str(super_state_export_end_date)
    if enable_legacy_backfill is not None:
        cfg.enable_legacy_backfill = bool(enable_legacy_backfill)
    if disable_legacy_equity_fallback_in_rolling is not None:
        cfg.disable_legacy_equity_fallback_in_rolling = bool(disable_legacy_equity_fallback_in_rolling)
    return cfg


__all__ = ["PipelineConfig", "build_pipeline_config"]
