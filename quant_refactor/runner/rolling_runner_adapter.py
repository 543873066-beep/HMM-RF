from typing import Dict, List, Optional, Tuple

import pandas as pd


def default_seed_triplet(i: int) -> Tuple[int, int, int]:
    # TODO: port from rolling_runner.py::default_seed_triplet
    raise NotImplementedError


def run_engine_once(
    run_dir: str,
    run_id: str,
    rs_5m: int,
    rs_30m: int,
    rs_1d: int,
    data_end_dt,
    trade_start_dt=None,
    trade_end_dt=None,
):
    # TODO: port from rolling_runner.py::run_engine_once
    raise NotImplementedError


def robust_select_top5(df: pd.DataFrame):
    # TODO: port from rolling_runner.py::robust_select_top5
    raise NotImplementedError


def extract_eval_metrics_from_summary(summ: Optional[dict]) -> Dict:
    # TODO: port from rolling_runner.py::extract_eval_metrics_from_summary
    raise NotImplementedError


def resolve_njobs() -> Tuple[int, int]:
    # TODO: port from rolling_runner.py::resolve_njobs
    raise NotImplementedError


def pick_backstep_endpoints(days: List[pd.Timestamp], step: int, min_start: pd.Timestamp) -> List[pd.Timestamp]:
    # TODO: port from rolling_runner.py::pick_backstep_endpoints
    raise NotImplementedError
