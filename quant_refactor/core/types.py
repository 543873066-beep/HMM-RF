from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class PipelineArtifacts:
    """Container for staged pipeline outputs."""

    states_5m: Optional[pd.DataFrame] = None
    states_30m: Optional[pd.DataFrame] = None
    states_1d: Optional[pd.DataFrame] = None
    overlay: Optional[pd.DataFrame] = None
    super_state: Optional[pd.DataFrame] = None
    rf_trades: Optional[pd.DataFrame] = None
    metrics: Optional[Dict] = None
