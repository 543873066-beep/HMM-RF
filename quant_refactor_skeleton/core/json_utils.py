"""JSON helpers migrated from legacy scripts."""

import json
import os
from typing import Dict, Optional

import numpy as np


def load_run_summary_optional(run_dir: str) -> Optional[Dict]:
    for root, _, files in os.walk(run_dir):
        if "run_summary.json" in files:
            try:
                with open(os.path.join(root, "run_summary.json"), "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return None
    return None


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if (not np.isfinite(obj)) or np.isnan(obj):
            return None
        return float(obj)
    return obj
