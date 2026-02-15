from __future__ import annotations

# assembly-only module imports for migration wiring
from quant_refactor_skeleton.data import loaders as data_loaders
from quant_refactor_skeleton.data import resample as data_resample
from quant_refactor_skeleton.features import feature_builder as feature_builder_mod
from quant_refactor_skeleton.hmm import model as hmm_model
from quant_refactor_skeleton.hmm import portrait as hmm_portrait
from quant_refactor_skeleton.hmm import states as hmm_states
from quant_refactor_skeleton.overlay import overlay_builder as overlay_builder_mod
from quant_refactor_skeleton.super_state import super_hmm as super_hmm_mod


def run_msp_pipeline(cfg):
    raise NotImplementedError("assembly scaffold only")


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
