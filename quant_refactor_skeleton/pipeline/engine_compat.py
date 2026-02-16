from __future__ import annotations

import importlib
import os
import sys
from typing import Optional, Sequence


def _invoke_legacy_main(module_name: str, argv: Optional[Sequence[str]] = None) -> int:
    mod = importlib.import_module(module_name)
    main = getattr(mod, "main", None)
    if main is None:
        raise AttributeError(f"{module_name}.main not found")

    old_argv = sys.argv[:]
    sys.argv = [module_name + ".py"] + list(argv or [])
    try:
        ret = main()
        if isinstance(ret, int):
            return ret
        return 0
    except SystemExit as e:
        code = e.code
        return int(code) if isinstance(code, int) else 0
    finally:
        sys.argv = old_argv


def run_legacy_engine_main(argv: Optional[list[str]] = None) -> int:
    return _invoke_legacy_main("msp_engine_ewma_exhaustion_opt_atr_momo", argv=argv)


def run_new_engine_main(argv: Optional[list[str]] = None) -> int:
    from quant_refactor_skeleton.pipeline.msp_pipeline import run_msp_pipeline

    return int(run_msp_pipeline(argv=argv))


def run_pipeline(mode: str, argv: Optional[list[str]] = None) -> int:
    route = os.getenv("QRS_PIPELINE_ROUTE", "legacy").strip().lower()
    if route in {"legacy", "", "0", "false"}:
        return run_legacy_engine_main(argv=argv)
    if route in {"new", "refactor", "1", "true"}:
        return run_new_engine_main(argv=argv)
    return run_legacy_engine_main(argv=argv)
