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


def run_legacy_rolling_runner(argv: Optional[list[str]] = None) -> int:
    return _invoke_legacy_main("rolling_runner", argv=argv)


def run_refactor_rolling(argv: Optional[list[str]] = None) -> int:
    route = os.getenv("QRS_ROLLING_ROUTE", "legacy").strip().lower()
    if route in {"legacy", "", "0", "false"}:
        return run_legacy_rolling_runner(argv=argv)
    if route in {"new", "refactor", "1", "true"}:
        # Hook only. Keep behavior unchanged until refactor rolling pipeline is wired.
        return run_legacy_rolling_runner(argv=argv)
    return run_legacy_rolling_runner(argv=argv)


__all__ = ["run_legacy_rolling_runner", "run_refactor_rolling"]
