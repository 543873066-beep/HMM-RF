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


def run_new_rolling_runner(argv: Optional[list[str]] = None) -> int:
    if argv and any(a in ("-h", "--help") for a in argv):
        print("usage: qrs-new-rolling [--help]")
        print("placeholder rolling route (new)")
        return 0
    print("[QRS:new] rolling placeholder route active")
    return 0


def run_refactor_rolling(argv: Optional[list[str]] = None) -> int:
    route = os.getenv("QRS_ROLLING_ROUTE", "legacy").strip().lower()
    if route in {"legacy", "", "0", "false"}:
        return run_legacy_rolling_runner(argv=argv)
    if route in {"new", "refactor", "1", "true"}:
        return run_new_rolling_runner(argv=argv)
    return run_legacy_rolling_runner(argv=argv)


__all__ = ["run_legacy_rolling_runner", "run_new_rolling_runner", "run_refactor_rolling"]
