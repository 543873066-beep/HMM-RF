# L8 Smoke Test Commands

## Compile

```bash
python -m compileall quant_refactor_skeleton/pipeline quant_refactor_skeleton/runner scripts
```

## Imports

```bash
python -c "import quant_refactor_skeleton.pipeline as p; import quant_refactor_skeleton.runner as r; print('ok: imports')"
python -c "from quant_refactor_skeleton.pipeline import engine_compat; print(engine_compat.run_pipeline('engine', ['--help']))"
```

## Legacy Diff Guard

```bash
git diff -- msp_engine_ewma_exhaustion_opt_atr_momo.py rolling_runner.py
```

## Compat Entrypoints

```bash
python scripts/run_engine_compat.py --route legacy -- --help
python scripts/run_rolling_compat.py --route legacy -- --help
```

## Regression Compare Example

```bash
python tools/regression_compare.py --old-equity <old_equity_csv> --new-equity <new_equity_csv> --out artifacts/regression_diff.csv
```
