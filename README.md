# motorwrap (MVP)

Thin wrapper for peril-level CatBoost Poisson models with exposure handling, Optuna tuning, and robust scoring.

## Install (editable)
```bash
pip install -e .
```

## Fit
```bash
motorwrap fit-peril data.parquet --target-col claims --weight-col exposure --model-dir runs/perilA
```

## Score
```bash
motorwrap score-peril new_data.parquet --model-dir runs/perilA --output-path scored.parquet
```

## Notes
- For Poisson, we fit on `target/weight` with `case weights = weight` and exponentiate raw margin at predict time.
- Schema JSON ensures dtype and column order are consistent at scoring time.
