"""Motorwrap: a thin wrapper to fit/tune/score CatBoost Poisson models for insurance perils.

MVP contains:
- schema inference + coercion (Polars)
- single-peril fit with Optuna
- scoring with dtype coercion
- minimal CLI (typer)

Requires: polars, catboost, optuna, numpy, matplotlib, typer
"""

from .schema import infer_schema, coerce_like_schema, save_schema, load_schema
from .catboost_poisson import fit_peril
from .score import score_peril
