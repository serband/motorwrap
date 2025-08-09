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
from .predict_with_model import predict_with_model
from .explainability import explain_model_with_shap

__all__ = [
    "fit_peril",
    "predict_with_model",
    "explain_model_with_shap"
]
