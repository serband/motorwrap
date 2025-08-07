from __future__ import annotations
import os, json
from typing import Dict, List, Optional
import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

from .schema import load_schema, coerce_like_schema

def _safe_exp(x: np.ndarray) -> np.ndarray:
    # guard against overflow in exp
    x = np.clip(x, -50, 50)
    return np.exp(x)

def score_peril(
    df: pl.DataFrame,
    model_dir: str,
    output_col: str = "pred_rate",
) -> pl.DataFrame:
    """Load model + schema from model_dir, coerce df columns, and return predicted rates.
    For CatBoost Poisson, raw margin is on log scale, so we exponentiate to get expected count/rate.
    """
    schema_path = os.path.join(model_dir, "schema.json")
    model_path = os.path.join(model_dir, "model.cbm")
    if not (os.path.exists(schema_path) and os.path.exists(model_path)):
        raise FileNotFoundError(f"Missing model or schema in {model_dir}")

    schema = load_schema(schema_path)
    X = coerce_like_schema(df, schema)
    cat_idx = schema["cat_features_idx"]

    model = CatBoostRegressor()
    model.load_model(model_path)

    pool = Pool(X.to_pandas(), cat_features=cat_idx)
    # CatBoost predict returns raw margin by default; for Poisson we exp to get mean lambda
    raw = model.predict(pool, prediction_type="RawFormulaVal")
    rate = _safe_exp(raw)
    return df.with_columns(pl.Series(output_col, rate))
