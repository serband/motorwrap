from __future__ import annotations
import os, json, time, math
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
import optuna

from .schema import infer_schema, coerce_like_schema, save_schema

def _default_search_space(trial: optuna.Trial) -> Dict:
    return {
        "depth": trial.int("depth", 4, 10),
        "l2_leaf_reg": trial.loguniform("l2_leaf_reg", 1e-2, 1e2),
        "learning_rate": trial.loguniform("learning_rate", 0.01, 0.2),
        "min_data_in_leaf": trial.int("min_data_in_leaf", 20, 1000),
        "bagging_temperature": trial.uniform("bagging_temperature", 0.0, 1.0),
        "leaf_estimation_iterations": trial.int("leaf_estimation_iterations", 1, 10),
        "random_strength": trial.uniform("random_strength", 0.0, 1.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
        "one_hot_max_size": trial.int("one_hot_max_size", 2, 255),
    }

def _train_valid_split(
    df: pl.DataFrame,
    train_test_col: Optional[str],
    valid_size: float,
    random_state: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if train_test_col and train_test_col in df.columns:
        mask = df[train_test_col]
        # Accept boolean or 0/1
        if mask.dtype == pl.Boolean:
            train_df = df.filter(mask)
            valid_df = df.filter(~mask)
        else:
            train_df = df.filter(pl.col(train_test_col) == 1)
            valid_df = df.filter(pl.col(train_test_col) == 0)
        if valid_df.height == 0 or train_df.height == 0:
            # Fallback to random split
            rng = np.random.RandomState(random_state)
            idx = rng.permutation(df.height)
            cut = int((1.0 - valid_size) * df.height)
            train_df = df[idx[:cut].tolist(), :]
            valid_df = df[idx[cut:].tolist(), :]
    else:
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(df.height)
        cut = int((1.0 - valid_size) * df.height)
        train_df = df[idx[:cut].tolist(), :]
        valid_df = df[idx[cut:].tolist(), :]
    return train_df, valid_df

def fit_peril(
    df: pl.DataFrame,
    target_col: str,
    weight_col: Optional[str] = None,
    train_test_col: Optional[str] = None,
    cat_cols: Optional[List[str]] = None,
    model_dir: str = "./model_run",
    random_state: int = 42,
    n_trials: int = 40,
    timeout: Optional[int] = None,
    valid_size: float = 0.2,
) -> Dict:
    """Fit a Poisson CatBoost model with Optuna.
    - Target is divided by weights (exposure): y_adj = target/weight
    - Case weights = weight (default = 1 if None)
    Saves model + schema + metrics in model_dir.
    Returns dict of paths and best metrics.
    """
    os.makedirs(model_dir, exist_ok=True)

    if weight_col is None or weight_col not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("_mw_weight_temp"))
        weight_col_internal = "_mw_weight_temp"
    else:
        weight_col_internal = weight_col

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in df")

    # y_adj = target / weight
    df = df.with_columns(
        (pl.col(target_col).cast(pl.Float64) / pl.col(weight_col_internal).cast(pl.Float64)).alias("_y_adj"),
        pl.col(weight_col_internal).cast(pl.Float64).alias("_wts")
    )

    # Feature set = all columns except target/weight/train_test helper columns
    ignore = {target_col, weight_col_internal, "_y_adj", "_wts"}
    if train_test_col:
        ignore.add(train_test_col)
    feature_cols = [c for c in df.columns if c not in ignore]

    # Infer schema on full data features (dtypes, roles)
    schema = infer_schema(df, feature_cols, cat_cols=cat_cols)

    # Split train/valid
    train_df, valid_df = _train_valid_split(df, train_test_col, valid_size, random_state)

    # Coerce to schema (order + types)
    train_X = coerce_like_schema(train_df, schema)
    valid_X = coerce_like_schema(valid_df, schema)
    y_tr = train_df["_y_adj"].to_numpy()
    y_va = valid_df["_y_adj"].to_numpy()
    w_tr = train_df["_wts"].to_numpy()
    w_va = valid_df["_wts"].to_numpy()

    cat_idx = schema["cat_features_idx"]

    train_pool = Pool(train_X.to_pandas(), label=y_tr, weight=w_tr, cat_features=cat_idx)
    valid_pool = Pool(valid_X.to_pandas(), label=y_va, weight=w_va, cat_features=cat_idx)

    def objective(trial: optuna.Trial) -> float:
        params = _default_search_space(trial)
        model = CatBoostRegressor(
            loss_function="Poisson",
            random_seed=random_state,
            eval_metric="Poisson",
            od_type="Iter",
            od_wait=50,
            **params,
        )
        model.fit(
            train_pool,
            eval_set=valid_pool,
            verbose=False,
            use_best_model=True,
        )
        # NOTE: For Poisson, CatBoost predicts on the log-link scale by default.
        # We evaluate Poisson metric directly via best_score_.
        best_score = float(model.best_score_["validation"]["Poisson"])
        return best_score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    # Train a final model on combined train+valid with best params and early stopping using a small holdout from valid
    model = CatBoostRegressor(
        loss_function="Poisson",
        random_seed=random_state,
        eval_metric="Poisson",
        od_type="Iter",
        od_wait=50,
        **best_params,
    )
    model.fit(
        train_pool,
        eval_set=valid_pool,
        verbose=False,
        use_best_model=True,
    )

    # Persist artifacts
    model_path = os.path.join(model_dir, "model.cbm")
    model.save_model(model_path)

    schema_path = os.path.join(model_dir, "schema.json")
    save_schema(schema, schema_path)

    # Metrics
    metrics = {
        "best_value": study.best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "features": schema["features"],
        "cat_features_idx": schema["cat_features_idx"],
        "target_col": target_col,
        "weight_col": None if weight_col is None else weight_col,
        "notes": "Predictions at scoring time will be exp(raw) to recover Poisson mean.",
    }
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save trials
    try:
        import pandas as pd
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(model_dir, "optuna_trials.csv"), index=False)
    except Exception:
        pass

    return {
        "model_path": model_path,
        "schema_path": schema_path,
        "metrics_path": os.path.join(model_dir, "metrics.json"),
        "optuna_trials_path": os.path.join(model_dir, "optuna_trials.csv"),
        "best_value": study.best_value,
        "best_params": best_params,
    }
