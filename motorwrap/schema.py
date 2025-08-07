from __future__ import annotations
import json
from typing import Dict, List, Optional, Tuple
import polars as pl

SENTINEL_CAT = "__NA__"

def _pl_dtype_name(dt: pl.datatypes.DataType) -> str:
    # Serialize Polars dtype to a stable string
    return str(dt)

def _from_dtype_name(name: str) -> pl.DataType:
    # Best-effort mapping back to Polars dtype
    # Default numeric -> Float64, string -> Utf8
    if "Utf8" in name or "String" in name:
        return pl.Utf8
    if "Int" in name:
        return pl.Int64
    if "Float" in name:
        return pl.Float64
    if "Boolean" in name:
        return pl.Boolean
    # Fallback
    return pl.Utf8

def infer_schema(
    df: pl.DataFrame,
    feature_cols: List[str],
    cat_cols: Optional[List[str]] = None,
) -> Dict:
    """Infer a simple schema:
    - features: ordered list of feature names
    - roles: {col: 'numeric'|'categorical'}
    - dtypes: {col: dtype_name}
    - cat_features_idx: indices (relative to features) of categoricals
    """
    if cat_cols is None:
        # Auto-detect categoricals as Utf8/String columns
        cat_cols = [c for c in feature_cols if df.schema[c] == pl.Utf8]
    roles = {}
    dtypes = {}
    for c in feature_cols:
        roles[c] = "categorical" if c in set(cat_cols) else "numeric"
        dtypes[c] = _pl_dtype_name(df.schema[c])
    cat_idx = [i for i, c in enumerate(feature_cols) if roles[c] == "categorical"]
    schema = {
        "version": 1,
        "features": feature_cols,
        "roles": roles,
        "dtypes": dtypes,
        "cat_features_idx": cat_idx,
        "sentinel_cat": SENTINEL_CAT,
    }
    return schema

def coerce_like_schema(df: pl.DataFrame, schema: Dict) -> pl.DataFrame:
    """Coerce df columns to match schema roles & dtypes.
    - Ensure all features present; add missing with sentinel/0
    - Cast categoricals to Utf8 and fill nulls with sentinel
    - Cast numerics to Float64 (safe), fill nulls with 0
    Columns not in schema are ignored.
    """
    feats = schema["features"]
    roles = schema["roles"]
    dtypes = schema["dtypes"]
    sentinel = schema.get("sentinel_cat", "__NA__")

    # Start with only schema features (order preserved)
    out = []
    for c in feats:
        if c in df.columns:
            s = df[c]
        else:
            # Create missing column
            if roles[c] == "categorical":
                s = pl.Series(name=c, values=[sentinel] * df.height, dtype=pl.Utf8)
            else:
                s = pl.Series(name=c, values=[0.0] * df.height, dtype=pl.Float64)

        if roles[c] == "categorical":
            s = s.cast(pl.Utf8).fill_null(sentinel)
        else:
            # Cast numerics to Float64 to be safe for CatBoost
            # If not numeric, try to coerce
            try:
                s = s.cast(pl.Float64)
            except Exception:
                s = s.cast(pl.Utf8).str.to_decimal(10, 2).cast(pl.Float64)
            s = s.fill_null(0.0)
        out.append(s)
    return pl.DataFrame(out)

def save_schema(schema: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

def load_schema(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
