from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import polars as pl
import json

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
    """Coerce df to match schema (column order + types)"""
    out = []
    
    # Process each feature column according to the schema
    for name in schema["features"]:
        if name not in df.columns:
            # Add missing column with default value
            out.append(pl.Series(name, [None] * df.height, dtype=pl.Float64))
            continue
            
        s = df[name]
        
        # Check if this column should be categorical according to schema
        if schema["roles"][name] == "categorical":
            # Ensure categorical columns are strings
            if s.dtype != pl.Utf8:
                out.append(s.cast(pl.Utf8))
            else:
                out.append(s)
        else:
            # For numeric columns, ensure they are numeric
            if s.dtype == pl.Categorical or s.dtype == pl.Utf8:
                # Try to convert string/categorical to numeric
                try:
                    # Try direct conversion to float
                    out.append(s.cast(pl.Float64))
                except:
                    # If that fails, fill with None
                    out.append(pl.Series(name, [None] * df.height, dtype=pl.Float64))
            else:
                # Convert to appropriate numeric type
                try:
                    out.append(s.cast(pl.Float64))
                except:
                    # If conversion fails, fill with None
                    out.append(pl.Series(name, [None] * df.height, dtype=pl.Float64))
    
    return pl.DataFrame(out)

def save_schema(schema: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

def load_schema(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
