import polars as pl
import os
from catboost import CatBoostRegressor
from typing import Optional
from .schema import load_schema, coerce_like_schema

def predict_with_model(
    df: pl.DataFrame, 
    model_path: str, 
    prediction_col: str = "prediction"
) -> pl.DataFrame:
    """
    Generate predictions using a trained model.
    
    This function loads a previously trained CatBoost Poisson model and generates
    predictions for the provided dataset. It handles schema matching and 
    categorical column conversion automatically.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing features for prediction
    model_path : str
        Path to the model directory containing:
        - model.cbm: Trained CatBoost model
        - schema.json: Feature schema and column types
    prediction_col : str, optional
        Name for the prediction column (default: "prediction")
        
    Returns
    -------
    pl.DataFrame
        Original dataframe with predictions appended as a new column
        
    Raises
    ------
    FileNotFoundError
        If model files are not found
    ValueError
        If required columns are missing
        
    Examples
    --------
    >>> import polars as pl
    >>> import motorwrap as mw
    >>> 
    >>> # Load your data
    >>> df = pl.read_csv("data.csv")
    >>> 
    >>> # Generate predictions
    >>> df_with_predictions = mw.predict_with_model(df, "/path/to/model")
    >>> 
    >>> # Save results
    >>> df_with_predictions.write_csv("predictions.csv")
    """
    # Validate model path
    model_file = os.path.join(model_path, "model.cbm")
    schema_file = os.path.join(model_path, "schema.json")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    
    # Load the model
    model = CatBoostRegressor()
    model.load_model(model_file)
    
    # Load schema
    schema = load_schema(schema_file)
    
    # Validate that all required features are present
    missing_features = set(schema["features"]) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Coerce dataframe to match schema
    df_coerced = coerce_like_schema(df, schema)
    
    # Convert to pandas for CatBoost prediction
    df_pd = df_coerced.to_pandas()
    
    # Set categorical columns
    cat_feature_names = [df_coerced.columns[i] for i in schema["cat_features_idx"]]
    for col_name in cat_feature_names:
        if col_name in df_pd.columns:
            df_pd[col_name] = df_pd[col_name].astype('category')
    
    # Generate predictions
    predictions = model.predict(df_pd)
    
    # Add predictions to original dataframe
    result_df = df.with_columns(
        pl.Series(name=prediction_col, values=predictions)
    )
    
    return result_df