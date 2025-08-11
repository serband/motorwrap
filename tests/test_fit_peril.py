import pandas as pd
import polars as pl
import motorwrap as mw
import pytest
import tempfile
import os

def load_external_dataframe():
    import rdata
    import urllib.request
    url = "https://github.com/dutangc/CASdatasets/raw/master/data/freMTPL2freq.rda"
    with urllib.request.urlopen(url) as response:
        data = response.read()
    parsed_data = rdata.parser.parse_data(data)
    converted_data = rdata.conversion.convert(parsed_data)
    df = converted_data['freMTPL2freq']
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    return df

def create_test_dataframe():
    # Create test data with consistent lengths
    data = {
        'nu_cl': [1, 2, 3, 0, 1, 2],
        'ex': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'county': ['A', 'B', 'C', 'A', 'B', 'C'],
        'veh_age': [1, 2, 3, 4, 5, 6],
        'co_ownership': ['N', 'Y', 'N', 'Y', 'N', 'Y']
    }
    return pl.DataFrame(data)

def test_fit_peril():
    # df = create_test_dataframe()
    df = load_external_dataframe().head(1000)  # Use external data for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, "model")
        mw.fit_peril(
            df=df,
            target_col='ClaimNb',  # Use correct column names for your dataset
            weight_col='Exposure',
            model_dir=model_dir
        )
        assert os.path.exists(model_dir)
        assert os.path.exists(os.path.join(model_dir, "model.cbm"))
        assert os.path.exists(os.path.join(model_dir, "schema.json"))
        print("Test passed: Model fitted and artifacts created")

def test_predict_with_model():
    # df = create_test_dataframe()
    df = load_external_dataframe().head(1000)
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, "model")
        mw.fit_peril(
            df=df,
            target_col='ClaimNb',
            weight_col='Exposure',
            model_dir=model_dir
        )
        df_with_predictions = mw.predict_with_model(
            df=df,
            model_path=model_dir
        )
        assert "prediction" in df_with_predictions.columns
        assert len(df_with_predictions) == len(df)
        print("Test passed: Predictions generated successfully")

def test_explain_model_with_shap():
    # df = create_test_dataframe()
    df = load_external_dataframe().head(1000)
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, "model")
        shap_dir = os.path.join(temp_dir, "shap")
        mw.fit_peril(
            df=df,
            target_col='ClaimNb',
            weight_col='Exposure',
            model_dir=model_dir
        )
        mw.explain_model_with_shap(
            model_path=model_dir,
            df=df,
            output_path=shap_dir,
            sample_rows=10
        )
        assert os.path.exists(shap_dir)
        shap_files = os.listdir(shap_dir)
        assert len(shap_files) > 0
        print("Test passed: SHAP explanations generated successfully")

