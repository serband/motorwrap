import pandas as pd
import polars as pl
import motorwrap as mw
import pytest
import tempfile
import os

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
    df = create_test_dataframe()
    
    # Create a temporary directory for model artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, "model")
        
        # Test the fit_peril function
        try:
            mw.fit_peril(
                df=df,
                target_col='nu_cl',
                weight_col='ex',
                model_dir=model_dir
            )
            # Check if model artifacts were created
            assert os.path.exists(model_dir)
            assert os.path.exists(os.path.join(model_dir, "model.cbm"))
            assert os.path.exists(os.path.join(model_dir, "schema.json"))
            print("Test passed: Model fitted and artifacts created")
        except Exception as e:
            print(f"Test failed: {e}")
            raise e

def test_predict_with_model():
    df = create_test_dataframe()
    
    # Create a temporary directory for model artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, "model")
        
        # First fit a model
        mw.fit_peril(
            df=df,
            target_col='nu_cl',
            weight_col='ex',
            model_dir=model_dir
        )
        
        # Test prediction function
        try:
            df_with_predictions = mw.predict_with_model(
                df=df,
                model_path=model_dir
            )
            
            # Check if predictions were added
            assert "prediction" in df_with_predictions.columns
            assert len(df_with_predictions) == len(df)
            print("Test passed: Predictions generated successfully")
        except Exception as e:
            print(f"Test failed: {e}")
            raise e

def test_explain_model_with_shap():
    df = create_test_dataframe()
    
    # Create a temporary directory for model artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, "model")
        shap_dir = os.path.join(temp_dir, "shap")
        
        # First fit a model
        mw.fit_peril(
            df=df,
            target_col='nu_cl',
            weight_col='ex',
            model_dir=model_dir
        )
        
        # Test SHAP explanation function
        try:
            mw.explain_model_with_shap(
                model_path=model_dir,
                df=df,
                output_path=shap_dir,
                sample_rows=3  # Use small sample for faster testing
            )
            
            # Check if SHAP plots were created
            assert os.path.exists(shap_dir)
            # Check for at least one SHAP plot file
            shap_files = os.listdir(shap_dir)
            assert len(shap_files) > 0
            print("Test passed: SHAP explanations generated successfully")
        except Exception as e:
            print(f"Test failed: {e}")
            raise e