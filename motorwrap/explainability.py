import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import polars as pl
import numpy as np
from typing import Optional

def explain_model_with_shap(
    model_path: str, 
    df: pl.DataFrame, 
    output_path: str, 
    sample_rows: Optional[int] = None
) -> None:
    """
    Generate SHAP explanations for a trained model, including top interaction analysis.
    """
    # Load the model
    model = CatBoostRegressor()
    model.load_model(os.path.join(model_path, "model.cbm"))
    
    # Load schema
    from motorwrap.schema import load_schema, coerce_like_schema
    schema = load_schema(os.path.join(model_path, "schema.json"))
    
    # Coerce dataframe to match schema
    df_coerced = coerce_like_schema(df, schema)
    
    # Sample rows if specified
    if sample_rows is not None and sample_rows < len(df_coerced):
        df_coerced = df_coerced.sample(n=sample_rows, seed=42)
    
    # Convert to pandas for CatBoost prediction
    df_pd = df_coerced.to_pandas()
    
    # Set categorical columns
    cat_feature_names = [df_coerced.columns[i] for i in schema["cat_features_idx"]]
    for col_name in cat_feature_names:
        if col_name in df_pd.columns:
            df_pd[col_name] = df_pd[col_name].astype('category')
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(df_pd)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, df_pd, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary plot with feature names
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, df_pd, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate dependence plots for top features
    # Calculate feature importance from SHAP values
    feature_importance = np.abs(shap_values).mean(0)
    top_features_idx = feature_importance.argsort()[::-1][:5]  # Top 5 features
    
    for idx in top_features_idx:
        feature_name = df_pd.columns[idx]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(idx, shap_values, df_pd, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"shap_dependence_{feature_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # --- Interaction analysis ---
    interactions_dir = os.path.join(output_path, "interactions")
    os.makedirs(interactions_dir, exist_ok=True)
    feature_names = list(df_pd.columns)
    shap_interaction_values = explainer.shap_interaction_values(df_pd)
    n_features = shap_interaction_values.shape[1]
    interaction_scores = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            score = np.abs(shap_interaction_values[:, i, j]).mean()
            interaction_scores.append({
                "feature_1": feature_names[i],
                "feature_2": feature_names[j],
                "mean_abs_interaction": score
            })
    # Sort and select top-10 interactions
    top_interactions = sorted(interaction_scores, key=lambda x: x["mean_abs_interaction"], reverse=True)[:10]
    interactions_df = pd.DataFrame(top_interactions)
    interactions_df.to_csv(os.path.join(interactions_dir, "top_interactions.csv"), index=False)
    # Plot SHAP interaction plots for top-10 interactions
    for idx, row in interactions_df.iterrows():
        f1, f2 = row["feature_1"], row["feature_2"]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            (feature_names.index(f1), feature_names.index(f2)),
            shap_interaction_values, df_pd, feature_names=feature_names, interaction_index=feature_names.index(f2),
            show=False
        )
        plt.title(f"SHAP Interaction: {f1} & {f2}")
        plt.tight_layout()
        plt.savefig(os.path.join(interactions_dir, f"interaction_{idx+1}_{f1}_{f2}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"SHAP analysis completed. Plots saved to {output_path}")
    print(f"Top interactions and plots saved to {interactions_dir}")

def analyze_interactions(model, X, output_dir, feature_names=None, top_n=10):
    """
    Analyze and plot top-N feature interactions using SHAP.
    Args:
        model: Trained CatBoost model.
        X: DataFrame or numpy array used for SHAP analysis.
        output_dir: Directory to save outputs.
        feature_names: List of feature names (optional).
        top_n: Number of top interactions to analyze.
    """
    interactions_dir = os.path.join(output_dir, "interactions")
    os.makedirs(interactions_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X)

    # Compute mean absolute interaction values for each feature pair
    n_features = shap_interaction_values.shape[1]
    interaction_scores = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            score = np.abs(shap_interaction_values[:, i, j]).mean()
            interaction_scores.append({
                "feature_1": feature_names[i] if feature_names else str(i),
                "feature_2": feature_names[j] if feature_names else str(j),
                "mean_abs_interaction": score
            })

    # Sort and select top-N interactions
    top_interactions = sorted(interaction_scores, key=lambda x: x["mean_abs_interaction"], reverse=True)[:top_n]
    interactions_df = pd.DataFrame(top_interactions)
    interactions_df.to_csv(os.path.join(interactions_dir, "top_interactions.csv"), index=False)

    # Plot SHAP interaction plots for top-N interactions
    for idx, row in interactions_df.iterrows():
        f1, f2 = row["feature_1"], row["feature_2"]
        plt.figure()
        shap.dependence_plot(
            (feature_names.index(f1), feature_names.index(f2)) if feature_names else (int(f1), int(f2)),
            shap_interaction_values, X, feature_names=feature_names, interaction_index=feature_names.index(f2) if feature_names else int(f2),
            show=False
        )
        plt.title(f"SHAP Interaction: {f1} & {f2}")
        plt.tight_layout()
        plt.savefig(os.path.join(interactions_dir, f"interaction_{idx+1}_{f1}_{f2}.png"))
        plt.close()