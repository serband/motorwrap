from __future__ import annotations
import os, json
import polars as pl
import typer
from typing import Optional, List

from .catboost_poisson import fit_peril as _fit_peril
from .score import score_peril as _score_peril

app = typer.Typer(add_completion=False)

@app.command()
def fit_peril(
    data_path: str = typer.Argument(..., help="Path to parquet/csv file."),
    target_col: str = typer.Option(...),
    model_dir: str = typer.Option("./model_run"),
    weight_col: Optional[str] = typer.Option(None),
    train_test_col: Optional[str] = typer.Option(None),
    cat_cols: Optional[str] = typer.Option(None, help="Comma-separated categorical columns"),
    n_trials: int = typer.Option(40),
    valid_size: float = typer.Option(0.2),
    random_state: int = typer.Option(42),
):
    """Fit a single peril model."""
    if data_path.endswith(".parquet"):
        df = pl.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        df = pl.read_csv(data_path)
    else:
        raise typer.BadParameter("Only parquet or csv supported.")
    cats = None if cat_cols is None else [c.strip() for c in cat_cols.split(",") if c.strip()]
    res = _fit_peril(
        df=df,
        target_col=target_col,
        weight_col=weight_col,
        train_test_col=train_test_col,
        cat_cols=cats,
        model_dir=model_dir,
        n_trials=n_trials,
        valid_size=valid_size,
        random_state=random_state,
    )
    typer.echo(json.dumps(res, indent=2))

@app.command()
def score_peril(
    data_path: str = typer.Argument(..., help="Path to parquet/csv with features to score."),
    model_dir: str = typer.Option("./model_run"),
    output_path: str = typer.Option("./scored.parquet"),
    output_col: str = typer.Option("pred_rate"),
):
    """Score a dataset with a fitted model."""
    if data_path.endswith(".parquet"):
        df = pl.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        df = pl.read_csv(data_path)
    else:
        raise typer.BadParameter("Only parquet or csv supported.")
    out = _score_peril(df=df, model_dir=model_dir, output_col=output_col)
    if output_path.endswith(".parquet"):
        out.write_parquet(output_path)
    elif output_path.endswith(".csv"):
        out.write_csv(output_path)
    else:
        raise typer.BadParameter("Only parquet or csv supported.")
    typer.echo(f"Saved: {output_path}")

def main():
    app()

if __name__ == "__main__":
    main()
