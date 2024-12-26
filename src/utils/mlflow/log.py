import mlflow
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas


def log_params(params: dict):
    mlflow.log_params(params)


def log_dataset(df: pd.DataFrame, context: str):
    dataset = from_pandas(df)
    mlflow.log_input(dataset, context)


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)
