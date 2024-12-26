from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from mlflow.sklearn import log_model as log_sklearn_model
from sklearn.base import BaseEstimator


@dataclass
class ModelLogger:
    artifact_path: str

    @abstractmethod
    def log(self, model: Any, x: pd.DataFrame) -> None:
        pass


@dataclass
class SklearnModelLogger(ModelLogger):
    artifact_path: str = "sklearn-model"

    def log(self, model: BaseEstimator, x: pd.DataFrame):
        log_sklearn_model(
            sk_model=model,
            artifact_path=self.artifact_path,
            input_example=x,
            registered_model_name=model.__class__.__name__,
        )
