from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from mlflow.sklearn import log_model as log_sklearn_model
from sklearn.base import BaseEstimator


@dataclass
class ModelWrapper(ABC):
    model: Any

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def train(self, x: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def log_model(self, x: pd.DataFrame) -> None:
        pass


@dataclass
class SklearnModelWrapper(ModelWrapper):
    model: BaseEstimator
    artifact_path: str = "sklearn-model"
    registered_model_name: str = "sklearn-model"

    def get_params(self):
        return self.model.get_params()

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def log_model(self, x):
        log_sklearn_model(
            sk_model=self.model,
            artifact_path=self.artifact_path,
            input_example=x,
            registered_model_name=self.registered_model_name,
        )
