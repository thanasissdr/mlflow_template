from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
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


@dataclass
class SklearnModelWrapper(ModelWrapper):
    model: BaseEstimator

    def get_params(self):
        return self.model.get_params()

    def train(self, x, y):
        self.model.fit(x, y)  # type: ignore

    def predict(self, x):
        return self.model.predict(x)  # type: ignore
