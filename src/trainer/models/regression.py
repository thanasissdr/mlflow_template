from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from src.trainer.models.wrappers import SklearnModelWrapper


@dataclass
class LinearRegressionWrapper(SklearnModelWrapper):
    model: BaseEstimator = LinearRegression()
