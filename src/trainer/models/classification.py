from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.trainer.models.wrappers import SklearnModelWrapper


@dataclass
class RandomForestClassifierWrapper(SklearnModelWrapper):
    model: BaseEstimator = RandomForestClassifier(random_state=42)
    registered_model_name: str = "random-forest-classifier"


@dataclass
class LogisticRegressionWrapper(SklearnModelWrapper):
    model: BaseEstimator = LogisticRegression(random_state=42)
    registered_model_name: str = "logistic-regression"
