from dataclasses import dataclass

import mlflow
import pandas as pd

from src.metrics.utils import (
    MetricRunnersRegistry,
    get_metrics,
)
from src.trainer.models.wrappers import ModelWrapper


@dataclass(kw_only=True)
class Trainer:
    model_wrapper: ModelWrapper
    metric_runners_registry: MetricRunnersRegistry

    def run(self, x_train: pd.DataFrame, y_train: pd.Series):
        self.model_wrapper.train(x_train, y_train)

        model_params = self.model_wrapper.get_params()

        y_train_pred = self.model_wrapper.predict(x_train)
        metrics = get_metrics(
            self.metric_runners_registry, y_true=y_train, y_pred=y_train_pred
        )

        self.log_into_mlflow(model_params, metrics, x_train)

    def log_into_mlflow(self, model_params: dict, metrics: dict, x_train: pd.DataFrame):
        with mlflow.start_run() as _:
            mlflow.log_params(model_params)
            mlflow.log_metrics(metrics)
            self.model_wrapper.log_model(x_train)


def trainer_factory(
    model_wrapper: ModelWrapper, metric_runners_registry: MetricRunnersRegistry
) -> Trainer:
    return Trainer(
        model_wrapper=model_wrapper, metric_runners_registry=metric_runners_registry
    )
