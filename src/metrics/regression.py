from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal

from sklearn.metrics import mean_squared_error

from .abstract import MetricRunner


@dataclass
class RegresssionMetricRunner(MetricRunner):
    pass


@dataclass
class MeanSquaredErrorRunner(RegresssionMetricRunner):
    sample_weight: list | None = None
    multioutput: list | Literal["raw_values", "uniform_average"] = "uniform_average"

    def run(self, y_true, y_pred):  # type: ignore
        return mean_squared_error(y_true, y_pred, **asdict(self))


class RegressionMetricRunnerFactory(ABC):
    @abstractmethod
    def create(self, **kwargs) -> RegresssionMetricRunner:
        pass


class MeanSquaredErrorRunnerFactory(RegressionMetricRunnerFactory):
    def create(self, **kwargs):
        return MeanSquaredErrorRunner(**kwargs)


def get_regression_metric_runner_factory(
    metric_name: str,
) -> RegressionMetricRunnerFactory:
    match metric_name:
        case "mean_squared_error":
            return MeanSquaredErrorRunnerFactory()
        case _:
            raise ValueError(
                f"Metric {metric_name} is not associated with a metric runner factory class"
            )
