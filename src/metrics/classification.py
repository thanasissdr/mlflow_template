from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal

from sklearn.metrics import f1_score, precision_score, recall_score

from .abstract import MetricRunner


@dataclass
class ClassificationMetricRunner(MetricRunner):
    labels: list | None = None
    pos_label: str | int = 1
    average: Literal["micro", "macro", "samples", "weighted", "binary"] | None = (
        "binary"
    )
    sample_weight: list | None = None
    zero_division: int | Literal["warn"] = "warn"


@dataclass
class PresicionRunner(ClassificationMetricRunner):
    def run(self, y_true, y_pred):  # type: ignore
        return precision_score(y_true, y_pred, **asdict(self))


@dataclass
class RecallRunner(ClassificationMetricRunner):
    def run(self, y_true, y_pred):  # type: ignore
        return recall_score(y_true, y_pred, **asdict(self))


@dataclass
class F1ScoreRunner(ClassificationMetricRunner):
    def run(self, y_true, y_pred):  # type: ignore
        return f1_score(y_true, y_pred, **asdict(self))


class ClassificationMetricRunnerFactory(ABC):
    @abstractmethod
    def create(self, **kwargs) -> ClassificationMetricRunner:
        pass


class PrecisionRunnerFactory(ClassificationMetricRunnerFactory):
    def create(self, **kwargs):
        return PresicionRunner(**kwargs)


class RecallRunnerFactory(ClassificationMetricRunnerFactory):
    def create(self, **kwargs):
        return RecallRunner(**kwargs)


class F1ScoreRunnerFactory(ClassificationMetricRunnerFactory):
    def create(self, **kwargs):
        return F1ScoreRunner(**kwargs)


def get_classification_metric_runner_factory(
    metric_name: str,
) -> ClassificationMetricRunnerFactory:
    match metric_name:
        case "precision":
            return PrecisionRunnerFactory()
        case "recall":
            return RecallRunnerFactory()
        case "f1_score":
            return F1ScoreRunnerFactory()
        case _:
            raise ValueError(
                f"Metric {metric_name} is not associated with a metric runner factory class"
            )
