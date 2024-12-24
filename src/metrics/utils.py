from dataclasses import dataclass, field

import pandas as pd

from .abstract import MetricRunner
from .runner import get_metric_runner_factory_fn


@dataclass
class MetricRunnersRegistry:
    metric_runners: dict[str, MetricRunner] = field(default_factory=lambda: {})

    def register_runner(self, metric_name: str, metric_runner: MetricRunner) -> None:
        self.metric_runners[metric_name] = metric_runner

    def delete_runner(self, metric_name: str):
        del self.metric_runners[metric_name]

    def get(self, metric_name: str):
        metric_runner = self.metric_runners.get(metric_name)
        if metric_runner is None:
            raise KeyError(f"Metric {metric_name} is not supported")


def metric_runners_registry_factory(
    type_of_problem: str,
    metrics_config: dict[str, dict],
) -> MetricRunnersRegistry:
    metric_runners_registry = MetricRunnersRegistry()

    for metric_name, metric_config in metrics_config.items():
        metric_runner_factory_fn = get_metric_runner_factory_fn(type_of_problem)
        metric_runner_factory = metric_runner_factory_fn(metric_name)
        metric_runner = metric_runner_factory.create(**metric_config)

        metric_runners_registry.register_runner(metric_name, metric_runner)
    return metric_runners_registry


def get_metrics(
    metric_runners_registry: MetricRunnersRegistry, y_true: pd.Series, y_pred: pd.Series
) -> dict:
    metrics = {}
    for (
        metric_name,
        metric_runner,
    ) in metric_runners_registry.metric_runners.items():
        metric = metric_runner.run(y_true=y_true, y_pred=y_pred)
        metrics.setdefault(metric_name, metric)
    return metrics
