from src.metrics.classification import get_classification_metric_runner_factory
from src.metrics.regression import get_regression_metric_runner_factory


def get_metric_runner_factory_fn(type_of_problem: str):
    match type_of_problem:
        case "classification":
            return get_classification_metric_runner_factory

        case "regression":
            return get_regression_metric_runner_factory

        case _:
            raise ValueError(
                f"Type of problem should be either 'classification' or 'regression', not '{type_of_problem}'"
            )
