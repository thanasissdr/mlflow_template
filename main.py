from src.dataset.load import load_dataset
from src.metrics.utils import metric_runners_registry_factory
from src.trainer.models.classification import (
    LogisticRegressionWrapper,
    RandomForestClassifierWrapper,
)
from src.trainer.train import trainer_factory

MODEL_WRAPPERS = [
    RandomForestClassifierWrapper(),
    LogisticRegressionWrapper(),
]

METRICS_CONFIGURATION = {
    "precision": {"average": "weighted"},
    "recall": {"average": "weighted"},
    "f1_score": {"average": "weighted"},
}

SKLEARN_DATASET = "breast_cancer"
TYPE_OF_PROBLEM = "classification"


def main():
    global MODEL_WRAPPERS, METRICS_CONFIGURATION, SKLEARN_DATASET

    X, y = load_dataset(SKLEARN_DATASET)

    metric_runners_registry = metric_runners_registry_factory(
        TYPE_OF_PROBLEM, METRICS_CONFIGURATION
    )

    for model_wrapper in MODEL_WRAPPERS:
        trainer_runner = trainer_factory(model_wrapper, metric_runners_registry)
        trainer_runner.run(X, y)


if __name__ == "__main__":
    main()
