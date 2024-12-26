from dotenv import load_dotenv

from src.dataset.load import load_dataset
from src.metrics.utils import metric_runners_registry_factory
from src.trainer.models.classification import (
    LogisticRegressionWrapper,
    RandomForestClassifierWrapper,
    SGDClassifierWrapper,
)
from src.trainer.models.logger import SklearnModelLogger
from src.trainer.train import trainer_factory
from src.utils.mlflow.experiment import set_experiment

load_dotenv()


DATASET_NAME = "breast_cancer"
TYPE_OF_PROBLEM = "classification"


MODEL_WRAPPERS = [
    RandomForestClassifierWrapper(),
    LogisticRegressionWrapper(),
    SGDClassifierWrapper(),
]


MODEL_LOGGER = SklearnModelLogger()


METRICS_CONFIGURATION = {
    "precision": {"average": "weighted"},
    "recall": {"average": "weighted"},
    "f1_score": {"average": "weighted"},
}


def main():
    global MODEL_WRAPPERS, METRICS_CONFIGURATION, DATASET_NAME

    set_experiment(TYPE_OF_PROBLEM, DATASET_NAME)
    X, y = load_dataset(DATASET_NAME, as_frame=True)

    metric_runners_registry = metric_runners_registry_factory(
        TYPE_OF_PROBLEM, METRICS_CONFIGURATION
    )

    for model_wrapper in MODEL_WRAPPERS:
        trainer = trainer_factory(model_wrapper, MODEL_LOGGER, metric_runners_registry)
        trainer.run(X, y)


if __name__ == "__main__":
    main()
