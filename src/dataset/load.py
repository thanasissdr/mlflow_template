from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.utils import Bunch


def load_data_factory(dataset_name: str) -> Bunch | tuple:
    match dataset_name:
        case "iris":
            return load_iris()
        case "breast_cancer":
            return load_breast_cancer()

        case _:
            raise ValueError(f"dataset_name {dataset_name} is not supported")


def load_dataset(dataset_name: str) -> tuple:
    data = load_data_factory(dataset_name)

    X = data["data"]  # type: ignore
    y = data["target"]  # type: ignore

    return X, y
