from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.utils import Bunch


def load_data_factory(dataset_name: str, as_frame: bool) -> Bunch | tuple:
    match dataset_name:
        case "iris":
            return load_iris(as_frame=as_frame)
        case "breast_cancer":
            return load_breast_cancer(as_frame=as_frame)
        case "wine":
            return load_wine(as_frame=as_frame)
        case "diabetes":
            return load_diabetes(as_frame=as_frame)
        case _:
            raise ValueError(f"dataset_name {dataset_name} is not supported")


def load_dataset(dataset_name: str, as_frame: bool) -> tuple:
    data = load_data_factory(dataset_name, as_frame=as_frame)

    X = data["data"]  # type: ignore
    y = data["target"]  # type: ignore

    return X, y
