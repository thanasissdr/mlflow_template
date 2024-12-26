import mlflow


def set_experiment(type_of_problem: str, dataset_name: str) -> None:
    mlflow.set_experiment(f"{type_of_problem}_{dataset_name}")
