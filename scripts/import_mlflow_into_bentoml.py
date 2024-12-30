import os

from src.utils.bentoml_model.import_mlflow_model import import_mlflow_model_into_bentoml

BUCKET_NAME = os.environ["BUCKET_NAME"]
EXPERIMENT_ID = os.environ["EXPERIMENT_ID"]
RUN_ID = os.environ["RUN_ID"]


MODEL_PATH = f"{EXPERIMENT_ID}/{RUN_ID}/artifacts/sklearn-model"


if __name__ == "__main__":
    import_mlflow_model_into_bentoml(BUCKET_NAME, MODEL_PATH, "models")
