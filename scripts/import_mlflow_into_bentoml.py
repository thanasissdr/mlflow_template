from src.utils.bentoml.import_mlflow_model import import_mlflow_model_into_bentoml

BUCKET_NAME = "mlflow"

EXPERIMENT_ID = "542485645421180081"
RUN_ID = "ffe3f2d5f1e943ff9b9168fe53fce50d"
MODEL_PATH = f"{EXPERIMENT_ID}/{RUN_ID}/artifacts/sklearn-model"


if __name__ == "__main__":
    import_mlflow_model_into_bentoml(BUCKET_NAME, MODEL_PATH, "models")
