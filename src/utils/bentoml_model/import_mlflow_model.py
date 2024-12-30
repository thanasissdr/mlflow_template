import bentoml

from src.utils.mlflow.client import create_client
from src.utils.mlflow.transfer.download import download_folder_locally


def import_mlflow_model_into_bentoml(
    bucket_name: str, remote_path: str, local_root_dir: str
) -> None:
    """
    Download the files of a remote model dir locally
    and import the mlflow into bentoml
    """
    client = create_client()
    download_folder_locally(client, bucket_name, remote_path, local_root_dir)
    bentoml.mlflow.import_model("my_model", model_uri=f"{local_root_dir}/{remote_path}")
