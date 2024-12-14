import logging
import os
from tempfile import NamedTemporaryFile
from typing import Any

import mlflow

from src.utils.serialization.deserialize import deserialize_object

logging.basicConfig(
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def download_file(
    client: Any, bucket_name: str, remote_path: str, local_path: str
) -> None:
    client.download_file(bucket_name, remote_path, local_path)


def download_file_locally(
    client: str, bucket_name: str, remote_path: str, local_path: str
) -> None:
    try:
        download_file(client, bucket_name, remote_path, local_path)
        logger.info(
            f"Downloaded {bucket_name}/{remote_path} successfully into {local_path}"
        )
    except Exception as e:
        logger.critical(f"Could not download artifact {remote_path}. {e}")


def download_file_object_in_memory(
    client: Any, bucket_name: str, remote_path: str
) -> Any:
    with NamedTemporaryFile(delete=False) as fh:
        client.download_fileobj(bucket_name, remote_path, fh)
        fh.seek(0)
    data = deserialize_object(fh)
    return data


def download_data_in_memory(client: Any, bucket_name: str, remote_path: str) -> Any:
    try:
        data = download_file_object_in_memory(client, bucket_name, remote_path)
        return data
    except Exception as e:
        logger.critical(
            f"Could not retrieve data from {bucket_name}/{remote_path} into memory. {e}"
        )


def download_artifact(
    client: Any, bucket_name: str, remote_path: str, local_path: str | None = None
) -> None | Any:
    if isinstance(local_path, str):
        download_file_locally(client, bucket_name, remote_path, local_path)
    elif local_path is None:
        return download_data_in_memory(client, bucket_name, remote_path)
    else:
        raise TypeError(
            f"Local path type  {type(local_path)} is not supported. Should be either None/missing or string"
        )
