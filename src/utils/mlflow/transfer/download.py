import logging
import os
from tempfile import NamedTemporaryFile
from typing import Any

import mlflow

from src.utils.file.characteristics import get_dirname
from src.utils.file.create import create_dir_if_not_exists
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


def download_file_locally_with_exception(
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


def download_file_object_in_memory_with_exception(
    client: Any, bucket_name: str, remote_path: str
) -> Any:
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
        download_file_locally_with_exception(
            client, bucket_name, remote_path, local_path
        )
    elif local_path is None:
        return download_file_object_in_memory_with_exception(
            client, bucket_name, remote_path
        )
    else:
        raise TypeError(
            f"Local path type  {type(local_path)} is not supported. Should be either None/missing or string"
        )


def download_folder_locally(
    client: Any, bucket_name: str, artifact_folder: str, local_root_dir: str
) -> None:
    objects = client.list_objects(Bucket=bucket_name, Prefix=artifact_folder)

    contents = objects.get("Contents")
    if contents is None:
        raise KeyError(
            f"Contents key does not exist in the response when trying to download contents of {artifact_folder}"
        )

    c = contents[0]
    full_local_path = f"{local_root_dir}/{c["Key"]}"
    main_dir = get_dirname(full_local_path)
    create_dir_if_not_exists(main_dir)

    for c in contents:
        full_local_path = f"{local_root_dir}/{c["Key"]}"
        download_file(client, bucket_name, c["Key"], full_local_path)

    logger.info(f"Files have been saved successfully in {main_dir}")
