import logging
import os
from tempfile import _TemporaryFileWrapper
from typing import Any, Callable

import mlflow
from boto3.exceptions import S3UploadFailedError

from src.utils.mlflow.bucket import create_bucket_if_not_exists
from src.utils.serialization.serialize import serialize_object

logging.basicConfig(
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def upload_file(client: Any, object: str, bucket_name: str, remote_path: str):
    client.upload_file(object, bucket_name, remote_path)


def upload_file_object(
    client: Any, object: _TemporaryFileWrapper, bucket_name: str, remote_path: str
):
    with open(object.name, "rb") as fh:
        client.upload_fileobj(fh, bucket_name, remote_path)


def upload_fn_factory(object: str | _TemporaryFileWrapper) -> Callable:
    if isinstance(object, str):
        return upload_file
    elif isinstance(object, _TemporaryFileWrapper):
        return upload_file_object
    else:
        raise TypeError(f"Object of type {type(object)} is not supported by client")


def upload_artifact(
    client: Any,
    object: str | _TemporaryFileWrapper,
    bucket_name: str,
    remote_path: str,
) -> None:
    create_bucket_if_not_exists(client, bucket_name)
    upload_fn = upload_fn_factory(object)
    try:
        upload_fn(client, object, bucket_name, remote_path)
        logger.info(f"Uploaded {object} into {bucket_name}/{remote_path}")
    except S3UploadFailedError:
        logger.critical(
            f"Could not upload {object} to bucket {bucket_name}/{remote_path}"
        )


def serialize_and_upload_artifact(
    client: Any, object: str | Any, bucket_name: str, remote_path: str
) -> None:
    temp_file_wrapper = serialize_object(object)
    upload_artifact(client, temp_file_wrapper, bucket_name, remote_path)
