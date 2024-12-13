import logging
from typing import Any

from boto3.exceptions import S3UploadFailedError

from src.utils.mlflow.bucket import create_bucket_if_not_exists

logging.basicConfig(
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def upload_artifact(
    client: Any, local_path: str, bucket_name: str, remote_path: str
) -> None:
    create_bucket_if_not_exists(client, bucket_name)

    try:
        client.upload_file(local_path, bucket_name, remote_path)
        logger.info(f"Uploaded {local_path} into {bucket_name}/{remote_path}")
    except S3UploadFailedError:
        logger.critical(
            f"Could not upload {local_path} to bucket {bucket_name}/{remote_path}"
        )
