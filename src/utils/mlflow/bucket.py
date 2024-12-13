import logging
from typing import Any

logging.basicConfig(
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def bucket_exists(client: Any, bucket: str) -> bool:
    try:
        client.head_bucket(Bucket=bucket)
        return True
    except Exception:
        return False


def create_bucket(client: Any, bucket: str) -> None:
    try:
        client.create_bucket(Bucket=bucket)
        logger.info(f"Bucket {bucket} created successfully")
    except Exception:
        logger.critical(f"Bucket {bucket} could not be created")


def create_bucket_if_not_exists(client: Any, bucket: str):
    if not bucket_exists(client, bucket):
        create_bucket(client, bucket)
