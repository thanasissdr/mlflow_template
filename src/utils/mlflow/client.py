import os

import boto3
import mlflow
from dotenv import load_dotenv

load_dotenv("./infrastructure/.env")


def create_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def set_tracking_uri() -> None:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
