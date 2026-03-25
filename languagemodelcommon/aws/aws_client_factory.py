import os
from typing import Any, cast

import boto3
from boto3 import Session
from botocore.config import Config

from types_boto3_bedrock_runtime.client import BedrockRuntimeClient
from types_boto3_s3.client import S3Client
from types_boto3_textract.client import TextractClient


class AwsClientFactory:
    @staticmethod
    def _get_float_env(*, name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None or value.strip() == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _get_int_env(*, name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None or value.strip() == "":
            return default
        try:
            return int(value)
        except ValueError:
            return default

    # noinspection PyMethodMayBeStatic
    def create_bedrock_client(self) -> BedrockRuntimeClient:
        """Create and return a Bedrock client"""
        retries_config = cast(
            Any,
            {
                "mode": "adaptive",
                "max_attempts": self._get_int_env(
                    name="AWS_BEDROCK_MAX_ATTEMPTS",
                    default=4,
                ),
            },
        )
        bedrock_config = Config(
            connect_timeout=self._get_float_env(
                name="AWS_BEDROCK_CONNECT_TIMEOUT_SECONDS",
                default=10.0,
            ),
            read_timeout=self._get_float_env(
                name="AWS_BEDROCK_READ_TIMEOUT_SECONDS",
                default=180.0,
            ),
            retries=retries_config,
            tcp_keepalive=True,
        )
        session: Session = boto3.Session(
            profile_name=os.environ.get("AWS_CREDENTIALS_PROFILE")
        )
        bedrock_client: BedrockRuntimeClient = session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            config=bedrock_config,
        )
        return bedrock_client

    # noinspection PyMethodMayBeStatic
    def create_s3_client(self) -> S3Client:
        session: Session = boto3.Session(
            profile_name=os.environ.get("AWS_CREDENTIALS_PROFILE")
        )
        s3_client: S3Client = session.client(
            service_name="s3",
            region_name="us-east-1",
        )
        return s3_client

    # noinspection PyMethodMayBeStatic
    def create_textract_client(self) -> TextractClient:
        session: Session = boto3.Session(
            profile_name=os.environ.get("AWS_CREDENTIALS_PROFILE")
        )
        textract_client: TextractClient = session.client(
            service_name="textract",
            region_name="us-east-1",
        )
        return textract_client
