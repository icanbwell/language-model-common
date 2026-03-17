import pytest
from typing import Any
from unittest.mock import patch, MagicMock
from languagemodelcommon.configs.config_reader.s3_config_reader import S3ConfigReader


@pytest.mark.asyncio
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.boto3")
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.UrlParser")
async def test_read_model_configs_success(
    mock_urlparser: MagicMock, mock_boto3: MagicMock
) -> None:
    # Mock URL parsing
    mock_urlparser.parse_s3_uri.return_value = ("bucket", "prefix/")
    # Mock S3 paginator and objects
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_paginator = MagicMock()
    mock_client.get_paginator.return_value = mock_paginator
    mock_page_iterator = [
        {
            "Contents": [
                {"Key": "prefix/modelA.json"},
                {"Key": "prefix/modelB.json"},
                {"Key": "prefix/ignore.txt"},
            ]
        }
    ]
    mock_paginator.paginate.return_value = mock_page_iterator

    # Mock S3 get_object and JSON
    def get_object_side_effect(Bucket: Any, Key: Any) -> Any:
        data = {"name": "A"} if "A" in Key else {"name": "B"}
        return {
            "Body": MagicMock(
                read=MagicMock(return_value=b'{"name": "%s"}' % data["name"].encode())
            )
        }

    mock_client.get_object.side_effect = get_object_side_effect

    # Patch ChatModelConfig to a simple class
    with patch(
        "languagemodelcommon.configs.config_reader.s3_config_reader.ChatModelConfig",
        side_effect=lambda **kwargs: type("C", (), kwargs)(),
    ):
        reader = S3ConfigReader()
        configs = await reader.read_model_configs(s3_url="s3://bucket/prefix/")
        assert [c.name for c in configs] == ["A", "B"]


@pytest.mark.asyncio
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.UrlParser")
async def test_read_model_configs_empty_bucket(mock_urlparser: MagicMock) -> None:
    mock_urlparser.parse_s3_uri.return_value = ("", "prefix")
    reader = S3ConfigReader()
    with pytest.raises(ValueError):
        await reader.read_model_configs(s3_url="s3://bucket/prefix/")


@pytest.mark.asyncio
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.UrlParser")
async def test_read_model_configs_empty_prefix(mock_urlparser: MagicMock) -> None:
    mock_urlparser.parse_s3_uri.return_value = ("bucket", "")
    reader = S3ConfigReader()
    with pytest.raises(ValueError):
        await reader.read_model_configs(s3_url="s3://bucket/prefix/")


@pytest.mark.asyncio
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.boto3")
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.UrlParser")
async def test_read_model_configs_client_error(
    mock_urlparser: MagicMock, mock_boto3: MagicMock
) -> None:
    mock_urlparser.parse_s3_uri.return_value = ("bucket", "prefix/")
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_paginator = MagicMock()
    mock_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "prefix/modelA.json"}]}
    ]
    # Simulate S3 client error
    from botocore.exceptions import ClientError

    mock_client.get_object.side_effect = ClientError({"Error": {}}, "GetObject")
    with patch(
        "languagemodelcommon.configs.config_reader.s3_config_reader.ChatModelConfig",
        side_effect=lambda **kwargs: type("C", (), kwargs)(),
    ):
        reader = S3ConfigReader()
        configs = await reader.read_model_configs(s3_url="s3://bucket/prefix/")
        assert configs == []


@pytest.mark.asyncio
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.boto3")
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.UrlParser")
async def test_read_model_configs_json_decode_error(
    mock_urlparser: MagicMock, mock_boto3: MagicMock
) -> None:
    mock_urlparser.parse_s3_uri.return_value = ("bucket", "prefix/")
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_paginator = MagicMock()
    mock_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "prefix/modelA.json"}]}
    ]
    # Simulate JSON decode error
    mock_client.get_object.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=b"notjson"))
    }
    with patch(
        "languagemodelcommon.configs.config_reader.s3_config_reader.ChatModelConfig",
        side_effect=lambda **kwargs: type("C", (), kwargs)(),
    ):
        reader = S3ConfigReader()
        configs = await reader.read_model_configs(s3_url="s3://bucket/prefix/")
        assert configs == []


@pytest.mark.asyncio
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.boto3")
@patch("languagemodelcommon.configs.config_reader.s3_config_reader.UrlParser")
async def test_read_model_configs_top_level_exception(
    mock_urlparser: MagicMock, mock_boto3: MagicMock
) -> None:
    mock_urlparser.parse_s3_uri.return_value = ("bucket", "prefix/")
    mock_boto3.client.side_effect = Exception("fail")
    reader = S3ConfigReader()
    with pytest.raises(Exception):
        await reader.read_model_configs(s3_url="s3://bucket/prefix/")
