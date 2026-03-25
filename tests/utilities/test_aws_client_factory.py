from typing import Any

from botocore.config import Config

from languagemodelcommon.aws.aws_client_factory import AwsClientFactory


class _FakeSession:
    def __init__(self) -> None:
        self.client_calls: list[dict[str, Any]] = []

    def client(self, **kwargs: Any) -> object:
        self.client_calls.append(kwargs)
        return object()


def test_create_bedrock_client_uses_default_timeout_and_retry_values(
    monkeypatch,
) -> None:
    fake_session = _FakeSession()
    captured_profile_name: dict[str, str | None] = {"value": None}

    def _fake_boto3_session(*, profile_name: str | None = None) -> _FakeSession:
        captured_profile_name["value"] = profile_name
        return fake_session

    monkeypatch.delenv("AWS_CREDENTIALS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_BEDROCK_CONNECT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("AWS_BEDROCK_READ_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("AWS_BEDROCK_MAX_ATTEMPTS", raising=False)
    monkeypatch.setattr(
        "languagemodelcommon.aws.aws_client_factory.boto3.Session",
        _fake_boto3_session,
    )

    AwsClientFactory().create_bedrock_client()

    assert captured_profile_name["value"] is None
    assert len(fake_session.client_calls) == 1
    client_call = fake_session.client_calls[0]
    assert client_call["service_name"] == "bedrock-runtime"
    assert client_call["region_name"] == "us-east-1"
    assert isinstance(client_call["config"], Config)
    assert client_call["config"].connect_timeout == 10.0
    assert client_call["config"].read_timeout == 180.0


def test_create_bedrock_client_honors_environment_timeout_and_retry_overrides(
    monkeypatch,
) -> None:
    fake_session = _FakeSession()

    def _fake_boto3_session(*, profile_name: str | None = None) -> _FakeSession:
        assert profile_name == "dev-profile"
        return fake_session

    monkeypatch.setenv("AWS_CREDENTIALS_PROFILE", "dev-profile")
    monkeypatch.setenv("AWS_BEDROCK_CONNECT_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("AWS_BEDROCK_READ_TIMEOUT_SECONDS", "222")
    monkeypatch.setenv("AWS_BEDROCK_MAX_ATTEMPTS", "6")
    monkeypatch.setattr(
        "languagemodelcommon.aws.aws_client_factory.boto3.Session",
        _fake_boto3_session,
    )

    AwsClientFactory().create_bedrock_client()

    client_call = fake_session.client_calls[0]
    config = client_call["config"]
    assert isinstance(config, Config)
    assert config.connect_timeout == 7.0
    assert config.read_timeout == 222.0
    retries = config.retries
    max_attempts = retries.get("max_attempts") or retries.get("total_max_attempts")
    if (
        isinstance(max_attempts, int)
        and max_attempts > 6
        and "total_max_attempts" in retries
    ):
        assert max_attempts in {6, 7}
    else:
        assert max_attempts == 6
    assert retries.get("mode") == "adaptive"
