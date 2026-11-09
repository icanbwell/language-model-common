import pytest
from pydantic import ValidationError

from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.configs.schemas.mcp_json_schema import McpServerEntry


@pytest.mark.parametrize(
    "auth",
    ["None", "jwt_token", "pass_through", "oauth", "headers", "identity_assertion"],
)
def test_every_documented_auth_value_is_accepted(*, auth: str) -> None:
    assert AgentConfig.model_validate({"name": "t", "auth": auth}).auth == auth
    assert (
        McpServerEntry.model_validate({"url": "https://x.invalid/", "auth": auth}).auth
        == auth
    )


@pytest.mark.parametrize("auth", ["vendor_token", "identity-assertion", "Bearer", ""])
def test_an_unknown_auth_value_is_rejected(*, auth: str) -> None:
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({"name": "t", "auth": auth})
    with pytest.raises(ValidationError):
        McpServerEntry.model_validate({"url": "https://x.invalid/", "auth": auth})
