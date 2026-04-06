"""Tests for MCP interceptor types."""

from languagemodelcommon.mcp.interceptors.types import MCPToolCallRequest


class TestMCPToolCallRequest:
    def test_basic_creation(self) -> None:
        req = MCPToolCallRequest(name="tool1", args={"key": "val"}, server_name="s1")
        assert req.name == "tool1"
        assert req.args == {"key": "val"}
        assert req.server_name == "s1"
        assert req.headers is None

    def test_override_name(self) -> None:
        req = MCPToolCallRequest(name="tool1", args={}, server_name="s1")
        modified = req.override(name="tool2")
        assert modified.name == "tool2"
        assert req.name == "tool1"  # original unchanged

    def test_override_args(self) -> None:
        req = MCPToolCallRequest(name="tool1", args={"a": 1}, server_name="s1")
        modified = req.override(args={"b": 2})
        assert modified.args == {"b": 2}
        assert req.args == {"a": 1}

    def test_override_headers(self) -> None:
        req = MCPToolCallRequest(name="tool1", args={}, server_name="s1")
        modified = req.override(headers={"Authorization": "Bearer token"})
        assert modified.headers == {"Authorization": "Bearer token"}
        assert req.headers is None

    def test_override_preserves_other_fields(self) -> None:
        req = MCPToolCallRequest(
            name="tool1", args={"a": 1}, server_name="s1", headers={"X-Key": "val"}
        )
        modified = req.override(name="tool2")
        assert modified.args == {"a": 1}
        assert modified.server_name == "s1"
        assert modified.headers == {"X-Key": "val"}
