from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    ChatModelConfig,
    McpOAuthConfig,
)
from languagemodelcommon.configs.config_reader.mcp_json_reader import (
    McpJsonConfig,
    McpServerEntry,
    resolve_mcp_servers,
)


def _make_model(agents: list[AgentConfig]) -> ChatModelConfig:
    return ChatModelConfig(id="test", name="test", agents=agents)


class TestAuthProviderNormalization:
    def test_same_client_id_same_auth_server_share_provider(self) -> None:
        mcp_config = McpJsonConfig(
            mcpServers={
                "server-a": McpServerEntry(
                    type="http",
                    url="https://a.example.com/mcp",
                    oauth=McpOAuthConfig(
                        client_id="shared",
                        token_url="https://auth.example.com/token",
                        scopes=["read"],
                    ),
                ),
                "server-b": McpServerEntry(
                    type="http",
                    url="https://b.example.com/mcp",
                    oauth=McpOAuthConfig(
                        client_id="shared",
                        token_url="https://auth.example.com/token",
                        scopes=["write"],
                    ),
                ),
            }
        )
        agent_a = AgentConfig(name="tool-a", mcp_server="server-a")
        agent_b = AgentConfig(name="tool-b", mcp_server="server-b")
        resolve_mcp_servers([_make_model([agent_a, agent_b])], mcp_config)

        assert agent_a.auth_providers == agent_b.auth_providers
        assert agent_a.auth_providers is not None
        assert "shared" in agent_a.auth_providers[0]

    def test_same_client_id_union_scopes(self) -> None:
        mcp_config = McpJsonConfig(
            mcpServers={
                "server-a": McpServerEntry(
                    type="http",
                    url="https://a.example.com/mcp",
                    oauth=McpOAuthConfig(
                        client_id="shared",
                        token_url="https://auth.example.com/token",
                        scopes=["read", "profile"],
                    ),
                ),
                "server-b": McpServerEntry(
                    type="http",
                    url="https://b.example.com/mcp",
                    oauth=McpOAuthConfig(
                        client_id="shared",
                        token_url="https://auth.example.com/token",
                        scopes=["write", "profile"],
                    ),
                ),
            }
        )
        agent_a = AgentConfig(name="tool-a", mcp_server="server-a")
        agent_b = AgentConfig(name="tool-b", mcp_server="server-b")
        resolve_mcp_servers([_make_model([agent_a, agent_b])], mcp_config)

        assert agent_a.oauth is not None
        assert agent_a.oauth.scopes is not None
        assert set(agent_a.oauth.scopes) == {"read", "write", "profile"}

    def test_different_client_ids_separate_providers(self) -> None:
        mcp_config = McpJsonConfig(
            mcpServers={
                "server-a": McpServerEntry(
                    type="http",
                    url="https://a.example.com/mcp",
                    oauth=McpOAuthConfig(
                        client_id="client-1", token_url="https://auth.example.com/token"
                    ),
                ),
                "server-b": McpServerEntry(
                    type="http",
                    url="https://b.example.com/mcp",
                    oauth=McpOAuthConfig(
                        client_id="client-2", token_url="https://auth.example.com/token"
                    ),
                ),
            }
        )
        agent_a = AgentConfig(name="tool-a", mcp_server="server-a")
        agent_b = AgentConfig(name="tool-b", mcp_server="server-b")
        resolve_mcp_servers([_make_model([agent_a, agent_b])], mcp_config)

        assert agent_a.auth_providers != agent_b.auth_providers

    def test_no_client_id_dcr_uses_server_key(self) -> None:
        mcp_config = McpJsonConfig(
            mcpServers={
                "dcr-server": McpServerEntry(
                    type="http",
                    url="https://dcr.example.com/mcp",
                    oauth=McpOAuthConfig(
                        registration_url="https://auth.example.com/register",
                        authorization_url="https://auth.example.com/authorize",
                        token_url="https://auth.example.com/token",
                    ),
                ),
            }
        )
        agent = AgentConfig(name="tool-dcr", mcp_server="dcr-server")
        resolve_mcp_servers([_make_model([agent])], mcp_config)

        assert agent.auth_providers == ["dcr-server"]
        assert agent.auth == "jwt_token"
