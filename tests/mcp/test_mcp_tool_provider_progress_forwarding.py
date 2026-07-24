from unittest.mock import AsyncMock, patch

import pytest

from languagemodelcommon.mcp.callbacks import CallbackContext
from languagemodelcommon.mcp.mcp_tool_provider import MCPToolProvider


@pytest.mark.asyncio
async def test_on_mcp_tool_progress_dispatches_custom_event() -> None:
    with patch(
        "languagemodelcommon.mcp.mcp_tool_provider.adispatch_custom_event",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        await MCPToolProvider.on_mcp_tool_progress(
            progress=2.0,
            total=10.0,
            message="Validating skill",
            context=CallbackContext(
                server_name="skills-publisher", tool_name="propose_skill"
            ),
        )

    mock_dispatch.assert_awaited_once()
    call_args = mock_dispatch.call_args
    assert call_args.args[0] == "mcp_task_progress"
    payload = call_args.args[1]
    assert payload["server_name"] == "skills-publisher"
    assert payload["tool_name"] == "propose_skill"
    assert "Validating skill" in payload["message"]


@pytest.mark.asyncio
async def test_on_mcp_tool_progress_swallows_runtime_error_outside_callback_context() -> (
    None
):
    """adispatch_custom_event raises RuntimeError when called outside an
    active LangChain callback-manager context (e.g. this direct unit test
    invocation without a real run). on_mcp_tool_progress must not propagate
    that error -- mirrors the existing pattern in
    _execute_tool_as_task (tool_invocation.py:203-209)."""
    with patch(
        "languagemodelcommon.mcp.mcp_tool_provider.adispatch_custom_event",
        new_callable=AsyncMock,
        side_effect=RuntimeError("no callback manager in context"),
    ):
        await MCPToolProvider.on_mcp_tool_progress(
            progress=1.0,
            total=None,
            message=None,
            context=CallbackContext(server_name="skills-publisher"),
        )
    # No exception raised -- test passes by not raising.
