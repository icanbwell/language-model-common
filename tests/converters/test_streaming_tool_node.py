import pytest
from unittest.mock import AsyncMock, patch

from oidcauthlib.auth.exceptions.authorization_needed_exception import (
    AuthorizationNeededException,
)

from languagemodelcommon.converters.streaming_tool_node import StreamingToolNode


class TestStreamingToolNode:
    @pytest.mark.asyncio
    async def test_successful_invocation_yields_result(self) -> None:
        node = StreamingToolNode(tools=[])
        expected_result = {"messages": ["tool result"]}
        with patch.object(node, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = expected_result
            results = [
                chunk
                async for chunk in node.astream(
                    input={"tool_call": {"name": "test_tool", "args": {}}}
                )
            ]
        assert results == [expected_result]

    @pytest.mark.asyncio
    async def test_authorization_needed_exception_propagates(self) -> None:
        node = StreamingToolNode(tools=[])
        with patch.object(node, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = AuthorizationNeededException(
                message="login required"
            )
            with pytest.raises(AuthorizationNeededException):
                _ = [chunk async for chunk in node.astream(input={})]

    @pytest.mark.asyncio
    async def test_generic_exception_wrapped_with_tool_context(self) -> None:
        node = StreamingToolNode(tools=[])
        with patch.object(node, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = ValueError("something broke")
            with pytest.raises(
                Exception, match="Exception in tool my_tool"
            ) as exc_info:
                _ = [
                    chunk
                    async for chunk in node.astream(
                        input={"tool_call": {"name": "my_tool", "args": {"x": 1}}}
                    )
                ]
            assert "something broke" in str(exc_info.value)
            assert isinstance(exc_info.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_non_dict_input_handled_gracefully(self) -> None:
        node = StreamingToolNode(tools=[])
        with patch.object(node, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = RuntimeError("fail")
            with pytest.raises(Exception, match="Exception in tool None"):
                _ = [chunk async for chunk in node.astream(input="not_a_dict")]
