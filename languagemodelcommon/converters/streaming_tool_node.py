from typing import (
    Any,
    AsyncIterator,
    Optional,
    override,
    cast,
)

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input
from langchain_core.runnables.utils import (
    Output,
)
from langgraph.prebuilt import ToolNode
from oidcauthlib.auth.exceptions.authorization_needed_exception import (
    AuthorizationNeededException,
)


class StreamingToolNode(ToolNode):
    @override
    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        try:
            yield await self.ainvoke(input, config, **kwargs)
        except AuthorizationNeededException:
            raise
        except Exception as e:
            tool_call: dict[str, Any] = (
                cast(dict[str, Any], input.get("tool_call"))
                if isinstance(input, dict)
                else {}
            )
            raise Exception(
                f"Exception in tool {tool_call.get('name')} {tool_call.get('args')}: {type(e)}: {e}"
            ) from e
