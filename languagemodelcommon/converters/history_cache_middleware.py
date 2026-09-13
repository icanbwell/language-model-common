from typing import Any, Awaitable, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_aws import ChatAnthropicBedrock
from langchain_core.messages import AIMessage, ToolMessage


class HistoryCacheMiddleware(AgentMiddleware):
    """Caches conversation/tool-call history within a multi-round tool-calling turn.

    Anchors a `cache_control` breakpoint at the end of the message list before
    each model call after the first in a turn, so repeat calls read prior
    history from cache instead of resending it at full price. `ChatAnthropicBedrock`
    (via `langchain_anthropic`'s `_apply_cache_control_to_last_eligible_block`)
    recomputes the breakpoint's position from the live message list on every
    call, so it always re-anchors to the current last message rather than a
    fixed offset -- including after any history trimming -- with no
    bookkeeping needed here.

    Scoped to within-turn only (BAI-706 ADR, Phase 2 / Option C). Cross-turn
    history caching is a separate, larger follow-up requiring its own PHI
    review and is explicitly out of scope.
    """

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        if not self._should_cache(request):
            return await handler(request)
        return await handler(self._with_cache_control(request))

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        if not self._should_cache(request):
            return handler(request)
        return handler(self._with_cache_control(request))

    @staticmethod
    def _should_cache(request: ModelRequest[Any]) -> bool:
        """Only cache from the second model call in a turn onward.

        A `ToolMessage` in the message list means at least one full
        model -> tool -> result round-trip already happened, i.e. this is a
        subsequent call. The first call has no prior history worth reading
        from cache, and tagging it would only add a write premium with no
        matching read if the turn never calls a tool.
        """
        if not isinstance(request.model, ChatAnthropicBedrock):
            return False
        return any(isinstance(message, ToolMessage) for message in request.messages)

    @staticmethod
    def _with_cache_control(request: ModelRequest[Any]) -> ModelRequest[Any]:
        model_settings = dict(request.model_settings)
        model_settings["cache_control"] = {"type": "ephemeral"}
        return request.override(model_settings=model_settings)
