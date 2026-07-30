from __future__ import annotations

from typing import Dict, Protocol

from langchain_core.messages import AnyMessage


class MessagePreprocessor(Protocol):
    async def preprocess(
        self, *, messages: list[AnyMessage], headers: Dict[str, str] | None = None
    ) -> list[AnyMessage]: ...
