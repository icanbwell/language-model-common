from __future__ import annotations

from typing import Protocol

from langchain_core.messages import AnyMessage


class MessagePreprocessor(Protocol):
    async def preprocess(self, *, messages: list[AnyMessage]) -> list[AnyMessage]: ...
