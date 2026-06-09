from __future__ import annotations

from typing import Dict, Sequence

from langchain_core.messages import AnyMessage

from languagemodelcommon.utilities.message_preprocessing.message_preprocessor import (
    MessagePreprocessor,
)


class CompositeMessagePreprocessor:
    def __init__(self, *, preprocessors: Sequence[MessagePreprocessor]) -> None:
        self._preprocessors = list(preprocessors)

    async def preprocess(
        self, *, messages: list[AnyMessage], headers: Dict[str, str] | None = None
    ) -> list[AnyMessage]:
        result = messages
        for preprocessor in self._preprocessors:
            result = await preprocessor.preprocess(messages=result, headers=headers)
        return result
