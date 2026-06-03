from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AnyMessage

from languagemodelcommon.utilities.message_preprocessing.message_preprocessor import (
    MessagePreprocessor,
)


class CompositeMessagePreprocessor:
    def __init__(self, *, preprocessors: Sequence[MessagePreprocessor]) -> None:
        self._preprocessors = list(preprocessors)

    async def preprocess(self, *, messages: list[AnyMessage]) -> list[AnyMessage]:
        result = messages
        for preprocessor in self._preprocessors:
            result = await preprocessor.preprocess(messages=result)
        return result
