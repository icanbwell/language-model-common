from __future__ import annotations

from typing import Protocol

from languagemodelcommon.pipeline.context import PipelineContext


class PipelineStep(Protocol):
    """Protocol for a single step in the request processing pipeline."""

    async def run(self, *, context: PipelineContext) -> None: ...
