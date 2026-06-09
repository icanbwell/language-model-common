from datetime import datetime, timezone

from langchain_core.messages import SystemMessage

from languagemodelcommon.pipeline.context import PipelineContext


class InsertDatetimeStep:
    """Insert a datetime context system message at the start of the message list."""

    async def run(self, *, context: PipelineContext) -> None:
        now = datetime.now(tz=timezone.utc)
        content = (
            f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}. "
            f"Day of week: {now.strftime('%A')}."
        )
        context.messages.insert(0, SystemMessage(content=content))
