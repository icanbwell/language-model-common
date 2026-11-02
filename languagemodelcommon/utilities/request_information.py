from typing import Optional, Dict

from pydantic import BaseModel, ConfigDict, Field

from oidcauthlib.auth.models.auth import AuthInformation

from languagemodelcommon.utilities.tool_display_name_mapper import (
    ToolDisplayNameMapper,
)


class RequestInformation(BaseModel):
    """
    Represents the information about the request being processed.
    """

    model_config = ConfigDict(
        extra="forbid",  # Prevents any additional properties
        arbitrary_types_allowed=True,
    )

    auth_information: Optional[AuthInformation] = Field(
        default=None,
        description="The authentication information associated with the request, if available.",
    )

    user_id: Optional[str] = Field(
        default=None,
        description="The user ID associated with the request, if available.",
    )

    user_email: Optional[str] = Field(
        default=None,
        description="The user email associated with the request, if available.",
    )

    user_name: Optional[str] = Field(
        default=None,
        description="The user name associated with the request, if available.",
    )

    request_id: str = Field(description="The unique identifier for the request.")

    conversation_thread_id: Optional[str] = Field(
        default=None,
        description="The conversation thread identifier for the request, if applicable.",
    )

    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="The HTTP headers associated with the request.",
    )

    client_id: str | None = Field(
        default=None,
        description="The client ID associated with the request, if available.",
    )

    enable_debug_logging: bool = Field(
        default=False,
        description="Indicates whether debug logging is enabled for this request.",
    )

    tool_display_name_mapper: ToolDisplayNameMapper | None = Field(
        default=None,
        description="Optional mapper for friendly tool display names in streaming output.",
    )

    request_started_at: float | None = Field(
        default=None,
        description=(
            "A time.perf_counter() reading taken by the caller (same process) "
            "when the request arrived. When set, the Responses API stream "
            "emits one prepare item with the elapsed time before the first "
            "model call (BAI-1440)."
        ),
    )

    prepare_event_emitted: bool = Field(
        default=False,
        description=(
            "Whether the prepare stream item has already been emitted for "
            "this request. Scoped to this per-request RequestInformation "
            "instance so only the first model call of a turn emits it; use "
            "one RequestInformation per turn, since a reused instance "
            "suppresses the item on later turns."
        ),
    )

    compaction_notice_shown: bool = Field(
        default=False,
        description=(
            "Whether the tool_result_compacted UI notice has already been "
            "surfaced for this request. OldToolResultTruncator re-truncates "
            "and re-dispatches tool_result_compacted on every model call for "
            "as long as a message stays outside its keep-recent window, so "
            "this flag (scoped to this per-request RequestInformation "
            "instance, not the singleton LangGraphStreamingManager) limits "
            "the user-visible notice to once per request (BAI-920)."
        ),
    )
