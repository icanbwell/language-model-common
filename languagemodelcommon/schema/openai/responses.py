import logging
import warnings
from typing import Annotated, Literal, Optional, Union, Any, get_args

from openai.types.responses import ResponseInputParam
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

MAX_INSTRUCTIONS_LENGTH: int = 2000
MAX_CONVERSATION_ID_LENGTH: int = 256

ConversationId = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_CONVERSATION_ID_LENGTH)
]


# Outcome of loading stored history for a turn (BAI-1263). Producers only emit
# these values; the upstream `conversation` payload model allows extra keys and
# the fields are optional, so a consumer reading a value added later does not fail.
HistoryStatus = Literal["loaded", "empty", "failed"]
HISTORY_STATUSES: tuple[str, ...] = get_args(HistoryStatus)


def validate_conversation_id(*, value: str) -> str:
    """Return `value` if it is a usable conversation id, else raise ValueError.

    The value is untrusted client input. This only checks its shape (not blank,
    no surrounding whitespace, no ASCII control characters, bounded length). It
    does not check that the caller owns the conversation: the backend must
    verify ownership before reading history or echoing the id.
    """
    if not value.strip():
        raise ValueError("conversation id must not be blank")
    if value != value.strip():
        raise ValueError("conversation id must not have leading or trailing whitespace")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError("conversation id must not contain control characters")
    if len(value) > MAX_CONVERSATION_ID_LENGTH:
        raise ValueError(
            f"conversation id must be at most {MAX_CONVERSATION_ID_LENGTH} characters"
        )
    return value


class ConversationParam(BaseModel):
    """The `{ id }` form of the OpenAI Responses API `conversation` parameter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: ConversationId

    @field_validator("id")
    @classmethod
    def _id_is_not_blank(cls, value: str) -> str:
        return validate_conversation_id(value=value)


class ResponsesRequest(BaseModel):
    """Request model for OpenAI Responses API."""

    model_config = ConfigDict(
        extra="forbid"  # Prevents any additional properties
    )

    # Required parameters
    model: str = Field(..., description="The model to use for generating the response.")
    input: Union[str, ResponseInputParam] = Field(
        ..., description="The input text or data to generate a response for."
    )

    # Optional parameters
    stream: bool = False
    instructions: Optional[str] = Field(
        None,
        max_length=MAX_INSTRUCTIONS_LENGTH,
        description="System instructions to apply for this response.",
    )
    previous_response_id: Optional[str] = None
    conversation: Optional[Union[ConversationId, ConversationParam]] = Field(
        None,
        description=(
            "Conversation id, as a string or `{ id }` (OpenAI Responses API "
            "`conversation` parameter). Cannot be combined with "
            "`previous_response_id`. The server decides which id is used."
        ),
    )
    store: Optional[bool] = False

    # Generation parameters
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    max_output_tokens: Optional[int] = Field(None, gt=0)

    # Additional parameters
    truncation: Optional[str] = None
    service_tier: Optional[str] = None
    user: Optional[str] = None
    include: Optional[list[str]] = None

    # Tools (not fully implemented yet)
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    # LangGraph context
    thread_id: Optional[str] = Field(
        None,
        description="Accepted for compatibility; not used by this package.",
        deprecated=True,
    )
    user_id: Optional[str] = Field(
        None,
        description="Accepted for compatibility; not used by this package.",
        deprecated=True,
    )
    metadata: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _warn_on_deprecated_fields(self) -> "ResponsesRequest":
        # Read `__dict__` rather than the attributes: attribute access on a deprecated
        # field raises a DeprecationWarning of its own. A None value (an explicit null
        # or a model_dump round-trip) is not a use of the field.
        for field_name in ("thread_id", "user_id"):
            if self.__dict__.get(field_name) is None:
                continue
            message = (
                f"`{field_name}` is deprecated and ignored; it will be removed "
                "in a future release."
            )
            # DeprecationWarning is hidden by default outside tests, so also log it.
            logger.warning(message)
            warnings.warn(
                message,
                DeprecationWarning,
                # validator -> pydantic's __init__/model_validate -> the caller
                stacklevel=3,
            )
        return self

    @field_validator("conversation")
    @classmethod
    def _conversation_string_is_not_blank(
        cls, value: Optional[Union[str, ConversationParam]]
    ) -> Optional[Union[str, ConversationParam]]:
        if isinstance(value, str):
            validate_conversation_id(value=value)
        return value

    @model_validator(mode="after")
    def _reject_conversation_with_previous_response_id(self) -> "ResponsesRequest":
        if self.conversation is not None and self.previous_response_id is not None:
            raise ValueError(
                "conversation cannot be used in conjunction with previous_response_id"
            )
        return self

    @property
    def requested_conversation_id(self) -> Optional[str]:
        """The conversation id the client asked for, whichever form it used."""
        if isinstance(self.conversation, ConversationParam):
            return self.conversation.id
        return self.conversation
