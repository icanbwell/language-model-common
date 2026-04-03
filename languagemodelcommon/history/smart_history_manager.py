"""
Smart history management with server-first, client-fallback strategy.
"""

import logging
from typing import Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from languagemodelcommon.history.conversation_history_manager import (
    ConversationHistoryManager,
)
from languagemodelcommon.state.messages_state import MyMessagesState

logger = logging.getLogger(__name__)


class SmartHistoryManager:
    """
    Manages conversation history with server-first, client-fallback strategy.

    This manager prioritizes server-stored conversation history (checkpoints)
    over client-provided history. It uses client history only to bootstrap
    new conversations when no server checkpoint exists.

    Strategy:
    1. Check if server has checkpoint for thread_id
    2. If yes: Use server history + extract new message from client
    3. If no: Use full client history to bootstrap
    4. Trim/manage the selected history to fit context window

    Attributes:
        checkpointer: Optional checkpoint storage for server-side state
        history_manager: Manager for trimming/summarizing history
        llm: Language model for generating summaries
    """

    def __init__(
        self,
        checkpointer: Optional[BaseCheckpointSaver[Any]],
        history_manager: ConversationHistoryManager,
        llm: BaseChatModel,
    ):
        """
        Initialize the smart history manager.

        Args:
            checkpointer: Checkpoint saver for server-side state persistence
            history_manager: History manager for context window optimization
            llm: Language model for summarization
        """
        self.checkpointer = checkpointer
        self.history_manager = history_manager
        self.llm = llm

    async def select_history(
        self, state: MyMessagesState, config: RunnableConfig
    ) -> MyMessagesState:
        """
        Select and manage conversation history.

        Implements server-first strategy:
        - If server checkpoint exists: Use it as source of truth
        - If no checkpoint: Bootstrap from client history
        - Always trim/manage to fit context window

        Args:
            state: Current state with client-provided messages
            config: Runnable config containing thread_id

        Returns:
            Updated state with managed history
        """
        client_messages = state.get("messages", [])
        thread_id = config.get("configurable", {}).get("thread_id")

        logger.info(
            "[HISTORY_SELECT] Client sent %s messages, thread_id=%s",
            len(client_messages),
            thread_id,
        )

        # Attempt to load server checkpoint
        server_checkpoint = await self._load_checkpoint(config, thread_id)

        # Decision logic: server-first, client-fallback
        if server_checkpoint:
            state = await self._use_server_history(
                state, client_messages, server_checkpoint
            )
        else:
            state = self._use_client_history(state, client_messages)

        # Manage/trim the selected history
        total_before = len(state.get("messages", []))
        managed_state = await self.history_manager.manage_history(state, self.llm)
        total_after = len(managed_state.get("messages", []))

        logger.info(
            "[HISTORY_SELECT] History trimmed: %s -> %s messages",
            total_before,
            total_after,
        )

        return managed_state

    async def _load_checkpoint(
        self, config: RunnableConfig, thread_id: Optional[str]
    ) -> Any:
        """
        Attempt to load server checkpoint.

        Args:
            config: Runnable config for checkpoint lookup
            thread_id: Thread identifier

        Returns:
            Checkpoint data if found, None otherwise
        """
        if not thread_id or not self.checkpointer:
            logger.debug("[HISTORY_SELECT] No thread_id or checkpointer available")
            return None

        try:
            checkpoint = await self.checkpointer.aget(config)
            if checkpoint:
                checkpoint_id = getattr(checkpoint, "id", None)
                logger.info(
                    "[HISTORY_SELECT] Found server checkpoint: %s",
                    checkpoint_id,
                )
                return checkpoint
            else:
                logger.info(
                    "[HISTORY_SELECT] No checkpoint found for thread %s",
                    thread_id,
                )
                return None
        except Exception as e:
            logger.warning(
                "[HISTORY_SELECT] Failed to load checkpoint: %s", e, exc_info=True
            )
            return None

    async def _use_server_history(
        self,
        state: MyMessagesState,
        client_messages: Sequence[BaseMessage],
        server_checkpoint: Any,
    ) -> MyMessagesState:
        """
        Use server checkpoint as source of truth.

        Extracts new message from client and appends to server history.

        Args:
            state: Current state to update
            client_messages: Messages from client
            server_checkpoint: Server checkpoint data

        Returns:
            Updated state with server history + new message
        """
        if "messages" not in server_checkpoint.channel_values:
            logger.warning("[HISTORY_SELECT] Checkpoint has no messages, using client")
            return self._use_client_history(state, client_messages)

        server_messages = server_checkpoint.channel_values["messages"]
        logger.info(
            "[HISTORY_SELECT] Using server history (%s messages)",
            len(server_messages),
        )

        # Extract new message from client (assume last message is new)
        if client_messages:
            new_message = client_messages[-1]

            # Check for duplicates
            if self._is_duplicate(new_message, server_messages):
                logger.info("[HISTORY_SELECT] New message is duplicate, skipping")
                state["messages"] = server_messages
            else:
                logger.info(
                    "[HISTORY_SELECT] Appending new %s",
                    new_message.__class__.__name__,
                )
                state["messages"] = server_messages + [new_message]
        else:
            # No new message from client
            logger.info("[HISTORY_SELECT] No new message from client")
            state["messages"] = server_messages

        return state

    def _use_client_history(
        self, state: MyMessagesState, client_messages: Sequence[BaseMessage]
    ) -> MyMessagesState:
        """
        Bootstrap from client history.

        Used when no server checkpoint exists.

        Args:
            state: Current state to update
            client_messages: Full history from client

        Returns:
            Updated state with client history
        """
        logger.info(
            "[HISTORY_SELECT] No server checkpoint, using client history (%s messages)",
            len(client_messages),
        )
        state["messages"] = list(client_messages)  # type: ignore[arg-type]
        return state

    def _is_duplicate(
        self,
        new_message: BaseMessage,
        existing_messages: Sequence[BaseMessage],
        check_last_n: int = 5,
    ) -> bool:
        """
        Check if message already exists in history.

        Args:
            new_message: Message to check
            existing_messages: Existing message history
            check_last_n: How many recent messages to check

        Returns:
            True if duplicate found, False otherwise
        """
        if not existing_messages:
            return False

        # Check last N messages for efficiency
        recent_messages = existing_messages[-check_last_n:]

        for msg in recent_messages:
            if msg.content == new_message.content and type(msg) is type(new_message):
                return True

        return False
