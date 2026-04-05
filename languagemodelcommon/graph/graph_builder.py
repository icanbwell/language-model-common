"""
LangGraph graph builder with smart history management.
"""

import logging
from typing import Any, List, Sequence

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.middleware.skills_middleware import SkillMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent as create_agent
from langgraph.store.base import BaseStore

from languagemodelcommon.history.conversation_history_manager import (
    ConversationHistoryManager,
)
from languagemodelcommon.history.smart_history_manager import SmartHistoryManager
from languagemodelcommon.state.messages_state import MyMessagesState

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builder for creating LangGraph conversation graphs.

    Handles graph construction with smart history management,
    tool integration, and optional state persistence.
    """

    def __init__(self, skill_loader: SkillLoaderProtocol) -> None:
        """
        Initialize the graph builder.

        Args:
            skill_loader: Default skill loader for agent middleware
        """
        self.skill_loader = skill_loader

    async def create_graph_for_llm_async(
        self,
        *,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        store: BaseStore | None,
        checkpointer: BaseCheckpointSaver[str] | None,
        system_prompts: List[str] | None = None,
        skill_loader: SkillLoaderProtocol | None = None,
        max_messages: int = 20,
        max_tokens: int = 4000,
    ) -> CompiledStateGraph[MyMessagesState]:
        """
        Create a graph for the language model asynchronously.

        Implements smart history management:
        - Server checkpoint is source of truth if it exists
        - Client history is used to bootstrap if no checkpoint
        - History is trimmed to fit context window
        - Optionally includes health safety evaluation

        Args:
            llm: The language model to use
            tools: List of tools available to the agent
            store: Optional store for cross-conversation persistence
            checkpointer: Optional checkpointer for state management
            system_prompts: Optional list of system prompts to prepend
            skill_loader: Optional override for the skill loader (per-request scoping)
            max_messages: Maximum number of messages before trimming
            max_tokens: Maximum tokens allowed in conversation history

        Returns:
            Compiled state graph ready for invocation

        Raises:
            TypeError: If skill_loader is not SkillLoaderProtocol
        """
        # Resolve skill loader
        resolved_skill_loader = skill_loader or self.skill_loader
        if not isinstance(resolved_skill_loader, SkillLoaderProtocol):
            raise TypeError(
                "skill_loader must be SkillLoaderProtocol, got "
                f"{type(resolved_skill_loader)}"
            )

        # Build system prompt
        system_prompt = self._build_system_prompt(system_prompts)

        # Log configuration
        self._log_configuration(
            tools=tools,
            store=store,
            checkpointer=checkpointer,
            system_prompt=system_prompt,
            max_messages=max_messages,
            max_tokens=max_tokens,
        )

        # Create history managers
        conversation_history_manager = ConversationHistoryManager(
            max_messages=max_messages,
            max_tokens=max_tokens,
        )

        smart_history_manager = SmartHistoryManager(
            checkpointer=checkpointer,
            history_manager=conversation_history_manager,
            llm=llm,
        )

        # Create react agent
        react_agent_runnable = create_agent(
            model=llm,
            tools=tools,
            state_schema=MyMessagesState,
            store=store,
            checkpointer=None,  # Workflow handles checkpointing
            system_prompt=system_prompt,
            middleware=[SkillMiddleware(skill_loader=resolved_skill_loader)],
        )

        # Build workflow
        workflow = self._build_workflow(
            smart_history_manager=smart_history_manager,
            react_agent_runnable=react_agent_runnable,
        )

        # Compile with checkpointer
        compiled_state_graph = workflow.compile(checkpointer=checkpointer)

        logger.info("Graph compilation complete")

        return compiled_state_graph

    def _build_system_prompt(self, system_prompts: List[str] | None) -> str | None:
        """
        Build system prompt from list of prompt strings.

        Args:
            system_prompts: List of prompt strings to combine

        Returns:
            Combined system prompt or None
        """
        if not system_prompts:
            return None

        cleaned_prompts = [p.strip() for p in system_prompts if p and p.strip()]

        if not cleaned_prompts:
            return None

        system_prompt = "\n\n".join(cleaned_prompts)
        logger.debug(
            "Using system prompt from config (%s chars)",
            len(system_prompt),
        )

        return system_prompt

    def _log_configuration(
        self,
        *,
        tools: Sequence[BaseTool],
        store: BaseStore | None,
        checkpointer: BaseCheckpointSaver[Any] | None,
        system_prompt: str | None,
        max_messages: int,
        max_tokens: int,
    ) -> None:
        """
        Log graph configuration for debugging.

        Args:
            tools: Available tools
            store: Store instance
            checkpointer: Checkpointer instance
            system_prompt: System prompt
            max_messages: Max message threshold
            max_tokens: Max token threshold
        """
        logger.debug(
            "Creating LLM graph with configuration: "
            "tools=%s, store=%s, checkpointer=%s, system_prompt=%s, "
            "max_messages=%s, max_tokens=%s",
            len(tools) if tools else 0,
            "provided" if store else "none",
            "provided" if checkpointer else "none",
            "provided" if system_prompt else "none",
            max_messages,
            max_tokens,
        )

    def _build_workflow(
        self,
        *,
        smart_history_manager: SmartHistoryManager,
        react_agent_runnable: Any,
    ) -> StateGraph[MyMessagesState]:
        """
        Build the workflow graph.

        Args:
            smart_history_manager: History manager for state selection
            react_agent_runnable: React agent runnable

        Returns:
            Configured workflow graph
        """
        workflow: StateGraph[MyMessagesState] = StateGraph(MyMessagesState)

        # Add smart history selection node
        workflow.add_node(
            "select_history", RunnableLambda(smart_history_manager.select_history)
        )

        # Add react agent node
        workflow.add_node("react_agent", react_agent_runnable)

        # Define flow
        workflow.set_entry_point("select_history")
        workflow.add_edge("select_history", "react_agent")
        workflow.add_edge("react_agent", END)

        logger.debug("Workflow graph constructed")

        return workflow
