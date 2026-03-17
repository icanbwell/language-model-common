"""
Graph builder for BaileyAI LangGraph workflows.

Provides functions to create compiled LangGraph state graphs with optional
health safety evaluation.

Graph Structure
---------------
When health safety evaluation is disabled:

    ┌─────────────┐
    │ react_agent │ ──────► END
    └─────────────┘

When health safety evaluation is enabled:

    ┌─────────────┐     ┌─────────────────────────┐
    │ react_agent │ ──► │ health_safety_evaluator │ ──► END
    └─────────────┘     └─────────────────────────┘
                                    │
                                    │ (if failed, currently routes to END anyway)
                                    ▼
                                   END

Node Descriptions
-----------------
react_agent:
    A LangGraph prebuilt react agent that processes user messages, invokes tools
    as needed, and generates AI responses. This is the main agent that handles
    the conversation.

health_safety_evaluator:
    Evaluates the last AI message from the react agent across 5 dimensions:
    - Patient communication (clarity, empathy)
    - Information accuracy (factual correctness)
    - Scope boundaries (staying within appropriate limits)
    - Privacy (protecting sensitive information)
    - Uncertainty (appropriate hedging)

    The evaluator scores each dimension and detects violations. If the score
    is below the target threshold, it uses a corrector LLM to refine the
    response internally (up to max_iterations). The final response replaces
    the original AI message in the state.

    Note: Currently, should_continue_after_evaluation always returns "end",
    so failed evaluations do not route back to the react_agent for regeneration.
    The correction loop happens internally within the health_safety_evaluator node.
"""

import os
import uuid
from typing import Sequence, Optional, Literal

from langchain.agents import create_agent
from langchain_ai_skills_framework.loaders.skill_loader import SkillLoaderProtocol
from langchain_ai_skills_framework.middleware.skills_middleware import SkillMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from oidcauthlib.utilities.environment.oidc_environment_variables import (
    OidcEnvironmentVariables,
)

from languagemodelcommon.langgraph.state import MyMessagesState
from languagemodelcommon.langgraph.health_safety.config import HealthSafetyConfig
from languagemodelcommon.langgraph.health_safety.nodes.evaluation_node import (
    HealthSafetyNode,
)
from languagemodelcommon.utilities.logger.log_levels import logger


def should_continue_after_evaluation(
    state: MyMessagesState,
) -> Literal["continue", "end"]:
    """
    Conditional edge function to determine if processing should continue after evaluation.

    This can be used to route the graph based on evaluation results.
    """
    passed = state["passed_evaluation"]
    if passed is None or passed:
        return "end"
    else:
        # Could route to a re-generation node if evaluation fails
        # For now, we just end
        return "end"


def add_health_safety_to_graph(
    workflow: StateGraph[MyMessagesState],
    agent_node_name: str = "react_agent",
    config: Optional[HealthSafetyConfig] = None,
) -> StateGraph[MyMessagesState]:
    """
    Add health safety evaluation to an existing LangGraph workflow.

    Args:
        workflow: The StateGraph to modify
        agent_node_name: The name of the agent node to follow with evaluation
        config: Optional configuration for the health safety evaluator

    Returns:
        The modified workflow with health safety evaluation
    """
    health_safety_node = HealthSafetyNode(config)

    # Add the health safety node
    workflow.add_node("health_safety_evaluator", health_safety_node)

    # Add edge from agent to evaluator
    workflow.add_edge(agent_node_name, "health_safety_evaluator")

    # Add conditional edge from evaluator
    workflow.add_conditional_edges(
        "health_safety_evaluator",
        should_continue_after_evaluation,
        {
            "continue": agent_node_name,  # Could regenerate if needed
            "end": END,
        },
    )

    return workflow


async def create_agent_graph(
    *,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    store: Optional[BaseStore] = None,
    checkpointer: Optional[BaseCheckpointSaver[str]] = None,
    enable_health_safety: Optional[bool] = None,
    health_safety_config: Optional[HealthSafetyConfig] = None,
    system_prompts: Optional[Sequence[str]] = None,
    skill_loader: SkillLoaderProtocol,
) -> CompiledStateGraph[MyMessagesState]:
    """
    Create a compiled LangGraph state graph for the agent.

    Creates a react agent with optional health safety evaluation. The health
    safety node can evaluate and refine AI responses before they are returned
    to the user.

    Args:
        llm: The language model to use for the agent
        tools: List of tools available to the agent
        store: Optional store for persistence
        checkpointer: Optional checkpointer for state management
        enable_health_safety: Whether to enable health safety evaluation.
                              If None, reads from HEALTH_SAFETY_ENABLE_EVALUATOR env var.
        health_safety_config: Optional configuration for the health safety evaluator
        system_prompts: Optional list of system prompts to prepend to the agent.
                        These are joined with newlines and used as the system message.
        skill_loader: The skill loader to use for providing skill information to the agent

    Returns:
        A compiled state graph ready for invocation
    """
    # Build system prompt from config if provided
    system_prompt: str | None = None
    if system_prompts:
        # Strip and filter empty prompts to avoid accidental separators
        cleaned_prompts = [p.strip() for p in system_prompts if p and p.strip()]
        if cleaned_prompts:
            system_prompt = "\n\n".join(cleaned_prompts)
            logger.debug(
                f"[GRAPH] {uuid.uuid4()} Using system prompt from config ({len(system_prompt)} chars): {system_prompt} )"
            )

    # Create the react agent with optional system prompt
    react_agent_runnable = create_agent(
        model=llm,
        tools=tools,
        state_schema=MyMessagesState,
        store=store,
        checkpointer=checkpointer,
        system_prompt=system_prompt,
        middleware=[SkillMiddleware(skill_loader=skill_loader)],
    )

    # Build the workflow
    workflow: StateGraph[MyMessagesState] = StateGraph(MyMessagesState)
    workflow.add_node("react_agent", react_agent_runnable)
    workflow.set_entry_point("react_agent")

    # Determine if health safety should be enabled
    if enable_health_safety is None:
        enable_health_safety = OidcEnvironmentVariables.str2bool(
            os.getenv("HEALTH_SAFETY_ENABLE_EVALUATOR", "false")
        )

    if enable_health_safety:
        logger.debug("🏥 [GRAPH] Adding health safety evaluation node to graph")

        # Create health safety config if not provided
        if health_safety_config is None:
            health_safety_config = HealthSafetyConfig(enabled=True)

        # Create and add the health safety node
        health_safety_node = HealthSafetyNode(health_safety_config)
        workflow.add_node("health_safety_evaluator", health_safety_node)

        # Add edge from agent to evaluator
        workflow.add_edge("react_agent", "health_safety_evaluator")

        # Add conditional edge from evaluator to end (or back to agent for regeneration)
        workflow.add_conditional_edges(
            "health_safety_evaluator",
            should_continue_after_evaluation,
            {
                "continue": "react_agent",
                "end": END,
            },
        )
    else:
        logger.debug("[GRAPH] Health safety evaluation disabled")
        # No health safety - just end after react_agent
        workflow.add_edge("react_agent", END)

    compiled_state_graph = workflow.compile()

    return compiled_state_graph  # type: ignore[return-value]
