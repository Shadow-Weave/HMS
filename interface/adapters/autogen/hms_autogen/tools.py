"""AutoGen tool definitions for HMS memory operations.

Provides a factory function that creates AutoGen-compatible ``FunctionTool``
instances backed by HMS's retain/recall/reflect APIs. These tools can
be passed directly to ``AssistantAgent(tools=[...])``.
"""

from __future__ import annotations

import logging
from typing import Any

from autogen_core.tools import FunctionTool
from hms_client import HMS

from ._client import resolve_client
from .config import (
    DEFAULT_BUDGET,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RECALL_TAGS_MATCH,
    Budget,
    TagsMatch,
    get_config,
)
from .errors import HMSError

logger = logging.getLogger(__name__)


def create_hms_tools(
    *,
    bank_id: str,
    client: HMS | None = None,
    hms_api_url: str | None = None,
    api_key: str | None = None,
    budget: Budget | None = None,
    max_tokens: int | None = None,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch | None = None,
    # Retain options
    retain_metadata: dict[str, str] | None = None,
    retain_document_id: str | None = None,
    # Recall options
    recall_types: list[str] | None = None,
    recall_include_entities: bool = False,
    # Reflect options
    reflect_context: str | None = None,
    reflect_max_tokens: int | None = None,
    reflect_response_schema: dict[str, Any] | None = None,
    reflect_tags: list[str] | None = None,
    reflect_tags_match: TagsMatch | None = None,
    include_retain: bool = True,
    include_recall: bool = True,
    include_reflect: bool = True,
) -> list[FunctionTool]:
    """Create HMS memory tools for an AutoGen agent.

    Returns a list of ``FunctionTool`` instances compatible with AutoGen's
    ``AssistantAgent(tools=[...])``.

    Args:
        bank_id: The HMS memory bank to operate on.
        client: Pre-configured HMS client (preferred).
        hms_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall/reflect budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        tags: Tags applied when storing memories via retain.
        recall_tags: Tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        retain_metadata: Default metadata dict for retain operations.
        retain_document_id: Default document_id for retain (groups/upserts memories).
        recall_types: Fact types to filter (world, experience, opinion, observation).
        recall_include_entities: Include entity information in recall results.
        reflect_context: Additional context for reflect operations.
        reflect_max_tokens: Max tokens for reflect results (defaults to max_tokens).
        reflect_response_schema: JSON schema to constrain reflect output format.
        reflect_tags: Tags to filter memories used in reflect (defaults to recall_tags).
        reflect_tags_match: Tag matching for reflect (defaults to recall_tags_match).
        include_retain: Include the retain (store) tool.
        include_recall: Include the recall (search) tool.
        include_reflect: Include the reflect (synthesize) tool.

    Returns:
        List of AutoGen FunctionTool instances.

    Raises:
        HMSError: If no client or API URL can be resolved.
    """
    resolved_client = resolve_client(client, hms_api_url, api_key)

    config = get_config()
    effective_tags = tags if tags is not None else (config.tags if config else None)
    effective_recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
    effective_recall_tags_match = (
        recall_tags_match
        if recall_tags_match is not None
        else (config.recall_tags_match if config else DEFAULT_RECALL_TAGS_MATCH)
    )
    effective_budget = budget if budget is not None else (config.budget if config else DEFAULT_BUDGET)
    effective_max_tokens = (
        max_tokens if max_tokens is not None else (config.max_tokens if config else DEFAULT_MAX_TOKENS)
    )

    tools: list[FunctionTool] = []

    if include_retain:

        async def hms_retain(content: str) -> str:
            """Store information to long-term memory for later retrieval.

            Use this to save important facts, user preferences, decisions,
            or any information that should be remembered across conversations.

            Args:
                content: The information to store in memory.
            """
            try:
                retain_kwargs: dict[str, Any] = {"bank_id": bank_id, "content": content}
                if effective_tags:
                    retain_kwargs["tags"] = effective_tags
                if retain_metadata:
                    retain_kwargs["metadata"] = retain_metadata
                if retain_document_id:
                    retain_kwargs["document_id"] = retain_document_id
                await resolved_client.aretain(**retain_kwargs)
                return "Memory stored successfully."
            except HMSError:
                raise
            except Exception as e:
                logger.error("Retain failed: %s", e)
                raise HMSError(f"Retain failed: {e}") from e

        tools.append(
            FunctionTool(
                hms_retain,
                description="Store information to long-term memory for later retrieval.",
                name="hms_retain",
            )
        )

    if include_recall:

        async def hms_recall(query: str) -> str:
            """Search long-term memory for relevant information.

            Use this to find previously stored facts, preferences, or context.
            Returns a numbered list of matching memories.

            Args:
                query: What to search for in memory.
            """
            try:
                recall_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": query,
                    "budget": effective_budget,
                    "max_tokens": effective_max_tokens,
                }
                if effective_recall_tags:
                    recall_kwargs["tags"] = effective_recall_tags
                    recall_kwargs["tags_match"] = effective_recall_tags_match
                if recall_types:
                    recall_kwargs["types"] = recall_types
                if recall_include_entities:
                    recall_kwargs["include_entities"] = True
                response = await resolved_client.arecall(**recall_kwargs)
                if not response.results:
                    return "No relevant memories found."
                lines = []
                for i, result in enumerate(response.results, 1):
                    lines.append(f"{i}. {result.text}")
                return "\n".join(lines)
            except HMSError:
                raise
            except Exception as e:
                logger.error("Recall failed: %s", e)
                raise HMSError(f"Recall failed: {e}") from e

        tools.append(
            FunctionTool(
                hms_recall,
                description="Search long-term memory for relevant information.",
                name="hms_recall",
            )
        )

    if include_reflect:

        async def hms_reflect(query: str) -> str:
            """Synthesize a thoughtful answer from long-term memories.

            Use this when you need a coherent summary or reasoned response
            about what you know, rather than raw memory facts.

            Args:
                query: The question to reflect on using stored memories.
            """
            try:
                reflect_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": query,
                    "budget": effective_budget,
                }
                if reflect_context:
                    reflect_kwargs["context"] = reflect_context
                effective_reflect_max = reflect_max_tokens or effective_max_tokens
                if effective_reflect_max:
                    reflect_kwargs["max_tokens"] = effective_reflect_max
                if reflect_response_schema:
                    reflect_kwargs["response_schema"] = reflect_response_schema
                # Reflect tags: use reflect-specific or fall back to recall tags
                effective_reflect_tags = reflect_tags if reflect_tags is not None else effective_recall_tags
                effective_reflect_tags_match = reflect_tags_match or effective_recall_tags_match
                if effective_reflect_tags:
                    reflect_kwargs["tags"] = effective_reflect_tags
                    reflect_kwargs["tags_match"] = effective_reflect_tags_match
                response = await resolved_client.areflect(**reflect_kwargs)
                return response.text or "No relevant memories found."
            except HMSError:
                raise
            except Exception as e:
                logger.error("Reflect failed: %s", e)
                raise HMSError(f"Reflect failed: {e}") from e

        tools.append(
            FunctionTool(
                hms_reflect,
                description="Synthesize a thoughtful answer from long-term memories.",
                name="hms_reflect",
            )
        )

    return tools
