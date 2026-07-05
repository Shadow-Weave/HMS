"""
hms-agentcore
===================

Persistent memory for Amazon Bedrock AgentCore Runtime agents using HMS.

AgentCore Runtime sessions are ephemeral — they terminate on inactivity and
reprovision fresh. This package adds durable cross-session memory so agents
remember users, decisions, and patterns across any number of Runtime sessions.

Quick start:
    from hms_agentcore import HMSRuntimeAdapter, TurnContext, configure

    configure(
        hms_api_url="https://api.hms.local",
        api_key=os.environ["HMS_API_KEY"],
    )

    adapter = HMSRuntimeAdapter(agent_name="support-agent")

    # In your AgentCore Runtime handler:
    context = TurnContext(
        runtime_session_id=event["sessionId"],
        user_id=event["userId"],           # from validated auth context
        agent_name="support-agent",
        tenant_id=event.get("tenantId"),
    )

    result = await adapter.run_turn(
        context=context,
        payload={"prompt": user_message},
        agent_callable=my_agent,
    )
"""

from __future__ import annotations

from .adapter import HMSRuntimeAdapter, RecallPolicy, RetentionPolicy
from .bank import BankResolver, TurnContext, default_bank_resolver
from .config import (
    HMSAgentCoreConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import BankResolutionError, HMSAgentCoreError

__version__ = "0.1.0"

__all__ = [
    # Adapter
    "HMSRuntimeAdapter",
    "RecallPolicy",
    "RetentionPolicy",
    # Identity + bank
    "TurnContext",
    "BankResolver",
    "default_bank_resolver",
    # Configuration
    "configure",
    "get_config",
    "reset_config",
    "HMSAgentCoreConfig",
    # Errors
    "HMSAgentCoreError",
    "BankResolutionError",
]
