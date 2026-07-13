"""Two-turn OpenAI Responses API demo with automatic HMS retain and recall."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from hms_litellm import wrap_openai
from openai import OpenAI


def required_env(name: str, fallback: str | None = None) -> str:
    candidates = [os.getenv(name)]
    if fallback:
        candidates.append(os.getenv(fallback))
    value = next((candidate for candidate in candidates if candidate and not candidate.endswith("_change_me")), None)
    if not value:
        fallback_hint = f" or {fallback}" if fallback else ""
        raise RuntimeError(f"Set {name}{fallback_hint} in .env before running the demo.")
    return value


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    return "(No text output returned.)"


def main() -> int:
    llm_api_key = required_env("OPENAI_API_KEY", "HMS_API_LLM_API_KEY")
    llm_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("HMS_API_LLM_BASE_URL") or None
    model = os.getenv("OPENAI_MODEL") or os.getenv("HMS_API_LLM_MODEL") or "gpt-4o-mini"

    hms_api_url = os.getenv("HMS_MEMORY_API_URL", "http://127.0.0.1:18080")
    hms_api_key = required_env("HMS_MEMORY_API_KEY", "HMS_API_TENANT_API_KEY")
    bank_id = os.getenv("HMS_MEMORY_BANK_ID", "automatic-memory-demo")
    session_id = os.getenv("HMS_MEMORY_SESSION_ID", f"demo-{int(time.time())}")

    native_client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    client = wrap_openai(
        native_client,
        hms_api_url=hms_api_url,
        api_key=hms_api_key,
        bank_id=bank_id,
        session_id=session_id,
        inject_memories=True,
        store_conversations=True,
        budget="mid",
        verbose=True,
    )

    print("\n[1/2] First call: the exchange is automatically retained.")
    first = client.responses.create(
        model=model,
        instructions="Answer naturally and briefly.",
        input=(
            "My name is Alice. I prefer dark mode, and my current project is a "
            "holographic memory system. Please acknowledge this information."
        ),
    )
    print(response_text(first))

    print("\n[2/2] Second call: relevant memories are automatically recalled and injected.")
    second = client.responses.create(
        model=model,
        instructions="Use recalled memory when it is relevant. Answer in one sentence.",
        input="What interface theme do I prefer, and what project am I working on?",
    )
    print(response_text(second))

    print("\nAutomatic memory demo completed.")
    print(f"HMS bank: {bank_id}")
    print(f"Session: {session_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nDemo failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
