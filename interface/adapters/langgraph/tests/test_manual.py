"""Manual integration test — requires running HMS API on localhost:8888."""

import asyncio

from hms_client import HMS
from hms_langgraph import create_hms_tools


async def main():
    client = HMS(base_url="http://localhost:8888")

    # Create a test bank
    await client.acreate_bank("langgraph-test", name="LangGraph Test")

    tools = create_hms_tools(client=client, bank_id="langgraph-test")
    retain, recall, reflect = tools

    # Test retain
    print("--- Retain ---")
    result = await retain.ainvoke("The user's favorite language is Python")
    print(result)

    result = await retain.ainvoke("The user lives in San Francisco")
    print(result)

    # Give the engine a moment to process
    await asyncio.sleep(2)

    # Test recall
    print("\n--- Recall ---")
    result = await recall.ainvoke("What programming language does the user like?")
    print(result)

    # Test reflect
    print("\n--- Reflect ---")
    result = await reflect.ainvoke("What do you know about the user?")
    print(result)

    # Cleanup
    await client.adelete_bank("langgraph-test")
    print("\n--- Done, bank cleaned up ---")


if __name__ == "__main__":
    asyncio.run(main())
