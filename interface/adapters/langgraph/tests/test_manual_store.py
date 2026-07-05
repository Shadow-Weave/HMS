"""Manual test of HMSStore as a LangGraph BaseStore."""

import asyncio

from hms_client import HMS
from hms_langgraph import HMSStore


async def main():
    client = HMS(base_url="http://localhost:8888")
    store = HMSStore(client=client)

    ns = ("user", "test-store-123")

    # Put some values
    print("--- Storing via put ---")
    await store.aput(ns, "pref-theme", {"preference": "dark mode", "category": "ui"})
    await store.aput(ns, "pref-lang", {"preference": "Python", "category": "coding"})
    print("Stored 2 items")

    await asyncio.sleep(2)

    # Search
    print("\n--- Searching ---")
    results = await store.asearch(ns, query="programming language preference")
    for item in results:
        print(f"  key={item.key} value={item.value} score={item.score:.2f}")

    # Get specific
    print("\n--- Get by key ---")
    item = await store.aget(ns, "pref-theme")
    if item:
        print(f"  key={item.key} value={item.value}")

    # List namespaces
    print("\n--- List namespaces ---")
    namespaces = await store.alist_namespaces()
    for ns_item in namespaces:
        print(f"  {ns_item}")

    # Cleanup (bank_id uses "." separator: "user.test-store-123")
    await client.adelete_bank("user.test-store-123")
    print("\n--- Done, bank cleaned up ---")


if __name__ == "__main__":
    asyncio.run(main())
