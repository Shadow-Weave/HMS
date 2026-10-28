"""HMS Elite quickstart: six offline scenarios — recall, revision, temporal, multi-session, work state, detective, rollback.
HMS Elite 快速上手：六个离线场景——召回、修订、时间、多会话、工作状态、侦探、回滚。
"""

# Purpose (EN): Runnable demo of the CONTRACT §19 golden scenarios S1–S6 plus consolidate()/graph_stats() on one HMS.
# 目的（中）：在同一个 HMS 上可运行地演示 CONTRACT §19 黄金场景 S1–S6，以及 consolidate()/graph_stats()。

# Input (EN): Nothing external; every message and structured agent event is inlined below (no network, no API key).
# 输入（中）：无外部输入；所有消息与结构化 Agent 事件都内联在下方（无网络、无 API key）。

# Output (EN): Printed S1–S6 results: the Work State Card, the why() chain, transition tuples, consolidate report.
# 输出（中）：打印 S1–S6 的结果：工作状态卡、why() 证据链、转移元组、consolidate 报告。

# State effect (EN): Writes only to an in-memory SQLite store (rule LLM + hash embedder) discarded on close().
# 状态影响（中）：只写入内存 SQLite 存储（rule LLM + hash 嵌入器），close() 后即丢弃。

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `python examples/quickstart.py` work from a source checkout that was not `pip install -e .`-ed:
# 让未执行 `pip install -e .` 的源码检出也能运行 `python examples/quickstart.py`：
# put the repository root (the parent of examples/) ahead of everything else on sys.path.
# 把仓库根目录（examples/ 的父目录）放到 sys.path 的最前面。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hms_elite import HMS

U = "alice"
S1 = {"user_id": U, "run_id": "run-coffee"}
SCOPE = {"user_id": U, "agent_id": "coder", "run_id": "run-billing"}
SARAH = [
    ("Sarah joined the billing team in March.", "2026-03-02T09:00:00Z", "run-a"),
    ("Sarah shipped the webhook retry design on May 6.", "2026-05-06T09:00:00Z", "run-b"),
    ("Sarah is reviewing the checkout refactor this week.", "2026-06-11T09:00:00Z", "run-c"),
]
EVENTS_FAIL = [
    {
        "object_type": "file_diff",
        "role": "tool",
        "actor_id": "coder",
        "timestamp": "2026-06-15T10:00:00Z",
        "payload": {
            "path": "src/billing/retry.ts",
            "patch": "+ backoff = 2 ** attempt",
            "summary": "add exponential backoff",
            "intent": "fix checkout webhook retry",
        },
    },
    {
        "object_type": "test_log",
        "role": "tool",
        "actor_id": "coder",
        "timestamp": "2026-06-15T10:04:00Z",
        "payload": {
            "suite": "billing_e2e",
            "passed": 11,
            "failed": 3,
            "failures": [{"name": "subscription_state_regression", "message": "state machine changed"}],
        },
    },
]


def main() -> None:
    """Run S1–S6, consolidate() and graph_stats() on one in-memory HMS and print the results.
    在同一个内存 HMS 上运行 S1–S6、consolidate() 与 graph_stats() 并打印结果。

    Input / 输入：
        none.
        无。

    Output / 输出：
        None; every result is printed to stdout.
        无；所有结果打印到标准输出。

    State effect / 状态影响：
        Creates one in-memory store, writes S1–S6 into it and closes it.
        创建一个内存存储，写入 S1–S6 后关闭。
    """
    # rule LLM + hash 嵌入器 + sqlite 索引；无需 API key
    hms = HMS({"store": {"path": ":memory:"}})  # rule LLM + hash embedder + sqlite index; no API key

    # S1 preference → revision (same Mem0 id, new version)
    # S1 偏好 → 修订（Mem0 id 不变，版本号递增）
    first = hms.add("I like oat-milk lattes in the morning.", timestamp="2026-04-02T08:00:00Z", **S1)["results"][0]
    hms.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z", **S1)
    print("S1:", hms.get(first["id"])["memory"], "| history:", [h["event"] for h in hms.history(first["id"])])

    # S2 temporal + S3 entity across sessions (one run per message)
    # S2 时间 + S3 实体，跨会话（每条消息一个 run）
    for text, at, run in SARAH:
        hms.add(text, timestamp=at, user_id=U, run_id=run)
    print("S2:", [r["memory"] for r in hms.search("what happened in May", filters={"user_id": U})["results"]])
    print("S3:", [r["memory"] for r in hms.get_all(filters={"user_id": U, "entity": "Sarah"})["results"]])

    # S4 constraint + structured agent events (no LLM) → Work State Card
    # S4 约束 + 结构化 Agent 事件（无 LLM）→ 工作状态卡
    hms.add("Do not modify the subscription state machine.", timestamp="2026-06-15T09:00:00Z", **SCOPE)
    hms.add_events(EVENTS_FAIL, **SCOPE)
    card = hms.compile("fix checkout webhook retry", filters=SCOPE, budget_tokens=800, as_of="2026-06-15T12:00:00Z")
    print(card.to_text())

    # S5 detective
    # S5 侦探
    print("S5:", hms.why("why must retry.ts not be modified", filters=SCOPE)["explanation"])

    # S6 checkpoint → feedback → rollback → transitions
    # S6 检查点 → 反馈 → 回滚 → 转移
    constraint_id = hms.get_all(filters={**SCOPE, "obs_type": "constraint"})["results"][-1]["id"]
    cp = hms.checkpoint(filters=SCOPE, label="before-feedback")
    fb = hms.feedback(constraint_id, "state_outdated", correction="retry.ts may change once billing_e2e is isolated.")
    print("S6 feedback:", [a["action"] for a in fb["actions"]])
    print("S6 rollback:", hms.rollback(cp)["restored"])
    for _s_t, a_t, _s_t1, r_t, _kappa_t, w_t in hms.transitions(filters=SCOPE):
        print("S6 transition:", a_t, "->", r_t, "| witnesses:", len(w_t))

    print(json.dumps(hms.consolidate(filters=SCOPE)["phases"]["consolidate"]), hms.graph_stats(filters=SCOPE)["edges"])
    hms.close()


if __name__ == "__main__":
    main()
