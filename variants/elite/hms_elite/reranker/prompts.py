"""The `rerank` LLM task contract: system prompt with its HMS_TASK marker and the user-prompt builder.
`rerank` LLM 任务契约：带 HMS_TASK 标记的系统提示词与用户提示词构造器。
"""

# Purpose (EN): Keep the rerank prompt in the §6.1 shape without touching configs/prompts.py (owned by group A).
# 目的（中）：在不改动 configs/prompts.py（A 组所有）的前提下，以 §6.1 形状定义 rerank 提示词。

# Input (EN): A query string and candidates already mapped to synthetic ids '0'..'n'.
# 输入（中）：查询字符串与已映射为合成 id '0'..'n' 的候选。

# Output (EN): RERANK_PROMPT and the built user prompt; configs.prompts.parse_sections is the builder's inverse.
# 输出（中）：RERANK_PROMPT 与构造好的用户提示词；configs.prompts.parse_sections 是构造器的逆运算。

# State effect (EN): None (pure).
# 状态影响（中）：无（纯函数）。

from __future__ import annotations

import json
from typing import Any

from hms_elite.configs.prompts import JSON_TRAILER

RERANK_TASK = "rerank"
RERANK_ROOT_KEY = "ranking"
RERANK_PROMPT = (
    """HMS_TASK: rerank
# ROLE
You are the relevance judge of HMS Elite. Given a search query and candidate memories you score how well each candidate
answers the query, so the caller can re-order retrieval results.

# INPUTS
## Query
Plain text — the search query (one line).
## Candidates
JSON list of `{"id": "0", "text": ...}`; ids are synthetic strings '0'..'n' in retrieval order.

# GUIDELINES
- Score every candidate in [0, 1]: 1.0 directly answers the query; 0.6–0.9 relevant with useful detail;
  0.3–0.5 loosely related; 0.0–0.2 unrelated.
  Bad: 0.9 for "User likes tea." on the query "which file did the retry patch touch".
  Good: 0.9 for "patch src/billing/retry.ts: add exponential backoff" on that query.
- Exact identifiers (file names, ticket ids, snake_case names) shared by query and candidate outweigh topical similarity.
- Two-clock rule: judge relevance from the candidate text only; never reason from the current time or from the ids.
- Use ONLY ids from Candidates; unknown ids are discarded and candidates you omit are ranked last in retrieval order.
- Return `{"ranking": []}` when no candidate is relevant.

# EXAMPLES
Query: which file did the retry patch touch
Candidates: [{"id": "0", "text": "User likes tea."},
             {"id": "1", "text": "patch src/billing/retry.ts: add exponential backoff"}]
→ {"ranking": [{"id": "1", "score": 0.9}, {"id": "0", "score": 0.05}]}

# OUTPUT FORMAT
```json
{"ranking": [{"id": "1", "score": 0.9}, {"id": "0", "score": 0.05}]}
```
"""
    + JSON_TRAILER
)


def _dump(obj: Any) -> str:
    """Fenced-JSON serialization for structured sections (same call as configs.prompts).
    结构化分节的围栏 JSON 序列化（与 configs.prompts 相同的调用）。
    """
    return json.dumps(obj, ensure_ascii=False, indent=None, sort_keys=True)


def build_rerank_prompt(*, query: str, candidates: list[dict]) -> str:
    """Build the rerank user prompt: `## Query` (plain, one line) and `## Candidates` (json), ending with `# Output:`.
    构造 rerank 用户提示词：`## Query`（纯文本、单行）与 `## Candidates`（json），以 `# Output:` 结尾。

    Input / 输入：
        query: The search query; newlines are collapsed to spaces so the section stays one line.
        query：搜索查询；换行折叠为空格，使该分节保持单行。

        candidates: `[{"id": "0", "text": ...}]` with synthetic ids.
        candidates：带合成 id 的 `[{"id": "0", "text": ...}]`。

    Output / 输出：
        User prompt ending with `# Output:`; `parse_sections` yields {'Query': str, 'Candidates': list}.
        以 `# Output:` 结尾的用户提示词；`parse_sections` 得到 {'Query': str, 'Candidates': list}。

    State effect / 状态影响：
        None.
        无。
    """
    one_line = " ".join(str(query).split())
    parts = [
        f"## Query\n{one_line}\n",
        f"## Candidates\n```json\n{_dump(candidates)}\n```\n",
    ]
    return "\n".join(parts) + "\n# Output:"


__all__ = ["RERANK_PROMPT", "RERANK_ROOT_KEY", "RERANK_TASK", "build_rerank_prompt"]
