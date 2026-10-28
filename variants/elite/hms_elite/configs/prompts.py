"""Prompt constants with HMS_TASK markers, user-prompt builders and their parsers.
带 HMS_TASK 标记的提示词常量、用户提示词构造器及其解析器。
"""

# Purpose (EN): Define the six LLM task contracts (system prompts + user prompt shapes) shared by every provider, plus
# the composite memory_extraction task that folds observation / EP / revision into one call, and the compact twins
# (observation_extraction_compact / memory_extraction_compact) that ask for COMPACT_OBSERVATION_KEYS only, and the
# lines twin (memory_extraction_lines) whose reply is tab-separated text parsed by ingest.lines.
# 目的（中）：定义所有提供者共享的六个 LLM 任务契约（系统提示词 + 用户提示词形状），把观察 / EP / 修订
# 折叠为一次调用的复合任务 memory_extraction，以及只要求 COMPACT_OBSERVATION_KEYS 的紧凑孪生任务
# （observation_extraction_compact / memory_extraction_compact），以及回复为 tab 分隔文本、由 ingest.lines 解析的
# 行式孪生任务（memory_extraction_lines）。

# Input (EN): Structured inputs (memories, messages, nodes, pairs) already mapped to synthetic refs.
# 输入（中）：已映射为合成引用的结构化输入（记忆、消息、节点、配对）。

# Output (EN): System/user prompt strings and the parsed sections of a built user prompt.
# 输出（中）：系统/用户提示词字符串，以及已构造用户提示词的分节解析结果。

# State effect (EN): None (pure).
# 状态影响（中）：无（纯函数）。

from __future__ import annotations

import json
from typing import Any

TASKS = (
    "observation_extraction",
    "observation_extraction_compact",
    "ep_extraction",
    "revision_decision",
    "edge_proposal",
    "coarsen",
    "procedural_summary",
)
# The observation keys the compact contracts (`*_compact` tasks) ask the LLM for; every other observation field is
# derived locally (ingest.derive). `supersedes_hint_ids`, `when` and `valid_to` are optional and omitted when absent.
# 紧凑契约（`*_compact` 任务）要求 LLM 输出的观察键；其余观察字段都在本地推导（ingest.derive）。
# `supersedes_hint_ids`、`when`、`valid_to` 为可选键，没有时省略。
COMPACT_OBSERVATION_KEYS = (
    "id", "event_idx", "text", "type", "speaker", "supersedes_hint_ids", "when", "valid_to", "entities",
)
# The one-letter type vocabulary of a lines-contract OBS line (memory_extraction_lines, §6.6) and its inverse.
# 行式契约 OBS 行的单字母类型词表（memory_extraction_lines，§6.6）及其逆映射。
LINE_TYPES: dict[str, str] = {
    "F": "fact",
    "P": "preference_event",
    "H": "habit",
    "C": "constraint",
    "T": "temporal_fact",
}
TYPE_LETTERS: dict[str, str] = {name: letter for letter, name in LINE_TYPES.items()}
PAST_MESSAGE_TRUNCATION_LIMIT = 300
JSON_TRAILER = "Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers."
MARKDOWN_TRAILER = "Return ONLY the markdown above. No JSON, no wrappers."
EDGE_TYPE_DEFINITIONS: dict[str, str] = {
    "temporal": "A happened before B in the same run (time order only, no claim of influence).",
    "causal": "A's outcome triggered B's action (cause → effect).",
    "revision": "B supersedes A (a newer version of the same fact, constraint or action).",
    "conflict": "A and B are incompatible and neither is clearly newer or truer.",
    "use": "B relies on A (a constraint, tool result or prior EP consumed by B).",
    "distill": "B is derived from A (a constraint or skill distilled from an EP or InterfaceEP).",
}
_NODE_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("ref", ""),
    ("kind", "observation"),
    ("u", ""),
    ("b", ""),
    ("a", ""),
    ("y", ""),
    ("r", "unknown"),
    ("subject", None),
    ("predicate", None),
    ("objects", []),
    ("time", None),
    ("entities", []),
    ("topics", []),
    ("anchor_refs", []),
)
PROCEDURAL_SUMMARY_HEADINGS = (
    "## Summary of the agent's execution history",
    "**Task Objective**:",
    "**Progress Status**:",
    "**Agent Action**:",
    "**Action Result**:",
    "**Key Findings**:",
    "**Errors & Challenges**:",
    "**Current Context**:",
)
PROCEDURAL_SKELETON = """## Summary of the agent's execution history

**Task Objective**: ...
**Progress Status**: ...

1. **Agent Action**: ...
   **Action Result**: ...
2. **Agent Action**: ...
   **Action Result**: ...

**Key Findings**: ...
**Errors & Challenges**: ...
**Current Context**: ..."""

OBSERVATION_EXTRACTION_PROMPT = """HMS_TASK: observation_extraction
# ROLE
You are the observation extractor of HMS Elite. You turn new conversation messages into typed, dated, witnessed
observations about the user (or the assistant / the project) that are worth remembering later.

# INPUTS
## Last k Messages
Recent context as plain `role: text` lines (already stored; never re-extract them).
## Existing Memories
JSON list of `{"id": "0", "text": ...}`; cite these ids in `supersedes_hint_ids` when a new observation replaces one.
## New Messages
JSON list of `{"idx": 0, "role": "user", "text": ..., "timestamp": ...}`; every `event_idx` MUST index this list.
## Observation Date
`YYYY-MM-DD` — when the messages were written; resolve relative dates ONLY against it.
## Current Date
`YYYY-MM-DD` — when the system runs; NEVER use it to resolve dates in the messages.
## Custom Instructions
Optional caller guidance (present only when given).

# GUIDELINES
- Two-clock rule: `date_observed` is resolved against Observation Date (the user's clock), never Current Date (the
  system's clock).
  Bad: "yesterday" → the day before Current Date.
  Good: "yesterday" → the day before Observation Date.
- One observation per self-contained statement, third person, 5–80 words.
  Bad: "I like oat-milk lattes"
  Good: "User likes oat-milk lattes in the morning."
- Type by intent: preference_event (likes/dislikes), constraint (must / must not), habit (usually / every day),
  temporal_fact (a dated event), fact (everything else); `predicate` is a lowercase snake verb.
  Bad: "Do not modify the subscription state machine." as type fact.
  Good: type constraint, predicate must_not, relation constraint.
- A supersession cue (now, switched to, no longer, instead, actually) sets relation supersession and cites the
  superseded memory.
  Bad: "User likes black coffee." with relation single_fact and no hint after "Actually I switched to black coffee now."
  Good: relation supersession, supersedes_hint_ids ["0"].
- Never invent ids: supersedes_hint_ids ⊂ Existing Memories ids; event_idx ⊂ New Messages idx.
- Questions, greetings, and assistant boilerplate yield nothing; assistant statements about its own actions are
  facts with subject "assistant" and attributed_to "assistant".
- `attributed_to` is exactly "user" or "assistant" (which side of the conversation said it), never a person's
  name; `speaker` is the named person who said it in a multi-speaker transcript (also in `subject` and
  `entity_mentions`), or null in a plain user/assistant chat.
  Bad: "attributed_to": "Katrina", "speaker": null
  Good: "attributed_to": "user", "speaker": "Katrina", "subject": "Katrina", "entity_mentions": ["Katrina"]

# EXAMPLES
Existing Memories: [{"id": "0", "text": "User likes oat-milk lattes in the morning."}]
New Messages: [{"idx": 0, "role": "user", "text": "Actually I switched to black coffee now.",
               "timestamp": "2026-04-20T08:00:00+00:00"}]
Observation Date: 2026-04-20
→ {"observations": [{"id": "0", "event_idx": 0, "type": "preference_event", "subject": "user", "predicate": "likes",
   "objects": ["black coffee"], "relation": "supersession", "text": "User likes black coffee.",
   "date_observed": "2026-04-20", "valid_from": null, "valid_to": null, "confidence": 0.9, "attributed_to": "user",
   "speaker": null, "sentiment": "positive", "topic_tags": ["coffee", "preference"], "entity_mentions": [],
   "supersedes_hint_ids": ["0"]}]}

# OUTPUT FORMAT
```json
{
  "observations": [
    {"id": "0", "event_idx": 0,
     "type": "preference_event",
     "subject": "user", "predicate": "likes", "objects": ["oat-milk lattes in the morning"],
     "relation": "single_fact",
     "text": "User likes oat-milk lattes in the morning.",
     "date_observed": "2026-04-02", "valid_from": null, "valid_to": null,
     "confidence": 0.8, "attributed_to": "user", "speaker": null,
     "sentiment": "positive",
     "topic_tags": ["coffee", "preference"], "entity_mentions": [],
     "supersedes_hint_ids": []}
  ]
}
```
Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers."""

# Compact variant of observation_extraction (ingest.extraction_output='compact'): the LLM emits the nine
# COMPACT_OBSERVATION_KEYS only (eight fields plus the optional `entities` hint); subject / predicate / objects /
# relation / date_observed / valid_from / confidence / attributed_to / sentiment / topic_tags are derived locally
# (ingest.derive) and entity_mentions is the union of `entities`, the extractor's hits in `text` and `speaker`. Same
# user prompt builder and section titles as the full prompt, so parse_sections and the rule handlers read it unchanged.
# observation_extraction 的紧凑变体（ingest.extraction_output='compact'）：LLM 只输出 COMPACT_OBSERVATION_KEYS 九个键
# （八个字段加可选的 `entities` 提示）；subject / predicate / objects / relation / date_observed / valid_from /
# confidence / attributed_to / sentiment / topic_tags 在本地推导（ingest.derive），entity_mentions 是 `entities`、
# 抽取器在 `text` 中的命中与 `speaker` 的并集。与完整提示词共用同一个用户提示词构造器与分节标题，
# parse_sections 与规则处理器无需改动即可读取。
OBSERVATION_EXTRACTION_COMPACT_PROMPT = """HMS_TASK: observation_extraction_compact
# ROLE
You are the observation extractor of HMS Elite (compact output). You turn new conversation messages into short,
witnessed statements about the user (or the assistant / the project) worth remembering later. Emit ONLY the keys
listed under OUTPUT FORMAT; every other field is derived by the system.

# INPUTS
## Last k Messages
Plain `role: text` context lines (already stored; never re-extract them).
## Existing Memories
`[{"id": "0", "text"}]`; cite these ids in `supersedes_hint_ids` when a new statement replaces one.
## New Messages
`[{"idx", "role", "text", "timestamp"}]`; every `event_idx` MUST index this list.
## Observation Date
`YYYY-MM-DD` when the messages were written; resolve relative dates ONLY against it, never against Current Date.
## Current Date
`YYYY-MM-DD` when the system runs.
## Custom Instructions
Optional caller guidance (present only when given).

# GUIDELINES
- `text`: third person, ONE statement, at most 25 words; never state the same fact twice across items.
  Bad: "I like oat-milk lattes"
  Good: "User likes oat-milk lattes in the morning."
- `type` is exactly one of: fact | preference_event | habit | constraint | temporal_fact (preference_event =
  likes/dislikes, constraint = must / must not, habit = usually / every day, temporal_fact = a dated event, fact =
  everything else).
- `speaker`: the named person who said it in a multi-speaker transcript, else null. Assistant statements about its
  own actions are statements whose text starts with "Assistant".
- `entities`: the named things the statement mentions, 0–6 short strings; omit when none.
- `supersedes_hint_ids`: ONLY when a supersession cue is present (now, switched to, no longer, instead, actually)
  and an Existing Memories id is replaced; omit the key otherwise. Never invent ids.
- `when`: ONLY when the statement itself carries a date (ISO `YYYY-MM-DD`) or datetime (ISO with time) different
  from the message date, resolved against Observation Date; `valid_to` ONLY for an explicit expiry. Omit otherwise.
- Questions, greetings and assistant boilerplate yield nothing.

# EXAMPLES
Existing Memories: [{"id": "0", "text": "User likes oat-milk lattes in the morning."}]
New Messages: [{"idx": 0, "role": "user", "text": "Actually I switched to black coffee now.",
               "timestamp": "2026-04-20T08:00:00+00:00"}]
Observation Date: 2026-04-20
→ {"observations": [{"id": "0", "event_idx": 0, "text": "User likes black coffee.", "type": "preference_event",
   "speaker": null, "entities": ["black coffee"], "supersedes_hint_ids": ["0"]}]}

# OUTPUT FORMAT
```json
{
  "observations": [
    {"id": "0", "event_idx": 0, "text": "User likes black coffee.", "type": "preference_event", "speaker": null,
     "entities": ["black coffee"], "supersedes_hint_ids": ["1"], "when": "2026-04-18", "valid_to": null}
  ]
}
```
Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers."""

EP_EXTRACTION_PROMPT = """HMS_TASK: ep_extraction
# ROLE
You are the experience-package extractor of HMS Elite. From chat messages you build EPs: a trigger u, background b,
action a, outcome y, reward r and constraints κ — one EP per action → outcome closure.

# INPUTS
## Events
JSON list of `{"idx", "role", "object_type", "text", "timestamp"}`; every `event_idx` MUST index this list.
## Known Constraints
JSON list of `{"id": "c0", "text"}`; copy matching texts into `b` and `kappa.constraints`.
## Recent EPs
JSON list of `{"id": "e0", "u", "a", "y", "r"}`; `uses_ep_ids` ⊂ these ids.
## Observation Date
`YYYY-MM-DD` — the day the events happened; resolve relative dates ONLY against it.

# GUIDELINES
- Emit an EP only when a trigger AND an action AND (an outcome OR a constraint) are present.
  Bad: an EP for "I will look into it later" (no action, no outcome).
  Good: an EP for "I patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed."
- `a` keeps identifiers intact (retry.ts, billing_e2e); `y` is the shortest clause carrying the outcome word.
  Bad: a = "patched retry"
  Good: a = "patched retry.ts to add backoff", y = "billing_e2e failed"
- `r` is negative for failed/broken/regression outcomes, positive for passed/works outcomes, unknown when there is no
  outcome, neutral otherwise.
- Two-clock rule: `time_anchor` is the event's timestamp, never the current time.
- Never invent ids: uses_ep_ids ⊂ Recent EPs ids; event_idx ⊂ Events idx.

# EXAMPLES
Events: [{"idx": 0, "role": "assistant", "object_type": "message",
          "text": "I patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed.",
          "timestamp": "2026-06-15T10:04:00+00:00"}]
→ {"eps": [{"id": "0", "event_idx": 0, "scale": "L1_micro", "u": "the webhook kept timing out", "b": "",
   "a": "patched retry.ts to add backoff", "y": "billing_e2e failed", "r": "negative",
   "kappa": {"constraints": [], "visibility": "private"}, "time_anchor": "2026-06-15T10:04:00+00:00",
   "entity_mentions": ["retry.ts", "billing_e2e"], "topic_tags": ["engineering", "billing"], "uses_ep_ids": [],
   "confidence": 0.7}]}

# OUTPUT FORMAT
```json
{
  "eps": [
    {"id": "0", "event_idx": 1, "scale": "L1_micro",
     "u": "billing_e2e failed after the retry patch",
     "b": "Do not modify the subscription state machine.",
     "a": "patch retry.ts to add exponential backoff",
     "y": "billing_e2e failed: subscription_state_regression",
     "r": "negative",
     "kappa": {"constraints": ["Do not modify the subscription state machine."], "visibility": "team"},
     "time_anchor": "2026-06-15T10:04:00+00:00",
     "entity_mentions": ["retry.ts", "billing_e2e"], "topic_tags": ["engineering", "billing"],
     "uses_ep_ids": ["e1"], "confidence": 0.7}
  ]
}
```
Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers."""

REVISION_DECISION_PROMPT = """HMS_TASK: revision_decision
# ROLE
You are the revision judge of HMS Elite. For every new memory you decide whether it adds to, revises, conflicts with or
duplicates an existing memory.

# INPUTS
## New Memories
JSON list of `{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed"}`.
## Candidate Existing Memories
JSON list of `{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed",
"valid_from"}`.
## Current Date
`YYYY-MM-DD` — the system clock; ordering between memories uses their own `date_observed` (two-clock rule).

# GUIDELINES
- Exactly one decision per New Memories id; `target_id` is REQUIRED for REVISE / CONFLICT / NONE and MUST be a
  Candidate Existing Memories id; `null` for ADD.
- ADD = compatible or additive; REVISE = the new memory supersedes the target; CONFLICT = incompatible and neither is
  clearly newer or truer; NONE = duplicate of the target.
  Bad: REVISE for "User likes tea." against "User likes coffee." with no supersession cue.
  Good: REVISE for "User likes black coffee." (relation supersession, newer date) against "User likes oat-milk lattes."
- Different subjects or types never revise each other; temporal facts and habits are always ADD.
  Bad: NONE for two different dated events about Sarah.
  Good: ADD for each dated event.
- Exclusive attributes (lives_in, works_at, is_named, prefers) with disjoint objects: REVISE when the new memory is
  newer or same-dated, CONFLICT when it is older.

# EXAMPLES
New Memories: [{"id": "0", "type": "preference_event", "subject": "user", "predicate": "likes",
                "objects": ["black coffee"], "relation": "supersession", "text": "User likes black coffee.",
                "date_observed": "2026-04-20"}]
Candidate Existing Memories: [{"id": "2", "type": "preference_event", "subject": "user", "predicate": "likes",
                "objects": ["oat-milk lattes in the morning"], "relation": "single_fact",
                "text": "User likes oat-milk lattes in the morning.", "date_observed": "2026-04-02",
                "valid_from": "2026-04-02T08:00:00+00:00"}]
→ {"decisions": [{"new_id": "0", "decision": "REVISE", "target_id": "2", "reason": "supersession cue: 'switched to'"}]}

# OUTPUT FORMAT
```json
{"decisions": [{"new_id": "0", "decision": "REVISE", "target_id": "2", "reason": "supersession cue: 'switched to'"}]}
```
Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers."""

MEMORY_EXTRACTION_PROMPT = """HMS_TASK: memory_extraction
# ROLE
You are the memory extractor of HMS Elite. In ONE pass you (1) turn new conversation messages into typed, dated,
witnessed observations, (2) build experience packages (EPs: trigger u, background b, action a, outcome y, reward r,
constraints κ) from the action → outcome closures of the same messages, and (3) judge every new observation against
the candidate existing memories (ADD / REVISE / CONFLICT / NONE). One reply carries all three sections.

# INPUTS
## Last k Messages
Recent context as plain `role: text` lines (already stored; never re-extract them).
## Existing Memories
JSON list of `{"id": "0", "text": ...}`; cite these ids in `supersedes_hint_ids` when a new observation replaces one.
## New Messages
JSON list of `{"idx": 0, "role": "user", "text": ..., "timestamp": ...}`; every observation `event_idx` MUST index
this list.
## Observation Date
`YYYY-MM-DD` — when the messages were written; resolve relative dates ONLY against it.
## Current Date
`YYYY-MM-DD` — when the system runs; NEVER use it to resolve dates in the messages.
## Custom Instructions
Optional caller guidance (present only when given).
## Events
JSON list of `{"idx", "role", "object_type", "text", "timestamp"}` — the messages that may close an EP; every EP
`event_idx` MUST index this list. An empty list means `"eps": []`.
## Known Constraints
JSON list of `{"id": "c0", "text"}`; copy matching texts into `b` and `kappa.constraints`.
## Recent EPs
JSON list of `{"id": "e0", "u", "a", "y", "r"}`; `uses_ep_ids` ⊂ these ids.
## Candidate Existing Memories
JSON list of `{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed",
"valid_from"}` — the same ids as Existing Memories; every decision `target_id` MUST be one of them.

# GUIDELINES
- Two-clock rule: `date_observed` and `time_anchor` come from the messages (Observation Date / event timestamps),
  never from Current Date; ordering between memories uses their own `date_observed`.
  Bad: "yesterday" → the day before Current Date.
  Good: "yesterday" → the day before Observation Date.
- Observations: one per self-contained statement, third person, 5–80 words; type by intent (preference_event,
  constraint, habit, temporal_fact, fact); `predicate` is a lowercase snake verb; a supersession cue (now, switched
  to, no longer, instead, actually) sets relation supersession and cites the superseded memory; questions, greetings
  and assistant boilerplate yield nothing; assistant statements about its own actions are facts with subject
  "assistant" and attributed_to "assistant"; `attributed_to` is exactly "user" or "assistant" (the conversation
  side), never a person's name — the named person of a multi-speaker transcript goes in `speaker` (and in `subject`
  and `entity_mentions`); `speaker` is null in a plain user/assistant chat.
  Bad: "I like oat-milk lattes"
  Good: "User likes oat-milk lattes in the morning."
- EPs: emit one only when a trigger AND an action AND (an outcome OR a constraint) are present; `a` keeps identifiers
  intact, `y` is the shortest clause carrying the outcome word; `r` is negative for failed/broken outcomes, positive
  for passed/works outcomes, unknown without an outcome, neutral otherwise.
  Bad: an EP for "I will look into it later" (no action, no outcome).
  Good: a = "patched retry.ts to add backoff", y = "billing_e2e failed", r = "negative".
- Decisions: exactly one per observation `id`; `target_id` is REQUIRED for REVISE / CONFLICT / NONE and MUST be a
  Candidate Existing Memories id, `null` for ADD; ADD = compatible or additive; REVISE = the new observation
  supersedes the target; CONFLICT = incompatible and neither is clearly newer or truer; NONE = duplicate of the target.
  Different subjects or types never revise each other; temporal facts and habits are always ADD; exclusive attributes
  (lives_in, works_at, is_named, prefers) with disjoint objects: REVISE when newer or same-dated, CONFLICT when older.
  Bad: REVISE for "User likes tea." against "User likes coffee." with no supersession cue.
  Good: REVISE for "User likes black coffee." (relation supersession, newer date) against "User likes oat-milk lattes."
- Never invent ids: supersedes_hint_ids ⊂ Existing Memories ids; observation event_idx ⊂ New Messages idx; EP
  event_idx ⊂ Events idx; uses_ep_ids ⊂ Recent EPs ids; target_id ⊂ Candidate Existing Memories ids. With no
  candidates every decision is ADD.

# EXAMPLES
Existing Memories: [{"id": "0", "text": "User likes oat-milk lattes in the morning."}]
New Messages: [{"idx": 0, "role": "user", "text": "Actually I switched to black coffee now.",
               "timestamp": "2026-04-20T08:00:00+00:00"}]
Events: []
Candidate Existing Memories: [{"id": "0", "type": "preference_event", "subject": "user", "predicate": "likes",
                "objects": ["oat-milk lattes in the morning"], "relation": "single_fact",
                "text": "User likes oat-milk lattes in the morning.", "date_observed": "2026-04-02",
                "valid_from": "2026-04-02T08:00:00+00:00"}]
Observation Date: 2026-04-20
→ {"observations": [{"id": "0", "event_idx": 0, "type": "preference_event", "subject": "user", "predicate": "likes",
   "objects": ["black coffee"], "relation": "supersession", "text": "User likes black coffee.",
   "date_observed": "2026-04-20", "valid_from": null, "valid_to": null, "confidence": 0.9, "attributed_to": "user",
   "speaker": null, "sentiment": "positive", "topic_tags": ["coffee", "preference"], "entity_mentions": [],
   "supersedes_hint_ids": ["0"]}],
   "eps": [],
   "decisions": [{"id": "0", "decision": "REVISE", "target_id": "0", "reason": "supersession cue: 'switched to'"}]}

# OUTPUT FORMAT
```json
{
  "observations": [
    {"id": "0", "event_idx": 0,
     "type": "preference_event",
     "subject": "user", "predicate": "likes", "objects": ["black coffee"],
     "relation": "supersession",
     "text": "User likes black coffee.",
     "date_observed": "2026-04-20", "valid_from": null, "valid_to": null,
     "confidence": 0.9, "attributed_to": "user", "speaker": null,
     "sentiment": "positive",
     "topic_tags": ["coffee", "preference"], "entity_mentions": [],
     "supersedes_hint_ids": ["0"]}
  ],
  "eps": [
    {"id": "0", "event_idx": 0, "scale": "L1_micro",
     "u": "the webhook kept timing out", "b": "",
     "a": "patched retry.ts to add backoff", "y": "billing_e2e failed", "r": "negative",
     "kappa": {"constraints": [], "visibility": "private"},
     "time_anchor": "2026-06-15T10:04:00+00:00",
     "entity_mentions": ["retry.ts", "billing_e2e"], "topic_tags": ["engineering", "billing"],
     "uses_ep_ids": [], "confidence": 0.7}
  ],
  "decisions": [
    {"id": "0", "decision": "REVISE", "target_id": "0", "reason": "supersession cue: 'switched to'"}
  ]
}
```
Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers."""

# Compact variant of memory_extraction (ingest.extraction_output='compact'): observations carry the nine
# COMPACT_OBSERVATION_KEYS only (eight fields plus the optional `entities` hint); EPs and decisions keep their full
# shapes and are emitted only when their triggers are present (an action → outcome closure; a non-empty candidate
# list). The DECISIONS rule is stated per NEW observation, never per candidate or by position, and NONE is reserved
# for a restatement (same fact, same subject, same value): under the earlier wording deepseek-flash paired
# observation i with candidate i and called it a duplicate (1,685 NONE on the 204-question EgoLife run against 23
# under the full prompt; benchmarks/egolife/results/replay_diag/summary.md). The example shows two candidates and
# a target id that differs from the observation id for the same reason. Same builder as the full prompt.
# memory_extraction 的紧凑变体（ingest.extraction_output='compact'）：观察只带 COMPACT_OBSERVATION_KEYS 九个键
# （八个字段加可选的 `entities` 提示）；EP 与决策保持完整形状，且只在触发条件出现时输出（行动 → 结果闭包；
# 非空候选列表）。DECISIONS 规则按"每条新观察一个决策"陈述，绝不按候选、绝不按位置配对，NONE 只留给复述
# （同一事实、同一主语、同一取值）：旧措辞下 deepseek-flash 把观察 i 与候选 i 配对并称其重复（204 题 EgoLife 运行中
# 1,685 个 NONE，完整提示词下为 23；benchmarks/egolife/results/replay_diag/summary.md）。示例出于同一原因给出两个
# 候选、且目标 id 与观察 id 不同。与完整提示词共用构造器。
MEMORY_EXTRACTION_COMPACT_PROMPT = """HMS_TASK: memory_extraction_compact
# ROLE
You are the memory extractor of HMS Elite (compact output). In ONE pass: (1) turn new messages into short,
witnessed statements, (2) build EPs (trigger u, background b, action a, outcome y, reward r, constraints κ) ONLY
from action → outcome closures, (3) judge each new statement against the candidates when shown. Emit ONLY the
OUTPUT FORMAT keys; the rest is derived.

# INPUTS
## Last k Messages
`role: text` lines already stored; never re-extract them.
## Existing Memories
`[{"id": "0", "text"}]`; cite an id in `supersedes_hint_ids` when a new statement replaces it.
## New Messages
`[{"idx", "role", "text", "timestamp"}]`; every observation `event_idx` MUST index this list.
## Observation Date
`YYYY-MM-DD` of the messages; resolve relative dates ONLY against it, never against Current Date.
## Current Date
`YYYY-MM-DD` of the run.
## Custom Instructions
Optional guidance.
## Events
`[{"idx", "role", "object_type", "text", "timestamp"}]`; every EP `event_idx` MUST index this list. Empty →
`"eps": []`.
## Known Constraints
`[{"id": "c0", "text"}]`; copy matching texts into `b` and `kappa.constraints`.
## Recent EPs
`[{"id": "e0", "u", "a", "y", "r"}]`; `uses_ep_ids` ⊂ these ids.
## Candidate Existing Memories
Existing Memories with fields; every decision `target_id` MUST be one of their ids, which are unrelated to
observation ids. Empty → `"decisions": []`.

# GUIDELINES
- Observations: `text` is third person, ONE statement, at most 25 words; never state the same fact twice across
  items. `type` is exactly one of: fact | preference_event | habit | constraint | temporal_fact (preference_event =
  likes/dislikes, constraint = must/must not, habit = usually/every day, temporal_fact = a dated event, else fact).
  `speaker` = the named person who said it in a multi-speaker transcript, else null. Assistant statements about
  its own actions start with "Assistant".
  Bad: "I like oat-milk lattes"
  Good: "User likes oat-milk lattes."
- `entities`: the named things the statement mentions, 0–6 short strings; omit when none.
- `supersedes_hint_ids`: ONLY with a supersession cue (now, switched to, no longer, instead, actually) replacing an
  Existing Memories id; omit otherwise. Never invent ids.
- `when`: ONLY an ISO date/datetime the statement itself carries, differing from the message date, resolved
  against Observation Date; `valid_to` ONLY for an explicit expiry. Omit otherwise.
- Questions, greetings and assistant boilerplate yield nothing.
- EPs: emit one ONLY when a trigger AND an action AND (an outcome OR a constraint) are present; `a` keeps
  identifiers intact, `y` = the shortest clause with the outcome word; `r`: negative for failed/broken, positive
  for passed/works, unknown without an outcome, else neutral.
- Decisions: ONLY when Candidate Existing Memories is non-empty; exactly one per NEW observation `id`, never one
  per candidate, never by position (most observations match no candidate). `target_id` MUST be a candidate id for REVISE / CONFLICT / NONE, `null` for ADD. NONE ONLY when a candidate states the SAME
  fact about the SAME subject with the SAME value (a restatement); a new detail, person, time or a different value
  is ADD. REVISE = the new statement supersedes the target; CONFLICT = incompatible, neither clearly newer or truer.
  Different subjects or types never revise each other; temporal facts and habits are always ADD.
  Bad: NONE for "Jake sets the table." against "Jake brought plates to the kitchen." (similar words, new fact).
  Good: ADD for it; NONE only against "Jake set the dinner table."

# EXAMPLES
Existing Memories: [{"id": "0", "text": "User works at Acme."}, {"id": "1", "text": "User likes oat-milk lattes."}]
New Messages: [{"idx": 0, "role": "user", "text": "Actually I switched to black coffee now.", "timestamp": null}]
Events: []
Candidate Existing Memories: [{"id": "0", "text": "User works at Acme."}, {"id": "1", "text": "User likes oat-milk lattes."}]
→ {"observations": [{"id": "0", "event_idx": 0, "text": "User likes black coffee.", "type": "preference_event",
   "speaker": null, "entities": ["black coffee"], "supersedes_hint_ids": ["1"]}],
   "eps": [],
   "decisions": [{"id": "0", "decision": "REVISE", "target_id": "1", "reason": "cue 'switched to'"}]}

# OUTPUT FORMAT
```json
{
  "observations": [
    {"id": "0", "event_idx": 0, "text": "User likes black coffee.", "type": "preference_event", "speaker": null,
     "entities": ["black coffee"], "supersedes_hint_ids": ["1"], "when": "2026-04-18", "valid_to": null}
  ],
  "eps": [
    {"id": "0", "event_idx": 0, "scale": "L1_micro", "u": "webhook timed out", "b": "",
     "a": "patched retry.ts to add backoff", "y": "billing_e2e failed", "r": "negative",
     "kappa": {"constraints": [], "visibility": "private"}, "time_anchor": "2026-06-15T10:04:00+00:00",
     "entity_mentions": ["retry.ts"], "topic_tags": ["billing"], "uses_ep_ids": [], "confidence": 0.7}
  ],
  "decisions": [
    {"id": "0", "decision": "REVISE", "target_id": "1", "reason": "cue 'switched to'"}
  ]
}
```
Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers."""

# Lines variant of memory_extraction (ingest.extraction_output='lines'): the reply is plain tab-separated text, never
# JSON. Output tokens are 83% of the full contract's write cost on the 4,851-window EgoLife run (2.40M output tokens
# of ¥11.5; compact v2 0.76M of ¥4.2; mem0 0.22M of ¥1.85 — the input-miss side is already at parity) and a JSON
# envelope costs 2–4× the text it carries, while the zero-API replay showed retrieval uses only the statement texts
# and the decisions (entities / type / when move hit@20 by 0). The compact prompt's own texts scored below the full
# prompt's (rows 56.4% vs 61.3%), so this prompt keeps the FULL prompt's statement rules verbatim (third person,
# 5–80 words, one self-contained statement, quantities / dates / times kept, supersession cues, questions yield
# nothing, assistant statements) and the rewritten compact DECISIONS rules (one decision per NEW observation, only
# for non-ADD; NONE only for a same-subject same-value restatement; temporal facts and habits always ADD), and only
# replaces the OUTPUT FORMAT. Every other field is derived locally exactly as in compact (ingest.derive); the
# parser is ingest.lines.parse_lines. Same builder as the full prompt. Single extraction mode only.
# memory_extraction 的行式变体（ingest.extraction_output='lines'）：回复是纯 tab 分隔文本，绝不是 JSON。在 4,851 窗口的
# EgoLife 运行上，输出 token 占完整契约写入成本的 83%（¥11.5 中输出 2.40M token；compact v2 ¥4.2 中 0.76M；mem0
# ¥1.85 中 0.22M——输入未命中侧已经持平），而 JSON 信封的开销是其承载文本的 2–4 倍；零 API 回放表明检索只用到陈述
# 文本与决策（entities / type / when 对 hit@20 的影响为 0）。紧凑提示词自己写出的文本低于完整提示词（记忆行 56.4% 对
# 61.3%），因此本提示词逐字保留完整提示词的陈述规则（第三人称、5–80 词、一条自足陈述、保留数量 / 日期 / 时刻、取代
# 线索、问句不产出、助手陈述）与重写后的紧凑 DECISIONS 规则（每条新观察一个决策、只写非 ADD；NONE 只留给同主语同取值
# 的复述；时态事实与习惯恒为 ADD），只替换 OUTPUT FORMAT。其余字段与 compact 完全一样在本地推导（ingest.derive）；
# 解析器是 ingest.lines.parse_lines。与完整提示词共用构造器。仅支持 single 抽取模式。
MEMORY_EXTRACTION_LINES_PROMPT = """HMS_TASK: memory_extraction_lines
# ROLE
You are the memory extractor of HMS Elite. In ONE pass you (1) turn new conversation messages into typed, dated,
witnessed observations, (2) build experience packages (EPs: trigger u, action a, outcome y, reward r, constraints κ)
from the action → outcome closures of the same messages, and (3) judge every new observation against the candidate
existing memories (ADD / REVISE / CONFLICT / NONE). One reply carries all three sections as plain tab-separated
lines — never JSON.

# INPUTS
## Last k Messages
Recent context as plain `role: text` lines (already stored; never re-extract them).
## Existing Memories
JSON list of `{"id": "0", "text": ...}` — what is already known; a new observation with a supersession cue replaces
one of them (see GUIDELINES and DEC).
## New Messages
JSON list of `{"idx": 0, "role": "user", "text": ..., "timestamp": ...}`; every OBS line's event_idx MUST index
this list.
## Observation Date
`YYYY-MM-DD` — when the messages were written; resolve relative dates ONLY against it.
## Current Date
`YYYY-MM-DD` — when the system runs; NEVER use it to resolve dates in the messages.
## Custom Instructions
Optional caller guidance (present only when given).
## Events
JSON list of `{"idx", "role", "object_type", "text", "timestamp"}` — the messages that may close an EP; every EP
line's event_idx MUST index this list. An empty list means no EP section.
## Known Constraints
JSON list of `{"id": "c0", "text"}`; copy matching texts into `k=`.
## Recent EPs
JSON list of `{"id": "e0", "u", "a", "y", "r"}` — context only.
## Candidate Existing Memories
JSON list of `{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed",
"valid_from"}` — the same ids as Existing Memories, unrelated to OBS line numbers; every DEC line's candidate id
MUST be one of them. An empty list means no DEC section.

# GUIDELINES
- Two-clock rule: every date in a statement or after `@` comes from the messages (Observation Date / event
  timestamps), never from Current Date; ordering between memories uses their own dates.
  Bad: "yesterday" → the day before Current Date.
  Good: "yesterday" → the day before Observation Date.
- Observations: one per self-contained statement, third person, 5–80 words; type by intent (preference_event,
  constraint, habit, temporal_fact, fact); keep the specific quantities, dates, times and names the message gives;
  a supersession cue (now, switched
  to, no longer, instead, actually) means the statement replaces an existing memory — keep the cue's meaning in the
  statement and judge it in DEC; questions, greetings and assistant boilerplate yield nothing; assistant statements
  about its own actions are facts that start with "Assistant"; the named person of a multi-speaker transcript is
  the subject of the statement ("Katrina must not modify the billing state machine."). Write every statement in
  the language of the source messages (Chinese messages → Chinese statements, names as written); never translate.
  Bad: "I like oat-milk lattes"
  Good: "User likes oat-milk lattes in the morning."
- `@date`: append it ONLY when the statement itself carries a date or datetime different from the message date,
  resolved against Observation Date (`@2026-06-20` or `@2026-06-20T09:30`); omit otherwise.
- EPs: emit one only when a trigger AND an action AND (an outcome OR a constraint) are present; `a` keeps identifiers
  intact, `y` is the shortest clause carrying the outcome word; `r` is negative for failed/broken outcomes, positive
  for passed/works outcomes, unknown otherwise.
  Bad: an EP for "I will look into it later" (no action, no outcome).
  Good: a=patched retry.ts to add backoff, y=billing_e2e failed, r=negative.
- Decisions: ONLY when Candidate Existing Memories is non-empty; at most one DEC line per NEW observation, written
  ONLY for a non-ADD verdict and naming the observation by its OBS line number (1 = the first OBS line); never one
  per candidate, never by position (most observations match no candidate). REVISE = the new observation supersedes
  the target; CONFLICT = incompatible and neither is clearly newer or truer; NONE ONLY when a candidate states the
  SAME fact about the SAME subject with the SAME value (a restatement); a new detail, person, time or a different
  value is ADD (no line). Different subjects or types never revise each other; temporal facts and habits are always
  ADD; exclusive attributes (lives_in, works_at, is_named, prefers) with disjoint objects: REVISE when newer or
  same-dated, CONFLICT when older.
  Bad: REVISE for "User likes tea." against "User likes coffee." with no supersession cue.
  Bad: NONE for "Jake sets the table." against "Jake brought plates to the kitchen." (similar words, new fact).
  Good: REVISE for "User likes black coffee." (supersession cue, newer date) against "User likes oat-milk lattes."
- Never invent indices or ids: OBS event_idx ⊂ New Messages idx; EP event_idx ⊂ Events idx; DEC candidate id ⊂
  Candidate Existing Memories ids.

# EXAMPLES
Existing Memories: [{"id": "0", "text": "User works at Acme."}, {"id": "1", "text": "User likes oat-milk lattes in the morning."}]
New Messages: [{"idx": 0, "role": "user", "text": "Actually I switched to black coffee now. Sarah's dentist visit is on June 20 at 9:30.",
               "timestamp": "2026-04-20T08:00:00+00:00"}]
Events: []
Candidate Existing Memories: [{"id": "0", "type": "fact", "text": "User works at Acme.", ...},
                {"id": "1", "type": "preference_event", "subject": "user", "predicate": "likes",
                "objects": ["oat-milk lattes in the morning"], "text": "User likes oat-milk lattes in the morning.", ...}]
Observation Date: 2026-04-20
→
OBS
0	P	User switched to black coffee, no longer taking oat-milk lattes in the morning.
0	T	Sarah has a dentist visit on June 20 at 9:30.	@2026-06-20T09:30
DEC
1	REVISE	1	supersession cue: 'switched to'

# OUTPUT FORMAT
Plain text. One item per line; fields separated by ONE tab character; a statement never contains a tab. No JSON,
no quotes, no bullets, no numbering, no commentary. Sections in this order, each opened by its header on its own
line:
OBS
<event_idx>	<T>	<statement>	@<YYYY-MM-DD or ISO datetime>
  one line per observation; T is one letter: F fact, P preference_event, H habit, C constraint, T temporal_fact;
  the trailing tab + @date is optional (see GUIDELINES).
EP
<event_idx>	u=<trigger>	a=<action>	y=<outcome>	r=positive|negative|unknown	k=<constraint>;<constraint>
  section present ONLY when an action → outcome closure exists; the trailing tab + k= is optional.
DEC
<OBS line number, 1-based>	REVISE|CONFLICT|NONE	<candidate id>	<short reason>
  section present ONLY for non-ADD verdicts; an absent section means every observation is ADD.
Return ONLY these lines. No JSON, no reasoning, no explanations, no wrappers."""

EDGE_PROPOSAL_PROMPT = """HMS_TASK: edge_proposal
# ROLE
You are the edge proposer of HMS Elite. Given candidate node pairs you propose typed, witnessed Support Graph edges.

# INPUTS
## Nodes
JSON list of `{"ref": "n0", "kind": "ep"|"interface_ep"|"observation"|"event"|"skill", "u", "b", "a", "y", "r",
"subject", "predicate", "objects": [...], "time", "entities": [...], "topics": [...], "anchor_refs": ["w0"]}` —
EP / InterfaceEP nodes carry u, b, a, y, r (their slots) with subject / predicate null and objects [];
observation / event / skill nodes carry subject / predicate / objects with u, b, y = "", a = the atom text and
r = "unknown". Every key is always present.
## Candidate Pairs
JSON list of `[["n0", "n1"], ...]`; only these pairs may receive an edge.
## Witnesses
JSON list of `{"ref": "w0", "node_ref": "n0", "source_ref", "time"}`; `witness_refs` ⊂ these refs and must belong to
the src or dst node.
## Allowed Edge Types
Plain lines `type: definition` (six types).

# GUIDELINES
- Time order: the earlier node is src for temporal / causal / use / distill edges.
- causal needs evidence: the earlier outcome (y) feeds the later trigger (u), an identifier of the earlier action
  appears in the later trigger, or a negative earlier outcome shares a topic with the later node.
  Bad: causal between two unrelated EPs that merely happened in sequence.
  Good: causal when "billing_e2e failed" (y of n0) is the trigger (u) of n1.
- use: the later node consumes the earlier one (an identifier of the earlier action/outcome in the later action or
  background); revision: same target, negative then positive; conflict: same (subject, predicate) with disjoint
  objects.
- Two-clock rule: node `time` values are event times; never reason with the current time.
- witness_refs are REQUIRED and non-empty; no self edges; no pairs outside Candidate Pairs; `{"edges": []}` when
  nothing is justified.

# EXAMPLES
Nodes: n0 = EP "patch retry.ts to add exponential backoff" → "billing_e2e failed: subscription_state_regression"
(r negative, time 2026-06-15T10:04:00+00:00, anchor_refs ["w0", "w1"]); n1 = EP whose u is
"billing_e2e failed: subscription_state_regression" (time 2026-06-15T10:14:00+00:00, anchor_refs ["w2", "w3"]).
Candidate Pairs: [["n0", "n1"]]
→ {"edges": [{"src_ref": "n0", "dst_ref": "n1", "edge_type": "causal",
              "label": "retry patch caused the billing regression", "witness_refs": ["w2"], "confidence": 0.8}]}

# OUTPUT FORMAT
```json
{"edges": [{"src_ref": "n0", "dst_ref": "n1", "edge_type": "causal",
            "label": "retry patch caused the billing regression",
            "witness_refs": ["w1"], "confidence": 0.8}]}
```
Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers."""

COARSEN_PROMPT = """HMS_TASK: coarsen
# ROLE
You are the coarsening operator of HMS Elite. From a support set of low-level nodes you build ONE InterfaceEP: an
interface to that subgraph (slots citing support nodes, invariants, residuals) — never a summary that loses evidence.

# INPUTS
## Support Nodes
JSON list of `{"ref": "n0", "kind", "u", "b", "a", "y", "r", "constraints": [...], "time", "anchor_refs"}` in time
order.
## Internal Edges
JSON list of `{"ref": "x0", "src": "n0", "dst": "n1", "type"}` — edges inside the support set.
## Boundary Edges
JSON list of `{"ref": "b0", "inner": "n1", "outer": "outside", "direction": "out", "type"}` — edges crossing the
boundary.
## Target Scale
`L2_meso` (from EPs) or `L3_macro` (from InterfaceEPs).

# GUIDELINES
- Every non-empty slot MUST cite ≥1 Support Nodes ref in `slot_support`; do not invent nodes.
  Bad: slot y = "billing_e2e passed" with slot_support y = [].
  Good: slot_support y = ["n1"] where n1 carries that outcome.
- u = the first node's trigger; a = the actions joined by " → "; y = the last non-empty outcome; r = negative when any
  member is negative and the last is not positive, else the last member's r; b and kappa.constraints = the union of
  constraints with each text once.
- invariants: a constraint present in ≥ 2 nodes (kind constraint); every negative node yields "do not: <a>" (kind
  failed_path); residuals: unknown outcomes (missing_outcome), conflict edges (unresolved_conflict), questions left
  open (open_question).
- Two-clock rule: node times are event times; never reason with the current time.

# EXAMPLES
Support Nodes: n0 (negative: patch retry.ts → billing_e2e failed, constraints ["Do not modify the subscription state
machine."]), n1 (positive: patch idempotency.ts → billing_e2e passed, same constraint).
→ {"interface_ep": {"slots": {"u": "fix checkout webhook retry", "b": "Do not modify the subscription state machine.",
   "a": "patch retry.ts: add exponential backoff → patch idempotency.ts: add idempotency key",
   "y": "billing_e2e passed", "r": "positive",
   "kappa": {"constraints": ["Do not modify the subscription state machine."], "visibility": "private"}},
   "slot_support": {"u": ["n0"], "b": ["n0", "n1"], "a": ["n0", "n1"], "y": ["n1"], "r": ["n1"], "kappa": ["n0", "n1"]},
   "invariants": [{"text": "Do not modify the subscription state machine.", "kind": "constraint",
                   "support_refs": ["n0", "n1"]},
                  {"text": "do not: patch retry.ts: add exponential backoff", "kind": "failed_path",
                   "support_refs": ["n0"]}],
   "residuals": []}}

# OUTPUT FORMAT
```json
{"interface_ep": {
   "slots": {"u": "...", "b": "...", "a": "...", "y": "...", "r": "positive",
             "kappa": {"constraints": ["..."], "visibility": "private"}},
   "slot_support": {"u": ["n0"], "b": ["n0", "n1"], "a": ["n0", "n1"], "y": ["n1"], "r": ["n1"], "kappa": ["n0", "n1"]},
   "invariants": [{"text": "Do not modify the subscription state machine.", "kind": "constraint", "support_refs": ["n0", "n1"]}],
   "residuals": [{"text": "outcome of n2 unknown", "kind": "missing_outcome", "support_refs": ["n2"]}]}}
```
Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers."""

PROCEDURAL_SUMMARY_PROMPT = (
    """HMS_TASK: procedural_summary
# ROLE
You are a memory summarization system that records and preserves the complete interaction history between a human
and an AI agent. You are provided with the agent's execution history over the past N steps. Your task is to produce a
comprehensive summary of the agent's output history that contains every detail necessary for the agent to continue
the task without ambiguity.

# INPUTS
## Conversation
The full conversation (system prompt, then every message in order, then the instruction to create procedural memory).

# GUIDELINES
- Preserve every detail: actions, results, findings, errors, current context; never compress steps together.
  Bad: "The agent fixed some tests."
  Good: "1. **Agent Action**: patched retry.ts to add backoff **Action Result**: billing_e2e failed (3 failures)."
- Two-clock rule: report times as they appear in the conversation, never the current time.
- Progress Status is "completed" when the last outcome is positive, "blocked" when negative, else "in progress".
- Keep the markdown structure below exactly; unknown fields say "n/a".

# EXAMPLES
## Summary of the agent's execution history

**Task Objective**: fix checkout webhook retry
**Progress Status**: blocked

1. **Agent Action**: patched retry.ts to add backoff
   **Action Result**: billing_e2e failed: subscription_state_regression

**Key Findings**: n/a
**Errors & Challenges**: billing_e2e failed after the retry patch
**Current Context**: the retry patch is reverted; idempotency key pending

# OUTPUT FORMAT
"""
    + PROCEDURAL_SKELETON
    + "\n\n"
    + MARKDOWN_TRAILER
)

PROMPTS: dict[str, str] = {
    "observation_extraction": OBSERVATION_EXTRACTION_PROMPT,
    "observation_extraction_compact": OBSERVATION_EXTRACTION_COMPACT_PROMPT,
    "ep_extraction": EP_EXTRACTION_PROMPT,
    "revision_decision": REVISION_DECISION_PROMPT,
    "edge_proposal": EDGE_PROPOSAL_PROMPT,
    "coarsen": COARSEN_PROMPT,
    "procedural_summary": PROCEDURAL_SUMMARY_PROMPT,
}
ROOT_KEY: dict[str, str] = {
    "observation_extraction": "observations",
    "observation_extraction_compact": "observations",
    "ep_extraction": "eps",
    "revision_decision": "decisions",
    "edge_proposal": "edges",
    "coarsen": "interface_ep",
}
# Composite tasks reply with a multi-root object, so they stay OUT of ROOT_KEY: generate_json() then parses the reply
# with key=None and the caller splits the roots itself (MEMORY_EXTRACTION_ROOTS).
# 复合任务回复多根对象，因此不进入 ROOT_KEY：generate_json() 以 key=None 解析回复，由调用方自行拆分各根
# （MEMORY_EXTRACTION_ROOTS）。
COMPOSITE_TASKS = ("memory_extraction", "memory_extraction_compact", "memory_extraction_lines")
COMPOSITE_PROMPTS: dict[str, str] = {
    "memory_extraction": MEMORY_EXTRACTION_PROMPT,
    "memory_extraction_compact": MEMORY_EXTRACTION_COMPACT_PROMPT,
    "memory_extraction_lines": MEMORY_EXTRACTION_LINES_PROMPT,
}
MEMORY_EXTRACTION_ROOTS = ("observations", "eps", "decisions")


# ---------------------------------------------------------------------------
# Private helpers
# 私有辅助函数
# ---------------------------------------------------------------------------
def _dump(obj: Any) -> str:
    """Fenced-JSON serialization for structured sections.
    结构化分节的围栏 JSON 序列化。
    """
    return json.dumps(obj, ensure_ascii=False, indent=None, sort_keys=True)


def _json_section(title: str, obj: Any) -> str:
    """Render one `## title` section with a fenced JSON body.
    渲染一个带围栏 JSON 正文的 `## title` 分节。
    """
    return f"## {title}\n```json\n{_dump(obj)}\n```\n"


def _text_section(title: str, text: str) -> str:
    """Render one `## title` section with a plain body.
    渲染一个带纯文本正文的 `## title` 分节。
    """
    return f"## {title}\n{text}\n"


def _text_of(content: Any) -> str | None:
    """Reduce a message content (str or parts list) to text; None when absent.
    将消息内容（字符串或分片列表）归约为文本；缺失时为 None。
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def _message_line(message: dict) -> str:
    """One `role: text` line truncated to PAST_MESSAGE_TRUNCATION_LIMIT chars.
    截断到 PAST_MESSAGE_TRUNCATION_LIMIT 字符的一行 `role: text`。
    """
    text = _text_of(message.get("text", message.get("content"))) or ""
    text = text.replace("\n", " ")[:PAST_MESSAGE_TRUNCATION_LIMIT]
    return f"{message.get('role', 'user')}: {text}"


def _parse_body(body: str) -> Any:
    """Parse a section body: fenced JSON → object, else stripped text.
    解析分节正文：围栏 JSON → 对象，否则为去空白文本。
    """
    if body.startswith("```json") and body.endswith("```"):
        inner = body[len("```json") : -3].strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            return body
    return body


# ---------------------------------------------------------------------------
# Public functions (CONTRACT §6.2)
# 公开函数（CONTRACT §6.2）
# ---------------------------------------------------------------------------
def task_of(system_prompt: str) -> str | None:
    """Read the `HMS_TASK:` marker from the first line of a system prompt.
    读取系统提示词首行的 `HMS_TASK:` 标记。

    Input / 输入：
        system_prompt: A system prompt string.
        system_prompt：系统提示词字符串。

    Output / 输出：
        Task name, or None when the first line carries no marker.
        任务名；首行无标记时为 None。

    State effect / 状态影响：
        None.
        无。
    """
    if not isinstance(system_prompt, str):
        return None
    first = system_prompt.lstrip().split("\n", 1)[0].strip()
    if first.startswith("HMS_TASK:"):
        name = first[len("HMS_TASK:") :].strip()
        return name or None
    return None


def output_format_section(system_prompt: str) -> str:
    """Return the text after the last `# OUTPUT FORMAT` heading of a system prompt.
    返回系统提示词中最后一个 `# OUTPUT FORMAT` 标题之后的文本。

    Input / 输入：
        system_prompt: A `<TASK>_PROMPT` constant.
        system_prompt：`<TASK>_PROMPT` 常量。

    Output / 输出：
        The OUTPUT FORMAT block (fenced JSON + trailer, or the markdown skeleton); '' when absent.
        OUTPUT FORMAT 块（围栏 JSON + 尾句，或 markdown 骨架）；不存在时为 ''。

    State effect / 状态影响：
        None.
        无。
    """
    marker = "# OUTPUT FORMAT"
    index = system_prompt.rfind(marker)
    return "" if index == -1 else system_prompt[index + len(marker) :].strip()


def parse_sections(user_prompt: str) -> dict[str, Any]:
    """Split a built user prompt into {section_title: parsed_json | text}; the `# Output:` trailer is dropped.
    将已构造的用户提示词拆为 {分节标题: 解析后的 JSON | 文本}；丢弃 `# Output:` 尾部。

    Input / 输入：
        user_prompt: Output of a build_* function.
        user_prompt：build_* 函数的输出。

    Output / 输出：
        Dict keyed by `## ` titles; fenced JSON bodies parsed, plain bodies stripped.
        以 `## ` 标题为键的字典；围栏 JSON 正文已解析，纯文本正文已去空白。

    State effect / 状态影响：
        None.
        无。
    """
    sections: dict[str, Any] = {}
    title: str | None = None
    buffer: list[str] = []

    def _flush() -> None:
        if title is not None:
            sections[title] = _parse_body("\n".join(buffer).strip())

    for line in user_prompt.split("\n"):
        if line.startswith("## "):
            _flush()
            title = line[3:].strip()
            buffer = []
        elif line.strip() == "# Output:":
            break
        elif title is not None:
            buffer.append(line)
    _flush()
    return sections


def build_observation_prompt(
    *,
    existing_memories: list[dict],
    new_messages: list[dict],
    last_messages: list[dict],
    observation_date: str,
    current_date: str,
    custom_instructions: str | None = None,
) -> str:
    """Build the observation_extraction user prompt.
    构造 observation_extraction 的用户提示词。

    Input / 输入：
        existing_memories: `[{"id": "0", "text": ...}]` with synthetic ids.
        existing_memories：带合成 id 的 `[{"id": "0", "text": ...}]`。

        new_messages: `[{"idx": 0, "role": ..., "text": ..., "timestamp": ...}]`.
        new_messages：`[{"idx": 0, "role": ..., "text": ..., "timestamp": ...}]`。

        last_messages: Recent context messages rendered as `role: text` lines (300-char truncation).
        last_messages：渲染为 `role: text` 行的近期上下文消息（截断到 300 字符）。

        observation_date: `YYYY-MM-DD` of the messages.
        observation_date：消息的 `YYYY-MM-DD`。

        current_date: `YYYY-MM-DD` of the system clock.
        current_date：系统时钟的 `YYYY-MM-DD`。

        custom_instructions: Optional caller guidance; section emitted only when given.
        custom_instructions：可选的调用方指导；给定时才输出该分节。

    Output / 输出：
        User prompt ending with `# Output:`.
        以 `# Output:` 结尾的用户提示词。

    State effect / 状态影响：
        None.
        无。
    """
    parts = [
        _text_section("Last k Messages", "\n".join(_message_line(m) for m in last_messages)),
        _json_section("Existing Memories", existing_memories),
        _json_section("New Messages", new_messages),
        _text_section("Observation Date", observation_date),
        _text_section("Current Date", current_date),
    ]
    if custom_instructions:
        parts.append(_text_section("Custom Instructions", custom_instructions))
    return "\n".join(parts) + "\n# Output:"


def build_ep_prompt(
    *, events: list[dict], known_constraints: list[dict], recent_eps: list[dict], observation_date: str
) -> str:
    """Build the ep_extraction user prompt.
    构造 ep_extraction 的用户提示词。

    Input / 输入：
        events: `[{"idx", "role", "object_type", "text", "timestamp"}]`.
        events：`[{"idx", "role", "object_type", "text", "timestamp"}]`。

        known_constraints: `[{"id": "c0", "text"}]`.
        known_constraints：`[{"id": "c0", "text"}]`。

        recent_eps: `[{"id": "e0", "u", "a", "y", "r"}]`.
        recent_eps：`[{"id": "e0", "u", "a", "y", "r"}]`。

        observation_date: `YYYY-MM-DD`.
        observation_date：`YYYY-MM-DD`。

    Output / 输出：
        User prompt ending with `# Output:`.
        以 `# Output:` 结尾的用户提示词。

    State effect / 状态影响：
        None.
        无。
    """
    parts = [
        _json_section("Events", events),
        _json_section("Known Constraints", known_constraints),
        _json_section("Recent EPs", recent_eps),
        _text_section("Observation Date", observation_date),
    ]
    return "\n".join(parts) + "\n# Output:"


def build_revision_prompt(*, new_items: list[dict], candidates: list[dict], current_date: str) -> str:
    """Build the revision_decision user prompt.
    构造 revision_decision 的用户提示词。

    Input / 输入：
        new_items: `[{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed"}]`.
        new_items：`[{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed"}]`。

        candidates: Same keys plus `valid_from`.
        candidates：同样的键外加 `valid_from`。

        current_date: `YYYY-MM-DD`.
        current_date：`YYYY-MM-DD`。

    Output / 输出：
        User prompt ending with `# Output:`.
        以 `# Output:` 结尾的用户提示词。

    State effect / 状态影响：
        None.
        无。
    """
    parts = [
        _json_section("New Memories", new_items),
        _json_section("Candidate Existing Memories", candidates),
        _text_section("Current Date", current_date),
    ]
    return "\n".join(parts) + "\n# Output:"


def build_memory_extraction_prompt(
    *,
    existing_memories: list[dict],
    new_messages: list[dict],
    last_messages: list[dict],
    observation_date: str,
    current_date: str,
    custom_instructions: str | None = None,
    events: list[dict],
    known_constraints: list[dict],
    recent_eps: list[dict],
    candidates: list[dict],
) -> str:
    """Build the memory_extraction user prompt: the observation sections, then the EP sections, then the candidates.
    构造 memory_extraction 的用户提示词：先是观察分节，再是 EP 分节，最后是候选。

    Input / 输入：
        existing_memories: `[{"id": "0", "text": ...}]` with synthetic ids (same ids as `candidates`).
        existing_memories：带合成 id 的 `[{"id": "0", "text": ...}]`（与 `candidates` 同一套 id）。

        new_messages: `[{"idx": 0, "role": ..., "text": ..., "timestamp": ...}]`.
        new_messages：`[{"idx": 0, "role": ..., "text": ..., "timestamp": ...}]`。

        last_messages: Recent context messages rendered as `role: text` lines (300-char truncation).
        last_messages：渲染为 `role: text` 行的近期上下文消息（截断到 300 字符）。

        observation_date: `YYYY-MM-DD` of the messages.
        observation_date：消息的 `YYYY-MM-DD`。

        current_date: `YYYY-MM-DD` of the system clock.
        current_date：系统时钟的 `YYYY-MM-DD`。

        custom_instructions: Optional caller guidance; section emitted only when given.
        custom_instructions：可选的调用方指导；给定时才输出该分节。

        events: `[{"idx", "role", "object_type", "text", "timestamp"}]` state-routed messages (may be empty).
        events：分流到 state 的消息 `[{"idx", "role", "object_type", "text", "timestamp"}]`（可为空）。

        known_constraints: `[{"id": "c0", "text"}]`.
        known_constraints：`[{"id": "c0", "text"}]`。

        recent_eps: `[{"id": "e0", "u", "a", "y", "r"}]`.
        recent_eps：`[{"id": "e0", "u", "a", "y", "r"}]`。

        candidates: Revision candidates `[{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed", "valid_from"}]`.
        candidates：修订候选 `[{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed", "valid_from"}]`。

    Output / 输出：
        User prompt ending with `# Output:`; its section titles are exactly those of the three staged builders, so parse_sections and the rule handlers read it unchanged.
        以 `# Output:` 结尾的用户提示词；分节标题与三个分段构造器完全一致，parse_sections 与规则处理器无需改动即可读取。

    State effect / 状态影响：
        None.
        无。
    """
    parts = [
        _text_section("Last k Messages", "\n".join(_message_line(m) for m in last_messages)),
        _json_section("Existing Memories", existing_memories),
        _json_section("New Messages", new_messages),
        _text_section("Observation Date", observation_date),
        _text_section("Current Date", current_date),
    ]
    if custom_instructions:
        parts.append(_text_section("Custom Instructions", custom_instructions))
    parts.extend(
        [
            _json_section("Events", events),
            _json_section("Known Constraints", known_constraints),
            _json_section("Recent EPs", recent_eps),
            _json_section("Candidate Existing Memories", candidates),
        ]
    )
    return "\n".join(parts) + "\n# Output:"


def build_edge_prompt(*, nodes: list[dict], pairs: list[list[str]], witnesses: list[dict]) -> str:
    """Build the edge_proposal user prompt; every node is normalized to the full key set.
    构造 edge_proposal 的用户提示词；每个节点被归一化为完整键集。

    Input / 输入：
        nodes: Node dicts; any of the 14 documented node keys that is missing is filled with its default.
        nodes：节点字典；14 个文档化节点键中缺失的键以默认值填充。

        pairs: `[["n0", "n1"], ...]` candidate pairs.
        pairs：候选配对 `[["n0", "n1"], ...]`。

        witnesses: `[{"ref": "w0", "node_ref": "n0", "source_ref", "time"}]`.
        witnesses：`[{"ref": "w0", "node_ref": "n0", "source_ref", "time"}]`。

    Output / 输出：
        User prompt with `## Allowed Edge Types` as six plain `type: definition` lines, ending with `# Output:`.
        用户提示词，`## Allowed Edge Types` 为六行纯文本 `type: definition`，以 `# Output:` 结尾。

    State effect / 状态影响：
        None.
        无。
    """
    normalized = []
    for node in nodes:
        full = {
            key: (node[key] if key in node else (list(default) if isinstance(default, list) else default))
            for key, default in _NODE_DEFAULTS
        }
        normalized.append(full)
    parts = [
        _json_section("Nodes", normalized),
        _json_section("Candidate Pairs", pairs),
        _json_section("Witnesses", witnesses),
        _text_section(
            "Allowed Edge Types",
            "\n".join(f"{name}: {definition}" for name, definition in EDGE_TYPE_DEFINITIONS.items()),
        ),
    ]
    return "\n".join(parts) + "\n# Output:"


def build_coarsen_prompt(
    *, support_nodes: list[dict], internal_edges: list[dict], boundary_edges: list[dict], target_scale: str
) -> str:
    """Build the coarsen user prompt.
    构造 coarsen 的用户提示词。

    Input / 输入：
        support_nodes: `[{"ref": "n0", "kind", "u", "b", "a", "y", "r", "constraints", "time", "anchor_refs"}]`.
        support_nodes：`[{"ref": "n0", "kind", "u", "b", "a", "y", "r", "constraints", "time", "anchor_refs"}]`。

        internal_edges: `[{"ref": "x0", "src": "n0", "dst": "n1", "type"}]`.
        internal_edges：`[{"ref": "x0", "src": "n0", "dst": "n1", "type"}]`。

        boundary_edges: `[{"ref": "b0", "inner": "n1", "outer": "outside", "direction": "out", "type"}]`.
        boundary_edges：`[{"ref": "b0", "inner": "n1", "outer": "outside", "direction": "out", "type"}]`。

        target_scale: 'L2_meso' | 'L3_macro'.
        target_scale：'L2_meso' | 'L3_macro'。

    Output / 输出：
        User prompt ending with `# Output:`.
        以 `# Output:` 结尾的用户提示词。

    State effect / 状态影响：
        None.
        无。
    """
    parts = [
        _json_section("Support Nodes", support_nodes),
        _json_section("Internal Edges", internal_edges),
        _json_section("Boundary Edges", boundary_edges),
        _text_section("Target Scale", target_scale),
    ]
    return "\n".join(parts) + "\n# Output:"


def build_procedural_messages(messages: list[dict], prompt: str | None) -> list[dict]:
    """Mem0 `_create_procedural_memory` message list: system prompt, the messages (text only), the create instruction.
    Mem0 `_create_procedural_memory` 的消息列表：系统提示词、消息（仅文本）、创建指令。

    Input / 输入：
        messages: Chat messages `{'role', 'content'}` (list content reduced to text parts; None content dropped).
        messages：聊天消息 `{'role', 'content'}`（列表内容归约为文本分片；None 内容丢弃）。

        prompt: Replacement system prompt, or None for PROCEDURAL_SUMMARY_PROMPT.
        prompt：替代的系统提示词；None 时使用 PROCEDURAL_SUMMARY_PROMPT。

    Output / 输出：
        `[system, *messages, user: "Create procedural memory of the above conversation."]`.
        `[system, *messages, user: "Create procedural memory of the above conversation."]`。

    State effect / 状态影响：
        None.
        无。
    """
    out: list[dict] = [{"role": "system", "content": prompt or PROCEDURAL_SUMMARY_PROMPT}]
    for message in messages:
        if not isinstance(message, dict):
            continue
        text = _text_of(message.get("content"))
        if text is None:
            continue
        out.append({"role": message.get("role", "user"), "content": text})
    out.append({"role": "user", "content": "Create procedural memory of the above conversation."})
    return out


__all__ = [
    "COARSEN_PROMPT",
    "COMPACT_OBSERVATION_KEYS",
    "COMPOSITE_PROMPTS",
    "COMPOSITE_TASKS",
    "EDGE_PROPOSAL_PROMPT",
    "EDGE_TYPE_DEFINITIONS",
    "EP_EXTRACTION_PROMPT",
    "JSON_TRAILER",
    "LINE_TYPES",
    "MARKDOWN_TRAILER",
    "MEMORY_EXTRACTION_COMPACT_PROMPT",
    "MEMORY_EXTRACTION_LINES_PROMPT",
    "MEMORY_EXTRACTION_PROMPT",
    "MEMORY_EXTRACTION_ROOTS",
    "OBSERVATION_EXTRACTION_COMPACT_PROMPT",
    "OBSERVATION_EXTRACTION_PROMPT",
    "PAST_MESSAGE_TRUNCATION_LIMIT",
    "PROCEDURAL_SKELETON",
    "PROCEDURAL_SUMMARY_HEADINGS",
    "PROCEDURAL_SUMMARY_PROMPT",
    "PROMPTS",
    "REVISION_DECISION_PROMPT",
    "ROOT_KEY",
    "TASKS",
    "TYPE_LETTERS",
    "build_coarsen_prompt",
    "build_edge_prompt",
    "build_ep_prompt",
    "build_memory_extraction_prompt",
    "build_observation_prompt",
    "build_procedural_messages",
    "build_revision_prompt",
    "output_format_section",
    "parse_sections",
    "task_of",
]
