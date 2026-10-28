"""Group R tests: reranker contract, overlap scoring/stability/top_k, LLM reranker paths, prompt shape, configs.
R 组测试：重排器契约、覆盖率打分/稳定性/top_k、LLM 重排器各路径、提示词形状、配置。
"""

# Purpose (EN): Pin hms_elite/reranker with fully offline, deterministic tests (scripted and rule LLMs only).
# 目的（中）：以完全离线、确定性的测试（仅脚本化与规则 LLM）固定 hms_elite/reranker 的行为。

# Input (EN): Hand-built Mem0-shaped rows and the shared conftest fixtures (scripted_llm, rule_llm, caplog).
# 输入（中）：手工构造的 Mem0 形状行与共享的 conftest 夹具（scripted_llm、rule_llm、caplog）。

# Output (EN): pytest assertions.
# 输出（中）：pytest 断言。

# State effect (EN): None (no store, no network, no home writes).
# 状态影响（中）：无（无存储、无网络、不写入用户目录）。

from __future__ import annotations

import inspect
import json
import logging

import pytest

import hms_elite
from hms_elite.configs.prompts import JSON_TRAILER, output_format_section, parse_sections, task_of
from hms_elite.exceptions import ConfigError, LLMError, ValidationError
from hms_elite.llms.base import LLMBase
from hms_elite.reranker import (
    MISSING_SCORE,
    RERANK_PROMPT,
    RERANK_TASK,
    LLMReranker,
    LLMRerankerConfig,
    OverlapReranker,
    OverlapRerankerConfig,
    RerankerBase,
    build_rerank_prompt,
    clamp_score,
    rank_rows,
    row_text,
)

LOGGER = "hms_elite.reranker.llm"
TEXTS = (
    "User likes black coffee.",
    "patch src/billing/retry.ts: add exponential backoff",
    "billing_e2e failed: subscription_state_regression",
    "User prefers tea in the afternoon.",
)


def _rows(texts: tuple[str, ...] = TEXTS) -> list[dict]:
    """Mem0-shaped rows in retrieval order with descending fused scores.
    按检索顺序、融合分数递减的 Mem0 形状行。
    """
    return [
        {"id": f"mem_{i:06d}", "memory": text, "hash": f"h{i}", "metadata": None, "score": round(0.9 - 0.1 * i, 2)}
        for i, text in enumerate(texts)
    ]


def _ids(rows: list[dict]) -> list[str]:
    """The 'id' column of a row list.
    行列表的 'id' 列。
    """
    return [row["id"] for row in rows]


# ---------------------------------------------------------------------------
# Contract and shared helpers
# 契约与共享辅助函数
# ---------------------------------------------------------------------------
def test_reranker_contract_shape_and_helpers():
    """RerankerBase.rerank has Mem0's shape with a keyword-only top_k; clamp_score/row_text/rank_rows are exact.
    RerankerBase.rerank 与 Mem0 同形且 top_k 为仅关键字参数；clamp_score/row_text/rank_rows 行为精确。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the abstract signature and the pure helpers.
        对抽象签名与纯辅助函数的断言。

    State effect / 状态影响：
        None.
        无。
    """
    sig = inspect.signature(RerankerBase.rerank)
    assert list(sig.parameters) == ["self", "query", "rows", "top_k"]
    assert sig.parameters["top_k"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["top_k"].default is None
    assert inspect.isabstract(RerankerBase)
    assert issubclass(OverlapReranker, RerankerBase) and issubclass(LLMReranker, RerankerBase)
    assert clamp_score(0.42) == 0.42 and clamp_score("0.7") == 0.7 and clamp_score(1.7) == 1.0
    assert clamp_score(-3) == 0.0 and clamp_score(True) is None and clamp_score(None) is None
    assert clamp_score("nan") is None and clamp_score(float("inf")) is None and clamp_score("x") is None
    assert row_text({"memory": "m", "text": "t"}) == "m" and row_text({"text": "t"}) == "t"
    assert row_text({"content": 7}) == "7" and row_text({"memory": ""}) == "" and row_text({}, "data") == ""
    assert row_text({"data": "d", "memory": "m"}, "data") == "d"
    rows = _rows()
    ranked = rank_rows(rows, [0.2, None, 0.9, 0.2])
    # scored rows first (0.9, then the two 0.2 ties in input order), the unscored row last with MISSING_SCORE
    # 已打分行在前（0.9，然后两个 0.2 并列按输入顺序），未打分行以 MISSING_SCORE 排在最后
    assert _ids(ranked) == ["mem_000002", "mem_000000", "mem_000003", "mem_000001"]
    assert [r["rerank_score"] for r in ranked] == [0.9, 0.2, 0.2, MISSING_SCORE]
    assert all("rerank_score" not in row for row in rows), "input rows must never be mutated"
    assert ranked[0] is not rows[2] and ranked[0]["memory"] == rows[2]["memory"]
    assert _ids(rank_rows(rows, [0.2, None, 0.9, 0.2], top_k=2)) == ["mem_000002", "mem_000000"]
    with pytest.raises(ValueError):
        rank_rows(rows, [0.1])


# ---------------------------------------------------------------------------
# OverlapReranker
# 覆盖率重排器
# ---------------------------------------------------------------------------
def test_overlap_reranker_orders_by_weighted_coverage():
    """Identifier hits (path-suffix aware, ×identifier_weight) outrank plain-word hits; scores are in [0, 1].
    标识符命中（路径后缀感知、×identifier_weight）优先于普通词命中；分数落在 [0, 1]。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the order, the scores and the immutability of the input rows.
        对顺序、分数与输入行不可变性的断言。

    State effect / 状态影响：
        None.
        无。
    """
    rows = _rows()
    reranker = OverlapReranker()
    assert reranker.identity == "overlap@v1"
    out = reranker.rerank("retry.ts backoff patch", rows)
    # query tokens: 'retry.ts' (identifier, weight 2) + 'backoff' + 'patch' → row 1 covers all three
    # 查询词元：'retry.ts'（标识符，权重 2）+ 'backoff' + 'patch' → 第 1 行全部覆盖
    assert _ids(out)[0] == "mem_000001" and out[0]["rerank_score"] == 1.0
    assert _ids(out)[1:] == ["mem_000000", "mem_000002", "mem_000003"]
    assert all(r["rerank_score"] == 0.0 for r in out[1:])
    assert all("rerank_score" not in row for row in rows)
    # the same identifier written with its path still matches the bare file name in the row (and vice versa)
    # 带路径书写的同一标识符仍能匹配行中的裸文件名（反之亦然）
    assert reranker.score("src/billing/retry.ts", "patch retry.ts now") == 1.0
    assert reranker.score("retry.ts", "patch src/billing/retry.ts now") == 1.0
    # identifier weight 2: 'billing_e2e failed' against a row with only 'failed' → 1 / (2 + 1)
    # 标识符权重 2：'billing_e2e failed' 对仅含 'failed' 的行 → 1 / (2 + 1)
    assert reranker.score("billing_e2e failed", "the deploy failed") == round(1 / 3, 6)
    assert OverlapReranker({"identifier_weight": 0.0}).score("billing_e2e failed", "the deploy failed") == 1.0
    # a purely topical query ranks the coffee row first through the plain-word channel
    # 纯主题查询通过普通词通道把咖啡行排在最前
    assert _ids(reranker.rerank("black coffee", rows))[0] == "mem_000000"
    # CJK queries use the bigram tokens of utils.text
    # 中文查询使用 utils.text 的二元词元
    zh = OverlapReranker().rerank("黑咖啡", [{"memory": "用户喜欢黑咖啡。"}, {"memory": "用户喜欢茶。"}])
    assert zh[0]["memory"] == "用户喜欢黑咖啡。" and zh[0]["rerank_score"] > zh[1]["rerank_score"]


def test_overlap_reranker_is_stable_on_ties_and_empty_inputs():
    """Equal scores keep retrieval order; a stopword-only query scores every row 0.0; empty rows → [].
    分数相同保持检索顺序；纯停用词查询使每行得 0.0；空行列表 → []。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on stability and the degenerate inputs.
        对稳定性与退化输入的断言。

    State effect / 状态影响：
        None.
        无。
    """
    reranker = OverlapReranker()
    rows = _rows(("alpha beta", "alpha beta", "alpha beta", "gamma"))
    out = reranker.rerank("alpha", rows)
    assert _ids(out) == ["mem_000000", "mem_000001", "mem_000002", "mem_000003"]
    assert [r["rerank_score"] for r in out] == [1.0, 1.0, 1.0, 0.0]
    stop = reranker.rerank("the and of", _rows())
    assert _ids(stop) == _ids(_rows()) and all(r["rerank_score"] == 0.0 for r in stop)
    assert reranker.rerank("anything", []) == []
    # the reranker reads config.text_key and falls back through memory/text/content
    # 重排器读取 config.text_key，并按 memory/text/content 回退
    custom = OverlapReranker({"text_key": "body"}).rerank("coffee", [{"body": "tea"}, {"body": "coffee"}, {"x": 1}])
    assert [r.get("body") for r in custom] == ["coffee", "tea", None]


def test_reranker_top_k_truncates_after_sorting():
    """top_k applies after ordering: the call argument wins over config.top_k; a non-positive value is VALIDATION_009.
    top_k 在排序之后生效：调用参数优先于 config.top_k；非正值为 VALIDATION_009。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the truncated results and the error code.
        对截断结果与错误码的断言。

    State effect / 状态影响：
        None.
        无。
    """
    rows = _rows()
    query = "billing_e2e regression"
    assert _ids(OverlapReranker().rerank(query, rows, top_k=1)) == ["mem_000002"]
    assert len(OverlapReranker({"top_k": 2}).rerank(query, rows)) == 2
    assert len(OverlapReranker({"top_k": 2}).rerank(query, rows, top_k=3)) == 3
    assert len(OverlapReranker().rerank(query, rows, top_k=99)) == 4
    with pytest.raises(ValidationError) as excinfo:
        OverlapReranker().rerank(query, rows, top_k=0)
    assert excinfo.value.error_code == "VALIDATION_009"
    with pytest.raises(ConfigError) as cfg_err:
        OverlapReranker({"top_k": 0})
    assert cfg_err.value.error_code == "CONFIG_001"


# ---------------------------------------------------------------------------
# LLMReranker
# LLM 重排器
# ---------------------------------------------------------------------------
def test_llm_reranker_success_path_uses_synthetic_ids(scripted_llm):
    """A well-formed ranking re-orders rows by score; the prompt shows '0'..'n' and never a real memory id.
    格式正确的 ranking 按分数重排各行；提示词只呈现 '0'..'n'，绝不出现真实记忆 id。

    Input / 输入：
        scripted_llm: ScriptedLLM factory.
        scripted_llm：ScriptedLLM 工厂。

    Output / 输出：
        Assertions on the order, scores, route, identity and the recorded prompt.
        对顺序、分数、路由、身份与记录的提示词的断言。

    State effect / 状态影响：
        None.
        无。
    """
    rows = _rows()
    reply = {"ranking": [{"id": "2", "score": 0.9}, {"id": 0, "score": "0.4"}, {"id": "1", "score": 0.1}, {"id": "3", "score": 0.4}]}
    llm = scripted_llm([reply])
    reranker = LLMReranker(llm=llm)
    assert reranker.identity == "llm:scripted@v1"
    out = reranker.rerank("billing regression", rows)
    assert _ids(out) == ["mem_000002", "mem_000000", "mem_000003", "mem_000001"]
    assert [r["rerank_score"] for r in out] == [0.9, 0.4, 0.4, 0.1]
    assert reranker.last_route == "llm"
    assert all("rerank_score" not in row for row in rows)
    assert len(llm.calls) == 1 and llm.calls[0]["task"] == RERANK_TASK
    assert llm.calls[0]["kwargs"] == {"response_format": {"type": "json_object"}}
    system, user = (m["content"] for m in llm.calls[0]["messages"])
    assert system == RERANK_PROMPT and "mem_" not in user
    sections = parse_sections(user)
    assert sections["Query"] == "billing regression"
    assert sections["Candidates"] == [{"id": str(i), "text": text} for i, text in enumerate(TEXTS)]
    # top_k truncates the LLM ordering too, and an empty ranking keeps retrieval order under the 'llm' route
    # top_k 同样截断 LLM 排序；空 ranking 在 'llm' 路由下保持检索顺序
    llm.push(reply, {"ranking": []})
    assert _ids(reranker.rerank("q", rows, top_k=2)) == ["mem_000002", "mem_000000"]
    empty = reranker.rerank("q", rows)
    assert _ids(empty) == _ids(rows) and reranker.last_route == "llm"
    assert all(r["rerank_score"] == MISSING_SCORE for r in empty)
    assert reranker.rerank("q", []) == [] and len(llm.calls) == 3


def test_llm_reranker_malformed_output_keeps_retrieval_order(scripted_llm, caplog):
    """Garbage, a wrong root key, a non-list ranking or a bare object → retrieval order, MISSING_SCORE, one warning.
    垃圾输出、错误根键、非列表 ranking 或裸对象 → 检索顺序、MISSING_SCORE、一条告警。

    Input / 输入：
        scripted_llm: ScriptedLLM factory.
        scripted_llm：ScriptedLLM 工厂。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions per malformed reply.
        对每种畸形回复的断言。

    State effect / 状态影响：
        None.
        无。
    """
    rows = _rows()
    replies = ["this is not json", {"edges": [{"id": "0", "score": 1}]}, {"ranking": {"id": "0"}}, "[]", "42"]
    llm = scripted_llm(replies)
    reranker = LLMReranker(llm=llm)
    for _ in replies:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = reranker.rerank("billing regression", rows)
        assert _ids(out) == _ids(rows) and all(r["rerank_score"] == MISSING_SCORE for r in out)
        assert reranker.last_route == "passthrough"
        assert sum("unusable ranking" in r.getMessage() for r in caplog.records) == 1
    # a bare non-empty JSON list is accepted as the ranking itself (pushed as text: ScriptedLLM only JSON-encodes dicts)
    # 裸的非空 JSON 列表被接受为 ranking 本身（以文本推入：ScriptedLLM 只对字典做 JSON 编码）
    llm.push(json.dumps([{"id": "3", "score": 0.8}]))
    out = reranker.rerank("q", rows)
    assert _ids(out)[0] == "mem_000003" and reranker.last_route == "llm"


def test_llm_reranker_drops_unknown_ids_and_tails_missing_rows(scripted_llm, caplog):
    """Unknown/malformed ids and unusable scores are dropped with warnings; omitted rows follow in retrieval order.
    未知/畸形 id 与不可用分数被告警丢弃；未提及的行按检索顺序排在其后。

    Input / 输入：
        scripted_llm: ScriptedLLM factory.
        scripted_llm：ScriptedLLM 工厂。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions on the partial ordering, clamping, duplicate handling and the warnings.
        对部分排序、截断、重复处理与告警的断言。

    State effect / 状态影响：
        None.
        无。
    """
    rows = _rows()
    reply = {
        "ranking": [
            {"id": "7", "score": 0.99},
            {"id": "x1", "score": 0.99},
            {"id": "-1", "score": 0.99},
            "not an object",
            {"id": "3", "score": 1.7},
            {"id": "3", "score": 0.1},
            {"id": "1", "score": "nan"},
            {"id": "1", "score": None},
        ]
    }
    reranker = LLMReranker(llm=scripted_llm([reply]))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = reranker.rerank("billing regression", rows)
    # row 3 is the only scored row (clamped to 1.0, first mention wins); rows 0, 1, 2 keep retrieval order
    # 第 3 行是唯一打分行（截断到 1.0、首次出现为准）；第 0、1、2 行保持检索顺序
    assert _ids(out) == ["mem_000003", "mem_000000", "mem_000001", "mem_000002"]
    assert [r["rerank_score"] for r in out] == [1.0, MISSING_SCORE, MISSING_SCORE, MISSING_SCORE]
    assert reranker.last_route == "llm"
    messages = [r.getMessage() for r in caplog.records]
    assert sum("unknown candidate id" in m for m in messages) == 3
    assert sum("non-object ranking item" in m for m in messages) == 1
    assert sum("unusable score" in m for m in messages) == 2


def test_llm_reranker_falls_back_to_overlap_on_rule_llm(rule_llm, caplog):
    """The rule LLM raises LLM_002 for `rerank`: results equal OverlapReranker's with a warning; fallback=False raises.
    规则 LLM 对 `rerank` 抛出 LLM_002：结果与 OverlapReranker 相同并告警；fallback=False 时抛出。

    Input / 输入：
        rule_llm: A RuleLLM.
        rule_llm：一个 RuleLLM。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions on the fallback result, the route, the warning and the strict path.
        对回退结果、路由、告警与严格路径的断言。

    State effect / 状态影响：
        None (the default LLMReranker() builds a rule LLM: zero network, zero keys).
        无（默认 LLMReranker() 构造规则 LLM：零网络、零密钥）。
    """
    rows = _rows()
    query = "retry.ts backoff patch"
    reranker = LLMReranker(llm=rule_llm)
    assert reranker.identity == "llm:rule@v1"
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = reranker.rerank(query, rows, top_k=3)
    expected = OverlapReranker().rerank(query, rows, top_k=3)
    assert out == expected and _ids(out)[0] == "mem_000001" and len(out) == 3
    assert reranker.last_route == "overlap"
    assert sum("falling back to overlap@v1" in r.getMessage() for r in caplog.records) == 1
    # the default construction path (no llm, no config.llm) resolves to the rule provider and behaves the same
    # 默认构造路径（无 llm、无 config.llm）解析为规则提供者，行为一致
    default = LLMReranker({"top_k": 3})
    assert isinstance(default.llm, LLMBase) and default.llm.identity == "rule@v1"
    assert default.rerank(query, rows) == expected and default.last_route == "overlap"
    # config.llm envelope goes through LlmFactory; fallback=False turns LLM_002 into a raised LLMError
    # config.llm 信封经由 LlmFactory 构造；fallback=False 时 LLM_002 直接抛出 LLMError
    strict = LLMReranker({"llm": {"provider": "rule", "config": {"language": "en"}}, "fallback": False})
    with pytest.raises(LLMError) as excinfo:
        strict.rerank(query, rows)
    assert excinfo.value.error_code == "LLM_002" and excinfo.value.details["task"] == RERANK_TASK


def test_llm_reranker_transport_errors_propagate_and_inputs_are_truncated(scripted_llm):
    """A transport LLMError (LLM_001) is never swallowed; query and candidate texts are cut to max_chars.
    传输 LLMError（LLM_001）绝不被吞掉；查询与候选文本被截断到 max_chars。

    Input / 输入：
        scripted_llm: ScriptedLLM factory.
        scripted_llm：ScriptedLLM 工厂。

    Output / 输出：
        Assertions on the raised error and the recorded prompt.
        对抛出的错误与记录的提示词的断言。

    State effect / 状态影响：
        None.
        无。
    """
    rows = _rows()
    llm = scripted_llm([LLMError("boom", "LLM_001"), {"ranking": [{"id": "0", "score": 1}]}])
    reranker = LLMReranker({"max_chars": 12}, llm=llm)
    with pytest.raises(LLMError) as excinfo:
        reranker.rerank("q", rows)
    assert excinfo.value.error_code == "LLM_001"
    reranker.rerank("a very long query that exceeds twelve characters", rows)
    user = llm.calls[-1]["messages"][1]["content"]
    sections = parse_sections(user)
    # 12 chars of the query is 'a very long ' and the builder collapses whitespace, so the section reads 'a very long'
    # 查询的前 12 个字符是 'a very long '，构造器折叠空白后该分节为 'a very long'
    assert sections["Query"] == "a very long"
    assert [c["text"] for c in sections["Candidates"]] == [text[:12] for text in TEXTS]
    # the exhausted scripted queue (LLM_003) also propagates rather than degrading silently
    # 脚本队列耗尽（LLM_003）同样向上传播，而不是静默降级
    with pytest.raises(LLMError) as exhausted:
        reranker.rerank("q", rows)
    assert exhausted.value.error_code == "LLM_003"


# ---------------------------------------------------------------------------
# Prompt and configs
# 提示词与配置
# ---------------------------------------------------------------------------
def test_rerank_prompt_shape_and_builder_round_trip():
    """RERANK_PROMPT follows §6.1 (marker, sections, OUTPUT FORMAT + trailer); parse_sections inverts the builder.
    RERANK_PROMPT 遵循 §6.1（标记、分节、OUTPUT FORMAT + 尾句）；parse_sections 是构造器的逆运算。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the prompt text and the builder round trip.
        对提示词文本与构造器往返的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert RERANK_PROMPT.split("\n", 1)[0] == "HMS_TASK: rerank" and task_of(RERANK_PROMPT) == RERANK_TASK
    for heading in ("# ROLE", "# INPUTS", "## Query", "## Candidates", "# GUIDELINES", "Bad:", "Good:", "# EXAMPLES"):
        assert heading in RERANK_PROMPT
    assert RERANK_PROMPT.endswith(JSON_TRAILER)
    block = output_format_section(RERANK_PROMPT)
    fenced = block.split("```json\n", 1)[1].split("\n```", 1)[0]
    assert json.loads(fenced) == {"ranking": [{"id": "1", "score": 0.9}, {"id": "0", "score": 0.05}]}
    candidates = [{"id": "0", "text": "User likes tea."}, {"id": "1", "text": "多行\n文本"}]
    user = build_rerank_prompt(query="which file\ndid the retry patch touch", candidates=candidates)
    assert user.endswith("\n# Output:")
    assert parse_sections(user) == {"Query": "which file did the retry patch touch", "Candidates": candidates}


def test_reranker_configs_forbid_extra_fields():
    """Every reranker config is extra='forbid' and reports unknown keys as CONFIG_001 with a did-you-mean hint.
    每个重排器配置都是 extra='forbid'，未知键报告为带 did-you-mean 提示的 CONFIG_001。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the config classes and the error messages.
        对配置类与错误消息的断言。

    State effect / 状态影响：
        None.
        无。
    """
    for cls in (OverlapRerankerConfig, LLMRerankerConfig):
        assert cls.model_config["extra"] == "forbid"
    with pytest.raises(ConfigError) as overlap_err:
        OverlapReranker({"identifier_weigth": 2.0})
    assert overlap_err.value.error_code == "CONFIG_001" and "identifier_weight" in str(overlap_err.value)
    # through the provider constructor (coerce_config unwraps the ConfigError raised inside the pydantic validator)
    # 经由提供者构造函数（coerce_config 解包 pydantic 校验器内部抛出的 ConfigError）
    with pytest.raises(ConfigError) as llm_err:
        LLMReranker({"max_char": 3})
    assert llm_err.value.error_code == "CONFIG_001" and "max_chars" in str(llm_err.value)
    with pytest.raises(ConfigError):
        OverlapReranker({"identifier_weight": -1})
    with pytest.raises(ConfigError):
        LLMReranker({"max_chars": 0})
    cfg = LLMRerankerConfig(llm={"provider": "scripted", "config": {"responses": []}}, top_k=5, text_key="text")
    assert cfg.model_dump() == {
        "top_k": 5,
        "text_key": "text",
        "llm": {"provider": "scripted", "config": {"responses": []}},
        "max_chars": 4000,
        "fallback": True,
    }
    built = LLMReranker(cfg)
    assert built.llm.identity == "scripted@v1" and built.config is cfg


def test_package_root_exports_reranker_classes():
    """The package root re-exports RerankerBase / OverlapReranker / LLMReranker (the same objects, listed in __all__).
    包根再导出 RerankerBase / OverlapReranker / LLMReranker（同一对象，并列入 __all__）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Identity and __all__ membership assertions.
        同一性与 __all__ 成员断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert hms_elite.RerankerBase is RerankerBase
    assert hms_elite.OverlapReranker is OverlapReranker and hms_elite.LLMReranker is LLMReranker
    assert {"RerankerBase", "OverlapReranker", "LLMReranker"} <= set(hms_elite.__all__)
