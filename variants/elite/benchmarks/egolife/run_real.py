"""EgoLife real-LLM streaming benchmark: mem0_real versus elite_rows / elite_hybrid (elite_card with --with-card) on deepseek-flash, concurrency 1.
EgoLife 真实 LLM 流式评测：mem0_real 对 elite_rows / elite_hybrid（--with-card 时加 elite_card），模型 deepseek-flash，并发 1。
"""

# Purpose (EN): Reuse the infer=False bench's windows / timeline (windows.py, timeline.py), retain in batches of
#               <= 25 windows through each engine's infer=True path, answer every question with deepseek-flash on
#               the engine's context, and write per-question / per-attempt records plus the five-dimension summary
#               (accuracy, latency, tokens, money, stability) as summary.json + summary.md.
# 目的（中）：复用 infer=False 实验台的窗口 / 时间线（windows.py、timeline.py），以 ≤ 25 窗口一批经各引擎的 infer=True
#            路径 retain，用 deepseek-flash 基于各引擎的上下文回答每道题，写逐题 / 逐尝试记录与五维汇总
#           （准确率、延迟、token、金额、稳定性）summary.json + summary.md。

# Input (EN): --limit (default 204) --out (default results/real) --engines --keep-db --batch-size; the .env key file.
# 输入（中）：--limit（默认 204）--out（默认 results/real）--engines --keep-db --batch-size；.env 密钥文件。

# Output (EN): <out>/{manifest.json, windows.jsonl, batches.json, calls.jsonl, retain_<engine>.jsonl,
#              questions_<label>.jsonl, arm_<engine>.json, summary.json, summary.md}. No key ever appears in them.
# 输出（中）：<out>/{manifest.json, windows.jsonl, batches.json, calls.jsonl, retain_<engine>.jsonl,
#            questions_<label>.jsonl, arm_<engine>.json, summary.json, summary.md}。任何产物都不含密钥。

# State effect (EN): Engine files under <out>/work/<engine>/ (deleted after measurement unless --keep-db); network
#                    calls to api.deepseek.com.
# 状态影响（中）：引擎文件在 <out>/work/<engine>/（度量后删除，除非 --keep-db）；对 api.deepseek.com 的网络调用。

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from accounting import (  # noqa: E402
    MODEL,
    PEAK_WINDOWS_UTC,
    PRICES_USD_PER_M,
    PRICING_SOURCE,
    STAGES,
    Meter,
)
from loader import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    QA_FILENAME,
    Question,
    load_questions,
    load_transcripts,
    transcript_files,
)
from real_arms import (  # noqa: E402
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    BACKOFF_S,
    BANK,
    BASE_URL,
    BATCH_SIZE,
    EMBED_MODEL,
    EMBEDDING_DIMS,
    ENGINE_NAMES,
    EXTRACT_MAX_TOKENS,
    EXTRACT_TEMPERATURE,
    MAX_RETRIES,
    RETRY_STATUSES,
    THINKING,
    THRESHOLD,
    TIMEOUT_S,
    TOP_K,
    Batch,
    DeepSeekClient,
    EliteRealArm,
    Mem0RealArm,
    answer,
    load_api_key,
    plan_batches,
    rows_to_context,
    their_protocol_digest,
)
from timeline import GOLD_HALF_WIDTH_CS, GOLD_NEIGHBOURHOOD_CS, Step, schedule  # noqa: E402
from windows import MAX_CHARS, MAX_SPAN_CS, Window, build_windows, clock_to_iso, write_windows_jsonl  # noqa: E402

from hms_elite.configs.base import IngestConfig  # noqa: E402
from hms_elite.ingest.extraction import EXTRACTION_OUTPUTS  # noqa: E402

# The --extraction-output default is read from the library so the benchmark cannot drift from IngestConfig
# ('lines' since 2026-09-18, CONTRACT §6.6 / changelog row 108; 'full' and 'compact' stay selectable); the value
# actually used is still recorded in manifest['elite'].
# --extraction-output 的默认值从库里读取，基准脚本不会与 IngestConfig 漂移（2026-09-18 起为 'lines'，CONTRACT §6.6 /
# 变更记录第 108 行；'full' 与 'compact' 仍可选）；实际使用的值仍记入 manifest['elite']。
DEFAULT_EXTRACTION_OUTPUT: str = str(IngestConfig().extraction_output)

ENV_PATH = HERE / ".env"
DEFAULT_OUT = HERE / "results" / "real"
# Persistent fastembed model cache (the default lives under tempfile.gettempdir(), which macOS purges): exported as
# FASTEMBED_CACHE_PATH before any arm is built so mem0's provider (which takes only a model name) uses it too.
# 持久的 fastembed 模型缓存（默认在 tempfile.gettempdir() 下，macOS 会清理）：在构建任何臂之前导出为
# FASTEMBED_CACHE_PATH，使只接受模型名的 mem0 提供者也使用它。
DEFAULT_FASTEMBED_CACHE_DIR = Path.home() / ".cache" / "hms-elite-bench" / "fastembed_cache"
# Default Elite labels since 2026-09-18: the search rows and the hybrid context of HMS.context() (session card prefix +
# timestamped rows, answered memories-first; READER_AB_2026-09-18.md §六). The per-question compile() card (elite_card)
# is opt-in through --with-card so old result directories keep parsing.
# 2026-09-18 起 Elite 的默认标签：检索行与 HMS.context() 的混合上下文（会话卡片前缀 + 带时间戳记忆行，记忆在前作答；
# READER_AB_2026-09-18.md §六）。逐题 compile() 卡片（elite_card）经 --with-card 选择加入，旧结果目录仍可解析。
ENGINE_LABELS: dict[str, tuple[str, ...]] = {"mem0_real": ("mem0_real",), "elite": ("elite_rows", "elite_hybrid")}
CARD_LABEL = "elite_card"
LABEL_ENGINE: dict[str, str] = {lab: eng for eng, labs in ENGINE_LABELS.items() for lab in labs}
LABEL_ENGINE[CARD_LABEL] = "elite"
ERROR_CATEGORIES = ("no_evidence_retrieved", "evidence_present_wrong_answer", "invalid_output", "api_error", "timeout")
PROMPT_TEMPLATE = (
    "<build_question_prompt(item)>\n\nMemories:\n<context>   (context = '- row text' lines for *_rows / mem0_real; the Work State Card text for elite_card; '(none)' when empty); "
    "elite_hybrid is memories-first: Memories:\n<session card>\n\nRecent relevant memories:\n<'- [YYYY-MM-DD HH:MM] text' rows>\n\n<build_question_prompt(item)> (HMS.context(); each part '(none)' when empty)"
)


def resolve_embed_dims(model: str) -> tuple[int, str]:
    """The vector size of a fastembed model: EMBEDDING_DIMS for the default model, else fastembed's registry (no download).
    fastembed 模型的向量维数：默认模型取 EMBEDDING_DIMS，否则查 fastembed 的注册表（不下载）。

    Input / 输入：
        model: fastembed model name.
        model：fastembed 模型名。

    Output / 输出：
        (dims, source) with source 'default' | 'fastembed.get_embedding_size'.
        (维数, 来源)，来源为 'default' | 'fastembed.get_embedding_size'。

    State effect / 状态影响：
        None (SystemExit when fastembed does not know the model).
        无（fastembed 不认识该模型时 SystemExit）。
    """
    if model == EMBED_MODEL:
        return EMBEDDING_DIMS, "default"
    try:
        from fastembed import TextEmbedding

        return int(TextEmbedding.get_embedding_size(model)), "fastembed.get_embedding_size"
    except Exception as exc:  # noqa: BLE001 - surfaced as a usage error / 作为用法错误抛出
        raise SystemExit(f"cannot resolve --embed-dims for {model!r} ({exc}); pass --embed-dims explicitly") from exc


def glossary_names() -> tuple[str, ...]:
    """The Chinese glossary names of translate_zh.NAME_GLOSSARY (the closed EgoLife-ZH person set), minus the wearer's 我.
    translate_zh.NAME_GLOSSARY 的中文术语表人名（EgoLife-ZH 的封闭人物集合），去掉佩戴者的 我。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Sorted unique names.
        排序去重后的人名。

    State effect / 状态影响：
        None (imports translate_zh lazily; its module level needs no API SDK).
        无（惰性导入 translate_zh；其模块级不需要 API SDK）。
    """
    from translate_zh import NAME_GLOSSARY

    return tuple(sorted({v for v in NAME_GLOSSARY.values() if len(v) >= 2}))


def default_entity_names(text_lang: str, cjk: str | None, flag: str | None) -> tuple[str, ...]:
    """The rule extractor's closed name list for a run: the --entity-names flag when given ('' → none), else the
    glossary names only for --text-lang zh with --cjk on (run B), else none — run A (--cjk off) keeps the untouched
    rule@entities-v1 extractor so it measures the English-centric pipeline alone (EGOLIFE_ZH_PLAN §4).
    一次运行中规则抽取器的封闭人名表：给了 --entity-names 就用它（'' → 无），否则只在 --text-lang zh 且 --cjk on（运行 B）时
    取术语表人名，其它情况为空——运行 A（--cjk off）保持未改动的 rule@entities-v1 抽取器，只度量英文中心流水线
    （EGOLIFE_ZH_PLAN §4）。

    Input / 输入：
        text_lang: 'en' | 'zh'.
        text_lang：'en' | 'zh'。

        cjk: None | 'off' | 'on'.
        cjk：None | 'off' | 'on'。

        flag: The raw --entity-names value (comma-separated) or None.
        flag：--entity-names 的原始值（逗号分隔）或 None。

    Output / 输出：
        Tuple of names.
        人名元组。

    State effect / 状态影响：
        None.
        无。
    """
    if flag is not None:
        return tuple(n.strip() for n in flag.split(",") if n.strip())
    return glossary_names() if (text_lang == "zh" and cjk == "on") else ()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


PERCENTILE_RULE = "nearest-rank: sorted[round(q*(n-1))] (no interpolation)"
FIRST50_RULE = "first-50 = the first 50 questions in QA-file order (order < 50), denominator 50; first-50 (stream) = first 50 answered in timeline order"
HIT_RULE = "hit_batch = a returned memory's source batch contains a gold window; k = returned_count (20 rows for *_rows / mem0_real; the card's items for elite_card; the session card's items + the kept rows for elite_hybrid); NOT accuracy"
TRUNCATION_RULE = "an extraction attempt with finish_reason == 'length' is rejected and its window slice re-submitted as two halves (recursively to 1 window); every sub-add is a priced request"


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[idx], 3)


def _dist(values: list[float]) -> dict[str, Any]:
    return {"median": _p(values, 0.5), "p95": _p(values, 0.95), "max": _p(values, 1.0), "n": len(values)}


LEAF_OUTCOMES = ("ok", "truncated_unresolved", "invalid_json", "schema_rejected_all", "empty_extraction", "api_error")


def classify_retain(row: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the sub-add records of one retain batch into extraction-parse / truncation counts.
    把一个 retain 批次的子 add 记录汇总为抽取解析 / 截断计数。

    Input / 输入：
        row: retain_with_bisect record (has 'subs').
        row：retain_with_bisect 的记录（含 'subs'）。

    Output / 输出：
        {'parse_error', 'non_object_reply', 'items_skipped', 'items_skipped_attributed_to', 'lines_dropped',
         'other_warnings', 'finish_length', 'adds', 'bisected', 'leaf_outcomes', 'windows_lost', 'failed',
         'truncation_recovered'}; failed = some window slice ended with an unusable reply (invalid_json or
         truncated_unresolved); lines_dropped = single lines the lines parser dropped (extraction_output=lines only).
        {...}；failed = 某个窗口切片最终得到不可用回复（invalid_json 或 truncated_unresolved）；lines_dropped = 行式解析器
        丢掉的单行数（仅 extraction_output=lines）。

    State effect / 状态影响：
        None.
        无。
    """
    subs = row.get("subs") or []
    leaves = [x for x in subs if x["outcome"] != "truncated"]
    warnings = [w for x in subs for w in x.get("warnings", [])]
    parse_error = sum(x.get("parse_error", 0) for x in subs)
    non_object = sum(x.get("non_object_reply", 0) for x in subs)
    skipped = sum(x.get("items_skipped", 0) for x in subs)
    lines_dropped = sum(x.get("lines_dropped", 0) for x in subs)
    outcomes = {o: sum(1 for x in leaves if x["outcome"] == o) for o in LEAF_OUTCOMES}
    return {
        "parse_error": parse_error,
        "items_skipped_attributed_to": sum(x.get("items_skipped_attributed_to", 0) for x in subs),
        "non_object_reply": non_object,
        "items_skipped": skipped,
        "lines_dropped": lines_dropped,
        "other_warnings": max(0, len(warnings) - parse_error - non_object - skipped - lines_dropped - sum(1 for w in warnings if "reply is not a JSON object" in w)),
        "finish_length": row.get("truncated_attempts", 0),
        "adds": len(subs),
        "bisected": bool(row.get("bisected")),
        "leaf_outcomes": outcomes,
        "windows_lost": row.get("windows_lost", 0),
        "failed": bool(outcomes["invalid_json"] or outcomes["truncated_unresolved"]),
        "truncation_recovered": bool(row.get("bisected")) and row.get("windows_lost", 0) == 0,
    }


def error_category(rec: dict[str, Any]) -> str | None:
    """Classify a wrong / missing answer: api_error, timeout, invalid_output, no_evidence_retrieved,
    evidence_present_wrong_answer; None when correct.
    给错答 / 缺答归类：api_error、timeout、invalid_output、no_evidence_retrieved、evidence_present_wrong_answer；答对为 None。

    Input / 输入：
        rec: Per-question record (answer_status, pred, gold, hit_batch).
        rec：逐题记录（answer_status、pred、gold、hit_batch）。

    Output / 输出：
        Category string or None.
        类别字符串或 None。

    State effect / 状态影响：
        None.
        无。
    """
    if rec["correct"]:
        return None
    status = rec["answer_status"]
    if status == "timeout":
        return "timeout"
    if status == "http_error":
        return "api_error"
    if rec["pred"] is None:
        return "invalid_output"
    return "evidence_present_wrong_answer" if rec["hit_batch"] else "no_evidence_retrieved"


def run_engine(
    engine: str,
    steps: list[Step],
    batches: list[Batch],
    *,
    client: DeepSeekClient,
    meter: Meter,
    work_root: Path,
    out: Path,
    keep_db: bool,
    extraction_output: str = DEFAULT_EXTRACTION_OUTPUT,
    with_card: bool = False,
    embed_model: str = EMBED_MODEL,
    embed_dims: int = EMBEDDING_DIMS,
    cjk: str | None = None,
    fastembed_cache_dir: str | None = None,
    entity_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Stream every step through one engine; answer under each of its context labels; write the per-question files.
    让每个步骤流过一个引擎；在其每个上下文标签下答题；写逐题文件。

    Input / 输入：
        engine: 'mem0_real' | 'elite'.
        engine：'mem0_real' | 'elite'。

        steps: Shared timeline.
        steps：共享时间线。

        batches: plan_batches output (shared by both engines).
        batches：plan_batches 的输出（两引擎共用）。

        client: Accounting client.
        client：记账客户端。

        meter: Ledger.
        meter：台账。

        work_root: Parent of the engine directory.
        work_root：引擎目录的父目录。

        out: Result directory.
        out：结果目录。

        keep_db: Keep engine files.
        keep_db：保留引擎文件。

        extraction_output: 'full' | 'compact' | 'lines' (elite only; IngestConfig.extraction_output, default = the library default, 'lines' since 2026-09-18; choices follow EXTRACTION_OUTPUTS).
        extraction_output：'full' | 'compact' | 'lines'（仅 elite；IngestConfig.extraction_output，默认 = 库默认值，2026-09-18 起为 'lines'；选项跟随 EXTRACTION_OUTPUTS）。

        with_card: elite only; also answer under elite_card (the per-question compile() card, the pre-2026-09-18 default label).
        with_card：仅 elite；同时在 elite_card（逐题 compile() 卡片，2026-09-18 之前的默认标签）下答题。

        embed_model: fastembed model for both engines (default EMBED_MODEL; the EgoLife-ZH runs use BAAI/bge-small-zh-v1.5).
        embed_model：两引擎共用的 fastembed 模型（默认 EMBED_MODEL；EgoLife-ZH 运行用 BAAI/bge-small-zh-v1.5）。

        embed_dims: The model's vector size (default EMBEDDING_DIMS).
        embed_dims：模型向量维数（默认 EMBEDDING_DIMS）。

        cjk: elite only; None = library defaults, 'on' / 'off' = the CONTRACT §9.6 switches (EliteRealArm).
        cjk：仅 elite；None = 库默认值，'on' / 'off' = CONTRACT §9.6 的开关（EliteRealArm）。

        fastembed_cache_dir: elite only; FastEmbedEmbedderConfig.cache_dir (mem0 reads FASTEMBED_CACHE_PATH instead).
        fastembed_cache_dir：仅 elite；FastEmbedEmbedderConfig.cache_dir（mem0 改读 FASTEMBED_CACHE_PATH）。

        entity_names: elite only; closed proper-name list for the rule entity extractor (RuleEntityExtractorConfig.names).
        entity_names：仅 elite；规则实体抽取器的封闭人名表（RuleEntityExtractorConfig.names）。

    Output / 输出：
        {'records': {label: [...]}, 'retain': [...], 'stats': ..., 'wall_ms'}.
        {'records': {标签: [...]}, 'retain': [...], 'stats': ..., 'wall_ms'}。

    State effect / 状态影响：
        Writes questions_<label>.jsonl and retain_<engine>.jsonl; network calls.
        写 questions_<label>.jsonl 与 retain_<engine>.jsonl；网络调用。
    """
    t_init = time.perf_counter()
    arm = (
        Mem0RealArm(work_root / engine, client, meter, embed_model=embed_model, embed_dims=embed_dims)
        if engine == "mem0_real"
        else EliteRealArm(
            work_root / engine, client, meter, extraction_output=extraction_output, embed_model=embed_model, embed_dims=embed_dims,
            cjk=cjk, fastembed_cache_dir=fastembed_cache_dir, entity_names=entity_names,
        )
    )  # fmt: skip
    init_ms = (time.perf_counter() - t_init) * 1000.0  # store creation + fastembed model load, before any API call / 建库 + 加载 fastembed，早于任何 API 调用
    labels = ENGINE_LABELS[engine] + ((CARD_LABEL,) if with_card and engine == "elite" else ())
    by_position: dict[int, list[Batch]] = defaultdict(list)
    for b in batches:
        by_position[b.position].append(b)
    batch_of_window = {w.window_id: b.batch_id for b in batches for w in b.windows}
    stored_windows_by_batch: dict[str, int] = {}  # filled as batches are retained / 随批次入库填充
    records: dict[str, list[dict[str, Any]]] = {lab: [] for lab in labels}
    retain_log: list[dict[str, Any]] = []
    handles = {lab: (out / f"questions_{lab}.jsonl").open("w", encoding="utf-8") for lab in labels}
    retain_fh = (out / f"retain_{engine}.jsonl").open("w", encoding="utf-8")
    raw_fh = (out / f"extract_raw_{engine}.jsonl").open("w", encoding="utf-8")
    client.raw_sink = lambda row: (raw_fh.write(json.dumps(row, ensure_ascii=False) + "\n"), raw_fh.flush())
    t_engine = time.perf_counter()
    try:
        for step in steps:
            q = step.question
            query_iso = clock_to_iso(q.query_day, q.query_cs)
            # ---- retain: this step's batches (local + extraction API) / 本步批次（本地 + 抽取 API）
            t0 = time.perf_counter()
            step_batches = [arm.retain_batch(b) for b in by_position.get(step.position, [])]
            retain_wall = (time.perf_counter() - t0) * 1000.0
            for r in step_batches:
                stored_windows_by_batch[r["batch_id"]] = r["windows_stored"]
            retain_model = sum(r["model_ms"] for r in step_batches)
            for r in step_batches:
                row = {k: v for k, v in r.items() if k != "calls"}
                row.update({"position": step.position, "question_id": q.id, "attempts": len(r["calls"]),
                            "statuses": [c["status"] for c in r["calls"]], "local_ms": r["wall_ms"] - r["model_ms"],
                            "extraction_parse": classify_retain(r)})
                retain_log.append(row)
                retain_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            # ---- recall (local) / 召回（本地）
            t0 = time.perf_counter()
            rows = arm.recall(q.question, query_iso)
            recall_ms = (time.perf_counter() - t0) * 1000.0
            gold_batches = {batch_of_window[w] for w in step.gold_retained_ids if w in batch_of_window}
            gold_wide_batches = {batch_of_window[w] for w in step.gold_wide_retained_ids if w in batch_of_window}
            contexts: dict[str, dict[str, Any]] = {}
            t0 = time.perf_counter()
            rows_text = rows_to_context(rows)
            rows_ctx_ms = (time.perf_counter() - t0) * 1000.0
            row_batches = []
            for r in rows:
                if r["batch_id"] and r["batch_id"] not in row_batches:
                    row_batches.append(r["batch_id"])
            rows_ctx = {"text": rows_text, "recall_ms": recall_ms, "context_ms": rows_ctx_ms, "returned_count": len(rows),
                        "returned_batches": row_batches, "top_scores": [round(r["score"], 4) for r in rows[:5]], "kind": "rows"}
            # Elite rows carry the explain signals (real_arms.EliteRealArm.recall): how many returned rows the keyword /
            # entity channels scored > 0 and the recall channel sizes — the EgoLife-ZH 'keyword hits non-zero' check.
            # Elite 的行带 explain 信号（real_arms.EliteRealArm.recall）：关键词 / 实体通道打分 > 0 的返回行数与各召回通道的
            # 规模——EgoLife-ZH 的"关键词命中非零"检查。
            if rows and "keyword" in rows[0]:
                rows_ctx["keyword_hits"] = sum(1 for r in rows if r.get("keyword", 0.0) > 0.0)
                rows_ctx["entity_hits"] = sum(1 for r in rows if r.get("entity", 0.0) > 0.0)
                rows_ctx["channel_sizes"] = dict(rows[0].get("channel_sizes") or {})
                rows_ctx["recalled_by"] = {ch: sum(1 for r in rows if ch in (r.get("recalled_by") or [])) for ch in sorted({c for r in rows for c in (r.get("recalled_by") or [])})}
            if engine == "mem0_real":
                contexts["mem0_real"] = rows_ctx
            else:
                contexts["elite_rows"] = rows_ctx
                # elite_hybrid = HMS.context(): it runs its own search (the same call as recall above, local fastembed
                # only), so the 'recall' stage embeds the question twice per step; recall_ms of this label is the whole
                # context() time (search + the memoized session card).
                # elite_hybrid = HMS.context()：它自行执行一次检索（与上面的 recall 同一调用，仅本地 fastembed），因此
                # 'recall' 阶段每步嵌入题干两次；本标签的 recall_ms 是整个 context() 的耗时（检索 + 记忆化会话卡片）。
                hyb = arm.hybrid(q.question, query_iso)  # type: ignore[union-attr]
                contexts["elite_hybrid"] = {
                    "text": hyb["text"], "recall_ms": hyb["context_ms"], "context_ms": 0.0, "returned_count": hyb["returned_count"],
                    "returned_batches": hyb["batch_ids"], "top_scores": [], "kind": "hybrid",
                    "hybrid": {k: hyb[k] for k in ("chars", "session_reused", "session_key", "session_chars", "rows_kept", "session_batch_ids", "row_batch_ids")},
                }
                if CARD_LABEL in labels:
                    card = arm.card(q.question, query_iso)  # type: ignore[union-attr]
                    contexts[CARD_LABEL] = {
                        "text": card["text"], "recall_ms": card["compile_ms"], "context_ms": 0.0, "returned_count": card["items"],
                        "returned_batches": card["batch_ids"], "top_scores": [], "kind": "card",
                        "card": {k: card[k] for k in ("chars", "used_tokens", "items", "mode", "compile_ms")},
                    }
            for lab, ctx in contexts.items():
                res = answer(client, q, ctx["text"], arm=lab, order="memories_first" if ctx["kind"] == "hybrid" else "question_first")
                hit = any(b in gold_batches for b in ctx["returned_batches"])
                hit_wide = any(b in gold_wide_batches for b in ctx["returned_batches"])
                end_to_end = retain_wall + ctx["recall_ms"] + ctx["context_ms"] + res.wall_ms
                rec: dict[str, Any] = {
                    "arm": lab, "engine": engine, "id": q.id, "order": q.order, "position": step.position, "type": q.type,
                    "query_day": q.query_day, "query_time": q.query_hhmmsscc,
                    "retained_count": step.retained_count, "new_windows": len(step.new_windows),
                    "retain_batches": len(step_batches), "retain_batches_failed": sum(1 for r in step_batches if not r["ok"]),
                    "retain_adds": sum(r["adds"] for r in step_batches), "retain_windows_lost": sum(r["windows_lost"] for r in step_batches),
                    "excluded_future": step.excluded_future, "gold_window_count": len(step.gold_ids),
                    "gold_retained_count": len(step.gold_retained_ids), "gold_batches": sorted(gold_batches),
                    "returned_count": ctx["returned_count"], "returned_batches": ctx["returned_batches"],
                    "top_scores": ctx["top_scores"], "context_kind": ctx["kind"], "context_chars": len(ctx["text"]),
                    "hit_batch": hit, "hit_batch_wide": hit_wide, "hit_k": ctx["returned_count"],
                    "gold_batches_stored": sorted(b for b in gold_batches if stored_windows_by_batch.get(b, 0) > 0),
                    "gold": q.answer, "pred": res.letter, "raw_output": res.raw, "correct": res.letter == q.answer,
                    "answer_status": res.status, "answer_attempts": len(res.calls),
                    "answer_cold_start": any(c.cold_start for c in res.calls),
                    "answer_usage": {
                        "prompt_tokens": res.calls[-1].prompt_tokens if res.calls else None,
                        "completion_tokens": res.calls[-1].completion_tokens if res.calls else None,
                        "cache_hit_tokens": res.calls[-1].cache_hit_tokens if res.calls else None,
                        "cache_miss_tokens": res.calls[-1].cache_miss_tokens if res.calls else None,
                        "rate_tier": res.calls[-1].rate_tier if res.calls else None,
                        "cost_usd_all_attempts": round(sum(c.cost_usd or 0.0 for c in res.calls), 8),
                        "unpriced_attempts": sum(1 for c in res.calls if c.cost_usd is None),
                    },
                    "timing_ms": {
                        "retain_wall": round(retain_wall, 3), "retain_model": round(retain_model, 3),
                        "retain_local": round(retain_wall - retain_model, 3),
                        "retain_cold_start": any(c["cold_start"] for r in step_batches for c in r["calls"]),
                        "retain_attempts": sum(len(r["calls"]) for r in step_batches),
                        "recall_local": round(ctx["recall_ms"], 3), "context_local": round(ctx["context_ms"], 3),
                        "answer_model": round(res.model_ms, 3), "answer_local": round(res.local_ms, 3),
                        "answer_wall": round(res.wall_ms, 3), "end_to_end": round(end_to_end, 3),
                    },
                }
                if "keyword_hits" in ctx:
                    rec["channels"] = {k: ctx[k] for k in ("keyword_hits", "entity_hits", "channel_sizes", "recalled_by")}
                if "card" in ctx:
                    rec["card"] = ctx["card"]
                if "hybrid" in ctx:
                    rec["hybrid"] = ctx["hybrid"]
                    rec["prompt_order"] = "memories_first"
                rec["error_category"] = error_category(rec)
                records[lab].append(rec)
                handles[lab].write(json.dumps(rec, ensure_ascii=False) + "\n")
            done = len(records[labels[0]])
            if done % 25 == 0 or done == len(steps):
                accs = {lab: sum(1 for r in records[lab] if r["correct"]) for lab in labels}
                spent = round(sum(c.cost_usd or 0.0 for c in meter.calls), 4)
                pf = sum(1 for r in retain_log if r["extraction_parse"]["failed"])
                tr = sum(r["truncated_attempts"] for r in retain_log)
                lost = sum(r["windows_lost"] for r in retain_log)
                print(f"  [{engine}] {done}/{len(steps)} correct={accs} ledger_cost_usd={spent} retain_batches={len(retain_log)} adds={sum(r['adds'] for r in retain_log)} truncated_attempts={tr} windows_lost={lost} extraction_parse_failures={pf}", flush=True)
    finally:
        for fh in handles.values():
            fh.close()
        retain_fh.close()
        raw_fh.close()
        client.raw_sink = None
    wall_ms = (time.perf_counter() - t_engine) * 1000.0
    arm.close()
    stats = arm.stats()
    stats["init_ms"] = round(init_ms, 3)
    stats["init_note"] = "arm construction (store creation + fastembed model load) measured before the first API call; not part of any per-question timing"
    if not keep_db:
        shutil.rmtree(work_root / engine, ignore_errors=True)
    return {"records": records, "retain": retain_log, "stats": stats, "wall_ms": round(wall_ms, 1)}


def _accuracy(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(recs)
    correct = sum(1 for r in recs if r["correct"])
    valid = [r for r in recs if r["pred"] is not None]
    scored = [r for r in recs if not r["excluded_future"]]
    first50 = [r for r in recs if r["order"] < 50]
    stream50 = recs[:50]
    gold_stored = [r for r in scored if r["gold_batches_stored"]]
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "n_first50": len(first50),
        "correct_first50": sum(1 for r in first50 if r["correct"]),
        "accuracy_first50": round(sum(1 for r in first50 if r["correct"]) / len(first50), 4) if first50 else None,
        "n_first50_stream": len(stream50),
        "correct_first50_stream": sum(1 for r in stream50 if r["correct"]),
        "accuracy_first50_stream": round(sum(1 for r in stream50 if r["correct"]) / len(stream50), 4) if stream50 else None,
        "first50_rule": FIRST50_RULE,
        "valid_n": len(valid),
        "accuracy_valid_only": round(sum(1 for r in valid if r["correct"]) / len(valid), 4) if valid else None,
        "excluded_future": n - len(scored),
        "accuracy_gold_retained_only": round(sum(1 for r in scored if r["correct"]) / len(scored), 4) if scored else None,
        "n_gold_stored": len(gold_stored),
        "accuracy_gold_stored_subset": round(sum(1 for r in gold_stored if r["correct"]) / len(gold_stored), 4) if gold_stored else None,
        "gold_stored_note": "diagnostic only: questions whose gold batch stored >= 1 window; the 204 denominator above is the reported accuracy",
        "hit_batch": round(sum(1 for r in scored if r["hit_batch"]) / len(scored), 4) if scored else None,
        "hit_batch_wide": round(sum(1 for r in scored if r["hit_batch_wide"]) / len(scored), 4) if scored else None,
        "hit_k": _dist([float(r["hit_k"]) for r in scored]),
        "hit_note": HIT_RULE,
        "error_categories": {c: sum(1 for r in recs if r["error_category"] == c) for c in ERROR_CATEGORIES},
        "no_evidence_gold_unstored": sum(1 for r in recs if r["error_category"] == "no_evidence_retrieved" and not r["gold_batches_stored"]),
        "random_baseline": 0.25,
    }


def _latency(recs: list[dict[str, Any]]) -> dict[str, Any]:
    t = [r["timing_ms"] for r in recs]
    with_retain = [x for x in t if x["retain_attempts"] > 0]
    warm = [r for r in recs if not r["answer_cold_start"]]
    return {
        "concurrency": 1,
        "retain": {
            "wall": _dist([x["retain_wall"] for x in with_retain]),
            "model": _dist([x["retain_model"] for x in with_retain]),
            "local": _dist([x["retain_local"] for x in with_retain]),
            "steps_with_retain": len(with_retain),
            "note": "per timeline step (all batches released before the question); model = sum of extraction attempts incl. retries",
        },
        "recall_local": _dist([x["recall_local"] for x in t]),
        "context_local": _dist([x["context_local"] for x in t]),
        "answer_model": _dist([x["answer_model"] for x in t]),
        "answer_model_excluding_cold_start": _dist([r["timing_ms"]["answer_model"] for r in warm]),
        "answer_local": _dist([x["answer_local"] for x in t]),
        "end_to_end": _dist([x["end_to_end"] for x in t]),
        "end_to_end_excluding_retain": _dist([x["end_to_end"] - x["retain_wall"] for x in t]),
        "cold_start_questions": sum(1 for r in recs if r["answer_cold_start"] or r["timing_ms"]["retain_cold_start"]),
        "answer_attempts_total": sum(r["answer_attempts"] for r in recs),
        "answer_retries": sum(r["answer_attempts"] - 1 for r in recs),
    }


def _parse_failures(retain: list[dict[str, Any]], batch_windows: dict[str, int] | None = None) -> dict[str, Any]:
    """Aggregate classify_retain over retain rows (unusable extraction replies are NOT silent; truncation is its own count).
    汇总 retain 行的 classify_retain（不可用的抽取回复不得静默；截断单独计数）。
    """
    kinds = [r.get("extraction_parse") or {} for r in retain]
    failed = [r for r, k in zip(retain, kinds) if k.get("failed")]
    n = len(retain)
    leaf = {o: sum(k.get("leaf_outcomes", {}).get(o, 0) for k in kinds) for o in LEAF_OUTCOMES}
    nothing = [r for r in retain if r.get("added", 0) == 0]
    return {
        "batches": len(failed),
        "rate_over_batches": round(len(failed) / n, 4) if n else None,
        "windows_affected": sum(r.get("windows_lost", 0) for r in failed),
        "truncated_attempts": sum(k.get("finish_length", 0) for k in kinds),
        "batches_truncated": sum(1 for k in kinds if k.get("finish_length", 0)),
        "batches_bisected": sum(1 for k in kinds if k.get("bisected")),
        "batches_truncation_recovered": sum(1 for k in kinds if k.get("truncation_recovered")),
        "adds_total": sum(k.get("adds", 1) for k in kinds),
        "leaf_outcomes": leaf,
        "invalid_json_batches": sum(1 for k in kinds if k.get("leaf_outcomes", {}).get("invalid_json")),
        "parse_error": sum(k.get("parse_error", 0) for k in kinds),
        "non_object_reply": sum(k.get("non_object_reply", 0) for k in kinds),
        "items_skipped": sum(k.get("items_skipped", 0) for k in kinds),
        "items_skipped_attributed_to": sum(k.get("items_skipped_attributed_to", 0) for k in kinds),
        "lines_dropped": sum(k.get("lines_dropped", 0) for k in kinds),
        "batches_nothing_stored": len(nothing),
        "batches_nothing_stored_rate": round(len(nothing) / n, 4) if n else None,
        "batches_nothing_stored_ids": [r["batch_id"] for r in nothing],
        "windows_lost": sum(r.get("windows_lost", 0) for r in retain),
        "windows_lost_by_cause": {
            o: sum(x["windows"] for r in retain for x in (r.get("subs") or []) if x["outcome"] == o and x["added"] == 0)
            for o in LEAF_OUTCOMES if o != "ok"
        },
        "other_warnings": sum(k.get("other_warnings", 0) for k in kinds),
        "finish_length_attempts": sum(k.get("finish_length", 0) for k in kinds),
        "batch_ids": [r["batch_id"] for r in failed],
        "truncation_rule": TRUNCATION_RULE,
        "note": "parse_error = library could not json-parse the reply, or (extraction_output=lines) the reply opened no OBS / EP / DEC section "
        "(a truncated reply is counted under truncated_attempts and bisected, not as invalid JSON; under lines the arm hands a finish=length reply to the library as '' so nothing of it is stored — that '' logs one no-section warning, so parse_error counts it while the sub's outcome stays truncated); "
        "invalid_json = finish_reason stop but unparsable; schema_rejected_all = parsed but every item rejected by the library's schema; "
        "empty_extraction = parsed, zero items; lines_dropped = single lines the lines parser dropped (not schema rejections); all are status ok at the API and are billed",
    }


def _money(engine_extract: dict[str, Any], answer_stage: dict[str, Any], acc: dict[str, Any], windows: int, batches: int,
           extract_calls: list[Any], answer_calls: list[Any]) -> dict[str, Any]:
    extract_cost = sum(c.cost_usd or 0.0 for c in extract_calls)
    answer_cost = sum(c.cost_usd or 0.0 for c in answer_calls)
    total = extract_cost + answer_cost
    n, correct = acc["n"], acc["correct"]
    unpriced = engine_extract["unpriced_attempts"] + answer_stage["unpriced_attempts"]
    by_tier = {t: round(sum(c.cost_usd or 0.0 for c in extract_calls + answer_calls if c.rate_tier == t), 6) for t in ("peak", "offpeak")}
    cache_hit = sum(c.cache_hit_tokens or 0 for c in extract_calls + answer_calls)
    prompt = sum(c.prompt_tokens or 0 for c in extract_calls + answer_calls)
    return {
        "pricing": {"model": MODEL, "usd_per_1m": PRICES_USD_PER_M, "source": PRICING_SOURCE},
        "extract_cost_usd": round(extract_cost, 6),
        "answer_cost_usd": round(answer_cost, 6),
        "total_cost_usd": round(total, 6),
        "cost_by_tier_usd": by_tier,
        "peak_share": round(by_tier["peak"] / total, 4) if total else None,
        "cache_hit_share_of_prompt_tokens": round(cache_hit / prompt, 4) if prompt else None,
        "cost_per_question_usd": round(total / n, 6) if n else None,
        "answer_cost_per_question_usd": round(answer_cost / n, 6) if n else None,
        "cost_per_correct_usd": round(total / correct, 6) if correct else None,
        "retain_cost_per_window_usd": round(extract_cost / windows, 8) if windows else None,
        "retain_cost_per_batch_usd": round(extract_cost / batches, 6) if batches else None,
        "unpriced_attempts": unpriced,
        "unpriced_note": "attempts whose reply carried no usage (timeouts / transport errors) cost 未实测; counted in attempts, not in USD",
        "includes_failed_and_retried_attempts": True,
        "retain_shared_note": "all elite_* labels (elite_rows / elite_hybrid / optional elite_card) share one physical retain; its cost is shown in full under each label, not split; see ledger.cost_usd_total for the run total",
    }


def summarize(out: Path, manifest: dict[str, Any], meter: Meter, engines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Assemble the five-dimension summary from the in-memory results and the ledger.
    从内存结果与台账组装五维汇总。

    Input / 输入：
        out: Result directory.
        out：结果目录。

        manifest: Manifest (windows / batch counts).
        manifest：manifest（窗口 / 批次数）。

        meter: Ledger.
        meter：台账。

        engines: run_engine outputs keyed by engine.
        engines：以引擎为键的 run_engine 输出。

    Output / 输出：
        The summary dict (also written as summary.json / summary.md).
        汇总字典（同时写为 summary.json / summary.md）。

    State effect / 状态影响：
        Writes summary.json and summary.md.
        写 summary.json 与 summary.md。
    """
    windows = manifest["windows_retained"]
    batches = manifest["batches"]
    summary: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "manifest": {k: v for k, v in manifest.items() if k not in ("schedule", "batch_plan")},
        # The three EgoLife-ZH control variables are repeated at the top level so compare_real / report_real read them without the manifest.
        # EgoLife-ZH 的三个控制变量在顶层重复一份，使 compare_real / report_real 不必翻 manifest。
        "embedding": manifest.get("embedding"),
        "text_lang": manifest.get("text_lang", "en"),
        "cjk": manifest.get("cjk"),
        "engines": {},
        "arms": {},
    }
    for eng, res in engines.items():
        meter_eng = meter.summary(eng)
        retain = res["retain"]
        summary["engines"][eng] = {
            "stats": res["stats"],
            "wall_ms": res["wall_ms"],
            "retain": {
                "batches": len(retain),
                "adds": sum(r["adds"] for r in retain),
                "batches_ok": sum(1 for r in retain if r["ok"]),
                "batches_failed": sum(1 for r in retain if not r["ok"]),
                "windows_in_failed_batches": sum(
                    len(b["window_ids"]) for b in manifest["batch_plan"] if any(r["batch_id"] == b["batch_id"] and not r["ok"] for r in retain)
                ),
                "windows": sum(r["windows"] for r in retain),
                "windows_stored": sum(r["windows_stored"] for r in retain),
                "windows_lost": sum(r["windows_lost"] for r in retain),
                "truncated_attempts": sum(r["truncated_attempts"] for r in retain),
                "batches_bisected": sum(1 for r in retain if r["bisected"]),
                "truncated_parsed_added": sum(r["truncated_parsed_added"] for r in retain),
                "memories_added": sum(r["added"] for r in retain),
                "events": {k: sum(r["events"].get(k, 0) for r in retain) for k in sorted({k for r in retain for k in r["events"]})},
                # Guard-downgraded verdicts (retain_with_bisect 'downgraded'; {} for mem0 and for runs recorded before the key existed).
                # 被守卫降级的裁决（retain_with_bisect 的 'downgraded'；mem0 与该键出现之前记录的运行为 {}）。
                "downgraded": {k: sum(r.get("downgraded", {}).get(k, 0) for r in retain) for k in sorted({k for r in retain for k in r.get("downgraded", {})})},
                "stored_count": res["stats"].get("stored_count"),
                "count_note": "memories_added = add() results (ADD + UPDATE + NONE events); stored_count = distinct rows in the store at close",
                "batch_wall_ms": _dist([r["wall_ms"] for r in retain]),
                "batch_model_ms": _dist([r["model_ms"] for r in retain]),
                "batch_local_ms": _dist([r["local_ms"] for r in retain]),
                "add_model_ms": _dist([x["model_ms"] for r in retain for x in r["subs"]]),
                "failed": [{k: r[k] for k in ("batch_id", "position", "error", "statuses")} for r in retain if not r["ok"]],
                "extraction_parse_failures": _parse_failures(retain, {b["batch_id"]: len(b["window_ids"]) for b in manifest["batch_plan"]}),
                "truncation_rule": TRUNCATION_RULE,
            },
            "tokens": {"extract": meter_eng["extract"], "embedding": meter_eng["embedding"]},
            "organize_note": "no LLM call in recall / organize / compile on either engine: API tokens 0 (local only)",
        }
        for lab, recs in res["records"].items():
            acc = _accuracy(recs)
            by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in recs:
                by_type[r["type"]].append(r)
            lab_meter = meter.summary(lab)
            summary["arms"][lab] = {
                "engine": eng,
                "accuracy": acc,
                "by_type": {t: _accuracy(v) for t, v in sorted(by_type.items())},
                "failures": [
                    {"id": r["id"], "type": r["type"], "gold": r["gold"], "pred": r["pred"], "raw": r["raw_output"],
                     "category": r["error_category"], "hit_batch": r["hit_batch"], "returned": r["returned_count"], "gold_stored": bool(r["gold_batches_stored"])}
                    for r in recs if not r["correct"]
                ],
                "latency": _latency(recs),
                "tokens": {
                    "extract": meter_eng["extract"]["tokens"],
                    "organize": "本地（无 API token）",
                    "answer": lab_meter["answer"]["tokens"],
                    "embedding": meter_eng["embedding"],
                    "usage_source": "API-returned usage only (prompt_tokens / completion_tokens / prompt_cache_hit_tokens / prompt_cache_miss_tokens); None = 未实测",
                },
                "money": _money(meter_eng["extract"], lab_meter["answer"], acc, windows, batches, meter.calls_for(eng, "extract"), meter.calls_for(lab, "answer")),
                "stability": {
                    "extract": {k: meter_eng["extract"][k] for k in ("attempts", "requests", "retries", "by_status", "timeout_rate", "failure_rate", "empty_rate")},
                    "answer": {k: lab_meter["answer"][k] for k in ("attempts", "requests", "retries", "by_status", "timeout_rate", "failure_rate", "empty_rate", "latency_ms")},
                    "answer_empty_question_rate": round(sum(1 for r in recs if r["pred"] is None and r["answer_status"] in ("ok", "empty")) / len(recs), 4) if recs else None,
                    "answer_failed_question_rate": round(sum(1 for r in recs if r["answer_status"] in ("timeout", "http_error")) / len(recs), 4) if recs else None,
                    "retain_batches_failed": sum(1 for r in retain if not r["ok"]),
                    "extraction_parse_failures": sum(1 for r in retain if (r.get("extraction_parse") or {}).get("failed")),
                    "extraction_parse_failure_rate": round(sum(1 for r in retain if (r.get("extraction_parse") or {}).get("failed")) / len(retain), 4) if retain else None,
                    "extraction_truncated_attempts": sum(r["truncated_attempts"] for r in retain),
                    "extraction_truncation_rate_over_adds": round(sum(r["truncated_attempts"] for r in retain) / sum(r["adds"] for r in retain), 4) if retain else None,
                    "extraction_parse_detail": summary["engines"][eng]["retain"]["extraction_parse_failures"],
                    "cold_start_note": "cold_start = first API call of the label (HTTP connection shared with the extract stage, so the answer stage's first call is on a warm connection); store + fastembed init is stats.init_ms",
                },
                "card": (
                    {
                        "chars": _dist([r["card"]["chars"] for r in recs]),
                        "used_tokens": _dist([r["card"]["used_tokens"] for r in recs]),
                        "items": _dist([r["card"]["items"] for r in recs]),
                        "compile_ms": _dist([r["card"]["compile_ms"] for r in recs]),
                        "budget_tokens": res["stats"].get("compiler_budget_tokens"),
                        "over_budget_questions": sum(1 for r in recs if r["card"]["used_tokens"] > (res["stats"].get("compiler_budget_tokens") or 0)),
                        "budget_note": "budget is soft: evidence rows may push used_tokens over it (hms_elite/compiler/coverage.py over_budget_evidence); used_tokens is HMS's own estimator, not the API's prompt_tokens",
                    }
                    if recs and "card" in recs[0]
                    else None
                ),
                "hybrid": (
                    {
                        "chars": _dist([r["hybrid"]["chars"] for r in recs]),
                        "session_chars": _dist([r["hybrid"]["session_chars"] for r in recs]),
                        "rows_kept": _dist([r["hybrid"]["rows_kept"] for r in recs]),
                        "session_reused": sum(1 for r in recs if r["hybrid"]["session_reused"]),
                        "session_reused_share": round(sum(1 for r in recs if r["hybrid"]["session_reused"]) / len(recs), 4),
                        "session_compiles": sum(1 for r in recs if not r["hybrid"]["session_reused"]),
                        "session_budget_tokens": res["stats"].get("context_session_budget_tokens"),
                        "rows_top_k": res["stats"].get("context_rows_top_k"),
                        "prompt_order": "memories_first",
                        "note": "HMS.context(): session card = compile(latest stored message text, as_of=its timestamp) memoized per scope until a write; rows = search top_k deduplicated, time-sorted, '[YYYY-MM-DD HH:MM]' stamped; text placed before the question (CONTRACT §6.7)",
                    }
                    if recs and "hybrid" in recs[0]
                    else None
                ),
            }
    summary["ledger"] = {
        "attempts": len(meter.calls),
        "cost_usd_total": round(sum(c.cost_usd or 0.0 for c in meter.calls), 6),
        "cost_usd_by_tier": {t: round(sum(c.cost_usd or 0.0 for c in meter.calls if c.rate_tier == t), 6) for t in ("peak", "offpeak")},
        "cost_usd_by_stage": {st: round(sum(c.cost_usd or 0.0 for c in meter.calls if c.stage == st), 6) for st in STAGES},
        "percentile_rule": PERCENTILE_RULE,
        "unpriced_attempts": sum(1 for c in meter.calls if c.cost_usd is None),
        "usage_reconciles_all": all(c.usage_reconciles for c in meter.calls if c.usage_reconciles is not None),
        "attempts_with_cache_fields": sum(1 for c in meter.calls if c.cache_hit_tokens is not None),
        "by_tier": {t: sum(1 for c in meter.calls if c.rate_tier == t) for t in ("peak", "offpeak")},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "summary.md").write_text(markdown_tables(summary), encoding="utf-8")
    return summary


def markdown_tables(summary: dict[str, Any]) -> str:
    """Render the five dimensions as Markdown tables only.
    只把五个维度渲染为 Markdown 表格。

    Input / 输入：
        summary: summarize() output.
        summary：summarize() 的输出。

    Output / 输出：
        Markdown text.
        Markdown 文本。

    State effect / 状态影响：
        None.
        无。
    """
    arms = list(summary["arms"])
    L: list[str] = []

    def pct(v: Any) -> str:
        return "-" if v is None else f"{100 * v:.1f}%"

    def f(v: Any, nd: int = 1) -> str:
        if v is None:
            return "未实测"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return f"{v:,}"
        if isinstance(v, float):
            return f"{v:,.{nd}f}"
        return str(v)

    def usd(v: Any) -> str:
        return "未实测" if v is None else f"${v:.5f}"

    m = summary["manifest"]
    L.append(f"# EgoLife real-LLM run {summary['run_id']} — {m['model']} @ {m['base_url']}, concurrency 1, limit {m['limit']}")
    L.append("")
    L.append(f"## 1. 准确率 accuracy (denominator = all questions in --limit; invalid/empty/API failure = wrong; {FIRST50_RULE}; hit@k is retrieval only — {HIT_RULE})")
    L.append("")
    L.append("| arm | n | correct | accuracy | first-50 (file order) | first-50 (stream order) | valid-only acc (n) | acc on gold-stored subset (n; diagnostic) | hit@k batch (k median) | hit@k batch ±60s | no_evidence (gold unstored) | evidence_wrong | invalid | api_error | timeout |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        s = summary["arms"][a]["accuracy"]
        e = s["error_categories"]
        L.append(
            f"| {a} | {s['n']} | {s['correct']} | {pct(s['accuracy'])} | {s['correct_first50']}/{s['n_first50']} = {pct(s['accuracy_first50'])} | {s['correct_first50_stream']}/{s['n_first50_stream']} = {pct(s['accuracy_first50_stream'])} | "
            f"{pct(s['accuracy_valid_only'])} ({s['valid_n']}) | {pct(s['accuracy_gold_stored_subset'])} ({s['n_gold_stored']}) | {pct(s['hit_batch'])} (k={f(s['hit_k']['median'], 0)}) | {pct(s['hit_batch_wide'])} | "
            f"{e['no_evidence_retrieved']} ({s['no_evidence_gold_unstored']}) | {e['evidence_present_wrong_answer']} | {e['invalid_output']} | {e['api_error']} | {e['timeout']} |"
        )
    L.append("")
    types = sorted({t for a in arms for t in summary["arms"][a]["by_type"]})
    L.append("### by type (accuracy / hit@k batch, n)")
    L.append("")
    L.append("| type | " + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for t in types:
        cells = []
        for a in arms:
            s = summary["arms"][a]["by_type"].get(t)
            cells.append("-" if s is None else f"{s['correct']}/{s['n']} = {pct(s['accuracy'])} / {pct(s['hit_batch'])}")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### failures")
    L.append("")
    L.append("| arm | id | type | gold | pred | raw | category | hit@k batch | returned (k) | gold batch stored |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        for x in summary["arms"][a]["failures"]:
            raw = (x["raw"] or "").replace("|", "\\|").replace("\n", " ")[:40]
            L.append(f"| {a} | {x['id']} | {x['type']} | {x['gold']} | {x['pred']} | {raw} | {x['category']} | {x['hit_batch']} | {x['returned']} | {x['gold_stored']} |")
    L.append("")
    L.append(f"## 2. 延迟 latency ms (median / p95, {PERCENTILE_RULE}; model = API attempts incl. retries and bisected sub-adds, local = in-process; retain per timeline step; cold-start = first API call of the label, on the HTTP connection already opened by the extract stage)")
    L.append("")
    L.append("| arm | retain wall | retain model | retain local | recall local | context local | answer model | answer model (warm) | answer local | end-to-end | e2e excl. retain | cold-start q | answer retries |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        t = summary["arms"][a]["latency"]

        def d(x: dict[str, Any]) -> str:
            return f"{f(x['median'])} / {f(x['p95'])}"

        L.append(
            f"| {a} | {d(t['retain']['wall'])} | {d(t['retain']['model'])} | {d(t['retain']['local'])} | {d(t['recall_local'])} | {d(t['context_local'])} | "
            f"{d(t['answer_model'])} | {d(t['answer_model_excluding_cold_start'])} | {d(t['answer_local'])} | {d(t['end_to_end'])} | {d(t['end_to_end_excluding_retain'])} | "
            f"{t['cold_start_questions']} | {t['answer_retries']} |"
        )
    L.append("")
    L.append(f"### retain per batch (<=25 windows per batch; {TRUNCATION_RULE}; wall / model / local median, p95)")
    L.append("")
    L.append("| engine | batches | adds (incl. bisect) | ok | api failed | truncated attempts (finish=length) | batches bisected | invalid JSON (finish=stop) | schema-rejected-all | empty extraction | truncated unresolved | nothing stored (batches) | windows stored / lost | memories added (ADD/UPDATE/NONE) | stored rows | batch wall | batch model | per-add model | batch local | init ms | extract first-call ms |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for eng, e in summary["engines"].items():
        r = e["retain"]
        pf = r["extraction_parse_failures"]
        lo = pf["leaf_outcomes"]
        ev = r["events"]
        L.append(
            f"| {eng} | {r['batches']} | {r['adds']} | {r['batches_ok']} | {r['batches_failed']} | {r['truncated_attempts']} | {r['batches_bisected']} | {lo['invalid_json']} | {lo['schema_rejected_all']} | {lo['empty_extraction']} | {lo['truncated_unresolved']} | "
            f"{pf['batches_nothing_stored']} ({pct(pf['batches_nothing_stored_rate'])}) | {r['windows_stored']} / {r['windows_lost']} | {r['memories_added']} ({ev.get('ADD', 0)}/{ev.get('UPDATE', 0)}/{ev.get('NONE', 0)}) | {f(r['stored_count'])} | "
            f"{f(r['batch_wall_ms']['median'])} / {f(r['batch_wall_ms']['p95'])} | {f(r['batch_model_ms']['median'])} / {f(r['batch_model_ms']['p95'])} | {f(r['add_model_ms']['median'])} / {f(r['add_model_ms']['p95'])} | "
            f"{f(r['batch_local_ms']['median'])} / {f(r['batch_local_ms']['p95'])} | {f(e['stats'].get('init_ms'))} | {f(e['tokens']['extract']['latency_ms']['cold_start_ms'])} |"
        )
    L.append("")
    L.append("## 3. Token 成本 tokens (API-returned usage only; None = 未实测; embedding is local, no API token)")
    L.append("")
    L.append("| arm | stage | attempts | prompt | completion | cache hit | cache miss | reasoning (hidden) | attempts w/o usage | hit+miss==prompt |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        tk = summary["arms"][a]["tokens"]
        eng = summary["arms"][a]["engine"]
        ex = summary["engines"][eng]["tokens"]["extract"]
        an = summary["arms"][a]["stability"]["answer"]
        for stage, tokens, attempts in (("extract (engine)", tk["extract"], ex["attempts"]), ("answer", tk["answer"], an["attempts"])):
            L.append(
                f"| {a} | {stage} | {attempts} | {f(tokens['prompt'])} | {f(tokens['completion'])} | {f(tokens['cache_hit'])} | {f(tokens['cache_miss'])} | "
                f"{f(tokens.get('reasoning'))} | {tokens['attempts_without_usage']} | {tokens['usage_reconciles_all']} |"
            )
        L.append(f"| {a} | organize / compile | - | {tk['organize']} | | | | | | |")
        emb = tk["embedding"]
        L.append(f"| {a} | embedding (engine) | {emb['calls']} | {emb['api_tokens']}; texts={emb['texts']}; local_ms={f(emb['local_ms_total'])}; first-call ms={f(emb['cold_start_ms'])} | | | | | | |")
    L.append("")
    L.append("## 4. 金额与效率 money (USD; same price list; failed + retried + bisected attempts included; peak/offpeak per attempt send time; tier sums rounded after summing)")
    L.append("")
    L.append("| arm | extract | answer | total | peak / offpeak (peak share) | cache-hit share of prompt tokens | per question | answer per question | per correct | retain per window | retain per batch | unpriced attempts |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        mo = summary["arms"][a]["money"]
        per_window = "未实测" if mo["retain_cost_per_window_usd"] is None else f"${mo['retain_cost_per_window_usd']:.7f}"
        L.append(
            f"| {a} | {usd(mo['extract_cost_usd'])} | {usd(mo['answer_cost_usd'])} | {usd(mo['total_cost_usd'])} | "
            f"{usd(mo['cost_by_tier_usd']['peak'])} / {usd(mo['cost_by_tier_usd']['offpeak'])} ({pct(mo['peak_share'])}) | {pct(mo['cache_hit_share_of_prompt_tokens'])} | {usd(mo['cost_per_question_usd'])} | {usd(mo['answer_cost_per_question_usd'])} | "
            f"{usd(mo['cost_per_correct_usd'])} | {per_window} | "
            f"{usd(mo['retain_cost_per_batch_usd'])} | {mo['unpriced_attempts']} |"
        )
    lg = summary["ledger"]
    L.append(f"| run total (shared extract counted once) | {usd(lg['cost_usd_by_stage']['extract'])} | {usd(lg['cost_usd_by_stage']['answer'])} | {usd(lg['cost_usd_total'])} | {usd(lg['cost_usd_by_tier']['peak'])} / {usd(lg['cost_usd_by_tier']['offpeak'])} | | | | | | | {lg['unpriced_attempts']} |")
    L.append("")
    L.append("## 5. 稳定性 stability (rates over attempts; failed attempts kept)")
    L.append("")
    L.append("| arm | stage | attempts | requests | retries | ok | timeout | http_error | empty | timeout rate | failure rate | empty rate |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        st = summary["arms"][a]["stability"]
        for stage in ("extract", "answer"):
            s = st[stage]
            b = s["by_status"]
            L.append(
                f"| {a} | {stage}{' (engine)' if stage == 'extract' else ''} | {s['attempts']} | {s['requests']} | {s['retries']} | {b['ok']} | {b['timeout']} | {b['http_error']} | {b['empty']} | "
                f"{pct(s['timeout_rate'])} | {pct(s['failure_rate'])} | {pct(s['empty_rate'])} |"
            )
        pd = st["extraction_parse_detail"]
        lo = pd["leaf_outcomes"]
        L.append(
            f"| {a} | question level | | | | | | | | empty-answer q rate {pct(st['answer_empty_question_rate'])} | failed-answer q rate {pct(st['answer_failed_question_rate'])} | "
            f"retain batches failed {st['retain_batches_failed']}; extraction truncated (finish=length) {st['extraction_truncated_attempts']} attempts ({pct(st['extraction_truncation_rate_over_adds'])} of {pd['adds_total']} adds; {pd['batches_bisected']} batches bisected, {pd['batches_truncation_recovered']} fully recovered); "
            f"unrecovered parse failures {st['extraction_parse_failures']} batches ({pct(st['extraction_parse_failure_rate'])}: invalid JSON {lo['invalid_json']}, truncated at 1 window {lo['truncated_unresolved']}); "
            f"schema-rejected-all {lo['schema_rejected_all']}, empty extraction {lo['empty_extraction']}; items skipped by schema {pd['items_skipped']} (attributed_to {pd['items_skipped_attributed_to']}); "
            f"batches with nothing stored {pd['batches_nothing_stored']} ({pct(pd['batches_nothing_stored_rate'])}); windows lost {pd['windows_lost']} {json.dumps(pd['windows_lost_by_cause'])} |"
        )
    L.append("")
    if any(summary["arms"][a].get("hybrid") for a in arms):
        L.append("## elite_hybrid HMS.context() (median / p95; session card memoized per scope, recomputed after an ingest)")
        L.append("")
        L.append("| arm | chars | session chars | rows kept | session reused (share) | compiles | session budget | rows top_k |")
        L.append("|---|---|---|---|---|---|---|---|")
        for a in arms:
            h = summary["arms"][a].get("hybrid")
            if h:
                L.append(f"| {a} | {f(h['chars']['median'])} / {f(h['chars']['p95'])} | {f(h['session_chars']['median'])} / {f(h['session_chars']['p95'])} | {f(h['rows_kept']['median'])} / {f(h['rows_kept']['p95'])} | {h['session_reused']} ({pct(h['session_reused_share'])}) | {h['session_compiles']} | {f(h['session_budget_tokens'])} | {f(h['rows_top_k'])} |")
        L.append("")
    if any(summary["arms"][a].get("card") for a in arms):
        L.append("## elite_card compile() (median / p95; budget is soft — evidence rows may exceed it; used_tokens = HMS estimator)")
        L.append("")
        L.append("| arm | chars | used_tokens | budget | questions over budget | items | compile ms |")
        L.append("|---|---|---|---|---|---|---|")
        for a in arms:
            c = summary["arms"][a].get("card")
            if c:
                L.append(f"| {a} | {f(c['chars']['median'])} / {f(c['chars']['p95'])} | {f(c['used_tokens']['median'])} / {f(c['used_tokens']['p95'])} | {f(c['budget_tokens'])} | {c['over_budget_questions']} | {f(c['items']['median'])} / {f(c['items']['p95'])} | {f(c['compile_ms']['median'])} / {f(c['compile_ms']['p95'])} |")
        L.append("")
    L.append("## ledger")
    L.append("")
    L.append("| check | value |")
    L.append("|---|---|")
    for k, v in summary["ledger"].items():
        L.append(f"| {k} | {v} |")
    for k in ("windows", "windows_retained", "batches", "batch_size", "tier_rule", "retry_policy", "thinking", "prompt_template", "projected_full_cost"):
        if k in m:
            L.append(f"| {k} | {json.dumps(m[k], ensure_ascii=False) if isinstance(m[k], (dict, list)) else m[k]} |")
    L.append("")
    return "\n".join(L)


def projection(meter: Meter, engines: dict[str, dict[str, Any]], full_batches: int, full_questions: int,
               full_windows: int = 0, windows_measured: int = 0, questions_measured: int = 0) -> dict[str, Any]:
    """Projected full-run (204 questions) cost from this run's mean priced cost per extract batch and per answer.
    用本次运行的每抽取批次 / 每答题平均已计价费用外推全量（204 题）费用。

    Input / 输入：
        meter: Ledger.
        meter：台账。

        engines: run_engine outputs.
        engines：run_engine 输出。

        full_batches: Batch count of the full schedule.
        full_batches：全量调度的批次数。

        full_questions: 204.
        full_questions：204。

    Output / 输出：
        {'formula', 'inputs', 'per_engine_extract_usd', 'per_label_answer_usd', 'total_usd'}.
        {'formula', 'inputs', 'per_engine_extract_usd', 'per_label_answer_usd', 'total_usd'}。

    State effect / 状态影响：
        None.
        无。
    """
    inputs: dict[str, Any] = {"full_batches": full_batches, "full_questions": full_questions}
    extract: dict[str, float] = {}
    for eng in engines:
        calls = meter.calls_for(eng, "extract")
        batches = {c.batch_id for c in calls}
        cost = sum(c.cost_usd or 0.0 for c in calls)
        per_batch = cost / len(batches) if batches else 0.0
        inputs[f"{eng}_extract_usd_per_batch"] = round(per_batch, 6)
        inputs[f"{eng}_batches_measured"] = len(batches)
        extract[eng] = round(per_batch * full_batches, 4)
    answers: dict[str, float] = {}
    for eng, res in engines.items():
        for lab in res["records"]:
            calls = meter.calls_for(lab, "answer")
            qs = {c.question_id for c in calls}
            cost = sum(c.cost_usd or 0.0 for c in calls)
            per_q = cost / len(qs) if qs else 0.0
            inputs[f"{lab}_answer_usd_per_question"] = round(per_q, 6)
            inputs[f"{lab}_questions_measured"] = len(qs)
            answers[lab] = round(per_q * full_questions, 4)
    # window-ratio method (the requested projection): extract cost × full_windows / windows_measured; answer × full_q / q
    # 窗口比例法（要求的外推口径）：抽取费用 × 全量窗口 / 已覆盖窗口；答题 × 全量题数 / 已测题数
    extract_w: dict[str, float] = {}
    for eng in engines:
        cost = sum(c.cost_usd or 0.0 for c in meter.calls_for(eng, "extract"))
        inputs[f"{eng}_extract_usd_measured"] = round(cost, 6)
        extract_w[eng] = round(cost * full_windows / windows_measured, 4) if windows_measured else 0.0
    answers_w: dict[str, float] = {}
    for eng, res in engines.items():
        for lab in res["records"]:
            cost = sum(c.cost_usd or 0.0 for c in meter.calls_for(lab, "answer"))
            inputs[f"{lab}_answer_usd_measured"] = round(cost, 6)
            answers_w[lab] = round(cost * full_questions / questions_measured, 4) if questions_measured else 0.0
    inputs.update({"full_windows": full_windows, "windows_measured": windows_measured, "questions_measured": questions_measured})
    return {
        "formula": "total = Σ_engine(measured extract USD × full_windows / windows_measured) + Σ_label(measured answer USD × full_questions / questions_measured); "
        "priced attempts incl. retries and failures; tier = the tiers observed here (not re-priced)",
        "inputs": inputs,
        "per_engine_extract_usd": extract_w,
        "per_label_answer_usd": answers_w,
        "total_usd": round(sum(extract_w.values()) + sum(answers_w.values()), 4),
        "by_batch": {
            "formula": "Σ_engine(mean USD per extract batch × full_batches) + Σ_label(mean USD per answered question × full_questions)",
            "per_engine_extract_usd": extract,
            "per_label_answer_usd": answers,
            "total_usd": round(sum(extract.values()) + sum(answers.values()), 4),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry.
    命令行入口。

    Input / 输入：
        argv: Arguments (None -> sys.argv[1:]).
        argv：参数（None -> sys.argv[1:]）。

    Output / 输出：
        Exit code 0.
        退出码 0。

    State effect / 状态影响：
        Writes the result directory; network calls.
        写结果目录；网络调用。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=204)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--engines", default=",".join(ENGINE_NAMES), help="comma-separated: mem0_real,elite")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--run-id", default=None)
    # elite only: IngestConfig.extraction_output, default = the library default (DEFAULT_EXTRACTION_OUTPUT); recorded in
    # manifest['elite'] / 仅 elite：IngestConfig.extraction_output，默认 = 库默认值（DEFAULT_EXTRACTION_OUTPUT）；记入 manifest['elite']
    parser.add_argument("--extraction-output", choices=EXTRACTION_OUTPUTS, default=DEFAULT_EXTRACTION_OUTPUT)
    # elite only: also answer under elite_card (the per-question compile() card; default label before 2026-09-18)
    # 仅 elite：同时在 elite_card（逐题 compile() 卡片；2026-09-18 之前的默认标签）下答题
    parser.add_argument("--with-card", action="store_true")
    # EgoLife-ZH (docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md): the parallel data_zh dataset needs --text-lang zh (windows.build_windows
    # swaps the text slots; boundaries and ids stay those of the English data), a Chinese embedder and, for elite, the
    # CONTRACT §9.6 switches. Defaults reproduce the English runs exactly (manifest gains three keys only).
    # EgoLife-ZH（docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md）：平行的 data_zh 数据集需要 --text-lang zh（windows.build_windows 交换文本槽位；
    # 边界与 id 仍是英文数据的）、中文嵌入器，以及 elite 的 CONTRACT §9.6 开关。默认值精确复现英文运行（manifest 只多三个键）。
    parser.add_argument("--text-lang", choices=("en", "zh"), default="en")
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--embed-dims", type=int, default=None, help=f"default {EMBEDDING_DIMS} for the default model; otherwise resolved from fastembed without a download")
    parser.add_argument("--cjk", choices=("off", "on"), default=None, help="elite only: off = text.cjk_tokens=none + zh_rules=False; on = bigram + zh_rules=True; omitted = library defaults")
    parser.add_argument("--fastembed-cache-dir", type=Path, default=DEFAULT_FASTEMBED_CACHE_DIR)
    parser.add_argument("--entity-names", default=None, help="elite only: comma-separated closed proper-name list for the rule extractor; default = the translate_zh glossary names when --text-lang zh AND --cjk on (run B), none otherwise (run A keeps the untouched rule@entities-v1 extractor); '' = none")
    args = parser.parse_args(argv)

    if args.embed_dims is None:
        args.embed_dims, dims_source = resolve_embed_dims(args.embed_model)
    else:
        dims_source = "flag"
    # The glossary names belong to the Chinese support under test (run B); run A is the English-centric pipeline with
    # every CJK switch off, so it gets no name list unless --entity-names says so (EGOLIFE_ZH_PLAN §4).
    # 术语表人名属于受测的中文支持（运行 B）；运行 A 是全部 CJK 开关关闭的英文中心流水线，除非 --entity-names 指定，否则
    # 不带人名表（EGOLIFE_ZH_PLAN §4）。
    entity_names = default_entity_names(args.text_lang, args.cjk, args.entity_names)
    cache_dir = str(args.fastembed_cache_dir) if args.fastembed_cache_dir else None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        # The flag is the documented control: it overrides a FASTEMBED_CACHE_PATH already in the shell so mem0's provider
        # and EliteRealArm read the same directory (the manifest records what both actually use).
        # 标志是文档化的控制项：覆盖 shell 里已有的 FASTEMBED_CACHE_PATH，使 mem0 的提供者与 EliteRealArm 读同一目录
        # （manifest 记录两者实际使用的值）。
        os.environ["FASTEMBED_CACHE_PATH"] = cache_dir

    load_api_key(ENV_PATH)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    work_root = out / "work"
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in engines:
        if e not in ENGINE_LABELS:
            raise SystemExit(f"unknown engine {e!r}; choose from {ENGINE_NAMES}")

    qa_path = args.data_root / QA_FILENAME
    all_questions = load_questions(qa_path)
    if args.limit <= 0 or args.limit > len(all_questions):
        raise SystemExit(f"--limit must be in 1..{len(all_questions)}")
    questions: list[Question] = all_questions[: args.limit]
    transcripts = load_transcripts(args.data_root)
    windows: list[Window] = build_windows(transcripts, text_lang=args.text_lang)
    windows_sha = write_windows_jsonl(windows, out / "windows.jsonl")
    steps = schedule(questions, windows)
    batches = plan_batches(steps, args.batch_size)
    full_steps = schedule(all_questions, windows)
    full_batches = len(plan_batches(full_steps, args.batch_size))
    batch_plan = [{"batch_id": b.batch_id, "position": b.position, "window_ids": b.window_ids} for b in batches]
    (out / "batches.json").write_text(json.dumps(batch_plan, ensure_ascii=False), encoding="utf-8")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": MODEL,
        "base_url": BASE_URL,
        "key_source": f"{ENV_PATH} (variable name DEEPSEEK_API_KEY; value never written)",
        "pricing_usd_per_1m": PRICES_USD_PER_M,
        "pricing_source": PRICING_SOURCE,
        "tier_rule": f"peak = Mon..Fri UTC {PEAK_WINDOWS_UTC} half-open hours; else offpeak; judged per attempt at send time",
        "retry_policy": {"timeout_s": TIMEOUT_S, "max_retries": MAX_RETRIES, "backoff_s": list(BACKOFF_S), "retry_on": sorted(RETRY_STATUSES) + ["timeout", "connection"], "concurrency": 1, "sdk_retries": 0},
        "extract_params": {
            "temperature": EXTRACT_TEMPERATURE, "max_tokens": EXTRACT_MAX_TOKENS, "top_p": "library default 0.1 (both)",
            # The lines contract (CONTRACT §6.6) is plain text: LLMBase.generate_lines sends no response_format at all.
            # 行式契约（CONTRACT §6.6）是纯文本：LLMBase.generate_lines 完全不发送 response_format。
            "response_format": (
                "mem0: json_object; elite: none (extraction_output=lines, LLMBase.generate_lines sends no response_format)"
                if args.extraction_output == "lines" else "json_object (both libraries' own prompts)"
            ),
        },
        "thinking": {
            "extra_body": {"thinking": THINKING},
            "applies_to": "every attempt of every arm and stage (injected by DeepSeekClient.create)",
            "probe_2026_09_10": "default: max_tokens=8 -> content '' reasoning_tokens=8 finish=length; disabled: content 'B', completion_tokens=1, finish=stop",
            "evidence": "calls.jsonl reasoning_tokens per attempt (expected 0 / null)",
        },
        "answer_params": {"temperature": ANSWER_TEMPERATURE, "max_tokens": ANSWER_MAX_TOKENS, "messages": "single user message", "parse": "their parse_choice (exactly one distinct A-D)"},
        "prompt_template": PROMPT_TEMPLATE,
        "their_protocol": their_protocol_digest(),
        "embedding": {
            "provider": "fastembed (both)", "model": args.embed_model, "dims": args.embed_dims, "dims_source": dims_source, "cache_dir": cache_dir,
            "fastembed_cache_path_env": os.environ.get("FASTEMBED_CACHE_PATH"),
            "note": (
                "local ONNX; no API token; mem0 and hms_elite providers previously verified bit-identical"
                if args.embed_model == EMBED_MODEL else "local ONNX; no API token; non-default model (mem0 / hms_elite parity not re-verified)"
            ),
        },
        # Absolute, so a results directory read from elsewhere (report_real / compare_real) names the dataset
        # unambiguously; the default is already absolute, so the English manifest is unchanged.
        # 绝对路径，使从别处读取结果目录（report_real / compare_real）时数据集无歧义；默认值本就是绝对路径，英文 manifest 不变。
        "data_root": str(args.data_root.resolve()),
        "text_lang": args.text_lang,
        "cjk": args.cjk,
        "library_switches": {"text.cjk_tokens": (None if args.cjk is None else ("bigram" if args.cjk == "on" else "none")), "ingest.zh_rules": (None if args.cjk is None else args.cjk == "on")},
        "entity_names": list(entity_names),
        "mem0": {"version": "2.0.20", "vector_store": "milvus-lite (local file), COSINE + built-in BM25 hybrid", "infer": True, "search": {"top_k": TOP_K, "threshold": THRESHOLD}},
        "elite": {"index": "sqlite", "extraction_mode": "single", "extraction_output": args.extraction_output, "infer": True, "search": {"top_k": TOP_K, "threshold": THRESHOLD, "reference_date": "query_time"}, "compile": {"as_of": "query_time", "context": "ContextIR.to_text()"}, "context": {"method": "HMS.context()", "as_of": "query_time (rows only)", "prompt_order": "memories_first", "top_k": TOP_K}},
        "qa_file": str(qa_path),
        "qa_sha256": _sha256_file(qa_path),
        "questions_in_file": len(all_questions),
        "limit": args.limit,
        "question_ids": [q.id for q in questions],
        "transcript_files": len(transcript_files(args.data_root)),
        "windows": len(windows),
        "windows_retained": sum(len(s.new_windows) for s in steps),
        "windows_sha256": windows_sha,
        "window_rule": {"max_chars": MAX_CHARS, "max_span_s": MAX_SPAN_CS / 100, "text": "english lines only" if args.text_lang == "en" else "chinese lines of the english-bearing cues (boundaries / ids from the english chars; windows_sha256 differs from the english runs by design)"},
        "gold_rule": {
            "primary": f"window contains target_time (half width {GOLD_HALF_WIDTH_CS / 100:g} s); hit judged at batch granularity",
            "secondary_nb": f"window overlaps ±{GOLD_NEIGHBOURHOOD_CS / 100:g} s (hit_batch_wide)",
        },
        "cutoff_rule": "same day: window visible iff end_cs <= query_time; earlier days: whole day; later days: invisible (timeline.py)",
        "truncation_rule": TRUNCATION_RULE,
        "first50_rule": FIRST50_RULE,
        "hit_rule": HIT_RULE,
        "percentile_rule": PERCENTILE_RULE,
        "batch_size": args.batch_size,
        "batches": len(batches),
        "batches_full_204": full_batches,
        "batch_plan": batch_plan,
        "bank": BANK,
        "engines": engines,
        "labels": [lab for e in engines for lab in ENGINE_LABELS[e] + ((CARD_LABEL,) if args.with_card and e == "elite" else ())],
        "with_card": bool(args.with_card),
        "excluded_future": sum(1 for s in steps if s.excluded_future),
        "schedule": [{"position": s.position, "id": s.question.id, "day": s.question.query_day, "time": s.question.query_hhmmsscc, "new": len(s.new_windows), "retained": s.retained_count} for s in steps],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    meter = Meter()
    client = DeepSeekClient(meter)
    results: dict[str, dict[str, Any]] = {}
    for eng in engines:
        print(f"[{run_id}] engine {eng}: {len(steps)} questions, {len(batches)} batches ({manifest['windows_retained']} windows)", flush=True)
        res = run_engine(
            eng, steps, batches, client=client, meter=meter, work_root=work_root, out=out, keep_db=args.keep_db, extraction_output=args.extraction_output, with_card=args.with_card,
            embed_model=args.embed_model, embed_dims=args.embed_dims, cjk=args.cjk, fastembed_cache_dir=cache_dir, entity_names=entity_names,
        )  # fmt: skip
        results[eng] = res
        arm_file = {"engine": eng, "run_id": run_id, "finished_utc": datetime.now(timezone.utc).isoformat(), "stats": res["stats"], "wall_ms": res["wall_ms"], "meter": meter.summary(eng), "labels": {lab: meter.summary(lab) for lab in res["records"]}}
        (out / f"arm_{eng}.json").write_text(json.dumps(arm_file, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        meter.write_jsonl(out / "calls.jsonl")  # flush after every engine / 每个引擎结束后落盘
        print(f"  done {eng}: wall={res['wall_ms']} ms stats={json.dumps({k: res['stats'][k] for k in ('stored_count', 'batches_ok', 'batches_failed')})}", flush=True)
    if not args.keep_db:
        shutil.rmtree(work_root, ignore_errors=True)
    meter.write_jsonl(out / "calls.jsonl")
    manifest["projected_full_cost"] = projection(
        meter, results, full_batches, len(all_questions),
        full_windows=len(windows), windows_measured=manifest["windows_retained"], questions_measured=len(steps),
    )
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(out, manifest, meter, results)
    print(f"[{run_id}] wrote {out}: attempts={summary['ledger']['attempts']} cost_usd={summary['ledger']['cost_usd_total']} "
          f"acc={ {a: summary['arms'][a]['accuracy']['accuracy'] for a in summary['arms']} }", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
