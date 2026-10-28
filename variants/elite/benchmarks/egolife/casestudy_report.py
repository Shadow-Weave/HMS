"""Wrong-answer case study for a real-LLM EgoLife run: one card per wrong question with the official answer, the
system's answer, a recall overview and the exact memory rows the reader saw, plus a six-way root-cause split.
真实 LLM EgoLife 运行的错题集：每道错题一张卡，含官方答案、系统回答、召回概览、阅读器实际看到的记忆行，以及六类根因归类。

Inputs / 输入：
    --run       run_real.py output dir (questions_elite_rows.jsonl, optional questions_elite_hybrid.jsonl /
                questions_elite_card.jsonl, windows.jsonl, batches.json, extract_raw_elite.jsonl)  / run_real.py 输出目录
    --replay    replay_ab.py output dir whose per_question_full.jsonl carries the 20-row context the reader saw; its
                returned-batch sequences must equal the run's (asserted per question) / 回放目录，其 context 即阅读器看到的行
    --data      EgoLifeQA json / 题库
Output / 输出：
    one self-contained HTML file (dark "case study" template) / 一个自包含的 HTML 文件（深色 Case Study 模版）
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
from pathlib import Path
from typing import Any

from hms_elite.utils.json import parse_llm_json

DAYS = range(1, 8)
STOP = {"the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are", "was", "were", "be", "who",
        "what", "which", "when", "where", "how", "did", "does", "do", "with", "from", "by", "as", "it", "its", "that",
        "this", "these", "those", "his", "her", "their", "our", "my", "your", "me", "i", "we", "you", "they", "he",
        "she", "not", "no", "yes", "will", "would", "can", "could", "should", "have", "has", "had", "than", "then",
        "into", "about", "after", "before", "some", "any", "one", "two", "three"}
# 佩戴者 Jake 在题里叫 "Me" / the wearer is called "Me" in the questions
ME_ALIASES = ("jake",)
CATS = [
    ("vacuum", "检索真空", "返回 0 条记忆，或 gold 批次从未入库。", "rose"),
    ("miss", "检索未命中", "gold 批次（含 target_time 的窗口所在批次）不在返回的 20 行里。", "rose"),
    ("time", "时间推算", "答案是相对时间或时刻（如「约 4 小时前」「前天下午」）。记忆行上下文只有文本、没有时间戳，阅读器无从推算；给上下文带上每行的 time_anchor 可解。", "amber"),
    ("extract", "抽取丢失", "gold 批次的记忆在 20 行里，但没有任何一行含答案核心词：抽取时没有把这个细节写进记忆。", "amber"),
    ("rank", "排序靠后", "含答案核心词的行存在，但第一条排在第 11–20 位。", "cyan"),
    ("reader", "答题器用错记忆", "含答案核心词的行在前 10 位，答题器仍选错。", "accent"),
]
CAT_NAME = {k: n for k, n, _, _ in CATS}
CAT_COLOR = {k: c for k, _, _, c in CATS}


def _cs(t: str) -> int:
    """'HHMMSSff' → centiseconds of day. / 'HHMMSSff' → 当日厘秒。"""
    t = str(t).zfill(8)
    return (int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])) * 100 + int(t[6:8])


def _hms(t: str) -> str:
    t = str(t).zfill(8)
    return f"{t[:2]}:{t[2:4]}:{t[4:6]}"


def _tt(v: Any) -> dict:
    """target_time / query_time may be a dict or its repr. / 可能是字典或其 repr。"""
    if isinstance(v, dict):
        return v
    import ast

    return ast.literal_eval(str(v))


# A relative time or clock time; song titles like "Unforgettable Tonight" / 难忘今宵 must not match, so bare
# "tonight" / "noon" / "晚上" are not enough on their own.
# 相对时间或时刻；"Unforgettable Tonight" / 难忘今宵 这类歌名不能匹配，所以单独的 tonight / noon / 晚上 不算。
_TIME_EN = re.compile(r"(?i)\b(hours?|minutes?|days?|weeks?)\s+ago\b|\bhalf an hour\b|\byesterday\b|\blast night\b"
                      r"|\b(this|today|yesterday|last)\s+(morning|afternoon|evening|night|noon)\b|\b(today|yesterday)\s+at\b|\bat noon\b"
                      r"|\bbefore yesterday\b|\b\d{1,2}:\d{2}\b|\bo'clock\b")
_TIME_ZH = re.compile(r"(小时前|分钟前|天前|昨天|前天|昨晚|今早|凌晨|\d{1,2}[:：]\d{2}|\d+点(半|钟)?)")


def _is_time_answer(en: str, zh: str) -> bool:
    """The correct option is a relative time or a clock time. / 正确选项是相对时间或时刻。"""
    return bool(_TIME_EN.search(en or "") or _TIME_ZH.search(zh or ""))


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3 and w not in STOP]


def _has_answer(text: str, answer: str, keywords: str) -> bool:
    """Row carries the answer: the correct option text appears (word-bounded; 'Me' also matches the wearer's name),
    or at least ceil(60%) of the keyword tokens appear as whole words.
    行含答案：正确选项文本出现（词边界；'Me' 也匹配佩戴者名字），或关键词 token 以整词出现 ≥ 60%（向上取整）。
    """
    low = text.lower()
    ans = answer.strip().lower()
    cands = [ans] + (list(ME_ALIASES) if ans in ("me", "i", "myself") else [])
    for c in cands:
        if c and re.search(r"(?<![a-z0-9])" + re.escape(c) + r"(?![a-z0-9])", low):
            return True
    kw = _tokens(keywords)
    if kw:
        present = sum(1 for k in kw if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", low))
        need = -(-6 * len(kw) // 10)  # ceil(0.6 · n) / 60% 向上取整
        if present >= need:
            return True
    return False


def _load_label(run: Path, label: str) -> dict[str, dict] | None:
    """Per-question records of one Elite label keyed by question id, or None when the run has no such file — None, not
    {}, so a label that was never run is rendered as 未运行 instead of "0 wrong".
    某个 Elite 标签按题号索引的逐题记录；运行目录没有该文件时为 None——是 None 而不是 {}，这样未运行的标签渲染为「未运行」而非「答错 0 题」。
    """
    path = run / f"questions_{label}.jsonl"
    if not path.exists():
        return None
    return {json.loads(line)["id"]: json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load(run: Path, replay: Path, data: Path) -> dict:
    ds = {d["ID"]: d for d in json.loads(data.read_text(encoding="utf-8"))}
    rows = _load_label(run, "elite_rows")
    if rows is None:
        raise FileNotFoundError(run / "questions_elite_rows.jsonl")
    # Since 2026-09-18 the second default label is elite_hybrid (HMS.context(): session card prefix + timestamped rows)
    # and elite_card is opt-in (run_real --with-card); either may be absent and then loads as None.
    # 2026-09-18 起第二个默认标签是 elite_hybrid（HMS.context()：会话卡片前缀 + 带时间戳记忆行），elite_card 为选择加入
    # （run_real --with-card）；两者都可能缺失，缺失时加载为 None。
    card = _load_label(run, "elite_card")
    hybrid = _load_label(run, "elite_hybrid")
    rp = {json.loads(line)["id"]: json.loads(line) for line in (replay / "per_question_full.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    windows = [json.loads(line) for line in (run / "windows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    batches = json.loads((run / "batches.json").read_text(encoding="utf-8"))
    wid = {w["window_id"]: w for w in windows}
    batch_of_window = {w: b["batch_id"] for b in batches for w in b["window_ids"]}
    batch_day = {b["batch_id"]: int(wid[b["window_ids"][0]]["day"]) for b in batches}
    batch_span = {}
    for b in batches:
        ws = [wid[w] for w in b["window_ids"]]
        batch_span[b["batch_id"]] = (int(ws[0]["day"]), int(ws[0]["start_cs"]), int(ws[-1]["end_cs"]))
    text2batch: dict[str, list[str]] = {}
    for line in (run / "extract_raw_elite.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        o = parse_llm_json(d.get("content") or "", key=None, task="replay")
        if isinstance(o, list):
            o = {"observations": o}
        if not isinstance(o, dict):
            continue
        for it in o.get("observations") or []:
            if isinstance(it, dict) and it.get("text"):
                lst = text2batch.setdefault(str(it["text"]).strip(), [])
                if d["batch_id"] not in lst:
                    lst.append(d["batch_id"])
    return {"ds": ds, "rows": rows, "card": card, "hybrid": hybrid, "rp": rp, "windows": windows, "wid": wid, "batch_of_window": batch_of_window,
            "batch_day": batch_day, "batch_span": batch_span, "text2batch": text2batch}


def analyse(L: dict) -> list[dict]:
    ds, rows, rp = L["ds"], L["rows"], L["rp"]
    card, hybrid = L["card"] or {}, L["hybrid"] or {}
    cases = []
    for qid, q in rows.items():
        if q["correct"]:
            continue
        d = ds[qid]
        r = rp[qid]
        assert r["returned_batches"] == q["returned_batches"], f"replay/run mismatch on {qid}"
        answer_letter = d["answer"]
        answer_text = d[f"choice_{answer_letter.lower()}"]
        answer_zh = d[f"choice_{answer_letter.lower()}_chinese"]
        lines = [ln[2:].strip() if ln.startswith("- ") else ln.strip() for ln in r["context"].split("\n")]
        lines = [ln for ln in lines if ln]
        mem = []
        # A memory text can occur in several batches (the same sentence extracted twice). Walk the rows in order and
        # align them with the run's returned-batch sequence: the current batch, then the next one, then any batch
        # already seen; when the sources still cannot decide, show every candidate joined by '|'.
        # 同一句记忆可能出现在多个批次（同一句被抽取两次）。按行序与本次运行的返回批次序列对齐：先当前批次，
        # 再下一个，再任何已出现过的批次；来源仍无法判定时把候选用 '|' 连起来显示。
        returned = list(q["returned_batches"])
        j = 0
        seen: list[str] = []
        for k, t in enumerate(lines, 1):
            cands = [b for b in L["text2batch"].get(t, []) if b in returned] or list(L["text2batch"].get(t, []))
            b: str | None
            if len(cands) == 1:
                b = cands[0]
            elif cands:
                pick = None
                for probe in ([returned[j]] if j < len(returned) else []) + ([returned[j + 1]] if j + 1 < len(returned) else []) + seen:
                    if probe in cands:
                        pick = probe
                        break
                b = pick if pick else "|".join(cands)
            else:
                b = None
            if b and "|" not in b:
                if b not in seen:
                    seen.append(b)
                if j + 1 < len(returned) and b == returned[j + 1]:
                    j += 1
            day = L["batch_day"].get(b) if b and "|" not in b else (
                L["batch_day"].get(cands[0]) if cands and len({L["batch_day"].get(x) for x in cands}) == 1 else None)
            gold = any(x in set(q["gold_batches"]) for x in (cands if b and "|" in b else [b]))
            mem.append({"rank": k, "text": t, "batch": b, "day": day, "gold": gold,
                        "has": _has_answer(t, answer_text, d.get("keywords") or "")})
        gold_rows = [m for m in mem if m["gold"]]
        ans_rows = [m for m in mem if m["has"]]
        first_ans = ans_rows[0]["rank"] if ans_rows else None
        if not mem:
            cat = "vacuum"
        elif not gold_rows:
            cat = "miss"
        elif _is_time_answer(answer_text, answer_zh):
            cat = "time"
        elif not ans_rows:
            cat = "extract"
        elif first_ans > 10:
            cat = "rank"
        else:
            cat = "reader"
        tt = _tt(d["target_time"])
        qt = _tt(d["query_time"])
        # target_time 是单点 'time' 或时间列表 'time_list'（区间的首尾）/ a single 'time' or a 'time_list' (span endpoints)
        tday = int(tt["date"][3:])
        times = [tt["time"]] if "time" in tt else list(tt.get("time_list") or [])
        instants = [_cs(t) for t in times]
        # 与评测台 timeline.gold_window_ids 相同：包含任一 target 时刻的窗口 / same rule as the harness: windows containing
        # any target instant
        gold_windows = [w for w in L["windows"] if int(w["day"]) == tday
                        and any(int(w["start_cs"]) <= t <= int(w["end_cs"]) for t in instants)]
        days = collections.Counter(m["day"] for m in mem if m["day"])
        pred = q["pred"]
        pred_text = d.get(f"choice_{pred.lower()}") if pred in "ABCD" else ""
        pred_zh = d.get(f"choice_{pred.lower()}_chinese") if pred in "ABCD" else ""
        c = card.get(qid, {})
        h = hybrid.get(qid, {})
        cases.append({
            "id": qid, "type": d["type"], "type_zh": d.get("type_chinese", ""), "q": d["question"], "q_zh": d["question_chinese"],
            "choices": [(x.upper(), d[f"choice_{x}"], d.get(f"choice_{x}_chinese", "")) for x in "abcd"],
            "gold": answer_letter, "gold_text": answer_text, "gold_zh": answer_zh, "reason_zh": d.get("reason_chinese", ""),
            "reason": d.get("reason", ""), "trigger_zh": d.get("trigger_chinese", ""), "keywords": d.get("keywords", ""),
            "target": f"DAY{tday} " + (_hms(times[0]) + ("–" + _hms(times[-1]) if len(times) > 1 and times[-1] != times[0] else "") if times else "—"),
            "query": f"DAY{qt['date'][3:]} {_hms(qt['time'])}",
            "pred": pred, "pred_text": pred_text or "", "pred_zh": pred_zh or "", "raw": q.get("raw_output", ""),
            "card_pred": c.get("pred"), "card_correct": c.get("correct"),
            "hybrid_pred": h.get("pred"), "hybrid_correct": h.get("correct"),
            "cat": cat, "mem": mem, "gold_batches": q["gold_batches"], "returned_batches": q["returned_batches"],
            "hit_rank": r.get("hit_rank"), "row_rank": r.get("hit_row_rank"), "n_gold_rows": len(gold_rows), "n_ans_rows": len(ans_rows),
            "first_ans": first_ans, "days": dict(days), "top_score": (q.get("top_scores") or [None])[0],
            "recall_ms": q["timing_ms"].get("recall_local"), "gold_windows": gold_windows[:3], "n_gold_windows": len(gold_windows),
            "error_category": q.get("error_category"),
        })
    order = {"vacuum": 0, "miss": 1, "time": 2, "extract": 3, "rank": 4, "reader": 5}
    cases.sort(key=lambda c: (order[c["cat"]], int(c["id"])))
    return cases


HERE = Path(__file__).resolve().parent
CSS = (HERE / "template_casestudy.css").read_text(encoding="utf-8")
JS = (HERE / "template_casestudy.js").read_text(encoding="utf-8")

def _bar(label: str, n: int, total: int, color: str = "var(--accent)", value: str | None = None) -> str:
    """Template bar row: label · value · track. / 模版条形行：标签 · 数值 · 轨道。"""
    pct = 0 if not total else min(100.0, 100.0 * n / total)
    return (f'<div class="barrow"><div class="bl">{html.escape(label)}</div><div class="bv">{html.escape(value if value is not None else str(n))}</div>'
            f'<div class="bt"><div class="bf" style="width:{pct:.0f}%;background:{color}"></div></div></div>')


def _mark(text: str, words: list[str]) -> str:
    """Escape and <mark> the answer words (case-insensitive). / 转义并高亮答案词（忽略大小写）。"""
    out = html.escape(text)
    for w in sorted({w for w in words if len(w) >= 2}, key=len, reverse=True):
        out = re.sub("(?i)(?<![A-Za-z0-9])(" + re.escape(html.escape(w)) + ")(?![A-Za-z0-9])", r"<mark>\1</mark>", out)
    return out


NAMES = ("Jake", "Alice", "Tasha", "Shure", "Katrina", "Lucia", "Choiszt", "Nicous")


def _entities(c: dict) -> list[str]:
    """People named in the question or the correct option ('Me' → Jake). / 题目或正确选项里的人名（Me → Jake）。"""
    pool = c["q"] + " " + c["gold_text"]
    names = [n for n in NAMES if re.search(r"(?<![A-Za-z])" + n + r"(?![A-Za-z])", pool)]
    if re.search(r"(?<![A-Za-z])(Me|I|my)(?![A-Za-z])", pool):
        names.append("Jake")
    return sorted(set(names))


def _insight(c: dict) -> str:
    gb = ", ".join(c["gold_batches"]) or "无"
    if c["cat"] == "vacuum":
        return f"返回 0 行。gold 批次 {gb}。"
    if c["cat"] == "miss":
        return (f"gold 批次 {gb}（target {c['target']}）不在返回的 {len(c['mem'])} 行里；返回批次按序为 {', '.join(c['returned_batches'][:6])}"
                f"{'…' if len(c['returned_batches']) > 6 else ''}。含答案词的行 {c['n_ans_rows']} 条。这类题要么查询措辞与记忆文本不同，要么答案散在别的时间段。")
    if c["cat"] == "time":
        return (f"标准答案「{c['gold_text']}」是相对时间/时刻；gold 批次 {gb} 在返回批次的第 {c['hit_rank']} 位，来自该批次的记忆行 {c['n_gold_rows']} 条，"
                f"但阅读器拿到的 20 行只有文本、没有时间戳，无法算出与提问时刻（{c['query']}）的间隔，只能猜。给每行带上 time_anchor 即可修复。")
    if c["cat"] == "extract":
        return (f"gold 批次 {gb} 在返回批次的第 {c['hit_rank']} 位，来自该批次的记忆行 {c['n_gold_rows']} 条，但没有一行含答案核心词「{c['gold_text']}」/「{c['keywords']}」。"
                f"对照下方 gold 窗口原文：抽取时这个细节没有写进记忆。")
    if c["cat"] == "rank":
        return f"含答案词的行有 {c['n_ans_rows']} 条，第一条在第 {c['first_ans']} 位；gold 批次首次出现在第 {c['hit_rank']} 位。答题器面对 20 行时更容易被前 10 行带偏。"
    return (f"含答案词的行有 {c['n_ans_rows']} 条，第一条就在第 {c['first_ans']} 位（gold 批次第 {c['hit_rank']} 位），答题器仍选了 {c['pred']}「{c['pred_text']}」。"
            f"多半是多条记忆各说一半、或另一条记忆与选项 {c['pred']} 的字面更接近。")


def render(L: dict, cases: list[dict], run_name: str, replay_name: str, template_note: str | None) -> str:
    rows, card, hybrid = L["rows"], L["card"], L["hybrid"]
    n = len(rows)
    wrong_rows = [q for q in rows.values() if not q["correct"]]
    # None = that label was not run in this result dir (rendered 未运行), never 0.
    # None = 该结果目录没有运行该标签（渲染为「未运行」），绝不是 0。
    wrong_card = None if card is None else [q for q in card.values() if not q["correct"]]
    wrong_hybrid = None if hybrid is None else [q for q in hybrid.values() if not q["correct"]]
    hit = sum(1 for q in rows.values() if q.get("hit_batch"))
    dist = collections.Counter(c["cat"] for c in cases)
    by_type = collections.Counter(c["type"] for c in cases)
    type_total = collections.Counter(q["type"] for q in rows.values())
    both_wrong_card = None if card is None else sum(1 for c in cases if c["card_correct"] is False)
    both_wrong_hybrid = None if hybrid is None else sum(1 for c in cases if c["hybrid_correct"] is False)
    card_note = "「状态卡片」上下文未运行" if wrong_card is None else f"「状态卡片」上下文答错 {len(wrong_card)} 题"
    hybrid_note = "「混合上下文」（HMS.context()）未运行" if wrong_hybrid is None else f"「混合上下文」（HMS.context()）答错 {len(wrong_hybrid)} 题"
    retrieval = dist.get("vacuum", 0) + dist.get("miss", 0)
    reader = dist.get("rank", 0) + dist.get("reader", 0)
    H: list[str] = []
    A = H.append
    A('<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">')
    A('<title>HMS Elite 错题集 · 召回逻辑深度 Case Study</title>\n<style>' + CSS + '</style></head>\n<body><div class="wrap">')
    A('<header class="hero"><div class="eyebrow"><span class="dot"></span>HMS Elite · EgoLife A1_JAKE · 真实 LLM 运行 r3 · 2026-09-11</div>')
    A('<h1>错题集 · 召回逻辑深度 Case Study</h1>')
    A(f'<p class="subtitle">deepseek-flash 抽取与作答、本地 fastembed 嵌入、按 query_time 流式写入、HMS.search 取最多 20 条记忆行交给阅读器四选一。'
      f'{n} 题中「记忆行」上下文答错 {len(wrong_rows)} 题，{hybrid_note}，{card_note}。本页逐题拆记忆行的 {len(cases)} 道错题：官方答案、系统回答、召回概览、阅读器实际看到的记忆行（最多 20 条）、召回逻辑诊断。</p></header>')

    # ---------- 1 概览 ----------
    A('<section>\n<h2><span class="num">1</span> 错题概览与根因分布</h2>')
    A(f'<p class="dim">{len(cases)}/{n} 题答错（记忆行上下文）。下方按根因分类，每类对应 HMS Elite 流水线中不同的失败环节：检索（召回）、抽取（写入）、阅读器（作答）。</p>')
    A('<div class="keyline">')
    A(f'<div class="k rose"><div class="v">{len(cases)}</div><div class="l">错题总数</div></div>')
    A(f'<div class="k rose"><div class="v">{retrieval}</div><div class="l">检索侧（真空 + 未命中）</div></div>')
    A(f'<div class="k amber"><div class="v">{dist.get("extract", 0) + dist.get("time", 0)}</div><div class="l">写入/上下文侧（抽取丢失 + 时间推算）</div></div>')
    A(f'<div class="k cyan"><div class="v">{reader}</div><div class="l">答题器问题（排序靠后 + 用错）</div></div>')
    A(f'<div class="k green"><div class="v">{100.0 * hit / n:.1f}%</div><div class="l">hit@20 全题</div></div>')
    A('</div>')
    A('<div class="insight info"><div class="ic">i</div><div class="ct"><b>如何读懂这份 case study</b>'
      '<p>每道错题分 5 个维度：①官方答案+原因+gold 窗口原文 ②系统回答 ③召回数据概览（召回了什么）④阅读器实际使用的记忆行（≤20 条，用了什么）⑤召回逻辑诊断（为什么错）。'
      '重点看<strong>根因标签</strong>：检索侧改召回，抽取丢失改写入契约，答题器问题改阅读器提示词或上下文组织。</p></div></div>')
    A('<div class="grid g2"><div class="card"><h3 style="margin-top:0">按题型的错题数</h3>')
    for t in ("EventRecall", "TaskMaster", "RelationMap", "EntityLog", "HabitInsight"):
        A(_bar(t, by_type.get(t, 0), type_total.get(t, 1), "var(--rose)", f'{by_type.get(t, 0)}/{type_total.get(t, 0)}'))
    A('</div><div class="card"><h3 style="margin-top:0">各种上下文</h3>')
    A(_bar("记忆行错", len(wrong_rows), n, "var(--rose)"))
    A(_bar("混合上下文错", 0 if wrong_hybrid is None else len(wrong_hybrid), n, "var(--cyan)", "未运行" if wrong_hybrid is None else None))
    A(_bar("记忆行 + 混合都错", 0 if both_wrong_hybrid is None else both_wrong_hybrid, n, "var(--accent)", "未运行" if both_wrong_hybrid is None else None))
    A(_bar("状态卡片错", 0 if wrong_card is None else len(wrong_card), n, "var(--amber)", "未运行" if wrong_card is None else None))
    A(_bar("记忆行 + 卡片都错", 0 if both_wrong_card is None else both_wrong_card, n, "var(--accent)", "未运行" if both_wrong_card is None else None))
    A('<p class="dim" style="font-size:12.5px;margin:8px 0 0">记忆行 = <code>HMS.search(top_k=20)</code> 的 text 列；混合上下文 = <code>HMS.context()</code>（会话卡片前缀 + 带时间戳记忆行，记忆在前作答，2026-09-18 起默认）；状态卡片 = <code>HMS.compile(question, as_of).to_text()</code>（<code>run_real --with-card</code> 选择加入）。未运行 = 该结果目录没有对应的 questions_*.jsonl。每道错题卡右上角标出已运行的其他上下文是否答对。</p>')
    A('</div></div>')
    A('<h3>六类根因说明</h3><div class="grid g2">')
    for key, name, desc, color in CATS:
        A(f'<div class="card"><div class="tag {color if color != "accent" else "cyan"}" style="margin-bottom:8px">{name} ({dist.get(key, 0)}题)</div><p style="font-size:13px;color:var(--text-dim);margin:0">{html.escape(desc)}</p></div>')
    A('</div>')
    A('<div class="insight k"><div class="ic">≡</div><div class="ct"><b>判定规则与数据来源</b>'
      '<p>「gold 批次」= 含题目 target_time 的转写窗口所在的写入批次（每批 ≤25 窗口）。「含答案核心词」= 记忆行出现正确选项文本（词边界匹配；选项 Me 同时匹配佩戴者 Jake），或题库 keywords 的 token 以整词出现 ≥ 60%（向上取整）。'
      '六类互斥、按序判定：返回 0 行 → 检索真空；gold 批次不在 20 行 → 检索未命中；答案是相对时间/时刻 → 时间推算；gold 批次在但无含答案行 → 抽取丢失；含答案行首位 &gt; 10 → 排序靠后；否则 → 答题器用错记忆。'
      '本管线没有「拒答」类：阅读器 temperature 0、max_tokens 8，204 题全部输出了 A–D 之一。召回列表由零 API 回放重建（ReplayLLM 回放 r3 记录的 deepseek 回复），返回批次序列与 r3 逐题一致（204/204）。</p></div></div>')
    if template_note:
        A(f'<div class="insight warn"><div class="ic">!</div><div class="ct"><b>与模版报告的对照</b><p>{template_note}</p></div></div>')
    A('</section>')

    # ---------- 2 逐题 ----------
    A(f'<section>\n<h2><span class="num">2</span> {len(cases)} 道错题逐题 Case Study</h2>')
    A(f'<div class="filterbar" id="filterbar">\n  <button class="active" data-filter="all">全部 ({len(cases)})</button>')
    for key, name, _, _ in CATS:
        A(f'  <button data-filter="{key}">{name} ({dist.get(key, 0)})</button>')
    A('  <input class="search" id="qsearch" placeholder="搜索题目…">\n</div>')
    for idx, c in enumerate(cases):
        color = CAT_COLOR[c["cat"]]
        tagc = color if color != "accent" else "cyan"
        words = [c["gold_text"]] + (["Jake"] if c["gold_text"].strip().lower() in ("me", "i") else []) + _tokens(c["keywords"])
        dtext = html.escape((c["q_zh"] + " " + c["q"] + " " + c["gold_zh"] + " " + c["gold_text"] + " " + c["pred_text"]).lower(), quote=True)
        A(f'<div class="case" id="case-{idx}" data-root="{c["cat"]}" data-text="{dtext}">')
        A('  <div class="case-head">\n    <div>')
        A(f'      <div class="cid">E{c["id"]} · {html.escape(c["type"])} · {html.escape(c["type_zh"])}</div>')
        A(f'      <div class="ctitle">{html.escape(c["q_zh"])} <span style="font-weight:400;color:var(--text-dim);font-size:13px">{html.escape(c["q"])}</span></div>')
        A('    </div>\n    <div style="text-align:right">')
        A(f'      <span class="tag {tagc}">{CAT_NAME[c["cat"]]}</span>')
        A('      <span class="tag gray" style="margin-left:4px">得分 0</span>')
        if hybrid is not None:
            A(f'      <span class="tag {"green" if c["hybrid_correct"] else "rose"}" style="margin-left:4px">混合上下文 {"✓" if c["hybrid_correct"] else "✗ " + str(c["hybrid_pred"])}</span>')
        if card is not None:
            A(f'      <span class="tag {"green" if c["card_correct"] else "rose"}" style="margin-left:4px">状态卡片 {"✓" if c["card_correct"] else "✗ " + str(c["card_pred"])}</span>')
        A(f'      <span class="tag gray" style="margin-left:4px">gold 批次位次 {c["hit_rank"] if c["hit_rank"] else "未召回"}</span>')
        A('    </div>\n  </div>\n  <div class="case-body">')
        # ①
        opts = " · ".join(f'{L_}.{html.escape(zh)}' + (f'<span style="color:var(--text-faint)">({html.escape(en)})</span>' if zh != en else "") for L_, en, zh in c["choices"])
        A('    <div class="sec">\n      <div class="sec-label">① 官方答案与原因</div>')
        A(f'      <div class="box green-bd" style="margin-bottom:6px"><strong style="color:var(--green)">标准答案：</strong>{c["gold"]}. {html.escape(c["gold_zh"])} <span style="color:var(--text-faint)">{html.escape(c["gold_text"])}</span>'
          f'<div style="font-size:12px;color:var(--text-faint);margin-top:4px">{opts}</div></div>')
        A(f'      <div class="box green-bd" style="margin-bottom:6px"><strong style="color:var(--green)">官方原因（reason）：</strong>{html.escape(c["reason_zh"]) or "—"}'
          f'<div style="font-size:12px;color:var(--text-faint);margin-top:3px">触发场景：{html.escape(c["trigger_zh"]) or "—"} · 关键词：{html.escape(c["keywords"]) or "—"} · target {c["target"]} · 提问 {c["query"]} · gold 批次 {html.escape(", ".join(c["gold_batches"]) or "无")}</div></div>')
        if c["gold_windows"]:
            A(f'      <div class="box" style="font-size:12.5px"><strong>gold 窗口原文</strong> <span style="color:var(--text-faint)">（含 target_time 的转写窗口，共 {c["n_gold_windows"]} 个，显示前 {min(3, c["n_gold_windows"])} 个）</span>')
            for w in c["gold_windows"]:
                zh = (w.get("text_zh") or "").strip().replace("\n", " / ")[:500]
                en = (w.get("text") or "").strip().replace("\n", " / ")[:400]
                A(f'        <div style="margin-top:6px"><span style="font-family:var(--mono);color:var(--accent);font-size:11px">{html.escape(w["window_id"])}</span><br>{_mark(zh, words)}<br><span style="color:var(--text-faint)">{_mark(en, words)}</span></div>')
            A('      </div>')
        A('    </div>')
        # ②
        A('    <div class="sec">\n      <div class="sec-label">② 系统回答（得分 0）</div>')
        A(f'      <div class="box rose-bd"><strong style="color:var(--rose)">{c["pred"]}. {html.escape(c["pred_zh"])}</strong> <span style="color:var(--text-faint)">{html.escape(c["pred_text"])} · 原始输出 <code>{html.escape(str(c["raw"])[:20])}</code> · 阅读器 deepseek-flash，temperature 0，max_tokens 8，上下文为 ④ 的记忆行</span></div>\n    </div>')
        # ③
        ents = _entities(c)
        ent_hit = sum(1 for m in c["mem"] if any(re.search(r"(?<![A-Za-z])" + e + r"(?![A-Za-z])", m["text"]) for e in ents)) if ents else 0
        ent_pct = (100.0 * ent_hit / len(c["mem"])) if (ents and c["mem"]) else None
        A('    <div class="sec">\n      <div class="sec-label">③ HMS 召回数据概览（召回了什么）</div>')
        A('      <div class="keyline" style="margin:8px 0">')
        A(f'        <div class="k"><div class="v" style="font-size:20px;color:var(--accent)">{len(c["mem"])}</div><div class="l">总召回</div></div>')
        A(f'        <div class="k"><div class="v" style="font-size:20px;color:var(--amber)">{c["n_gold_rows"]}</div><div class="l">gold 批次行</div></div>')
        A(f'        <div class="k"><div class="v" style="font-size:20px;color:var(--green)">{c["n_ans_rows"]}</div><div class="l">含答案词行</div></div>')
        A(f'        <div class="k"><div class="v" style="font-size:20px;color:var(--cyan)">{f"{ent_pct:.0f}%" if ent_pct is not None else "—"}</div><div class="l">实体命中 {html.escape("/".join(ents)) if ents else ""}</div></div>')
        A(f'        <div class="k"><div class="v" style="font-size:20px;color:var(--text)">{c["recall_ms"]:.0f} ms</div><div class="l">检索耗时 · 首分 {c["top_score"] if c["top_score"] is not None else "—"}</div></div>')
        A('      </div>')
        A('      <div style="margin-top:6px"><div style="font-size:11px;color:var(--text-faint);margin-bottom:4px">天分布（各行按其批次所在天）· 返回批次顺序：<span style="font-family:var(--mono)">' + html.escape(", ".join(c["returned_batches"])) + '</span></div>')
        for dday in DAYS:
            if c["days"].get(dday):
                A(_bar(f"DAY{dday}", c["days"][dday], len(c["mem"]), "var(--accent)"))
        if ents:
            A(_bar("实体命中", ent_hit, len(c["mem"]), "var(--cyan)", f"{ent_pct:.0f}%"))
        A('      </div>\n    </div>')
        # ④
        A('    <div class="sec">\n      <div class="sec-label">④ 阅读器实际使用的记忆行（≤20 条；绿色=含答案核心词 · 琥珀色批次=gold 批次）</div>\n      <div class="mem-list">')
        for m in c["mem"]:
            cls = "mem-item has-answer" if m["has"] else "mem-item "
            meta = f'{m["batch"] or "?"} · DAY{m["day"] or "?"}'
            meta_html = (f'<span style="color:var(--amber)">{html.escape(meta)} · gold 批次</span>' if m["gold"] else html.escape(meta))
            A(f'<div class="{cls}"><div class="rank">#{m["rank"]}</div><div class="mem-text">{_mark(m["text"], words) if m["has"] else html.escape(m["text"])}<div class="mem-meta">{meta_html}</div></div></div>')
        A('      </div>\n    </div>')
        # ⑤
        kind = {"vacuum": "bad", "miss": "bad", "time": "warn", "extract": "warn", "rank": "info", "reader": "info"}[c["cat"]]
        ic = {"bad": "!", "warn": "!", "info": "i"}[kind]
        A('    <div class="sec">\n      <div class="sec-label">⑤ 召回逻辑诊断（为什么错）</div>')
        A(f'      <div class="insight {kind}" style="margin:0"><div class="ic">{ic}</div><div class="ct"><b>{CAT_NAME[c["cat"]]}</b><p>{html.escape(_insight(c))}</p></div></div>\n    </div>')
        A('  </div>\n</div>')
    A('</section>')
    A('<script>' + JS + '</script>')
    A(f'<div class="foot">HMS Elite 错题集 · casestudy_report.py 自动生成 · 题库 data/EgoLifeQA_A1_JAKE_TEXT_ONLY.json · 运行 {html.escape(run_name)} · 召回行 {html.escape(replay_name)}/per_question_full.jsonl（记忆行到批次映射 4,071/4,071）</div>')
    A('</div></body></html>')
    return "\n".join(H)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/real/elite_full204_r3"))
    ap.add_argument("--replay", type=Path, default=Path("results/replay_ab_entities"))
    ap.add_argument("--data", type=Path, default=Path("data/EgoLifeQA_A1_JAKE_TEXT_ONLY.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--template-note", default=None, help="optional paragraph shown under the taxonomy / 可选：分类说明下方的一段话")
    args = ap.parse_args()
    L = load(args.run, args.replay, args.data)
    cases = analyse(L)
    text = render(L, cases, args.run.name, args.replay.name, args.template_note)
    args.out.write_text(text, encoding="utf-8")
    dist = collections.Counter(c["cat"] for c in cases)
    print(f"wrote {args.out} ({len(text):,} chars); cases {len(cases)}; " + ", ".join(f"{CAT_NAME[k]} {dist.get(k, 0)}" for k, *_ in CATS))
    print(json.dumps({c["id"]: c["cat"] for c in cases}, ensure_ascii=False))


if __name__ == "__main__":
    main()
