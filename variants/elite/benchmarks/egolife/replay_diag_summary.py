"""Collect the replay-variant grid (results/replay_diag/<name>/summary.json) into one table, check that W0 reproduces
the real compact run (per question and per batch) and V0 the real r3 run, and write results/replay_diag/summary.md.
把回放变体网格（results/replay_diag/<name>/summary.json）汇成一张表，核对 W0 是否复现真实紧凑 run（逐题、逐批）、
V0 是否复现真实 r3 run，并写出 results/replay_diag/summary.md。
"""

# Purpose (EN): One place that states, with provenance, the hit@20 / MRR / atoms of every variant and the exact command
#               that produced it, plus the reproduction gap of the two anchors (W0 vs real compact, V0 vs real r3).
# 目的（中）：在一处带出处地给出每个变体的 hit@20 / MRR / 原子数及其精确命令，以及两个锚点的复现差距
#            （W0 vs 真实紧凑 run，V0 vs 真实 r3）。

# Input (EN): results/replay_diag/<name>/{summary.json, per_question_<arm>.jsonl, retain_<arm>.jsonl, <name>.cmd};
#             results/real/{elite_compact_full204, elite_full204_r3}/{questions_elite_rows.jsonl, retain_elite.jsonl}.
# 输入（中）：results/replay_diag/<name>/{summary.json, per_question_<arm>.jsonl, retain_<arm>.jsonl, <name>.cmd}；
#            results/real/{elite_compact_full204, elite_full204_r3}/{questions_elite_rows.jsonl, retain_elite.jsonl}。

# Output (EN): results/replay_diag/summary.md and summary_grid.json.
# 输出（中）：results/replay_diag/summary.md 与 summary_grid.json。

# State effect (EN): Reads only; writes the two summary files. No API calls.
# 状态影响（中）：只读；写两个汇总文件。无 API 调用。

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DIAG = HERE / "results" / "replay_diag"
REAL = HERE / "results" / "real"
VARIANTS: list[tuple[str, str, str, str]] = [
    # name, arm, replies source, description / 名称、臂、回复来源、描述
    ("W0", "compact", "elite_compact_full204", "compact replies as recorded (anchor; real run hit@20 141/204)"),
    ("W1", "compact", "elite_compact_full204", "compact replies, --drop-entities (entities := [])"),
    ("W2", "compact", "elite_compact_full204", "compact replies, --force-type fact (every type -> fact)"),
    ("W3", "compact", "elite_compact_full204", "compact replies, --drop-when (when / valid_to removed)"),
    ("W4", "compact", "elite_compact_full204", "compact replies, --keep-only-text (id/event_idx/text/type only)"),
    ("V0", "full", "elite_full204_r3", "r3 full replies, full projection (anchor; real run hit@20 161/204)"),
    ("V2", "compact", "elite_full204_r3", "r3 full replies, compact projection, --drop-entities"),
    ("V3", "compact", "elite_full204_r3", "r3 full replies, compact projection, --force-type fact"),
    # Extra axis found during the diagnosis / 诊断中发现的额外轴
    ("W5", "compact", "elite_compact_full204", "compact replies, --force-add (every recorded decision := ADD)"),
    ("W6", "compact", "elite_compact_full204", "compact replies, --keep-only-text --force-add (texts only, everything stored)"),
]


INTERPRETATION = """
1. None of the four requested field axes moves hit@20. With the compact model's texts held fixed (W-series, 7,315
   observations, 288 replayable batches): entities := [] gives 141 (W1, 0), type := fact gives 141 (W2, 0), when/valid_to
   removed gives 142 (W3, +1), text-only items give 142 (W4, +1). The same holds on r3's texts (V-series): entities := []
   gives 160 (V2, -1), type := fact gives 159 (V3, -2). MRR moves by at most 0.006 in either series. The FTS tokens
   come from text only and the embedding is of the text, so the type / entity / when fields are not on the retrieval
   path that hit@20 measures; the entity axis and the event/state split are effectively inert for this metric.
2. The single transformation that moves hit@20 is outside the four axes: forcing every recorded DECISION to ADD
   (W5) lifts the compact replay from 141 to 154 (+13; MRR 0.387 -> 0.504, now above V0's 0.497) and the live atoms
   from 5,717 to 7,309. The compact model's replies carry 1,685 NONE verdicts (r3: 23) — 1,623 of them with
   target_id == id, in 156 of 288 batches exactly observations 0..9 with the reason "duplicate of existing memory"
   — i.e. the compact prompt makes the model pair observation i with candidate i by index and call it a duplicate.
   The library treats a NONE identically under both contracts (the atom is not stored; the target gets a support
   witness), so 1,595 stored-atom losses (21.8 % of what was extracted) are a prompt-induced decision artifact, not
   a field-derivation cost. Text-only items with all-ADD (W6) give 155, the same picture.
3. W0 vs V0 (different texts, each through its own pipeline) = 20 questions. Decomposition: 13 from the decisions
   block (W5 - W0); 3 questions (ids 230, 366, 472) whose only gold batch is b0108 or b0237, the two compact replies
   the parser reads as a bare list (0 atoms in the real run and in every W variant); the remaining 4 net (V0 161 vs
   W5 154 is 21 V0-only against 14 W5-only) is the compact TEXT content — 7,315 observations vs 8,470 (1.51 vs
   1.75 per window), a different but not uniformly worse set of retrievable statements. TYPES, ENTITIES and WHEN
   fields contribute nothing measurable.
4. Consequence for the compact contract: the observation schema is not the problem; the decisions instruction of
   the compact SYSTEM prompt is. The user prompt's Candidate Existing Memories rows come from the same builder in
   both contracts (build_memory_extraction_prompt), so the model saw the same candidates; only the compact system
   prompt's decision wording / example differs, and under it deepseek-flash index-pairs observation i with candidate i.
   Fixing that wording (or ignoring NONE verdicts whose target_id merely equals the observation id) should recover
   ~13 of the 20 questions without touching extraction fields. The two non-object replies are a separate defect: in
   each, one item omits the `type` key (a bare "constraint" / "temporal_fact" value), the object is unparsable and
   parse_llm_json salvages a list, which the library then discards as "not a JSON object".
1. 四个被要求的字段轴都不改变 hit@20。文本固定为紧凑模型的输出（W 系列，7,315 条观察、288 个可回放批次）：
   entities := [] 得 141（W1，0）、type := fact 得 141（W2，0）、去掉 when/valid_to 得 142（W3，+1）、只留文本得 142
   （W4，+1）。在 r3 的文本上同样成立（V 系列）：entities := [] 得 160（V2，-1）、type := fact 得 159（V3，-2）。
   两个系列 MRR 至多变动 0.006。FTS 词元只来自文本、嵌入也只对文本，类型 / 实体 / 时间字段不在 hit@20 度量的检索路径上。
2. 唯一能移动 hit@20 的变换在四个轴之外：把每条记录的决策强制为 ADD（W5）使紧凑回放从 141 升到 154（+13；MRR
   0.387 -> 0.504，已高于 V0 的 0.497），存活原子从 5,717 升到 7,309。紧凑模型的回复带 1,685 个 NONE 裁决（r3 为 23），
   其中 1,623 个 target_id == id，156/288 个批次恰好把观察 0..9 判为 "duplicate of existing memory"——紧凑提示词让
   模型按下标把观察 i 与候选 i 配对并称其重复。库对 NONE 的处理在两种契约下相同（不存原子、给目标加支持见证），
   因此 1,595 个原子的损失（占抽取量 21.8 %）是提示词诱发的决策伪影，不是字段推导的代价。
3. W0 vs V0（不同文本、各走各的管线）= 20 题。分解：13 题来自决策块（W5 - W0）；3 题（230、366、472）的金标批次
   只在 b0108 / b0237，即被解析器读成裸列表的两条紧凑回复（真实 run 与所有 W 变体均为 0 原子）；余下净 4 题
   （V0 161 vs W5 154，21 题 V0 独中对 14 题 W5 独中）来自紧凑模型的文本内容——7,315 vs 8,470 条观察（每窗 1.51 vs
   1.75），是一组不同但并非一致更差的可检索陈述。类型、实体、时间字段没有可测的贡献。
4. 对紧凑契约的结论：观察 schema 不是问题，紧凑系统提示词的决策指令才是。两种契约的用户提示词候选行来自同一个构造器
   （build_memory_extraction_prompt），模型看到的候选相同；不同的只是紧凑系统提示词的决策措辞 / 示例，deepseek-flash 在
   其下把观察 i 与候选 i 按下标配对。修这段措辞（或忽略 target_id 恰等于观察 id 的 NONE）应能在不动抽取字段的前提下
   找回约 13 题。两条非对象回复是另一个缺陷：各有一条条目漏掉 `type` 键（裸的 "constraint" / "temporal_fact" 值），
   对象无法解析，parse_llm_json 抢救出一个列表，库随后以"不是 JSON 对象"丢弃。
"""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file (blank lines skipped). / 读 JSONL（跳过空行）。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def variant_row(name: str, arm: str) -> dict[str, Any] | None:
    """The numbers of one variant, or None when its summary is not there yet.
    一个变体的数字；其汇总尚不存在时为 None。
    """
    path = DIAG / name / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    a = s["arms"][arm]
    cmd_path = DIAG / f"{name}.cmd"
    cmd = cmd_path.read_text(encoding="utf-8").strip() if cmd_path.exists() else ""
    cmd = cmd.split(": ", 1)[1] if ": " in cmd else cmd
    return {
        "name": name, "arm": arm, "hits": a["hits"], "n_scored": a["n_scored"], "hit20": a["hit20"], "mrr": a["mrr"],
        "mrr_row_rank": a["mrr_row_rank"], "atoms": a["atoms"], "atoms_added": a["atoms_added_sum"],
        "atoms_with_entities": a["atoms_with_entities"], "entity_ids_per_atom": a["entity_ids_per_atom_mean"],
        "leaves_replayed": a["leaves_replayed"], "leaves_skipped": a.get("leaves_invalid_json_skipped", 0),
        "items_skipped": a["items_skipped"], "by_type": a["by_type_hit20"], "by_type_n": a["by_type_n"],
        "counters": (a.get("replay") or {}).get("reduction_counters") or {}, "transforms": (s.get("variant") or {}).get("transforms"),
        "command": cmd, "out_dir": str(DIAG / name),
    }  # fmt: skip


def reproduction(name: str, arm: str, real_run: str) -> dict[str, Any] | None:
    """Per-question and per-batch comparison of one replay arm with the real run it replays.
    一个回放臂与其所回放的真实 run 的逐题与逐批比较。
    """
    pq = DIAG / name / f"per_question_{arm}.jsonl"
    if not pq.exists():
        return None
    replay = {r["id"]: r for r in _jsonl(pq)}
    real = {r["id"]: r for r in _jsonl(REAL / real_run / "questions_elite_rows.jsonl")}
    ids = [i for i in real if i in replay]
    both = sum(1 for i in ids if real[i]["hit_batch"] and replay[i]["hit_batch"])
    real_only = [i for i in ids if real[i]["hit_batch"] and not replay[i]["hit_batch"]]
    replay_only = [i for i in ids if replay[i]["hit_batch"] and not real[i]["hit_batch"]]
    same_returned = sum(1 for i in ids if real[i]["returned_batches"] == replay[i]["returned_batches"])
    real_added = {r["batch_id"]: r["added"] for r in _jsonl(REAL / real_run / "retain_elite.jsonl")}
    replay_added = {r["batch_id"]: r["added"] for r in _jsonl(DIAG / name / f"retain_{arm}.jsonl")}
    batches = sorted(set(real_added) | set(replay_added))
    diff_batches = [(b, real_added.get(b), replay_added.get(b)) for b in batches if real_added.get(b) != replay_added.get(b)]
    return {
        "real_run": real_run, "real_hits": sum(1 for i in ids if real[i]["hit_batch"]),
        "replay_hits": sum(1 for i in ids if replay[i]["hit_batch"]), "hit_both": both,
        "hit_real_only": real_only, "hit_replay_only": replay_only, "same_returned_batches": same_returned, "n": len(ids),
        "real_atoms_added": sum(real_added.values()), "replay_atoms_added": sum(replay_added.values()),
        "batches_with_different_added": diff_batches,
    }  # fmt: skip


def main() -> int:
    """Build the grid and write summary.md / summary_grid.json. / 构建网格并写 summary.md / summary_grid.json。"""
    rows = {name: variant_row(name, arm) for name, arm, _src, _d in VARIANTS}
    repro = {"W0": reproduction("W0", "compact", "elite_compact_full204"), "V0": reproduction("V0", "full", "elite_full204_r3")}
    lines = [
        "# Replay-variant diagnosis of the compact-contract loss (zero API)",
        "# 紧凑契约损失的回放变体诊断（零 API）",
        "",
        "Same 204 EgoLife questions / 4,851 windows / 290 batches of 25 (batches.json of elite_full204_r3 = elite_compact_full204, verified byte-equal). "
        "hit@20 = a returned row's batch (25-window grid) is a gold batch; MRR = 1/rank of the first gold batch in the deduplicated returned-batch order; "
        "scored n = 204 (no question excluded). HMS: extraction_mode single, extraction_output per arm, fastembed BAAI/bge-small-en-v1.5 (384), sqlite index, "
        "top_k 20, threshold 0.1, reference_date = query time. Replay: benchmarks/egolife/replay_ab.py (ReplayLLM keyed by window texts), driver replay_diag_run.sh.",
        "",
        "| name | replies | what is fed to HMS | hit@20 (n=204) | MRR | MRR row-rank | atoms live | atoms added | atoms with entity axis | leaves replayed / skipped | out dir |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, arm, src, desc in VARIANTS:
        r = rows[name]
        if r is None:
            lines.append(f"| {name} | {src} | {desc} | (not run) | | | | | | | |")
            continue
        lines.append(
            f"| {name} | {src} | {desc} | {r['hits']}/{r['n_scored']} ({100 * r['hit20']:.1f}%) | {r['mrr']} | {r['mrr_row_rank']} | {r['atoms']} | "
            f"{r['atoms_added']} | {r['atoms_with_entities']} | {r['leaves_replayed']} / {r['leaves_skipped']} | `results/replay_diag/{name}` |"
        )
    lines += ["", "## Deltas (hit@20 in questions) / 差值", ""]
    w0, v0 = rows.get("W0"), rows.get("V0")
    if w0 and v0:
        lines.append(f"- V0 - W0 (full replies vs compact replies, each through its own pipeline) = {v0['hits'] - w0['hits']}")
    for name in ("W1", "W2", "W3", "W4", "W5", "W6"):
        if rows.get(name) and w0:
            lines.append(f"- {name} - W0 = {rows[name]['hits'] - w0['hits']} (MRR {rows[name]['mrr'] - w0['mrr']:+.4f})")
    for name in ("V2", "V3"):
        if rows.get(name) and v0:
            lines.append(f"- {name} - V0 = {rows[name]['hits'] - v0['hits']} (MRR {rows[name]['mrr'] - v0['mrr']:+.4f})")
    lines += ["", "## hit@20 by question type / 按题型", ""]
    present = [n for n, *_ in VARIANTS if rows.get(n)]
    if present:
        types = sorted(rows[present[0]]["by_type"])
        lines.append("| type | n | " + " | ".join(present) + " |")
        lines.append("|---|---|" + "---|" * len(present))
        for t in types:
            cells = " | ".join(f"{100 * rows[n]['by_type'].get(t, 0):.1f}%" for n in present)
            lines.append(f"| {t} | {rows[present[0]]['by_type_n'][t]} | {cells} |")
    lines += ["", "## Reproduction check / 复现核对", ""]
    for name, rp in repro.items():
        if rp is None:
            lines.append(f"- {name}: not run")
            continue
        lines.append(
            f"- {name} vs real `{rp['real_run']}`: real hit@20 = {rp['real_hits']}, replay = {rp['replay_hits']} (n={rp['n']}); hit in both = {rp['hit_both']}, "
            f"real-only = {len(rp['hit_real_only'])} {rp['hit_real_only']}, replay-only = {len(rp['hit_replay_only'])} {rp['hit_replay_only']}; "
            f"identical returned-batch lists = {rp['same_returned_batches']}/{rp['n']}; atoms added real = {rp['real_atoms_added']}, replay = {rp['replay_atoms_added']}; "
            f"batches whose added-count differs = {len(rp['batches_with_different_added'])} {rp['batches_with_different_added'][:12]}"
        )
    lines += ["", "## Per-item transform counters (from replay_ab reduction_counters) / 逐条变换计数", ""]
    for name in present:
        c = rows[name]["counters"]
        keep = {k: v for k, v in c.items() if k.startswith("transform_") or k.startswith("observation") or k.startswith("compact_replies")}
        lines.append(f"- {name}: {json.dumps(keep, ensure_ascii=False)}")
    lines += ["", "## Exact commands (run from benchmarks/egolife; PY = the bench venv interpreter) / 精确命令", ""]
    for name in present:
        lines.append(f"- {name}: `{rows[name]['command']}`")
    lines += ["", "Driver: `./replay_diag_run.sh W0 W1 W2 W3`, `./replay_diag_run.sh V0 W4 V2 V3`, `./replay_diag_run.sh W5 W6` (three groups in parallel, "
              "so the wall-clock timings inside each summary.json are not comparable). This file: `PY replay_diag_summary.py`.", ""]  # fmt: skip
    lines += ["## Interpretation (numbers above; written 2026-09-11) / 解读", ""] + INTERPRETATION.strip("\n").split("\n")
    (DIAG / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DIAG / "summary_grid.json").write_text(json.dumps({"variants": rows, "reproduction": repro}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
