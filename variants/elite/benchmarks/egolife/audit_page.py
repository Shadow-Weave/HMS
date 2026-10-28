"""Render the per-question EgoLife audit (results/audit_2026-09-20/audit.json) as one filterable HTML page.
把逐题对账（results/audit_2026-09-20/audit.json）渲染成一页可筛选的 HTML。

Purpose (EN): one row per question, one cell per configuration (answer, right/wrong, whether the gold batch was in the
context), the four-way reason label, and the exact context text the model saw, so a reader can check every claimed gain.
目的（ZH）：每题一行、每种读法一格（答案、对错、记忆里是否有答案）、四类原因标签，以及模型看到的原文上下文，让读者能逐题核对。

Input (EN): audit.json produced by the audit build (schema: meta / questions[] / summary).
输入（ZH）：对账构建产出的 audit.json（meta / questions[] / summary）。

Output (EN): a content-only HTML file (starts with <title>; no html/head/body tags) and a standalone copy with the doctype wrapper.
输出（ZH）：仅内容的 HTML（以 <title> 开头，无 html/head/body 标签）与带 doctype 包装的独立副本。

State effect (EN): writes the two files; reads nothing else.
状态影响（ZH）：写两个文件；不读其它内容。
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_AUDIT = HERE / "results" / "audit_2026-09-20" / "audit.json"

# Column order and Chinese labels; a config missing from audit.json is skipped.
# 列顺序与中文名；audit.json 里没有的读法自动跳过。
CONFIGS: list[tuple[str, str, str]] = [
    ("mem0", "mem0", "官方 mem0，自己的库"),
    ("full_rows", "旧默认：full 记忆行", "full 契约真实运行 r3，20 条记忆行"),
    ("full_card", "full 卡片", "同一 full 运行，逐题卡片"),
    ("lines_rows_real", "lines 记忆行（真实运行）", "lines 契约真实运行，20 条记忆行"),
    ("rows", "记忆行（回放）", "同一回放库，20 条记忆行，按相关度排"),
    ("rows_ts", "记忆行 + 日期", "同 20 条，按时间排并标日期"),
    ("card_1200", "逐题卡片", "同一回放库，针对问题编一张卡"),
    ("hybrid", "新默认：会话卡 + 记忆行", "同一回放库，会话卡前缀 + 20 条标日期的记忆行，记忆在前"),
]
REASON_CLASS = {
    "找到并答对": "ok",
    "记忆里没有答案却答对（疑似猜对）": "lucky",
    "记忆里有答案，模型没用对": "reader",
    "记忆里没找到答案": "miss",
}

STYLE = """
<style>
  :root {
    --bg: #F2F4F7; --surface: #FFFFFF; --ink: #1D2230; --muted: #5C6376; --line: #D5DAE3;
    --accent: #1F7A72; --accent-soft: #DDEFEC;
    --ok: #1F7A72; --ok-soft: #DDEFEC; --bad: #A4443C; --bad-soft: #F7E4E2;
    --lucky: #B5701E; --lucky-soft: #FBF1E1; --reader: #4C5A8A; --reader-soft: #E8EBF6;
    --mono-bg: #EEF1F5;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14171E; --surface: #1C2029; --ink: #E6E9F0; --muted: #9AA2B3; --line: #303645;
      --accent: #57B9AE; --accent-soft: #1E3532;
      --ok: #57B9AE; --ok-soft: #1E3532; --bad: #E08078; --bad-soft: #3A2220;
      --lucky: #E0A458; --lucky-soft: #35291A; --reader: #9AA8E0; --reader-soft: #23283A;
      --mono-bg: #232835;
    }
  }
  :root[data-theme="dark"] {
    --bg: #14171E; --surface: #1C2029; --ink: #E6E9F0; --muted: #9AA2B3; --line: #303645;
    --accent: #57B9AE; --accent-soft: #1E3532;
    --ok: #57B9AE; --ok-soft: #1E3532; --bad: #E08078; --bad-soft: #3A2220;
    --lucky: #E0A458; --lucky-soft: #35291A; --reader: #9AA8E0; --reader-soft: #23283A;
    --mono-bg: #232835;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--ink); font-family: "Noto Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; font-size: 14px; line-height: 1.6; margin: 0; }
  .page { max-width: 1320px; margin: 0 auto; padding: 36px 24px 72px; display: grid; gap: 24px; }
  h1, h2 { font-family: "Noto Serif SC", "Songti SC", "SimSun", serif; text-wrap: balance; margin: 0; }
  h1 { font-size: 30px; font-weight: 700; line-height: 1.25; }
  h2 { font-size: 18px; font-weight: 600; }
  .eyebrow { font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); font-weight: 700; font-family: "JetBrains Mono", monospace; }
  header { display: grid; gap: 10px; }
  header p { max-width: 78ch; color: var(--muted); margin: 0; }
  .num { font-family: "JetBrains Mono", "SF Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }
  .strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
  .kpi { background: var(--surface); border: 1px solid var(--line); padding: 12px 14px; display: grid; gap: 2px; }
  .kpi.main { border-left: 3px solid var(--accent); }
  .kpi .label { font-size: 12px; color: var(--muted); }
  .kpi .value { font-size: 22px; font-weight: 600; line-height: 1.2; }
  .kpi .sub { font-size: 12px; color: var(--muted); }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
  .card { background: var(--surface); border: 1px solid var(--line); padding: 14px 16px; display: grid; gap: 8px; }
  .card h2 { font-size: 15px; }
  .card p { margin: 0; font-size: 13px; color: var(--muted); }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { font-weight: 500; color: var(--muted); font-size: 12px; background: var(--mono-bg); position: sticky; top: env(safe-area-inset-top, 0px); }
  td.num, th.num { text-align: right; }
  .tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); }
  .small { min-width: 0; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; font-size: 13px; }
  .controls label { display: flex; gap: 6px; align-items: center; color: var(--muted); }
  select, input[type=search] { font: inherit; padding: 4px 8px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 2px; }
  select:focus, input:focus, summary:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  .count { color: var(--muted); font-size: 13px; }
  .ans { display: inline-block; min-width: 2.2em; text-align: center; font-family: "JetBrains Mono", monospace; font-weight: 600; padding: 0 6px; border-radius: 2px; border: 1px solid var(--line); }
  .ans.ok { color: var(--ok); background: var(--ok-soft); border-color: var(--ok); }
  .ans.bad { color: var(--bad); background: var(--bad-soft); border-color: var(--bad); }
  .ans.none { color: var(--muted); }
  .flag { font-size: 11px; color: var(--muted); margin-left: 4px; white-space: nowrap; }
  .tag { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 2px; border: 1px solid var(--line); white-space: nowrap; }
  .tag.ok { color: var(--ok); background: var(--ok-soft); border-color: var(--ok); }
  .tag.lucky { color: var(--lucky); background: var(--lucky-soft); border-color: var(--lucky); }
  .tag.reader { color: var(--reader); background: var(--reader-soft); border-color: var(--reader); }
  .tag.miss { color: var(--bad); background: var(--bad-soft); border-color: var(--bad); }
  .q { max-width: 46ch; }
  details { border-top: 1px solid var(--line); }
  details > summary { cursor: pointer; list-style: none; padding: 8px 10px; display: grid; grid-template-columns: 3em 7em minmax(220px, 1.4fr) 3em repeat(var(--ncfg), minmax(88px, 0.6fr)) minmax(150px, 0.9fr); gap: 8px; align-items: start; }
  details > summary::-webkit-details-marker { display: none; }
  details[open] > summary { background: var(--mono-bg); }
  .body { padding: 10px 14px 16px 14px; display: grid; gap: 12px; background: var(--surface); }
  .opts { display: grid; gap: 2px; font-size: 13px; }
  .opts .gold { color: var(--ok); font-weight: 600; }
  .cfgs { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
  .cfg { border: 1px solid var(--line); padding: 8px 10px; display: grid; gap: 4px; font-size: 13px; }
  .cfg .name { font-weight: 600; }
  pre.ctx { margin: 0; white-space: pre-wrap; word-break: break-word; font: 12px/1.5 "JetBrains Mono", "SF Mono", Menlo, monospace; background: var(--mono-bg); padding: 10px 12px; max-height: 420px; overflow: auto; }
  .ctxs { display: grid; gap: 10px; }
  .ctxs details { border: 1px solid var(--line); }
  .ctxs summary { padding: 6px 10px; display: block; font-size: 13px; cursor: pointer; }
  .head { display: grid; grid-template-columns: 3em 7em minmax(220px, 1.4fr) 3em repeat(var(--ncfg), minmax(88px, 0.6fr)) minmax(150px, 0.9fr); gap: 8px; padding: 8px 10px; font-size: 12px; color: var(--muted); background: var(--mono-bg); border-bottom: 1px solid var(--line); }
  .list { background: var(--surface); border: 1px solid var(--line); overflow-x: auto; }
  .listinner { min-width: 1100px; }
  .sources { font-size: 12px; color: var(--muted); display: grid; gap: 3px; padding-left: 18px; margin: 0; }
  code { font-family: "JetBrains Mono", monospace; font-size: 0.92em; background: var(--mono-bg); padding: 1px 5px; }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""


def esc(x: Any) -> str:
    """html.escape for any value / 任意值的 html.escape。"""
    return html.escape("" if x is None else str(x), quote=True)


def pct(c: int, n: int) -> str:
    """'64.7%' / 百分比字串。"""
    return f"{100.0 * c / n:.1f}%" if n else "—"


def present_configs(questions: list[dict]) -> list[tuple[str, str, str]]:
    """CONFIGS present in at least one question / 至少一题里出现的读法。"""
    keys = set()
    for q in questions:
        keys.update((q.get("configs") or {}).keys())
    return [c for c in CONFIGS if c[0] in keys]


def summary_tiles(summary: dict, cfgs: list[tuple[str, str, str]], n: int) -> str:
    """KPI tiles: accuracy per config and the '疑似猜对' count / 每读法准确率与疑似猜对数。"""
    acc = summary.get("per_config") or {}
    out = []
    for key, label, _desc in cfgs:
        a = acc.get(key) or {}
        correct = a.get("correct")
        if correct is None:
            continue
        reasons = a.get("reasons") or {}
        lucky = reasons.get("记忆里没有答案却答对（疑似猜对）", 0)
        miss = reasons.get("记忆里没找到答案", 0)
        reader = reasons.get("记忆里有答案，模型没用对", 0)
        main = " main" if key == "hybrid" else ""
        out.append(
            f'<div class="kpi{main}"><div class="label">{esc(label)}</div>'
            f'<div class="value num">{pct(int(correct), n)} <span style="font-size:13px;color:var(--muted)">({correct}/{n})</span></div>'
            f'<div class="sub">疑似猜对 {lucky} · 有答案没用对 {reader} · 没找到 {miss}</div></div>'
        )
    return "\n".join(out)


def flip_cards(summary: dict) -> str:
    """2×2 flip matrices for hybrid vs baselines with the sign-test p / 新默认对各基线的逐题变化与 p 值。"""
    flips = summary.get("flips") or {}
    tests = summary.get("sign_tests") or {}
    names = {"full_rows": "旧默认 full 记忆行", "mem0": "mem0", "rows": "记忆行（回放）", "card_1200": "逐题卡片"}
    out = []
    for key, name in names.items():
        m = flips.get("vs_" + key)
        if not m:
            continue
        both = m.get("都对", 0)
        base_only = m.get("只有基线对", 0)
        new_only = m.get("只有新默认对", 0)
        neither = m.get("都错", 0)
        p = (tests.get("vs_" + key) or {}).get("p_two_sided")
        ptxt = f"p = {p:.4f}" if isinstance(p, (int, float)) else ""
        net = int(new_only) - int(base_only)
        out.append(
            f'<div class="card"><h2>新默认 对 {esc(name)}</h2>'
            f'<table class="small"><tbody>'
            f'<tr><td>都对</td><td class="num">{both}</td><td>都错</td><td class="num">{neither}</td></tr>'
            f'<tr><td>只有{esc(name)}对</td><td class="num">{base_only}</td><td>只有新默认对</td><td class="num">{new_only}</td></tr>'
            f"</tbody></table>"
            f'<p>净变化 <b class="num">{net:+d}</b> 题 · 配对符号检验 {ptxt}</p></div>'
        )
    return "\n".join(out)


def type_table(summary: dict, cfgs: list[tuple[str, str, str]]) -> str:
    """Per-type accuracy table / 按题型的答对数。"""
    pc = summary.get("per_config") or {}
    types = sorted({t for key, _l, _d in cfgs for t in ((pc.get(key) or {}).get("per_type") or {})})
    if not types:
        return ""
    head = "".join(f"<th>{esc(label)}</th>" for _k, label, _d in cfgs)
    rows = []
    for t in types:
        cells = []
        for key, _label, _d in cfgs:
            v = ((pc.get(key) or {}).get("per_type") or {}).get(t)
            if isinstance(v, dict):
                cells.append(f'<td class="num">{v.get("correct", "—")}/{v.get("n", "—")}</td>')
            else:
                cells.append('<td class="num">—</td>')
        rows.append(f"<tr><td>{esc(t)}</td>{''.join(cells)}</tr>")
    return f'<div class="tablewrap"><table><thead><tr><th>题型（对 / 题数）</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def question_block(q: dict, cfgs: list[tuple[str, str, str]]) -> str:
    """One <details> per question / 每题一个可展开块。"""
    configs = q.get("configs") or {}
    gold = q.get("gold")
    cells = []
    for key, _label, _d in cfgs:
        c = configs.get(key)
        if not c:
            cells.append('<div><span class="ans none">—</span></div>')
            continue
        pred = c.get("pred") or "—"
        cls = "ok" if c.get("correct") else "bad"
        flag = "有" if c.get("gold_in_context") else "无"
        cells.append(f'<div><span class="ans {cls}">{esc(pred)}</span><span class="flag">答案在记忆里：{flag}</span></div>')
    hy = configs.get("hybrid") or {}
    reason = hy.get("reason") or ""
    rtag = f'<span class="tag {REASON_CLASS.get(reason, "")}">{esc(reason)}</span>' if reason else ""
    part = q.get("hybrid_gold_part")
    part_txt = f'<span class="flag">答案来自：{esc(part)}</span>' if part else ""
    flips = q.get("flips") or {}
    opts = q.get("options") or {}
    opt_html = "".join(
        f'<div class="{"gold" if k == gold else ""}">{esc(k)}. {esc(v)}{" ✓" if k == gold else ""}</div>' for k, v in sorted(opts.items())
    )
    cfg_html = []
    for key, label, desc in cfgs:
        c = configs.get(key)
        if not c:
            continue
        r = c.get("reason") or ""
        cfg_html.append(
            f'<div class="cfg"><div class="name">{esc(label)}</div><div>{esc(desc)}</div>'
            f'<div>答案 <span class="ans {"ok" if c.get("correct") else "bad"}">{esc(c.get("pred") or "—")}</span>'
            f' · 正确答案 {esc(gold)} · 答案在记忆里：{"有" if c.get("gold_in_context") else "无"}</div>'
            f'<div><span class="tag {REASON_CLASS.get(r, "")}">{esc(r)}</span></div></div>'
        )
    ctx_html = []
    for key, label, _d in cfgs:
        c = configs.get(key)
        if not c or not c.get("context"):
            continue
        ctx_html.append(
            f"<details><summary>模型看到的记忆文字 · {esc(label)}（{len(c['context']):,} 字符）</summary>"
            f'<pre class="ctx">{esc(c["context"])}</pre></details>'
        )
    flip_txt = " · ".join(f"对{esc(k)}：{esc(v)}" for k, v in flips.items()) if flips else ""
    data_attrs = (
        f' data-type="{esc(q.get("type"))}" data-reason="{esc(reason)}"'
        f' data-flip-rows="{esc(flips.get("vs_rows"))}" data-flip-card="{esc(flips.get("vs_card_1200"))}" data-flip-full="{esc(flips.get("vs_full_rows"))}"'
        f' data-text="{esc((q.get("question") or "") + " " + " ".join(str(v) for v in opts.values()))}"'
    )
    shown_q = q.get("question_chinese") or q.get("question")
    orig_q = q.get("question") if q.get("question_chinese") else None
    orig_html = f'<div class="count">原题：{esc(orig_q)}</div>' if orig_q else ""
    return (
        f"<details{data_attrs}><summary>"
        f'<div class="num">{esc(q.get("position", ""))}</div><div>{esc(q.get("type_chinese") or q.get("type"))}</div>'
        f'<div class="q">{esc(shown_q)}</div><div class="num">{esc(gold)}</div>'
        f"{''.join(cells)}<div>{rtag}{part_txt}</div></summary>"
        f'<div class="body">{orig_html}<div class="opts">{opt_html}</div>'
        f'<div class="cfgs">{"".join(cfg_html)}</div>'
        f'<div class="count">{flip_txt}</div>'
        f'<div class="ctxs">{"".join(ctx_html)}</div></div></details>'
    )


def render(audit: dict) -> str:
    """Whole page (content-only) / 整页（仅内容）。"""
    questions = sorted(audit.get("questions") or [], key=lambda q: int(q.get("position", 0)))
    n = len(questions)
    summary = audit.get("summary") or {}
    cfgs = present_configs(questions)
    meta = audit.get("meta") or {}
    types = sorted({q.get("type") for q in questions if q.get("type")})
    type_opts = "".join(f'<option value="{esc(t)}">{esc(t)}</option>' for t in types)
    reason_opts = "".join(f'<option value="{esc(r)}">{esc(r)}</option>' for r in REASON_CLASS)
    head_cells = "".join(f"<div>{esc(label)}</div>" for _k, label, _d in cfgs)
    blocks = "\n".join(question_block(q, cfgs) for q in questions)
    prompt_note = esc(meta.get("prompt_shapes") or "")
    return f"""<title>EgoLife 逐题对账</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap">
{STYLE}
<div class="page" style="--ncfg:{len(cfgs)}">
  <header>
    <div class="eyebrow">EgoLife · 204 题 · 当前默认读法 · 每题答了什么、对不对、记忆里有没有答案</div>
    <h1>EgoLife 逐题对账</h1>
    <p>同样的 204 道四选一，逐题列出模型选了什么、对不对、正确答案所在的那段记忆有没有进上下文，每题下面是模型看到的记忆原文。所有数字从原始运行文件重算（<code>verify.py</code> 可复算），不是从汇总表抄的。</p>
    <p>四类原因只看两件事：答对了没有、答案在记忆里没有。「记忆里没有答案却答对」是运气或常识，不算系统的功劳，单独计数。</p>
  </header>

  <div class="strip">{summary_tiles(summary, cfgs, n)}</div>

  <div class="grid2">{flip_cards(summary)}</div>

  {type_table(summary, cfgs)}

  <div class="controls">
    <label>题型 <select id="f-type"><option value="">全部</option>{type_opts}</select></label>
    <label>结果 <select id="f-reason"><option value="">全部</option>{reason_opts}</select></label>
    <label{" hidden" if not summary.get("flips") else ""}>新默认 对 旧默认 full <select id="f-flip"><option value="">全部</option><option>都对</option><option>只有基线对</option><option>只有新默认对</option><option>都错</option></select></label>
    <label>搜索 <input type="search" id="f-text" placeholder="题目或选项里的词"></label>
    <span class="count" id="count"></span>
  </div>

  <div class="list"><div class="listinner">
    <div class="head"><div>#</div><div>题型</div><div>题目（点开看选项、原因和记忆原文）</div><div>答案</div>{head_cells}<div>新默认为什么对 / 错</div></div>
    <div id="rows">{blocks}</div>
  </div></div>

  <div>
    <h2>口径</h2>
    <ul class="sources" style="margin-top:8px">
      <li>「答案在记忆里」：记忆行读法 = 正确答案所在的那批消息进了前 20 条；卡片与新默认 = 编进上下文的任一条记忆来自那批消息（口径更宽）；mem0 与 full 用各自运行记录的命中字段。</li>
      <li>记忆行、逐题卡片、新默认三种读法用同一个回放出来的库，只换交给模型的文字；mem0 与 full 是各自独立的真实运行，库不同。</li>
      <li>提示词：{prompt_note or "记忆行与卡片读法把问题放前、记忆放后；新默认把记忆放前、问题放后。"}</li>
      <li>逐题卡片在库重建后会换边缘项（原子 id 随机打平），这一列有约 ±2 题的噪声；新默认对逐题卡片的差距还在误差范围内，对记忆行与 full 记忆行的差距不在。</li>
      <li>数据文件：<code>benchmarks/egolife/results/audit_2026-09-20/</code> 下的 audit.json / audit.xlsx / audit.csv / README.md / verify.py。</li>
    </ul>
  </div>
</div>
<script>
(function () {{
  var rows = Array.prototype.slice.call(document.querySelectorAll('#rows > details'));
  var fType = document.getElementById('f-type'), fReason = document.getElementById('f-reason'), fFlip = document.getElementById('f-flip'), fText = document.getElementById('f-text'), count = document.getElementById('count');
  function apply() {{
    var t = fType.value, r = fReason.value, f = fFlip.value, s = fText.value.trim().toLowerCase(), shown = 0;
    rows.forEach(function (d) {{
      var ok = (!t || d.dataset.type === t) && (!r || d.dataset.reason === r) && (!f || d.dataset.flipFull === f) && (!s || (d.dataset.text || '').toLowerCase().indexOf(s) >= 0);
      d.hidden = !ok; if (ok) shown++;
    }});
    count.textContent = '显示 ' + shown + ' / ' + rows.length + ' 题';
  }}
  [fType, fReason, fFlip].forEach(function (el) {{ el.addEventListener('change', apply); }});
  fText.addEventListener('input', apply);
  apply();
}})();
</script>
"""


def main(argv: list[str] | None = None) -> int:
    """CLI: --audit, --out, --standalone / 命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, required=True, help="content-only HTML / 仅内容的 HTML")
    parser.add_argument("--standalone", type=Path, default=None, help="doctype-wrapped copy / 带 doctype 的副本")
    parser.add_argument("--only", default="hybrid", help="comma list of configs to show; '' = all / 只显示这些读法，空串 = 全部")
    args = parser.parse_args(argv)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if args.only:
        keep = {k.strip() for k in args.only.split(",") if k.strip()}
        for q in audit.get("questions") or []:
            q["configs"] = {k: v for k, v in (q.get("configs") or {}).items() if k in keep}
            q["flips"] = {}
        summary = audit.get("summary") or {}
        summary["per_config"] = {k: v for k, v in (summary.get("per_config") or {}).items() if k in keep}
        summary["flips"] = {}
        summary["sign_tests"] = {}
    page = render(audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    if args.standalone is not None:
        wrap = (
            '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            "<style>:root{color-scheme:light dark}body{margin:0;padding:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif}"
            "img{max-width:100%}[hidden]{display:none!important}</style></head><body>\n" + page + "\n</body></html>\n"
        )
        args.standalone.parent.mkdir(parents=True, exist_ok=True)
        args.standalone.write_text(wrap, encoding="utf-8")
    print(f"wrote {args.out} ({len(page):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
