"""Render a Chinese + English Markdown pair as one content-only HTML page with a language toggle.
把一对中英文 Markdown 渲染成一页带语言切换的仅内容 HTML。

Purpose (EN): design notes are kept as Markdown twins (X.zh.md / X.en.md); this turns them into one page
(starts with <title>, no html/head/body tags) whose ZH / EN buttons switch the visible version.
目的（ZH）：设计文档以中英文孪生 Markdown 保存；此脚本把它们合成一页，ZH / EN 按钮切换可见版本。

Input (EN): --zh, --en (Markdown files), --title, --out, optional --standalone.
输入（ZH）：--zh、--en（Markdown 文件）、--title、--out、可选 --standalone。

Output (EN): the HTML file(s). Markdown subset: headings, paragraphs, tables, lists, blockquotes, bold, inline code, fenced code.
输出（ZH）：HTML 文件。支持的 Markdown 子集：标题、段落、表格、列表、引用、粗体、行内代码、围栏代码。

State effect (EN): writes the output files only.
状态影响（ZH）：只写输出文件。
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

STYLE = """
<style>
  :root {
    --bg: #F2F4F7; --surface: #FFFFFF; --ink: #1D2230; --muted: #5C6376; --line: #D5DAE3;
    --accent: #1F7A72; --accent-soft: #DDEFEC; --mono-bg: #EEF1F5;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14171E; --surface: #1C2029; --ink: #E6E9F0; --muted: #9AA2B3; --line: #303645;
      --accent: #57B9AE; --accent-soft: #1E3532; --mono-bg: #232835;
    }
  }
  :root[data-theme="dark"] {
    --bg: #14171E; --surface: #1C2029; --ink: #E6E9F0; --muted: #9AA2B3; --line: #303645;
    --accent: #57B9AE; --accent-soft: #1E3532; --mono-bg: #232835;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--ink); font-family: "Noto Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; font-size: 15px; line-height: 1.75; margin: 0; }
  .page { max-width: 880px; margin: 0 auto; padding: 36px 24px 80px; }
  .bar { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  .eyebrow { font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); font-weight: 700; font-family: "JetBrains Mono", monospace; }
  .toggle { display: inline-flex; border: 1px solid var(--line); border-radius: 2px; overflow: hidden; }
  .toggle button { font: inherit; font-size: 13px; padding: 4px 14px; border: 0; background: var(--surface); color: var(--muted); cursor: pointer; }
  .toggle button[aria-pressed="true"] { background: var(--accent); color: #fff; }
  .toggle button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  article h1, article h2, article h3 { font-family: "Noto Serif SC", "Songti SC", "SimSun", serif; text-wrap: balance; margin: 0; }
  article h1 { font-size: 30px; line-height: 1.25; margin-bottom: 18px; }
  article h2 { font-size: 20px; margin-top: 34px; margin-bottom: 10px; padding-top: 14px; border-top: 1px solid var(--line); }
  article h3 { font-size: 16px; margin-top: 22px; margin-bottom: 6px; }
  article p { margin: 0 0 12px; max-width: 72ch; }
  article ul, article ol { margin: 0 0 12px; padding-left: 22px; max-width: 72ch; }
  article li { margin: 2px 0; }
  article blockquote { margin: 0 0 12px; padding: 8px 14px; border-left: 3px solid var(--accent); background: var(--surface); color: var(--muted); }
  .tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); margin: 0 0 14px; }
  table { border-collapse: collapse; width: 100%; min-width: 640px; font-size: 13px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { font-weight: 500; color: var(--muted); font-size: 12px; background: var(--mono-bg); }
  tr:last-child td { border-bottom: 0; }
  code { font-family: "JetBrains Mono", "SF Mono", Menlo, monospace; font-size: 0.9em; background: var(--mono-bg); padding: 1px 5px; }
  pre { background: var(--mono-bg); padding: 10px 12px; overflow-x: auto; font-size: 12.5px; line-height: 1.5; margin: 0 0 12px; }
  pre code { background: none; padding: 0; }
  article[lang="en"] { font-family: "Source Serif 4", Georgia, "Times New Roman", serif; }
  article[lang="en"] h1, article[lang="en"] h2, article[lang="en"] h3 { font-family: "Source Serif 4", Georgia, serif; }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def inline(text: str) -> str:
    """Escape, then bold and inline code / 转义后处理粗体与行内代码。"""
    out = html.escape(text, quote=False)
    out = _INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _BOLD.sub(lambda m: f"<b>{m.group(1)}</b>", out)
    return out


def _table(lines: list[str]) -> str:
    """Pipe table → <table> / 竖线表格转 HTML。"""
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "") for c in r)]
    if not rows:
        return ""
    head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def md_to_html(md: str) -> str:
    """Markdown subset → HTML / Markdown 子集转 HTML。"""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            flush()
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(buf))}</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", ln)
        if m:
            flush()
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
            i += 1
            continue
        if ln.strip().startswith("|"):
            flush()
            j = i
            tbl = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                tbl.append(lines[j])
                j += 1
            out.append(_table(tbl))
            i = j
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            flush()
            ordered = bool(re.match(r"^\s*\d+\.", ln))
            j = i
            items = []
            while j < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[j]):
                items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[j]))
                j += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            i = j
            continue
        if ln.startswith(">"):
            flush()
            j = i
            q = []
            while j < len(lines) and lines[j].startswith(">"):
                q.append(lines[j].lstrip("> "))
                j += 1
            out.append(f"<blockquote>{inline(' '.join(q))}</blockquote>")
            i = j
            continue
        if not ln.strip():
            flush()
            i += 1
            continue
        para.append(ln.strip())
        i += 1
    flush()
    return "\n".join(out)


def render(zh_md: str, en_md: str | None, title: str, eyebrow: str = "HMS Elite · 设计回顾 · Design retrospective") -> str:
    """Whole content-only page; without en_md there is no toggle / 整页（仅内容）；没有英文稿就不出切换按钮。"""
    toggle = (
        '<div class="toggle" role="group" aria-label="language">'
        '<button type="button" id="btn-zh" aria-pressed="true">中文</button>'
        '<button type="button" id="btn-en" aria-pressed="false">English</button></div>'
        if en_md is not None else ""
    )
    en_article = f'<article id="art-en" lang="en" hidden>\n{md_to_html(en_md)}\n  </article>' if en_md is not None else ""
    script = SCRIPT if en_md is not None else ""
    return f"""<title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&family=Noto+Sans+SC:wght@400;500;700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
{STYLE}
<div class="page">
  <div class="bar">
    <div class="eyebrow">{html.escape(eyebrow)}</div>
    {toggle}
  </div>
  <article id="art-zh" lang="zh">
{md_to_html(zh_md)}
  </article>
  {en_article}
</div>
{script}
"""


SCRIPT = """<script>
(function () {
  var zh = document.getElementById('art-zh'), en = document.getElementById('art-en');
  var bz = document.getElementById('btn-zh'), be = document.getElementById('btn-en');
  function show(lang) {
    var isZh = lang !== 'en';
    zh.hidden = !isZh; en.hidden = isZh;
    bz.setAttribute('aria-pressed', String(isZh)); be.setAttribute('aria-pressed', String(!isZh));
    try { localStorage.setItem('hms-origin-lang', isZh ? 'zh' : 'en'); } catch (e) {}
  }
  bz.addEventListener('click', function () { show('zh'); });
  be.addEventListener('click', function () { show('en'); });
  var saved = null;
  try { saved = localStorage.getItem('hms-origin-lang'); } catch (e) {}
  show(saved === 'en' ? 'en' : 'zh');
})();
</script>"""


def main(argv: list[str] | None = None) -> int:
    """CLI / 命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zh", type=Path, required=True)
    parser.add_argument("--en", type=Path, default=None, help="optional English twin / 可选英文稿")
    parser.add_argument("--title", required=True)
    parser.add_argument("--eyebrow", default="HMS Elite · 设计回顾 · Design retrospective")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--standalone", type=Path, default=None)
    args = parser.parse_args(argv)
    en_md = args.en.read_text(encoding="utf-8") if args.en is not None else None
    page = render(args.zh.read_text(encoding="utf-8"), en_md, args.title, args.eyebrow)
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
