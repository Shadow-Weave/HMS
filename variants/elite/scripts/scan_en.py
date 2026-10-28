"""Report real comment blocks that carry no Chinese line, so the bilingual rule stays enforced.
报告完全没有中文行的真实注释块，使双语注释规则持续可校验。
"""

# Purpose (EN): Find comment blocks that violate the CONTRACT §0.3 bilingual rule.
# 目的（中）：找出违反 CONTRACT §0.3 双语规则的注释块。

# Input (EN): A project root path (defaults to the hms-elite repository root).
# 输入（中）：项目根路径（默认为 hms-elite 仓库根目录）。

# Output (EN): One line per offending block on stdout plus a TOTAL_ENGLISH_ONLY_BLOCKS summary.
# 输出（中）：每个违规块在标准输出打印一行，外加一行 TOTAL_ENGLISH_ONLY_BLOCKS 汇总。

# State effect (EN): None; the checker only reads files.
# 状态影响（中）：无；检查器只读取文件。
import ast
import io
import pathlib
import re
import sys
import tokenize

CJK = re.compile(r"[一-鿿]")
PRAGMA = re.compile(r"#\s*(fmt:|noqa|type:|pragma|isort:|ruff:)")
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/lazy/Documents/Claude Code/hms-elite")

total = 0
# CONTRACT §0.3 binds hms_elite/**, tests/** and examples/*; the audit probes under docs/design/gaps/scripts are
# one-off evidence scripts outside that scope and are skipped here.
# CONTRACT §0.3 约束 hms_elite/**、tests/** 与 examples/*；docs/design/gaps/scripts 下的审计探针是范围之外的一次性
# 证据脚本，此处跳过。
EXEMPT = ("docs/design/gaps/scripts",)
for path in sorted(ROOT.rglob("*.py")):
    if "__pycache__" in str(path) or ".venv" in str(path):
        continue
    if any(str(path.relative_to(ROOT)).startswith(prefix) for prefix in EXEMPT):
        continue
    text = path.read_text()
    try:
        comments = [
            (t.start[0], t.string.strip())
            for t in tokenize.generate_tokens(io.StringIO(text).readline)
            if t.type == tokenize.COMMENT
        ]
        tree = ast.parse(text)
        stop = tree.body[0].lineno if tree.body else 0
    except (SyntaxError, tokenize.TokenError):
        continue
    # Group the comment tokens into runs of consecutive source lines.
    # 把注释 token 按连续源码行分组成块。
    runs, cur = [], []
    for lineno, s in comments:
        if cur and lineno == cur[-1][0] + 1:
            cur.append((lineno, s))
        else:
            if cur:
                runs.append(cur)
            cur = [(lineno, s)]
    if cur:
        runs.append(cur)
    for run in runs:
        start = run[0][0]
        # The module header has its own four-section bilingual form, checked elsewhere.
        # 模块头有自己的四段式双语格式，由别处校验。
        if start < stop:
            continue
        body = [s for _, s in run if not (s.startswith("#!") or PRAGMA.search(s) or set(s) <= set("#-= "))]
        if not body or not any(re.search(r"[A-Za-z]{2}", s) for s in body):
            continue
        if any(CJK.search(s) for s in body):
            continue
        print(f"{path.relative_to(ROOT)}:{start}: {body[0][:96]}")
        total += 1
print(f"TOTAL_ENGLISH_ONLY_BLOCKS={total}")
