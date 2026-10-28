"""Command-line entry point: hms add/search/compile/why/transitions/consolidate/stats over the $HMS_DIR store.
命令行入口：基于 $HMS_DIR 存储的 hms add/search/compile/why/transitions/consolidate/stats。
"""

# Purpose (EN): A minimal argparse CLI over the HMS facade (console script `hms`, §17.6; also `python -m hms_elite.cli`).
# 目的（中）：基于 HMS 门面的最小 argparse 命令行（控制台脚本 `hms`，§17.6；也可 `python -m hms_elite.cli`）。

# Input (EN): argv and the HMS_DIR environment variable (default store location).
# 输入（中）：argv 与 HMS_DIR 环境变量（默认存储位置）。

# Output (EN): JSON (ensure_ascii=False) or card / JSONL text on stdout (exit 0); an HMSError → to_dict() JSON on stderr (exit 1).
# 输出（中）：标准输出上的 JSON（ensure_ascii=False）或卡片 / JSONL 文本（退出码 0）；HMSError → to_dict() JSON 到标准错误（退出码 1）。

# State effect (EN): add/consolidate/transitions write the store at $HMS_DIR/hms.db; the others only read it.
# 状态影响（中）：add/consolidate/transitions 写入 $HMS_DIR/hms.db；其余命令只读。

from __future__ import annotations

import argparse
import json
import sys

from hms_elite.exceptions import HMSError
from hms_elite.memory.main import HMS

_SCOPE_FLAGS = (("user_id", "user"), ("agent_id", "agent"), ("run_id", "run"))
# 命令名、在必填 --user 之后追加的参数、额外选项（flag, kwargs）
_COMMANDS = (  # name, arguments added after the mandatory --user, extra option (flag, kwargs)
    ("add", ("text", "--agent", "--run"), None),
    ("search", ("query",), ("--top-k", {"type": int, "default": 20, "dest": "top_k"})),
    ("compile", ("query", "--run"), ("--mode", {"choices": ["auto", "fast", "axis_graph", "deep"], "default": None})),
    ("why", ("target", "--run"), None),
    ("transitions", ("--run",), ("--fmt", {"choices": ["tuples", "records", "jsonl"], "default": "jsonl"})),
    ("consolidate", (), None),
    ("stats", (), None),
)


def main(argv: list[str] | None = None) -> int:
    """Parse argv, run one command against HMS() (store under $HMS_DIR) and print the result.
    解析 argv，对 HMS()（存储位于 $HMS_DIR）执行一条命令并打印结果。

    Input / 输入：
        argv: Argument list (None → sys.argv[1:]).
        argv：参数列表（None → sys.argv[1:]）。

    Output / 输出：
        Process exit code: 0 on success, 1 when the command raised an HMSError (its to_dict() JSON goes to stderr).
        进程退出码：成功为 0；命令抛出 HMSError 时为 1（其 to_dict() JSON 输出到标准错误）。

    State effect / 状态影响：
        Opens the store under $HMS_DIR; add/consolidate/transitions write to it.
        打开 $HMS_DIR 下的存储；add/consolidate/transitions 会写入。
    """
    parser = argparse.ArgumentParser(prog="hms", description="HMS Elite command line")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, arguments, extra in _COMMANDS:
        cmd = sub.add_parser(name)
        cmd.add_argument("--user", required=True)
        # 位置参数（text/query/target）与可选的 scope 标志
        for argument in arguments:  # positionals (text/query/target) and the optional scope flags
            cmd.add_argument(argument)
        if extra:
            cmd.add_argument(extra[0], **extra[1])
    ns = parser.parse_args(argv)
    filters = {key: getattr(ns, flag) for key, flag in _SCOPE_FLAGS if getattr(ns, flag, None)}
    try:
        with HMS() as hms:
            runners = {
                "add": lambda: hms.add(ns.text, **filters),
                "search": lambda: hms.search(ns.query, filters=filters, top_k=ns.top_k),
                "compile": lambda: hms.compile(ns.query, filters=filters, mode=ns.mode).to_text(),
                "why": lambda: hms.why(ns.target, filters=filters),
                "transitions": lambda: hms.transitions(filters=filters, fmt=ns.fmt),
                "consolidate": lambda: hms.consolidate(filters=filters),
                "stats": lambda: hms.graph_stats(filters=filters),
            }
            out = runners[ns.cmd]()
    except HMSError as exc:
        # Library errors (NOT_FOUND_001, VALIDATION_*, ...) become structured JSON on stderr, not a traceback.
        # 库错误（NOT_FOUND_001、VALIDATION_* 等）变成标准错误上的结构化 JSON，而不是回溯。
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, default=str), file=sys.stderr)
        return 1
    print(out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
