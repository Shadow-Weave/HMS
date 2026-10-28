"""Docs tests: every README ```python block runs offline and matches its ```text block; the §3 YAML is the real defaults.
文档测试：README 的每个 ```python 代码块都可离线运行并与其 ```text 块一致；§3 的 YAML 就是真实默认值。
"""

# Purpose (EN): Keep the README literals honest: outputs are diffed after masking ids and clock instants (CONTRACT §20.2).
# 目的（中）：让 README 的字面输出保持真实：屏蔽 id 与时钟时刻后逐行比对（契约 §20.2）。

# Input (EN): README.md at the repository root; a ```python block whose first line carries "（示意）" is skipped.
# 输入（中）：仓库根目录的 README.md；首行带 "（示意）" 的 ```python 块跳过。

# Output (EN): pytest results (one unified diff per mismatching block).
# 输出（中）：pytest 结果（每个不一致的块给出一个 unified diff）。

# State effect (EN): Runs the blocks in one namespace on a ticking clock pinned to the README capture day; writes tmp_path only.
# 状态影响（中）：在同一命名空间、固定于 README 采集日的滴答时钟上运行代码块；只写 tmp_path。

from __future__ import annotations

import contextlib
import difflib
import io
import pathlib
import re
from datetime import timedelta

import pytest

from hms_elite import timeutil
from hms_elite.configs.base import HMSConfig, default_db_path
from hms_elite.memory import storage
from tests.conftest import mask_instants

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
README_ZH = README.with_name("README.zh.md")
FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.S)
ID_RE = re.compile(r"\b([a-z]+)_[0-9a-f]{12}\b")
SKIP_MARK = "（示意）"
# README 中字面量被采集的那一天（>= FrozenClock.start → 会被掩码）
CLOCK_START = "2026-09-02T22:43:00+00:00"  # the day the README literals were captured (>= FrozenClock.start → masked)


class _TickingClock:
    """Deterministic wall clock: starts at CLOCK_START and advances one second per call.
    确定性墙钟：从 CLOCK_START 开始，每次调用前进一秒。
    """

    def __init__(self) -> None:
        self._t = timeutil.parse_iso(CLOCK_START)

    def __call__(self) -> str:
        self._t += timedelta(seconds=1)
        return timeutil.to_utc_iso(self._t)


def _mask(text: str) -> list[str]:
    """Mask ids (`mem_<12hex>` → `mem_…`) and every instant on/after the clock start, then split into lines.
    屏蔽 id（`mem_<12hex>` → `mem_…`）与时钟起点及之后的所有时刻，再按行拆分。
    """
    return mask_instants(ID_RE.sub(lambda m: m.group(1) + "_…", text)).rstrip("\n").split("\n")


def _yaml_blocks(path: pathlib.Path) -> list[str]:
    """The ```yaml blocks of a README, in order.
    README 中按顺序排列的 ```yaml 代码块。
    """
    return [body for lang, body in FENCE_RE.findall(path.read_text(encoding="utf-8")) if lang == "yaml"]


def _run_block(body: str, namespace: dict) -> str:
    """Execute one README python block in `namespace` and return what it printed.
    在 `namespace` 中执行一个 README python 块并返回其打印内容。
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        # README.md 属于仓库自有的源码
        exec(compile(body, "<README.md>", "exec"), namespace)  # README.md is repository-owned source
    return buffer.getvalue()


def test_readme_blocks_match_their_text_literals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every non-illustrative ```python block of README.md in order; each following ```text block must match.
    按顺序运行 README.md 中每个非示意的 ```python 块；其后的每个 ```text 块都必须一致。

    Input / 输入：
        monkeypatch: Pins the library clock (storage.now_iso) to a ticking clock starting at CLOCK_START.
        monkeypatch：把库时钟（storage.now_iso）固定为从 CLOCK_START 起步的滴答时钟。

    Output / 输出：
        Assertions: no diff between the masked actual output and the README literal; at least 10 blocks compared.
        断言：屏蔽后的实际输出与 README 字面无差异；至少比对 10 个块。

    State effect / 状态影响：
        Writes only under tmp_path (conftest redirects HMS_DIR); the README's stores are in-memory.
        只写 tmp_path（conftest 已重定向 HMS_DIR）；README 中的存储均为内存库。
    """
    clock = _TickingClock()
    monkeypatch.setattr(storage, "now_iso", clock)
    monkeypatch.setattr(timeutil, "now_iso", clock)
    namespace: dict = {"__name__": "readme"}
    pending: str | None = None
    compared, diffs = 0, []
    for lang, body in FENCE_RE.findall(README.read_text(encoding="utf-8")):
        if lang == "python":
            pending = None if SKIP_MARK in body.strip().split("\n", 1)[0] else _run_block(body, namespace)
        elif lang == "text" and pending is not None:
            compared += 1
            got, want = _mask(pending), _mask(body)
            if got != want:
                diffs.append("\n".join(difflib.unified_diff(want, got, "README.md", "actual", lineterm="")))
            pending = None
    assert compared >= 10, f"only {compared} README blocks were compared"
    assert not diffs, "README literals drifted:\n" + "\n\n".join(diffs)


def test_readme_config_yaml_is_the_defaults() -> None:
    """The §3 YAML of both READMEs carries no literal `~`/`$VAR` store path, loads via HMSConfig.from_dict and resolves to the default store path.
    两份 README 的 §3 YAML 不含字面 `~`/`$VAR` 存储路径，能经 HMSConfig.from_dict 加载，并解析为默认存储路径。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions: identical blocks in both READMEs, no `~`/`$` on the store line, db_path() == default_db_path().
        断言：两份 README 的代码块一致、store 行不含 `~`/`$`、db_path() == default_db_path()。

    State effect / 状态影响：
        None (HMS_DIR is redirected to tmp_path by conftest; no store is opened).
        无（conftest 已把 HMS_DIR 重定向到 tmp_path；不打开任何存储）。
    """
    blocks, blocks_zh = _yaml_blocks(README), _yaml_blocks(README_ZH)
    assert blocks and blocks == blocks_zh
    store_line = next(line for line in blocks[0].splitlines() if line.startswith("store:"))
    # sqlite3 receives store.path verbatim (CONTRACT §5): a documented `~` would create ./~/.hms_elite under the cwd.
    # sqlite3 原样接收 store.path（契约 §5）：文档里若写 `~`，会在 cwd 下创建 ./~/.hms_elite。
    store_value = store_line.split("#", 1)[0]
    assert "~" not in store_value and "$" not in store_value
    yaml = pytest.importorskip("yaml")
    config = HMSConfig.from_dict(yaml.safe_load(blocks[0]))
    assert config.db_path() == default_db_path() == HMSConfig.from_dict({}).db_path()
