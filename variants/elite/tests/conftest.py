"""Shared pytest fixtures: frozen clock, sequential ids, offline guards, store/HMS factories, fixture loaders, goldens.
共享 pytest 夹具：冻结时钟、顺序 id、离线守卫、store/HMS 工厂、夹具加载器与黄金比对。
"""

# Purpose (EN): The ONLY conftest (§18.1); every deterministic ingredient of the test suite lives here.
# 目的（中）：唯一的 conftest（§18.1）；测试套件的全部确定性要素都在这里。

# Input (EN): pytest hooks, HMS_DIR/network environment, tests/fixtures/*.
# 输入（中）：pytest 钩子、HMS_DIR/网络环境、tests/fixtures/*。

# Output (EN): Fixtures (clock, ids, store, hms, events, golden, ...) and the --update-golden option.
# 输出（中）：夹具（clock、ids、store、hms、events、golden……）与 --update-golden 选项。

# State effect (EN): Redirects HMS_DIR to tmp_path, blocks sockets; --update-golden rewrites golden files.
# 状态影响（中）：将 HMS_DIR 重定向到 tmp_path、阻断套接字；--update-golden 会重写黄金文件。

from __future__ import annotations

import json
import os
import pathlib
import re
import socket
from datetime import timedelta
from typing import Any

import pytest

from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import mask_ids
from hms_elite.timeutil import parse_iso, to_utc_iso

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
EVENTS_DIR = FIXTURES / "events"
GOLDEN_DIR = FIXTURES / "golden"
_INSTANT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00")


class FrozenClock:
    """A clock that only moves when told to (start 2026-09-02T10:00:00+00:00).
    只有被推动时才前进的时钟（起点 2026-09-02T10:00:00+00:00）。
    """

    start = "2026-09-02T10:00:00+00:00"

    def __init__(self, start: str | None = None) -> None:
        """Freeze at `start`.
        冻结在 `start`。

        Input / 输入：
            start: ISO instant; defaults to the class start.
            start：ISO 时刻；默认类起点。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self._t = parse_iso(start or self.start)

    def now(self) -> str:
        """Current frozen instant (does NOT auto-advance).
        当前冻结时刻（不自动前进）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Fixed-width UTC ISO string.
            定宽 UTC ISO 字符串。

        State effect / 状态影响：
            None.
            无。
        """
        return to_utc_iso(self._t)

    def tick(self, seconds: float = 0, days: float = 0) -> str:
        """Advance the clock.
        推进时钟。

        Input / 输入：
            seconds: Seconds to add.
            seconds：增加的秒数。

            days: Days to add.
            days：增加的天数。

        Output / 输出：
            The new instant.
            新的时刻。

        State effect / 状态影响：
            Mutates the clock.
            修改时钟。
        """
        self._t = self._t + timedelta(seconds=seconds, days=days)
        return self.now()


class SequentialIds:
    """Deterministic ids: one counter per prefix, `prefix_000001`.
    确定性 id：每个前缀一个计数器，`prefix_000001`。
    """

    def __init__(self) -> None:
        """Start every counter at zero.
        所有计数器从零开始。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self._counters: dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        """Next id for a prefix.
        某前缀的下一个 id。
        """
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}_{self._counters[prefix]:06d}"


class FixtureDict(dict):
    """A dict whose keys are also attributes (`fixture.events_fail`).
    键同时可作属性访问的字典（`fixture.events_fail`）。
    """

    def __getattr__(self, name: str) -> Any:
        """Attribute access falls back to item access.
        属性访问回退为键访问。
        """
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def load_events(name: str) -> FixtureDict:
    """Load tests/fixtures/events/<name>.json; a bare list is wrapped as {'messages': [...]}.
    加载 tests/fixtures/events/<name>.json；裸列表包装为 {'messages': [...]}。

    Input / 输入：
        name: Fixture stem (without .json).
        name：夹具文件名（不含 .json）。

    Output / 输出：
        FixtureDict.
        FixtureDict。

    State effect / 状态影响：
        None.
        无。
    """
    data = json.loads((EVENTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    if isinstance(data, list):
        return FixtureDict(messages=data)
    return FixtureDict(data)


def mask_instants(text: str) -> str:
    """Replace every instant >= FrozenClock.start with '<ts>'; fixture instants stay literal.
    将每个 >= FrozenClock.start 的时刻替换为 '<ts>'；夹具时刻保持原样。

    Input / 输入：
        text: Text possibly containing 32-char instants.
        text：可能含 32 字符时刻的文本。

    Output / 输出：
        Masked text.
        掩码后的文本。

    State effect / 状态影响：
        None.
        无。
    """
    start = to_utc_iso(FrozenClock.start)
    return _INSTANT_RE.sub(lambda m: "<ts>" if m.group(0) >= start else m.group(0), text)


class Golden:
    """Golden-file helpers: byte-for-byte `text` and id/instant-masked `masked`.
    黄金文件辅助：逐字节比对的 `text` 与掩码 id/时刻的 `masked`。
    """

    def __init__(self, request: pytest.FixtureRequest) -> None:
        """Bind to the pytest request (reads --update-golden).
        绑定到 pytest 请求（读取 --update-golden）。

        Input / 输入：
            request: The fixture request.
            request：夹具请求。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self.update = bool(request.config.getoption("--update-golden"))

    def _compare(self, name: str, actual: str) -> None:
        """Compare (or rewrite) one golden file.
        比对（或重写）一个黄金文件。
        """
        path = GOLDEN_DIR / name
        if self.update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(actual, encoding="utf-8")
            return
        assert path.exists(), f"golden file missing: {path} (run with --update-golden)"
        expected = path.read_text(encoding="utf-8")
        assert actual == expected, f"golden mismatch for {name}"

    def text(self, name: str, text: str) -> None:
        """assert_golden_text: byte-for-byte, no masking (cards, why_chain.txt).
        assert_golden_text：逐字节、不掩码（卡片、why_chain.txt）。

        Input / 输入：
            name: Golden file name under tests/fixtures/golden/.
            name：tests/fixtures/golden/ 下的文件名。

            text: The produced text.
            text：产出的文本。

        Output / 输出：
            None (asserts).
            无（断言）。

        State effect / 状态影响：
            Rewrites the file under --update-golden.
            --update-golden 下重写文件。
        """
        self._compare(name, text)

    def masked(self, name: str, obj: Any) -> None:
        """assert_golden_masked: JSON/JSONL after mask_ids and masking of instants >= FrozenClock.start.
        assert_golden_masked：经 mask_ids 与 >= FrozenClock.start 时刻掩码后的 JSON/JSONL。

        Input / 输入：
            name: Golden file name (.json or .jsonl).
            name：黄金文件名（.json 或 .jsonl）。

            obj: A string (used verbatim), or a list/dict serialized as JSONL (.jsonl) / indented JSON.
            obj：字符串（原样使用），或序列化为 JSONL（.jsonl）/ 缩进 JSON 的列表/字典。

        Output / 输出：
            None (asserts).
            无（断言）。

        State effect / 状态影响：
            Rewrites the file under --update-golden.
            --update-golden 下重写文件。
        """
        if isinstance(obj, str):
            text = obj
        elif name.endswith(".jsonl") and isinstance(obj, list):
            text = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in obj)
        else:
            text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
        if not text.endswith("\n"):
            text += "\n"
        self._compare(name, mask_instants(mask_ids(text)))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --update-golden.
    注册 --update-golden。

    Input / 输入：
        parser: pytest option parser.
        parser：pytest 选项解析器。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        Adds a command-line option.
        新增命令行选项。
    """
    parser.addoption("--update-golden", action="store_true", default=False, help="rewrite golden files")


class _BlockedSocket(socket.socket):
    """socket.socket replacement that refuses to open.
    拒绝打开的 socket.socket 替代品。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Raise unless HMS_TEST_ALLOW_NETWORK=1.
        除非 HMS_TEST_ALLOW_NETWORK=1 否则抛出。
        """
        if os.environ.get("HMS_TEST_ALLOW_NETWORK") != "1":
            raise RuntimeError("network access is disabled in tests (set HMS_TEST_ALLOW_NETWORK=1 to allow)")
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True)
def no_home_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Redirect HMS_DIR to tmp_path and drop provider API keys.
    将 HMS_DIR 重定向到 tmp_path 并移除提供者 API 密钥。
    """
    monkeypatch.setenv("HMS_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block socket creation unless HMS_TEST_ALLOW_NETWORK=1.
    除非 HMS_TEST_ALLOW_NETWORK=1 否则阻断套接字创建。
    """
    monkeypatch.setattr(socket, "socket", _BlockedSocket)


@pytest.fixture
def clock() -> FrozenClock:
    """A FrozenClock at the §19 start instant.
    位于 §19 起点时刻的 FrozenClock。
    """
    return FrozenClock()


@pytest.fixture
def ids() -> SequentialIds:
    """A SequentialIds factory.
    一个 SequentialIds 工厂。
    """
    return SequentialIds()


@pytest.fixture
def store(clock: FrozenClock, ids: SequentialIds) -> StateStore:
    """An in-memory StateStore with the frozen clock and sequential ids.
    使用冻结时钟与顺序 id 的内存 StateStore。
    """
    return StateStore(":memory:", now=clock.now, id_factory=ids)


@pytest.fixture(params=["sqlite", "memory"])
def index_provider(request: pytest.FixtureRequest) -> str:
    """Both index providers.
    两种索引提供者。
    """
    return request.param


@pytest.fixture(scope="session")
def milvus_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One milvus-lite database path shared by the Milvus provider tests (skipped when pymilvus is missing).
    Milvus 提供者测试共用的一个 milvus-lite 数据库路径（缺少 pymilvus 时跳过）。
    """
    pytest.importorskip("pymilvus")
    return str(tmp_path_factory.mktemp("milvus") / "hms_milvus.db")


@pytest.fixture
def config(index_provider: str):
    """HMSConfig with the parametrized index provider and an in-memory store path.
    带参数化索引提供者与内存存储路径的 HMSConfig。
    """
    from hms_elite.configs.base import HMSConfig

    return HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"})


@pytest.fixture
def hms(config, store: StateStore):
    """HMS on the rule LLM + hash embedder + parametrized index, sharing the `store` fixture.
    基于规则 LLM + 哈希嵌入器 + 参数化索引、共享 `store` 夹具的 HMS。
    """
    from hms_elite.memory.main import HMS

    return HMS(config, store=store)


@pytest.fixture
def hms_scripted(store: StateStore):
    """HMS with an empty ScriptedLLM queue (asserts zero LLM calls on structured paths).
    使用空 ScriptedLLM 队列的 HMS（用于断言结构化路径零 LLM 调用）。
    """
    from hms_elite.configs.base import HMSConfig
    from hms_elite.memory.main import HMS

    return HMS(
        HMSConfig(llm={"provider": "scripted", "config": {"responses": []}}, store={"path": ":memory:"}), store=store
    )


@pytest.fixture
def rule_llm():
    """A RuleLLM.
    一个 RuleLLM。
    """
    from hms_elite.llms.rule import RuleLLM

    return RuleLLM()


@pytest.fixture
def scripted_llm():
    """Factory `responses -> ScriptedLLM`.
    工厂 `responses -> ScriptedLLM`。
    """

    def factory(responses: list) -> Any:
        """Build a ScriptedLLM with the given queue.
        用给定队列构建 ScriptedLLM。
        """
        from hms_elite.llms.scripted import ScriptedLLM

        return ScriptedLLM({"responses": list(responses)})

    return factory


@pytest.fixture
def hash_embedder():
    """A 64-dim HashEmbedder.
    64 维 HashEmbedder。
    """
    from hms_elite.embeddings.hash import HashEmbedder

    return HashEmbedder({"embedding_dims": 64})


@pytest.fixture
def events():
    """Loader `name -> FixtureDict` over tests/fixtures/events/.
    面向 tests/fixtures/events/ 的加载器 `name -> FixtureDict`。
    """
    return load_events


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> Golden:
    """The Golden helper bound to this request.
    绑定到本请求的 Golden 辅助对象。
    """
    return Golden(request)


@pytest.fixture
def scope4() -> dict[str, str]:
    """The S4 scope dict.
    S4 范围字典。
    """
    return {"user_id": "alice", "agent_id": "coder", "run_id": "run-billing"}
