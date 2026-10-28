"""Milvus index provider tests: contract, bit-identical scores, chunked candidates, guards, HMS end-to-end parity.
Milvus 索引提供者测试：契约、三提供者分数逐位一致、候选分块、守卫、HMS 端到端。
"""

# Purpose (EN): Prove MilvusIndex is a drop-in IndexBase provider whose results equal the sqlite / memory providers.
# 目的（中）：证明 MilvusIndex 是可直接替换的 IndexBase 提供者，其结果与 sqlite / memory 提供者相等。

# Input (EN): The session-scoped milvus_uri fixture (skips without pymilvus), in-memory StateStore, hash embeddings.
# 输入（中）：会话级 milvus_uri 夹具（缺少 pymilvus 时跳过）、内存 StateStore、哈希嵌入。

# Output (EN): Assertions over search / get / count / reset / info / close, config guards and HMS search parity.
# 输出（中）：对 search / get / count / reset / info / close、配置守卫与 HMS 检索一致性的断言。

# State effect (EN): Writes milvus-lite collections under pytest's temporary directory only.
# 状态影响（中）：仅在 pytest 临时目录下写入 milvus-lite 集合。

from __future__ import annotations

import math
import pathlib
import random
import sys
from typing import Any

import pytest

from hms_elite import models
from hms_elite.configs import registry
from hms_elite.configs.base import HMSConfig
from hms_elite.exceptions import ConfigError, DependencyError, StoreError, ValidationError
from hms_elite.index.memory import MemoryIndex
from hms_elite.index.milvus import MilvusIndex, MilvusIndexConfig, default_milvus_uri, is_remote_uri
from hms_elite.index.sqlite import SQLiteIndex
from hms_elite.memory.storage import StateStore
from hms_elite.utils.factory import IndexFactory
from tests.conftest import FrozenClock, SequentialIds

SCOPE = models.Scope(user_id="alice")


def _unit_vectors(count: int, dims: int, seed: int) -> list[list[float]]:
    """Deterministic random unit vectors (seeded; tests may use random, the library may not).
    确定性的随机单位向量（带种子；测试可以用 random，库代码不可以）。
    """
    rng = random.Random(seed)
    out: list[list[float]] = []
    for _ in range(count):
        raw = [rng.gauss(0.0, 1.0) for _ in range(dims)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        out.append([x / norm for x in raw])
    return out


def _seed_atoms(store: StateStore, atom_ids: list[str]) -> None:
    """Insert one witnessed state atom per id (one shared anchor) so atom_vectors rows satisfy their foreign key.
    为每个 id 插入一个带见证的 state 原子（共用一个锚点），使 atom_vectors 行满足外键。
    """
    event = models.SourceEvent(scope=SCOPE, role="user", payload={"text": "seed"}, timestamp="2026-06-15T09:00:00Z")
    stored, _ = store.insert_source_event(event)
    anchor = store.get_or_create_anchor(stored)
    with store.transaction():
        for atom_id in atom_ids:
            store.insert_atom(
                models.MemoryAtom(
                    atom_id=atom_id,
                    atom_type="state",
                    content={"text": f"atom {atom_id}"},
                    text=f"atom {atom_id}",
                    scale_level="state",
                    source_kind="observation",
                    source_id=f"obs_{atom_id}",
                    extracted_by="rule@obs-v1",
                    scope=SCOPE,
                    anchor_ids=[anchor.anchor_id],
                    obs_type="fact",
                    subject="user",
                    predicate="states",
                    time_anchor="2026-06-15T09:00:00Z",
                )
            )


def _three_indexes(
    milvus_uri: str, store: StateStore, *, collection: str, dims: int, **milvus_config: Any
) -> tuple[MilvusIndex, SQLiteIndex, MemoryIndex]:
    """Build a milvus / sqlite / memory trio over the same store and dims.
    在同一 store 与 dims 上构建 milvus / sqlite / memory 三件套。
    """
    milvus = MilvusIndex(
        {"uri": milvus_uri, "collection": collection, "scalar_index": "none", **milvus_config}, store=store, dims=dims
    )
    return milvus, SQLiteIndex({"use_numpy": "auto"}, store=store, dims=dims), MemoryIndex(dims=dims)


def _load_all(indexes: tuple[Any, ...], items: list[tuple[str, list[float]]], embedder: str = "hash@64") -> None:
    """upsert_batch the same items into every index.
    把同一批向量 upsert_batch 到每个索引。
    """
    for index in indexes:
        index.upsert_batch(items, embedder=embedder)


# ---------------------------------------------------------------------------
# 1. IndexBase contract, mirrored against sqlite and memory
# 1. IndexBase 契约，与 sqlite、memory 对照
# ---------------------------------------------------------------------------
def test_milvus_contract_matches_sqlite_and_memory(milvus_uri, store, hash_embedder, tmp_path):
    """upsert / upsert_batch / get / delete / search / count / reset / info / close behave like the sqlite provider.
    upsert / upsert_batch / get / delete / search / count / reset / info / close 与 sqlite 提供者行为一致。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path.
        milvus_uri：共享的 milvus-lite 数据库路径。

        store: In-memory StateStore (atom_vectors host for the sqlite twin).
        store：内存 StateStore（sqlite 对照的 atom_vectors 宿主）。

        hash_embedder: Vector source.
        hash_embedder：向量来源。

        tmp_path: pytest temporary directory (HMS_DIR).
        tmp_path：pytest 临时目录（HMS_DIR）。

    Output / 输出：
        Assertions over the whole IndexBase contract plus the config guards.
        对整个 IndexBase 契约及配置守卫的断言。

    State effect / 状态影响：
        Creates the 'contract' and 'contract_hnsw' collections in the shared database.
        在共享数据库中创建 'contract' 与 'contract_hnsw' 集合。
    """
    texts = {
        "atm_000001": "User likes oat-milk lattes in the morning.",
        "atm_000002": "User likes black coffee.",
        "atm_000003": "billing_e2e failed after the retry patch",
        "atm_000004": "Sarah joined the billing team in March.",
    }
    _seed_atoms(store, list(texts))
    milvus, sqlite_index, memory_index = _three_indexes(milvus_uri, store, collection="contract", dims=64)
    items = [(atom_id, hash_embedder.embed(text)) for atom_id, text in texts.items()]
    milvus.upsert_batch(items[:2], embedder=hash_embedder.identity)
    for atom_id, vector in items[2:]:
        milvus.upsert(atom_id, vector, embedder=hash_embedder.identity)
    _load_all((sqlite_index, memory_index), items, embedder=hash_embedder.identity)
    assert milvus.count() == sqlite_index.count() == memory_index.count() == 4
    query = hash_embedder.embed("what coffee does the user like")
    full = milvus.search(query, candidate_ids=None, top_k=10)
    assert full == sqlite_index.search(query, candidate_ids=None, top_k=10)
    assert full == memory_index.search(query, candidate_ids=None, top_k=10)
    assert full[0][0] in {"atm_000001", "atm_000002"} and all(0.0 <= s <= 1.0 for _, s in full)
    restricted = ["atm_000003", "atm_000004", "atm_missing"]
    assert milvus.search(query, candidate_ids=restricted, top_k=10) == sqlite_index.search(
        query, candidate_ids=restricted, top_k=10
    )
    assert {h[0] for h in milvus.search(query, candidate_ids=restricted, top_k=10)} == {"atm_000003", "atm_000004"}
    assert milvus.search(query, candidate_ids=[], top_k=10) == []
    assert milvus.search(query, candidate_ids=None, top_k=0) == []
    assert milvus.search(query, candidate_ids=None, top_k=2) == full[:2]
    # get() returns the float32-stored vector, bit-for-bit what the sqlite BLOB decodes to.
    # get() 返回按 float32 存储的向量，与 sqlite BLOB 解码结果逐位相同。
    assert milvus.get("atm_000001") == sqlite_index.get("atm_000001") == memory_index.get("atm_000001")
    assert milvus.get("atm_none") is None
    milvus.delete("atm_000004")
    milvus.delete("atm_none")
    assert milvus.count() == 3
    # upsert of an existing id replaces the vector without growing the collection.
    # 对已有 id 的 upsert 会替换向量而不增加集合行数。
    replacement = [1.0] + [0.0] * 63
    milvus.upsert("atm_000001", replacement, embedder="hash@64b")
    assert milvus.get("atm_000001") == replacement and milvus.count() == 3
    with pytest.raises(ValidationError) as info:
        milvus.upsert("atm_000002", [1.0, 0.0], embedder="hash@2")
    assert info.value.details == {"atom_id": "atm_000002", "dims": 2, "expected": 64}
    info_dict = milvus.info()
    assert set(info_dict) == {"provider", "dims", "count", "numpy", "uri", "collection", "index_type"}
    assert info_dict["provider"] == "milvus" and info_dict["dims"] == 64 and info_dict["count"] == 3
    assert info_dict["uri"] == milvus_uri and info_dict["collection"] == "contract"
    assert info_dict["index_type"] == "AUTOINDEX" and info_dict["numpy"] == memory_index.uses_numpy
    milvus.reset()
    assert milvus.count() == 0 and milvus.search(query, candidate_ids=None, top_k=5) == []
    milvus.upsert_batch(items, embedder=hash_embedder.identity)
    assert milvus.count() == 4 and milvus.search(query, candidate_ids=None, top_k=10) == full
    milvus.close()
    milvus.close()
    with pytest.raises(StoreError) as info:
        milvus.count()
    assert info.value.error_code == "STORE_001" and "closed" in info.value.message
    # HNSW with a small ef: search raises ef to the limit, results still equal the exhaustive providers.
    # 小 ef 的 HNSW：搜索时把 ef 提高到 limit，结果仍与穷举式提供者相等。
    hnsw = MilvusIndex(
        {"uri": milvus_uri, "scalar_index": "none", "collection": "contract_hnsw", "index_type": "HNSW", "M": 8, "efConstruction": 64, "ef": 2},
        store=store,
        dims=64,
    )
    hnsw.upsert_batch(items, embedder=hash_embedder.identity)
    assert hnsw.search(query, candidate_ids=None, top_k=10) == full and hnsw.info()["index_type"] == "HNSW"
    hnsw.close()
    # Config guards: did-you-mean hint, '.db' suffix rule, PathLike uri, default uri under HMS_DIR, remote detection.
    # 配置守卫：did-you-mean 提示、'.db' 后缀规则、PathLike uri、HMS_DIR 下的默认 uri、远端识别。
    with pytest.raises(ConfigError) as info:
        MilvusIndex({"uri": milvus_uri, "colection": "x"}, store=store, dims=64)
    assert info.value.error_code == "CONFIG_001" and "Did you mean 'collection'?" in info.value.message
    with pytest.raises(ConfigError) as info:
        MilvusIndex({"uri": str(tmp_path / "plain")}, store=store, dims=64)
    assert info.value.error_code == "CONFIG_001" and "'.db'" in info.value.message
    with pytest.raises(ConfigError):
        MilvusIndex({"uri": milvus_uri}, store=store, dims=0)
    assert MilvusIndexConfig(uri=pathlib.Path(milvus_uri)).uri == milvus_uri
    assert default_milvus_uri() == str(tmp_path / "milvus.db") and MilvusIndexConfig().candidate_chunk == 1024
    assert MilvusIndexConfig().scalar_index == "INVERTED" and MilvusIndexConfig().ef_recall_factor == 4
    assert is_remote_uri("http://localhost:19530") and is_remote_uri("HTTPS://x") and is_remote_uri("grpc://h:1")
    assert not is_remote_uri("/tmp/m.db") and not is_remote_uri("m.db")
    assert MilvusIndexConfig(uri="http://localhost:19530").uri == "http://localhost:19530"


# ---------------------------------------------------------------------------
# 2. Bit-identical scores across the three providers
# 2. 三提供者分数逐位一致
# ---------------------------------------------------------------------------
def test_milvus_scores_bitwise_equal_across_providers(milvus_uri, store):
    """For random unit vectors, milvus / sqlite / memory return exactly equal (id, score) lists — no approx.
    对随机单位向量，milvus / sqlite / memory 返回完全相等的 (id, score) 列表——不用近似比较。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path.
        milvus_uri：共享的 milvus-lite 数据库路径。

        store: In-memory StateStore.
        store：内存 StateStore。

    Output / 输出：
        Equality assertions over full and restricted searches for ten queries.
        对十个查询的全量与限定搜索的相等断言。

    State effect / 状态影响：
        Creates the 'bitwise' collection.
        创建 'bitwise' 集合。
    """
    ids = [f"atm_{i:06d}" for i in range(200)]
    _seed_atoms(store, ids)
    trio = _three_indexes(milvus_uri, store, collection="bitwise", dims=64)
    milvus, sqlite_index, memory_index = trio
    items = list(zip(ids, _unit_vectors(200, 64, seed=7)))
    _load_all(trio, items)
    subset = ids[40:160]
    for query in _unit_vectors(10, 64, seed=99):
        for candidate_ids, top_k in ((None, 5), (None, 25), (subset, 50), (subset, 120)):
            hits = milvus.search(query, candidate_ids=candidate_ids, top_k=top_k)
            assert hits == sqlite_index.search(query, candidate_ids=candidate_ids, top_k=top_k)
            assert hits == memory_index.search(query, candidate_ids=candidate_ids, top_k=top_k)
            assert len(hits) == min(top_k, len(candidate_ids or ids)) and hits[0][1] > 0.0
    for atom_id in ids[:5]:
        assert milvus.get(atom_id) == sqlite_index.get(atom_id) == memory_index.get(atom_id)
    milvus.close()


# ---------------------------------------------------------------------------
# 3. candidate_ids chunking and merge
# 3. candidate_ids 分块与合并
# ---------------------------------------------------------------------------
def test_milvus_candidate_chunks_merge_like_sqlite(milvus_uri, store, monkeypatch):
    """3000 vectors, 2500 candidates (with missing and duplicate ids), top_k=20: chunk merge equals the sqlite answer.
    3000 个向量、2500 个候选（含缺失 id 与重复），top_k=20：分块搜索合并后与 sqlite 结果相同。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path.
        milvus_uri：共享的 milvus-lite 数据库路径。

        store: In-memory StateStore.
        store：内存 StateStore。

        monkeypatch: Counts the per-chunk searches.
        monkeypatch：统计每块的搜索次数。

    Output / 输出：
        Equality with the sqlite provider for top_k=20 and top_k=len(candidates); chunk counts for two chunk sizes.
        top_k=20 与 top_k=len(候选) 时与 sqlite 提供者相等；两种块大小下的分块计数。

    State effect / 状态影响：
        Creates the 'chunks' collection.
        创建 'chunks' 集合。
    """
    ids = [f"atm_{i:06d}" for i in range(3000)]
    _seed_atoms(store, ids)
    milvus, sqlite_index, _memory = _three_indexes(milvus_uri, store, collection="chunks", dims=64)
    items = list(zip(ids, _unit_vectors(3000, 64, seed=3)))
    _load_all((milvus, sqlite_index), items)
    assert milvus.count() == 3000
    rng = random.Random(11)
    candidates = rng.sample(ids, 2500) + [f"atm_missing_{i}" for i in range(30)]
    candidates += candidates[:40]
    rng.shuffle(candidates)
    calls: list[int] = []
    original = MilvusIndex._ann

    def counting(self, query, *, expr, limit, filtered=False, ef_floor=0):
        """Record the limit of every chunk search, then delegate.
        记录每块搜索的 limit，然后委托原实现。
        """
        calls.append(limit)
        return original(self, query, expr=expr, limit=limit, filtered=filtered, ef_floor=ef_floor)

    monkeypatch.setattr(MilvusIndex, "_ann", counting)
    fetched: list[int] = []
    original_vectors_of = MilvusIndex._vectors_of

    def counting_vectors_of(self, ids, expr=""):
        """Record the size of every whole-chunk primary-key read, then delegate.
        记录每次整块主键读取的块大小，然后委托原实现。
        """
        fetched.append(len(ids))
        return original_vectors_of(self, ids, expr)

    monkeypatch.setattr(MilvusIndex, "_vectors_of", counting_vectors_of)
    query = _unit_vectors(1, 64, seed=5)[0]
    hits = milvus.search(query, candidate_ids=candidates, top_k=20)
    assert hits == sqlite_index.search(query, candidate_ids=candidates, top_k=20)
    assert len(hits) == 20 and calls == [20, 20, 20] and fetched == []
    # The HMS scoring path asks for every candidate (top_k=len(ids)): each chunk is read whole by primary key,
    # never through the ANN.
    # HMS 打分路径会索取全部候选（top_k=len(ids)）：每块按主键整块读取，不走 ANN。
    calls.clear()
    everything = milvus.search(query, candidate_ids=candidates, top_k=len(candidates))
    assert everything == sqlite_index.search(query, candidate_ids=candidates, top_k=len(candidates))
    assert len(everything) == 2500 and calls == [] and fetched == [1024, 1024, 482]
    small = MilvusIndex({"uri": milvus_uri, "scalar_index": "none", "collection": "chunks", "candidate_chunk": 500}, store=store, dims=64)
    calls.clear()
    assert small.search(query, candidate_ids=candidates, top_k=20) == hits and calls == [20] * 6
    small.close()
    milvus.close()


# ---------------------------------------------------------------------------
# 4. Zero-score tail: truncated ANN results are completed in atom_id order
# 4. 零分尾部：被截断的 ANN 结果按 atom_id 顺序补齐
# ---------------------------------------------------------------------------
def test_milvus_zero_score_tail_matches_sqlite(milvus_uri, store, monkeypatch):
    """When fewer than top_k candidates have a positive cosine, the zero-score tail follows atom_id order like sqlite.
    当正余弦候选少于 top_k 时，零分尾部像 sqlite 一样按 atom_id 顺序排列。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path.
        milvus_uri：共享的 milvus-lite 数据库路径。

        store: In-memory StateStore.
        store：内存 StateStore。

        monkeypatch: Spies on the id back-fill.
        monkeypatch：监视 id 回填。

    Output / 输出：
        Equality with sqlite for the candidate and the whole-collection paths; the back-fill ran exactly once each.
        候选路径与全集合路径都与 sqlite 相等；回填各恰好执行一次。

    State effect / 状态影响：
        Creates the 'zeros' collection.
        创建 'zeros' 集合。
    """
    rng = random.Random(21)
    ids = [f"atm_{i:06d}" for i in range(43)]
    rng.shuffle(ids)
    vectors: list[list[float]] = []
    for position in range(43):
        raw = [rng.gauss(0.0, 1.0) for _ in range(8)]
        # Three positives along e1; forty negatives whose cosine order differs from their atom_id order.
        # 沿 e1 的三个正向量；四十个负向量，其余弦顺序与 atom_id 顺序不同。
        raw[0] = abs(raw[0]) + 1.0 if position < 3 else -(abs(raw[0]) + 0.5 + position / 100.0)
        norm = math.sqrt(sum(x * x for x in raw))
        vectors.append([x / norm for x in raw])
    _seed_atoms(store, ids)
    milvus, sqlite_index, _memory = _three_indexes(milvus_uri, store, collection="zeros", dims=8)
    _load_all((milvus, sqlite_index), list(zip(ids, vectors)), embedder="hash@8")
    query = [1.0] + [0.0] * 7
    fills: list[Any] = []
    original = MilvusIndex._existing_ids

    def spying(self, wanted, expr=""):
        """Record the back-fill scope, then delegate.
        记录回填范围，然后委托原实现。
        """
        fills.append(wanted)
        return original(self, wanted, expr)

    monkeypatch.setattr(MilvusIndex, "_existing_ids", spying)
    expected = sqlite_index.search(query, candidate_ids=ids, top_k=10)
    assert [s for _, s in expected][:3] > [0.0] * 3 and [s for _, s in expected][3:] == [0.0] * 7
    assert milvus.search(query, candidate_ids=ids, top_k=10) == expected and len(fills) == 1
    whole = milvus.search(query, candidate_ids=None, top_k=10)
    assert whole == sqlite_index.search(query, candidate_ids=None, top_k=10)
    assert len(fills) == 2 and fills[1] is None and sorted(fills[0]) == sorted(ids)
    # Untruncated searches never back-fill.
    # 未被截断的搜索从不回填。
    assert milvus.search(query, candidate_ids=ids, top_k=43) == sqlite_index.search(query, candidate_ids=ids, top_k=43)
    assert milvus.search(query, candidate_ids=ids[:2], top_k=10) == sqlite_index.search(
        query, candidate_ids=ids[:2], top_k=10
    )
    assert len(fills) == 2
    milvus.close()


# ---------------------------------------------------------------------------
# 5. dims mismatch and foreign collections
# 5. 维数不一致与外来集合
# ---------------------------------------------------------------------------
def test_milvus_dims_mismatch_and_foreign_collection_raise(milvus_uri, store):
    """Re-opening a collection with another dims, or an alien schema, raises ConfigError(CONFIG_001) with a hint.
    以另一维数重开集合、或打开异构模式的集合，抛出带提示的 ConfigError(CONFIG_001)。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path.
        milvus_uri：共享的 milvus-lite 数据库路径。

        store: In-memory StateStore.
        store：内存 StateStore。

    Output / 输出：
        Assertions over error codes, details and messages.
        对错误码、details 与消息的断言。

    State effect / 状态影响：
        Creates the 'dims_guard' and 'foreign' collections.
        创建 'dims_guard' 与 'foreign' 集合。
    """
    pymilvus = pytest.importorskip("pymilvus")
    first = MilvusIndex({"uri": milvus_uri, "scalar_index": "none", "collection": "dims_guard"}, store=store, dims=8)
    first.upsert("atm_000001", [1.0] + [0.0] * 7, embedder="hash@8")
    first.close()
    with pytest.raises(ConfigError) as info:
        MilvusIndex({"uri": milvus_uri, "scalar_index": "none", "collection": "dims_guard"}, store=store, dims=16)
    assert info.value.error_code == "CONFIG_001" and "8-dim" in info.value.message and "16-dim" in info.value.message
    assert info.value.details["collection_dims"] == 8 and info.value.details["embedder_dims"] == 16
    assert info.value.details["collection"] == "dims_guard" and "collection" in (info.value.suggestion or "")
    with pytest.raises(ConfigError) as info:
        IndexFactory.create("milvus", {"uri": milvus_uri, "collection": "dims_guard"}, store=store, dims=16)
    assert info.value.error_code == "CONFIG_001"
    reopened = IndexFactory.create("milvus", {"uri": milvus_uri, "collection": "dims_guard"}, store=store, dims=8)
    assert isinstance(reopened, MilvusIndex) and reopened.count() == 1
    reopened.close()
    client = pymilvus.MilvusClient(milvus_uri)
    client.create_collection("foreign", dimension=8)
    client.close()
    with pytest.raises(ConfigError) as info:
        MilvusIndex({"uri": milvus_uri, "scalar_index": "none", "collection": "foreign"}, store=store, dims=8)
    assert info.value.error_code == "CONFIG_001" and "not an HMS atom collection" in info.value.message


# ---------------------------------------------------------------------------
# 6. Dependency guard and registry wiring (runs even without pymilvus)
# 6. 依赖守卫与注册表接线（没有 pymilvus 也运行）
# ---------------------------------------------------------------------------
def test_milvus_dependency_guard_and_registry(monkeypatch, tmp_path, store):
    """Without pymilvus the constructor raises DependencyError(DEPS_001, extra='milvus'); the module still imports.
    没有 pymilvus 时构造函数抛 DependencyError(DEPS_001, extra='milvus')；模块本身仍可导入。

    Input / 输入：
        monkeypatch: Hides pymilvus through sys.modules.
        monkeypatch：通过 sys.modules 隐藏 pymilvus。

        tmp_path: pytest temporary directory.
        tmp_path：pytest 临时目录。

        store: In-memory StateStore.
        store：内存 StateStore。

    Output / 输出：
        Assertions over DependencyError, IndexFactory, registry entries and HMSConfig validation.
        对 DependencyError、IndexFactory、注册表条目与 HMSConfig 校验的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules['pymilvus'] (restored by monkeypatch).
        临时替换 sys.modules['pymilvus']（由 monkeypatch 恢复）。
    """
    assert registry.supported("index") == ["memory", "milvus", "sqlite"]
    assert registry.EXTRA_OF_PROVIDER["milvus"] == "milvus"
    assert IndexFactory.get_supported_providers() == ["memory", "milvus", "sqlite"]
    config = HMSConfig(vector_store={"provider": "milvus", "config": {"uri": "http://localhost:19530"}})
    assert config.vector_store.provider == "milvus"
    assert HMSConfig.from_dict({"vector_store": {"provider": "milvus"}}).vector_store.config == {}
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    for build in (
        lambda: MilvusIndex({"uri": str(tmp_path / "x.db")}, store=store, dims=8),
        lambda: IndexFactory.create("milvus", {"uri": str(tmp_path / "x.db")}, store=store, dims=8),
    ):
        with pytest.raises(DependencyError) as info:
            build()
        assert info.value.error_code == "DEPS_001"
        assert info.value.details == {"package": "pymilvus", "extra": "milvus"}
        assert info.value.suggestion == "pip install 'hms-elite[milvus]'"
        assert isinstance(info.value.__cause__, ImportError)
    assert not (tmp_path / "x.db").exists()


# ---------------------------------------------------------------------------
# 7. HMS end-to-end: add / search / consolidate / why equal the sqlite provider
# 7. HMS 端到端：add / search / consolidate / why 与 sqlite 提供者一致
# ---------------------------------------------------------------------------
def test_milvus_hms_end_to_end_matches_sqlite(milvus_uri, tmp_path):
    """HMS on the milvus index yields the same search rows, consolidate report and why() chain as on sqlite.
    基于 milvus 索引的 HMS 与 sqlite 上得到相同的 search 行、consolidate 报告与 why() 链。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path (only used to skip without pymilvus).
        milvus_uri：共享的 milvus-lite 数据库路径（仅用于在缺少 pymilvus 时跳过）。

        tmp_path: pytest temporary directory holding the dedicated 'm.db'.
        tmp_path：存放专用 'm.db' 的 pytest 临时目录。

    Output / 输出：
        Equality of search results, consolidate phases and why() between the two providers; vectors live in Milvus;
        HMS.close() has released the Milvus client.
        两种提供者的 search 结果、consolidate 阶段与 why() 相等；向量只存在 Milvus 中；HMS.close() 已释放 Milvus 客户端。

    State effect / 状态影响：
        Creates tmp_path/m.db (milvus-lite) and two in-memory stores.
        创建 tmp_path/m.db（milvus-lite）与两个内存 store。
    """
    from hms_elite.memory.main import HMS

    messages = [
        {"role": "user", "content": "I prefer oat-milk lattes in the morning."},
        {"role": "assistant", "content": "Noted: oat-milk lattes in the morning."},
        {"role": "user", "content": "billing_e2e failed after the retry patch, so the retry patch must not ship."},
        {"role": "user", "content": "Sarah joined the billing team in March."},
    ]
    runs: dict[str, Any] = {}
    for provider in ("sqlite", "milvus"):
        env = {"config": {"uri": tmp_path / "m.db", "scalar_index": "none"}} if provider == "milvus" else {}
        hms = HMS(
            HMSConfig(vector_store={"provider": provider, **env}, store={"path": ":memory:"}),
            store=StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds()),
        )
        added = hms.add(messages, user_id="alice")
        search = hms.search("what coffee does the user like", filters={"user_id": "alice"}, top_k=5)
        report = hms.consolidate(filters={"user_id": "alice"})
        target = search["results"][0]["id"]
        runs[provider] = {
            "added": [(r["id"], r["memory"]) for r in added["results"]],
            "search": [(r["id"], r["score"]) for r in search["results"]],
            "phases": report["phases"],
            "why": hms.why(target, filters={"user_id": "alice"}),
            "count": hms.index.count(),
            "atom_vectors": hms.store.count_vectors(),
            "index": hms.index,
            "info": hms.index.info(),
        }
        hms.close()
    sqlite_run, milvus_run = runs["sqlite"], runs["milvus"]
    assert isinstance(milvus_run["index"], MilvusIndex) and milvus_run["info"]["uri"] == str(tmp_path / "m.db")
    assert milvus_run["added"] == sqlite_run["added"] and len(milvus_run["added"]) >= 2
    assert milvus_run["search"] == sqlite_run["search"] and milvus_run["search"][0][1] > 0.0
    assert milvus_run["phases"] == sqlite_run["phases"]
    assert milvus_run["why"] == sqlite_run["why"] and milvus_run["why"]["nodes"]
    assert milvus_run["count"] == sqlite_run["count"] > 0
    assert milvus_run["atom_vectors"] == 0 and sqlite_run["atom_vectors"] == sqlite_run["count"]
    # hms.close() reached MilvusIndex.close(): the client is gone and a second close() is a no-op
    # hms.close() 已调到 MilvusIndex.close()：客户端已释放，再次 close() 是空操作
    with pytest.raises(StoreError) as closed:
        milvus_run["index"].count()
    assert closed.value.error_code == "STORE_001" and "closed" in closed.value.message
    milvus_run["index"].close()


# ---------------------------------------------------------------------------
# 7. Scalar gates: native filtered ANN equals the sqlite gate; attrs come from the store; update_attrs follows SQLite
# 7. 标量门控：原生过滤 ANN 等于 sqlite 门控；标量来自 store；update_attrs 跟随 SQLite
# ---------------------------------------------------------------------------
_AS_OF = "2026-06-15T09:00:00Z"
_QUOTED_RUN = 'run"q"\\1'


def _seed_gate_corpus(store: StateStore) -> list[str]:
    """120 atoms over three scopes, six months, two scales / obs_types; some expired by valid_to; ids in insert order.
    120 个原子，分布在三个范围、六个月、两种尺度 / obs_type；部分因 valid_to 过期；按插入顺序返回 id。
    """
    scopes = [
        models.Scope(user_id="alice"),
        models.Scope(user_id="bob"),
        models.Scope(user_id="alice", agent_id="agent", run_id=_QUOTED_RUN),
    ]
    anchors = {}
    for scope in scopes:
        event = models.SourceEvent(scope=scope, role="user", payload={"text": "seed"}, timestamp=_AS_OF)
        stored, _ = store.insert_source_event(event)
        anchors[scope.key()] = store.get_or_create_anchor(stored).anchor_id
    ids: list[str] = []
    with store.transaction():
        for i in range(120):
            scope = scopes[i % 3]
            atom_id = f"atm_{i:06d}"
            store.insert_atom(
                models.MemoryAtom(
                    atom_id=atom_id,
                    atom_type="state" if i % 4 else "event",
                    content={"text": f"atom {i}"},
                    text=f"atom {i}",
                    scale_level="state" if i % 4 else "event",
                    source_kind="observation",
                    source_id=f"obs_{i}",
                    extracted_by="rule@obs-v1",
                    scope=scope,
                    anchor_ids=[anchors[scope.key()]],
                    obs_type="preference_event" if i % 5 == 0 else "fact",
                    subject="user",
                    predicate="states",
                    time_anchor=f"2026-{(i % 6) + 1:02d}-10T09:00:00Z",
                    valid_to="2026-06-14T00:00:00Z" if i % 7 == 0 else None,
                )
            )
            ids.append(atom_id)
    return ids


def test_milvus_native_filter_matches_sqlite_gate(milvus_uri, store):
    """search(filter=IndexFilter) on a FLAT collection equals the sqlite provider over store.axis_ids(...), bit for bit.
    FLAT 集合上的 search(filter=IndexFilter) 与 sqlite 提供者在 store.axis_ids(...) 上的结果逐位相等。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path.
        milvus_uri：共享的 milvus-lite 数据库路径。

        store: In-memory StateStore (the gate's source of truth).
        store：内存 StateStore（门控的真源）。

    Output / 输出：
        Equality for scope / validity / scale / obs_type / time / include_expired filters and the quoted run_id, for
        filter ∩ candidate_ids, for the zero-score tail under a filter, and after archive / restore through the
        attribute listener; HNSW raises ef for filtered searches.
        对 scope / 有效期 / 尺度 / obs_type / 时间 / include_expired 过滤器与带引号 run_id、filter ∩ candidate_ids、
        过滤下的零分尾部、以及经属性监听器归档 / 恢复之后的相等断言；HNSW 在过滤搜索时提高 ef。

    State effect / 状态影响：
        Creates the 'gate' and 'gate_hnsw' collections.
        创建 'gate' 与 'gate_hnsw' 集合。
    """
    from hms_elite.index import IndexFilter

    ids = _seed_gate_corpus(store)
    store.archive_atom(ids[3], reason="revised", at=_AS_OF)  # archived before the vectors are loaded / 向量加载前归档
    milvus = MilvusIndex({"uri": milvus_uri, "scalar_index": "none", "collection": "gate", "index_type": "FLAT"}, store=store, dims=16)
    sqlite_index = SQLiteIndex({"use_numpy": "auto"}, store=store, dims=16)
    items = list(zip(ids, _unit_vectors(120, 16, seed=17)))
    _load_all((milvus, sqlite_index), items, embedder="hash@16")
    store.attr_listeners.append(milvus.update_attrs)
    store.archive_atom(ids[6], reason="deleted", at=_AS_OF)  # synced through the listener / 经监听器同步
    # Both providers push the gate down: Milvus as a scalar expression, SQLite as a numpy mask over its cache.
    # 两个提供者都下推门控：Milvus 以标量表达式，SQLite 以其缓存上的 numpy 掩码。
    assert milvus.supports_filter is True and sqlite_index.supports_filter is True
    alice, bob = models.Scope(user_id="alice"), models.Scope(user_id="bob")
    run = models.Scope(user_id="alice", agent_id="agent", run_id=_QUOTED_RUN)
    cases = [
        (IndexFilter(scope=alice, as_of=_AS_OF), dict(scope=alice, as_of=_AS_OF)),
        (IndexFilter(scope=bob, as_of=_AS_OF), dict(scope=bob, as_of=_AS_OF)),
        (IndexFilter(scope=run, as_of=_AS_OF), dict(scope=run, as_of=_AS_OF)),
        (IndexFilter(scope=alice, as_of=_AS_OF, include_expired=True), dict(scope=alice, include_expired=True)),
        (IndexFilter(scope=alice, as_of=_AS_OF, scale_in=("event",)), dict(scope=alice, as_of=_AS_OF, scale_constraint=["event"])),
        (
            IndexFilter(scope=alice, as_of=_AS_OF, obs_type_in=("preference_event",)),
            dict(scope=alice, as_of=_AS_OF, where_sql="memory_atoms.obs_type IN (?)", params=("preference_event",)),
        ),
        (
            IndexFilter(scope=alice, as_of=_AS_OF, time_from="2026-02-01", time_to="2026-03-31T23:59:59Z"),
            dict(
                scope=alice,
                as_of=_AS_OF,
                where_sql="memory_atoms.time_anchor >= ? AND memory_atoms.time_anchor <= ?",
                params=("2026-02-01T00:00:00.000000+00:00", "2026-03-31T23:59:59.000000+00:00"),
            ),
        ),
        (IndexFilter(scope=alice, as_of="2026-04-01"), dict(scope=alice, as_of="2026-04-01")),
        (IndexFilter(as_of=_AS_OF), dict(scope=models.Scope(), as_of=_AS_OF)),
    ]
    queries = _unit_vectors(3, 16, seed=23) + [[1.0] + [0.0] * 15]
    for spec, gate_kwargs in cases:
        gate = store.axis_ids(**gate_kwargs)
        assert gate and ids[3] not in gate and ids[6] not in gate or spec.include_expired
        for query in queries:
            for top_k in (5, 20, 200):
                expected = sqlite_index.search(query, candidate_ids=gate, top_k=top_k)
                assert milvus.search(query, filter=spec, top_k=top_k) == expected, (spec, top_k)
                assert sqlite_index.search(query, filter=spec, top_k=top_k) == expected, (spec, top_k)
        assert sqlite_index.count(spec) == len(sqlite_index.search(queries[0], filter=spec, top_k=10**6))
        assert sqlite_index.count(spec) == milvus.count(spec)
    # filter ∩ candidate_ids is the exact intersection (chunked id filters ANDed with the gate expression).
    # filter ∩ candidate_ids 是精确交集（分块 id 过滤 AND 门控表达式）。
    spec = IndexFilter(scope=alice, as_of=_AS_OF)
    gate = set(store.axis_ids(scope=alice, as_of=_AS_OF))
    subset = ids[::2] + ["atm_missing"]
    both = milvus.search(queries[0], candidate_ids=subset, filter=spec, top_k=30)
    assert both == sqlite_index.search(queries[0], candidate_ids=[i for i in subset if i in gate], top_k=30)
    assert {atom_id for atom_id, _score in both} <= gate and len(both) == 30
    # archive / restore reach Milvus after COMMIT: the row leaves and re-enters the filtered result.
    # 归档 / 恢复在 COMMIT 后到达 Milvus：该行离开又重新进入过滤结果。
    target = ids[9]  # alice, live, not expired / alice、存活、未过期
    assert target in {atom_id for atom_id, _score in milvus.search(queries[3], filter=spec, top_k=200)}
    store.archive_atom(target, reason="deleted", at=_AS_OF)
    assert target not in {atom_id for atom_id, _score in milvus.search(queries[3], filter=spec, top_k=200)}
    assert milvus.search(queries[3], filter=spec, top_k=200) == sqlite_index.search(
        queries[3], candidate_ids=store.axis_ids(scope=alice, as_of=_AS_OF), top_k=200
    )
    store.restore_atom(target, at=_AS_OF)
    assert target in {atom_id for atom_id, _score in milvus.search(queries[3], filter=spec, top_k=200)}
    store.set_valid_to(target, "2026-05-01T00:00:00Z", at=_AS_OF)  # valid_from is its time_anchor 2026-04-10 / valid_from 即其 time_anchor 2026-04-10
    assert target not in {atom_id for atom_id, _score in milvus.search(queries[3], filter=spec, top_k=200)}
    earlier = IndexFilter(scope=alice, as_of="2026-04-20")
    assert target in {atom_id for atom_id, _score in milvus.search(queries[3], filter=earlier, top_k=200)}
    assert milvus.search(queries[3], filter=earlier, top_k=200) == sqlite_index.search(
        queries[3], candidate_ids=store.axis_ids(scope=alice, as_of="2026-04-20"), top_k=200
    )
    store.attr_listeners.remove(milvus.update_attrs)
    milvus.close()
    # HNSW: a filtered search raises ef to ef_recall_factor × limit and returns only gated rows.
    # HNSW：过滤搜索把 ef 提高到 ef_recall_factor × limit，且只返回门控内的行。
    hnsw = MilvusIndex(
        {"uri": milvus_uri, "scalar_index": "none", "collection": "gate_hnsw", "index_type": "HNSW", "ef": 2, "ef_recall_factor": 4},
        store=store,
        dims=16,
    )
    assert hnsw._search_params(20, filtered=True)["params"]["ef"] == 80 and hnsw._search_params(20)["params"]["ef"] == 20
    hnsw.upsert_batch(items, embedder="hash@16")
    hits = hnsw.search(queries[0], filter=spec, top_k=10)
    assert len(hits) == 10 and {atom_id for atom_id, _score in hits} <= set(store.axis_ids(scope=alice, as_of=_AS_OF))
    hnsw.close()


def test_milvus_attrs_from_store_and_update_attrs(milvus_uri, store):
    """upsert without attrs reads the scalars from the store; explicit attrs win; update_attrs rewrites one row in place.
    不带 attrs 的 upsert 从 store 读取标量；显式 attrs 优先；update_attrs 原地重写一行。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path.
        milvus_uri：共享的 milvus-lite 数据库路径。

        store: In-memory StateStore.
        store：内存 StateStore。

    Output / 输出：
        Assertions on the stored scalar rows (get with output_fields), on defaults for ids the store does not know,
        on update_attrs keeping the vector, and on the no-op for absent ids.
        对已存标量行（带 output_fields 的 get）、store 不认识的 id 的默认值、update_attrs 保留向量、
        以及不存在 id 的空操作的断言。

    State effect / 状态影响：
        Creates the 'attrs' collection.
        创建 'attrs' 集合。
    """
    from hms_elite.index import INDEX_ATTR_KEYS, atom_index_attrs
    from hms_elite.index.milvus import ATTR_DEFAULTS, SCALAR_FIELDS

    ids = _seed_gate_corpus(store)
    milvus = MilvusIndex({"uri": milvus_uri, "scalar_index": "none", "collection": "attrs", "index_type": "FLAT"}, store=store, dims=16)
    vectors = _unit_vectors(3, 16, seed=5)
    fields = list(INDEX_ATTR_KEYS) + ["embedder"]

    def row(atom_id: str) -> dict:
        got = milvus._client.get("attrs", ids=[atom_id], output_fields=fields)
        return {key: got[0][key] for key in fields}

    milvus.upsert(ids[9], vectors[0], embedder="hash@16")
    expected = atom_index_attrs(store.get_atom(ids[9]))
    assert row(ids[9]) == {**expected, "embedder": "hash@16"} and expected["user_id"] == "alice"
    assert row(ids[9])["valid_to"] == "" and row(ids[9])["archived"] is False
    milvus.upsert(ids[0], vectors[0], embedder="hash@16")
    assert row(ids[0])["valid_to"] == "2026-06-14T00:00:00.000000+00:00"  # expired rows keep their boundary / 过期行保留其边界
    milvus.upsert(ids[2], vectors[1], embedder="hash@16")
    assert row(ids[2])["run_id"] == _QUOTED_RUN and row(ids[2])["agent_id"] == "agent"
    # Explicit attrs are layered over the store's row: only the given keys change.
    # 显式 attrs 叠加在 store 的行之上：只有给定的键改变。
    milvus.upsert(ids[7], vectors[2], embedder="hash@16", attrs={"user_id": "carol", "valid_to": None, "archived": 1})
    stored = row(ids[7])
    assert stored["user_id"] == "carol" and stored["archived"] is True and stored["scale_level"] == "state"
    assert stored["valid_to"] == "" and stored["time_anchor"] == store.get_atom(ids[7]).time_anchor
    assert set(name for name, _length in SCALAR_FIELDS) | {"archived"} == set(INDEX_ATTR_KEYS)
    # An id the store does not know (no atom_vectors foreign key here) gets the defaults ("" and archived=false).
    # store 不认识的 id（这里没有 atom_vectors 外键）得到默认值（"" 与 archived=false）。
    milvus.upsert("atm_unknown", vectors[0], embedder="hash@16")
    assert row("atm_unknown") == {**ATTR_DEFAULTS, "embedder": "hash@16"}
    milvus.store = None
    milvus.upsert_batch([("atm_detached", vectors[0])], embedder="hash@16")
    assert row("atm_detached") == {**ATTR_DEFAULTS, "embedder": "hash@16"}
    milvus.store = store
    # update_attrs: only the given keys change, the vector survives; absent ids are ignored.
    # update_attrs：只改给定键，向量保留；不存在的 id 忽略。
    milvus.update_attrs(ids[9], {"archived": True, "valid_to": "2026-07-01T00:00:00Z"})
    changed = row(ids[9])
    assert changed["archived"] is True and changed["valid_to"] == "2026-07-01T00:00:00.000000+00:00"
    assert changed["user_id"] == "alice" and milvus.get(ids[9]) == milvus._rows([(ids[9], vectors[0])], "x")[0]["vector"]
    milvus.update_attrs("atm_absent", {"archived": True})
    milvus.update_attrs(ids[9], {"bogus": 1})
    assert milvus.get("atm_absent") is None and row(ids[9]) == changed and milvus.count() == 6
    milvus.close()


def test_milvus_scalar_schema_guard(milvus_uri, store):
    """A collection without the scalar gate fields (an older HMS layout) is rejected with CONFIG_001 naming the missing fields.
    没有标量门控字段的集合（旧的 HMS 布局）被拒绝，CONFIG_001 指出缺失字段。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path.
        milvus_uri：共享的 milvus-lite 数据库路径。

        store: In-memory StateStore.
        store：内存 StateStore。

    Output / 输出：
        ConfigError details['missing'] lists every scalar field; the suggestion names consolidate().
        ConfigError 的 details['missing'] 列出每个标量字段；建议提到 consolidate()。

    State effect / 状态影响：
        Creates the 'legacy' collection by hand.
        手工创建 'legacy' 集合。
    """
    pymilvus = pytest.importorskip("pymilvus")
    client = pymilvus.MilvusClient(uri=milvus_uri)
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("atom_id", pymilvus.DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("vector", pymilvus.DataType.FLOAT_VECTOR, dim=8)
        schema.add_field("embedder", pymilvus.DataType.VARCHAR, max_length=128)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_collection("legacy", schema=schema, index_params=index_params)
    finally:
        client.close()
    with pytest.raises(ConfigError) as info:
        MilvusIndex({"uri": milvus_uri, "scalar_index": "none", "collection": "legacy"}, store=store, dims=8)
    assert info.value.error_code == "CONFIG_001" and "consolidate()" in (info.value.suggestion or "")
    assert info.value.details["missing"] == sorted(
        ["archived", "user_id", "agent_id", "run_id", "time_anchor", "valid_from", "valid_to", "scale_level", "obs_type", "atom_type"]
    )


def test_archive_hides_from_every_provider_and_rollback_restores(milvus_uri, tmp_path):
    """On sqlite / memory / milvus, a deleted memory leaves the semantic recall gate and returns after rollback().
    在 sqlite / memory / milvus 上，被删除的记忆离开语义召回门控，rollback() 之后回来。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path (skips without pymilvus).
        milvus_uri：共享的 milvus-lite 数据库路径（缺少 pymilvus 时跳过）。

        tmp_path: pytest temporary directory holding the dedicated 'sync.db'.
        tmp_path：存放专用 'sync.db' 的 pytest 临时目录。

    Output / 输出：
        For every provider: the deleted atom is absent from index.search (filter pushdown on milvus, axis_ids gate
        on sqlite / memory) and from HMS.search after delete(), present again after rollback().
        对每个提供者：delete() 之后被删原子不在 index.search（milvus 走过滤下推，sqlite / memory 走 axis_ids 门控）
        与 HMS.search 中；rollback() 之后再次出现。

    State effect / 状态影响：
        Creates tmp_path/sync.db (milvus-lite) and three in-memory stores.
        创建 tmp_path/sync.db（milvus-lite）与三个内存 store。
    """
    from hms_elite.index import IndexFilter
    from hms_elite.memory.main import HMS

    messages = [
        {"role": "user", "content": "I prefer oat-milk lattes in the morning."},
        {"role": "user", "content": "billing_e2e failed after the retry patch."},
        {"role": "user", "content": "Sarah joined the billing team in March."},
    ]
    for provider in ("sqlite", "memory", "milvus"):
        env = {"config": {"uri": tmp_path / "sync.db", "index_type": "FLAT", "scalar_index": "none"}} if provider == "milvus" else {}
        hms = HMS(
            HMSConfig(vector_store={"provider": provider, **env}, store={"path": ":memory:"}),
            store=StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds()),
        )
        assert hms.index.update_attrs in hms.store.attr_listeners
        added = hms.add(messages, user_id="alice", infer=False)
        target = next(r for r in added["results"] if "oat-milk" in r["memory"])
        atom_id = hms.store.resolve_head(target["id"]).atom_id
        query = hms.embedder.embed("what coffee does the user like")
        scope = models.Scope(user_id="alice")

        def recalled() -> set[str]:
            if hms.index.supports_filter:
                hits = hms.index.search(query, filter=IndexFilter(scope=scope, as_of=hms.store.now()), top_k=10)
            else:
                hits = hms.index.search(query, candidate_ids=hms.store.axis_ids(scope, as_of=hms.store.now()), top_k=10)
            return {hit for hit, _score in hits}

        def searched() -> set[str]:
            return {r["id"] for r in hms.search("what coffee does the user like", filters={"user_id": "alice"}, top_k=10)["results"]}

        assert atom_id in recalled() and target["id"] in searched()
        checkpoint_id = hms.checkpoint(filters={"user_id": "alice"})
        hms.delete(target["id"])
        assert atom_id not in recalled() and target["id"] not in searched(), provider
        assert hms.index.count() == 3  # the vector stays; only the gate scalars / SQL validity changed
        # 向量仍在；只有门控标量 / SQL 有效性改变了
        report = hms.rollback(checkpoint_id)
        assert atom_id in report["restored"] and atom_id in recalled() and target["id"] in searched(), provider
        hms.close()


def test_milvus_default_scalar_indexes(milvus_uri, store):
    """The default config builds one INVERTED index per SCALAR_INDEXED field; 'none' builds only the vector index.
    默认配置为每个 SCALAR_INDEXED 字段建一个 INVERTED 索引；'none' 只建向量索引。

    Input / 输入：
        milvus_uri: Shared milvus-lite database path.
        milvus_uri：共享的 milvus-lite 数据库路径。

        store: In-memory StateStore.
        store：内存 StateStore。

    Output / 输出：
        Assertions on list_indexes for both settings and on a filtered search working under both.
        对两种设置下 list_indexes 以及过滤搜索都能工作的断言。

    State effect / 状态影响：
        Creates the 'scalars' and 'scalars_none' collections (the former pays the one-time index build).
        创建 'scalars' 与 'scalars_none' 集合（前者付出一次性建索引成本）。
    """
    from hms_elite.index import IndexFilter
    from hms_elite.index.milvus import SCALAR_INDEXED

    ids = _seed_gate_corpus(store)[:12]
    items = list(zip(ids, _unit_vectors(12, 8, seed=9)))
    indexed = MilvusIndex({"uri": milvus_uri, "collection": "scalars", "index_type": "FLAT"}, store=store, dims=8)
    bare = MilvusIndex(
        {"uri": milvus_uri, "collection": "scalars_none", "index_type": "FLAT", "scalar_index": "none"}, store=store, dims=8
    )
    assert set(SCALAR_INDEXED) <= {name for name, _length in __import__("hms_elite.index.milvus", fromlist=["SCALAR_FIELDS"]).SCALAR_FIELDS} | {"archived"}
    with_scalars = {str(name) for name in indexed._client.list_indexes("scalars")}
    without = {str(name) for name in bare._client.list_indexes("scalars_none")}
    assert len(with_scalars) == len(SCALAR_INDEXED) + 1 and len(without) == 1
    spec = IndexFilter(scope=models.Scope(user_id="alice"), as_of=_AS_OF)
    gate = store.axis_ids(scope=models.Scope(user_id="alice"), as_of=_AS_OF)
    for index in (indexed, bare):
        index.upsert_batch(items, embedder="hash@8")
        hits = index.search(items[0][1], filter=spec, top_k=20)
        assert hits and {atom_id for atom_id, _score in hits} == {i for i in ids if i in gate}
        index.close()
