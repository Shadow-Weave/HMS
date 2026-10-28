"""The five arms behind one interface: mem0_faiss, mem0_milvus, elite_matched, elite_full, elite_infer.
统一接口后的五个臂：mem0_faiss、mem0_milvus、elite_matched、elite_full、elite_infer。
"""

# Purpose (EN): Wrap official mem0ai 2.0.20 (FAISS semantic-only; milvus-lite with its own BM25 hybrid; stub LLM;
#               shared hash embedder) and three HMS Elite configurations so run_egolife.py only sees
#               retain / recall / stats / close.
# 目的（中）：把官方 mem0ai 2.0.20（FAISS 仅语义；milvus-lite 带其自身 BM25 混合；桩 LLM；共享哈希嵌入器）与三种
#            HMS Elite 配置包成 retain / recall / stats / close，run_egolife.py 只看到这四个方法。

# Input (EN): Window objects (windows.py), question text and the query instant (ISO), an arm working directory.
# 输入（中）：Window 对象（windows.py）、题干与查询时刻（ISO）、臂的工作目录。

# Output (EN): Recall rows {window_id, text, score, memory_id}; stats with embedding / LLM call counts and DB bytes.
# 输出（中）：召回行 {window_id, text, score, memory_id}；统计含嵌入 / LLM 调用数与库字节数。

# State effect (EN): Each arm writes only inside its own directory; MEM0_TELEMETRY is forced off before mem0 is
#                    imported (no network); the mem0 factories gain two provider names (egolife_hash, egolife_stub).
# 状态影响（中）：每个臂只写自己的目录；导入 mem0 之前强制关闭 MEM0_TELEMETRY（不联网）；mem0 的工厂新增两个
#                提供者名（egolife_hash、egolife_stub）。

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

os.environ["MEM0_TELEMETRY"] = "False"  # must precede `import mem0` / 必须早于 `import mem0`

from mem0.configs.base import MemoryConfig  # noqa: E402
from mem0.configs.llms.base import BaseLlmConfig  # noqa: E402
from mem0.embeddings.base import EmbeddingBase  # noqa: E402
from mem0.embeddings.configs import EmbedderConfig as Mem0EmbedderConfig  # noqa: E402
from mem0.llms.base import LLMBase  # noqa: E402
from mem0.llms.configs import LlmConfig as Mem0LlmConfig  # noqa: E402
from mem0.memory.main import Memory  # noqa: E402
from mem0.utils.factory import EmbedderFactory, LlmFactory  # noqa: E402
from mem0.vector_stores.base import VectorStoreBase  # noqa: E402
from windows import Window  # noqa: E402

from hms_elite import HMS, HMSConfig  # noqa: E402
from hms_elite.embeddings.hash import HashEmbedder, HashEmbedderConfig  # noqa: E402

BANK = "egolife_a1_jake"
TOP_K = 20
THRESHOLD = 0.1
EMBEDDING_DIMS = 64
ARM_NAMES = ("mem0_faiss", "mem0_milvus", "elite_matched", "elite_full", "elite_infer")


def shared_hash_embedder() -> HashEmbedder:
    """The one embedder algorithm every arm uses: hms_elite HashEmbedder at 64 dims, default n-grams (2..3).
    所有臂使用的同一嵌入算法：hms_elite HashEmbedder，64 维，默认 n-gram（2..3）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        A fresh HashEmbedder instance (stateless apart from its usage meter).
        一个新的 HashEmbedder 实例（除用量计量器外无状态）。

    State effect / 状态影响：
        None.
        无。
    """
    return HashEmbedder(HashEmbedderConfig(embedding_dims=EMBEDDING_DIMS))


class SharedHashEmbedding(EmbeddingBase):
    """mem0 EmbeddingBase shell over hms_elite.embeddings.hash.HashEmbedder.embed (same function, same bits).
    包住 hms_elite.embeddings.hash.HashEmbedder.embed 的 mem0 EmbeddingBase 薄壳（同一函数、同一比特）。
    """

    def __init__(self, config: Any = None) -> None:
        """Build the inner HashEmbedder; mem0 passes a BaseEmbedderConfig(embedding_dims=64).
        构建内部 HashEmbedder；mem0 传入 BaseEmbedderConfig(embedding_dims=64)。

        Input / 输入：
            config: mem0 BaseEmbedderConfig (embedding_dims read, other fields ignored).
            config：mem0 BaseEmbedderConfig（读取 embedding_dims，其余忽略）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Starts the call counter at 0.
            调用计数从 0 开始。
        """
        super().__init__(config)
        dims = int(getattr(self.config, "embedding_dims", None) or EMBEDDING_DIMS)
        if dims != EMBEDDING_DIMS:
            raise ValueError(f"shared embedder must be {EMBEDDING_DIMS}-dim, got {dims}")
        self.config.embedding_dims = dims
        self.inner = shared_hash_embedder()
        self.calls = 0

    def embed(self, text: str, memory_action: Any = None) -> list[float]:
        """Delegate to HashEmbedder.embed and count the call.
        委托给 HashEmbedder.embed 并计数。

        Input / 输入：
            text: Text to embed.
            text：待嵌入文本。

            memory_action: Ignored (mem0 passes 'add' / 'search').
            memory_action：忽略（mem0 传 'add' / 'search'）。

        Output / 输出：
            64 floats.
            64 个浮点数。

        State effect / 状态影响：
            calls += 1.
            calls += 1。
        """
        self.calls += 1
        return self.inner.embed(text)


class StubLLM(LLMBase):
    """LLM stub for the infer=False arms: any generation call raises and is counted.
    infer=False 臂的 LLM 桩：任何生成调用都抛错并计数。
    """

    def __init__(self, config: Any = None) -> None:
        """Accept mem0's config envelope without validating a model name.
        接受 mem0 的配置信封，不校验模型名。

        Input / 输入：
            config: BaseLlmConfig or dict.
            config：BaseLlmConfig 或字典。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Starts the attempt counter at 0.
            尝试计数从 0 开始。
        """
        self.config = config if isinstance(config, BaseLlmConfig) else BaseLlmConfig(**(config or {}))
        self.calls = 0

    def generate_response(self, *args: Any, **kwargs: Any) -> str:
        """Count and refuse.
        计数并拒绝。

        Input / 输入：
            args, kwargs: Whatever mem0 passes.
            args、kwargs：mem0 传入的任何参数。

        Output / 输出：
            Never returns; raises RuntimeError.
            永不返回；抛 RuntimeError。

        State effect / 状态影响：
            calls += 1.
            calls += 1。
        """
        self.calls += 1
        raise RuntimeError("EgoLife control arm: the LLM must not be called (infer=False)")


# Register both shells under private provider names; the config envelopes bypass the whitelist via model_construct.
# 以私有提供者名注册两个薄壳；配置信封用 model_construct 绕过白名单。
EmbedderFactory.provider_to_class["egolife_hash"] = f"{__name__}.SharedHashEmbedding"
LlmFactory.provider_to_class["egolife_stub"] = (f"{__name__}.StubLLM", BaseLlmConfig)


def embedding_identity_check(texts: list[str]) -> dict[str, Any]:
    """Embed `texts` through the mem0 shell and through a plain hms_elite HashEmbedder; require list equality (==).
    用 mem0 薄壳和普通 hms_elite HashEmbedder 分别嵌入 `texts`；要求列表相等（==）。

    Input / 输入：
        texts: Probe texts (20 in the runner).
        texts：探针文本（运行器用 20 段）。

    Output / 输出：
        {'equal': bool, 'count': n, 'sha256': digest of the JSON-serialized vectors, 'dims': 64}.
        {'equal': 布尔, 'count': n, 'sha256': 向量 JSON 序列化的摘要, 'dims': 64}。

    State effect / 状态影响：
        None.
        无。
    """
    shell = SharedHashEmbedding(type("Cfg", (), {"embedding_dims": EMBEDDING_DIMS})())
    plain = shared_hash_embedder()
    a = [shell.embed(t, "add") for t in texts]
    b = [plain.embed(t) for t in texts]
    digest = hashlib.sha256(json.dumps(a).encode("utf-8")).hexdigest()
    return {"equal": a == b, "count": len(texts), "sha256": digest, "dims": EMBEDDING_DIMS}


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def _path_bytes(path: Path) -> int:
    """Bytes of a file, or of every file below a directory (milvus-lite keeps its db as a directory).
    文件的字节数，或目录下所有文件的字节数（milvus-lite 的库是一个目录）。

    Input / 输入：
        path: File or directory (missing -> 0).
        path：文件或目录（不存在 -> 0）。

    Output / 输出：
        Integer bytes.
        整数字节。

    State effect / 状态影响：
        None.
        无。
    """
    if not path.exists():
        return 0
    return path.stat().st_size if path.is_file() else _dir_bytes(path)


class Arm(ABC):
    """Common retain / recall / stats / close surface.
    共同的 retain / recall / stats / close 接口。
    """

    name: str = ""

    def __init__(self, workdir: Path) -> None:
        """Create the working directory and zero the counters.
        创建工作目录并清零计数器。

        Input / 输入：
            workdir: Per-arm directory (created).
            workdir：臂专属目录（创建）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            mkdir.
            建目录。
        """
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.retained_ids: list[str] = []
        self.retain_ms_total = 0.0
        self.closed = False

    def retain(self, windows: list[Window]) -> list[float]:
        """Add each window as one raw message; return per-window wall-clock milliseconds.
        把每个窗口作为一条原始消息加入；返回每个窗口的墙钟毫秒。

        Input / 输入：
            windows: Windows released by the timeline for this step.
            windows：时间线在本步释放的窗口。

        Output / 输出：
            List of milliseconds, one per window, same order.
            按序的毫秒列表，每个窗口一个。

        State effect / 状态影响：
            Appends to retained_ids and retain_ms_total.
            追加到 retained_ids 与 retain_ms_total。
        """
        out: list[float] = []
        for w in windows:
            t0 = time.perf_counter()
            self._add(w)
            ms = (time.perf_counter() - t0) * 1000.0
            out.append(ms)
            self.retained_ids.append(w.window_id)
            self.retain_ms_total += ms
        return out

    @abstractmethod
    def _add(self, window: Window) -> None: ...

    @abstractmethod
    def recall(self, question: str, query_iso: str) -> list[dict[str, Any]]:
        """Search with the shared top_k / threshold; rows carry window_id, text, score, memory_id.
        用共享的 top_k / threshold 检索；行含 window_id、text、score、memory_id。

        Input / 输入：
            question: Public English question text.
            question：公开英文题干。

            query_iso: Query instant (Elite reference_date; mem0 ignores it).
            query_iso：查询时刻（Elite 的 reference_date；mem0 不用）。

        Output / 输出：
            Rows in engine rank order.
            按引擎排名的行。

        State effect / 状态影响：
            None beyond provider counters.
            除提供者计数器外无。
        """

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Counters and sizes (DB bytes are only final after close()).
        计数器与尺寸（库字节数只有 close() 之后才是最终值）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with embedding_calls, llm_calls, stored_count, db_bytes, retain_ms_total, extra.
            含 embedding_calls、llm_calls、stored_count、db_bytes、retain_ms_total、extra 的字典。

        State effect / 状态影响：
            None.
            无。
        """

    @abstractmethod
    def close(self) -> None: ...


class Mem0Arm(Arm):
    """Official mem0ai Memory with a stub LLM, the shared hash embedder and infer=False; subclasses pick the store.
    官方 mem0ai Memory：桩 LLM、共享哈希嵌入器、infer=False；子类选择向量库。
    """

    name = "mem0"

    def __init__(self, workdir: Path) -> None:
        """Memory(MemoryConfig(...)) with model_construct envelopes (from_config would re-run the whitelist).
        用 model_construct 信封构造 Memory(MemoryConfig(...))（from_config 会重新跑白名单校验）。

        Input / 输入：
            workdir: Holds the store files and history.db.
            workdir：存放向量库文件与 history.db。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Creates the store files and the history SQLite file.
            创建向量库文件与历史 SQLite 文件。
        """
        super().__init__(workdir)
        self.history_db = self.workdir / "history.db"
        config = MemoryConfig(
            vector_store=self._vector_store_config(),
            embedder=Mem0EmbedderConfig.model_construct(provider="egolife_hash", config={"embedding_dims": EMBEDDING_DIMS}),
            llm=Mem0LlmConfig.model_construct(provider="egolife_stub", config={}),
            history_db_path=str(self.history_db),
        )
        self.memory = Memory(config)
        if not isinstance(self.memory.embedding_model, SharedHashEmbedding):
            raise RuntimeError("mem0 did not pick up the shared hash embedder")
        if not isinstance(self.memory.llm, StubLLM):
            raise RuntimeError("mem0 did not pick up the stub LLM")

    @abstractmethod
    def _vector_store_config(self) -> dict[str, Any]: ...

    @abstractmethod
    def _db_files(self) -> dict[str, int]: ...

    @abstractmethod
    def _stored_count(self) -> int: ...

    def _add(self, window: Window) -> None:
        self.memory.add(
            messages=[{"role": "user", "content": window.text}],
            user_id=BANK,
            metadata={
                "window_id": window.window_id,
                "day": window.day,
                "start": window.start_cs,
                "end": window.end_cs,
                "source": window.source,
            },
            infer=False,
        )

    def recall(self, question: str, query_iso: str) -> list[dict[str, Any]]:
        result = self.memory.search(question, filters={"user_id": BANK}, top_k=TOP_K, threshold=THRESHOLD)
        rows = []
        for row in result.get("results", []):
            meta = row.get("metadata") or {}
            rows.append(
                {
                    "window_id": meta.get("window_id") or row.get("window_id"),
                    "text": row.get("memory", ""),
                    "score": float(row.get("score", 0.0)),
                    "memory_id": row.get("id"),
                }
            )
        return rows

    def stats(self) -> dict[str, Any]:
        files = self._db_files()
        files["history.db"] = _path_bytes(self.history_db)
        store = self.memory.vector_store
        return {
            "embedding_calls": self.memory.embedding_model.calls,
            "llm_calls": self.memory.llm.calls,
            "stored_count": self._stored_count(),
            "db_bytes": sum(files.values()),
            "db_files": files,
            "retain_ms_total": self.retain_ms_total,
            "extra": {
                "vector_store": self.memory.config.vector_store.provider,
                "keyword_search": type(store).keyword_search is not VectorStoreBase.keyword_search,
                "history_db": str(self.history_db),
            },
        }


class Mem0FaissArm(Mem0Arm):
    """mem0 + FAISS inner-product flat index on disk: semantic-only (faiss has no keyword_search).
    mem0 + 磁盘 FAISS 内积平面索引：仅语义（faiss 没有 keyword_search）。
    """

    name = "mem0_faiss"

    def __init__(self, workdir: Path) -> None:
        self.faiss_dir = workdir / "faiss"
        super().__init__(workdir)

    def _vector_store_config(self) -> dict[str, Any]:
        return {
            "provider": "faiss",
            "config": {
                "collection_name": "egolife",
                "path": str(self.faiss_dir),
                "distance_strategy": "inner_product",  # hash vectors are unit-norm: IP == cosine / 单位向量：内积即余弦
                "embedding_model_dims": EMBEDDING_DIMS,
            },
        }

    def _db_files(self) -> dict[str, int]:
        return {p.name: p.stat().st_size for p in sorted(self.faiss_dir.rglob("*")) if p.is_file()} if self.faiss_dir.exists() else {}

    def _stored_count(self) -> int:
        return len(self.memory.vector_store.docstore)

    def close(self) -> None:
        if self.closed:
            return
        self.memory.vector_store._save()  # noqa: SLF001 - flush the on-disk copy before measuring / 量尺寸前落盘
        self.memory.db.close()
        self.closed = True


class Mem0MilvusArm(Mem0Arm):
    """mem0 + milvus-lite (local file): official mem0 hybrid = semantic COSINE + Milvus BM25 keyword_search.
    mem0 + milvus-lite（本地文件）：官方 mem0 混合检索 = 语义 COSINE + Milvus BM25 keyword_search。
    """

    name = "mem0_milvus"

    def __init__(self, workdir: Path) -> None:
        self.milvus_db = workdir / "milvus.db"
        super().__init__(workdir)
        store = self.memory.vector_store
        if not getattr(store, "_has_bm25_schema", False):
            raise RuntimeError("mem0 milvus store was created without the BM25 (text + sparse) schema")

    def _vector_store_config(self) -> dict[str, Any]:
        return {
            "provider": "milvus",
            "config": {
                "url": str(self.milvus_db),  # a local path -> milvus-lite, no server / 本地路径 -> milvus-lite，无服务
                "collection_name": "egolife",
                "embedding_model_dims": EMBEDDING_DIMS,
                "metric_type": "COSINE",  # unit-norm hash vectors: cosine == inner product / 单位向量：余弦即内积
                "db_name": "",
            },
        }

    def _db_files(self) -> dict[str, int]:
        return {"milvus.db": _path_bytes(self.milvus_db)}

    def _stored_count(self) -> int:
        if self.closed:
            return self._final_count
        rows = self.memory.vector_store.list(filters={"user_id": BANK}, top_k=16_384)
        return len(rows[0]) if rows else 0

    def close(self) -> None:
        if self.closed:
            return
        self._final_count = self._stored_count()
        self.memory.vector_store.client.close()
        self.memory.db.close()
        self.closed = True


class EliteArm(Arm):
    """HMS Elite over a file SQLite store; subclasses fix the scoring weights and infer flag.
    基于文件 SQLite 存储的 HMS Elite；子类决定打分权重与 infer 标志。
    """

    name = "elite"
    infer = False
    scoring: dict[str, float] = {}

    def __init__(self, workdir: Path) -> None:
        """HMS(HMSConfig(store=<workdir>/hms.db, scoring=<class weights>)); embedder/LLM/index are the offline defaults.
        HMS(HMSConfig(store=<workdir>/hms.db, scoring=<类权重>))；嵌入器 / LLM / 索引为离线默认。

        Input / 输入：
            workdir: Holds hms.db.
            workdir：存放 hms.db。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Creates the SQLite file.
            创建 SQLite 文件。
        """
        super().__init__(workdir)
        self.db_path = self.workdir / "hms.db"
        # The offline arms follow the library default extraction contract ('lines' since 2026-09-18, CONTRACT changelog
        # row 108); only EliteInferArm (the offline rule LLM) is affected. The offline tables in
        # docs/design/EGOLIFE_BENCH_2026-09-10.md were produced under the full contract and are reproducible with
        # ingest={"extraction_output": "full"}.
        # 离线臂跟随库默认的抽取契约（2026-09-18 起为 'lines'，CONTRACT 变更记录第 108 行）；只有 EliteInferArm（离线规则
        # LLM）受影响。docs/design/EGOLIFE_BENCH_2026-09-10.md 的离线表是在完整契约下产生的，用
        # ingest={"extraction_output": "full"} 可复现。
        self.config = HMSConfig(store={"path": str(self.db_path)}, scoring=dict(self.scoring))
        self.hms = HMS(self.config)
        if self.hms.embedder.identity != f"hash@{EMBEDDING_DIMS}":
            raise RuntimeError(f"unexpected Elite embedder {self.hms.embedder.identity}")
        self.compile_stats: list[dict[str, Any]] = []

    def _add(self, window: Window) -> None:
        self.hms.add(
            [{"role": "user", "content": window.text}],
            user_id=BANK,
            metadata={
                "window_id": window.window_id,
                "day": window.day,
                "start": window.start_cs,
                "end": window.end_cs,
                "source": window.source,
            },
            timestamp=window.start_iso,
            infer=self.infer,
        )

    def recall(self, question: str, query_iso: str) -> list[dict[str, Any]]:
        result = self.hms.search(
            question, filters={"user_id": BANK}, top_k=TOP_K, threshold=THRESHOLD, reference_date=query_iso
        )
        rows = []
        for row in result.get("results", []):
            meta = row.get("metadata") or {}
            rows.append(
                {
                    "window_id": meta.get("window_id"),
                    "text": row.get("memory", ""),
                    "score": float(row.get("score", 0.0)),
                    "memory_id": row.get("id"),
                }
            )
        return rows

    def compile_card(self, question: str, query_iso: str) -> dict[str, Any]:
        """elite_infer only: compile(question) and measure the card (chars, used_tokens, item count, ms).
        仅 elite_infer：compile(question) 并度量卡片（字符数、used_tokens、条目数、毫秒）。

        Input / 输入：
            question: Public question text.
            question：公开题干。

            query_iso: as_of instant.
            query_iso：as_of 时刻。

        Output / 输出：
            Measurement dict (also appended to compile_stats).
            度量字典（同时追加到 compile_stats）。

        State effect / 状态影响：
            Provider counters advance.
            提供者计数器前进。
        """
        t0 = time.perf_counter()
        ir = self.hms.compile(question, filters={"user_id": BANK}, as_of=query_iso)
        ms = (time.perf_counter() - t0) * 1000.0
        items = ir.items()
        chars = sum(len(str(getattr(item, "text", "") or getattr(item, "memory", ""))) for item in items)
        card = {
            "card_chars": chars,
            "card_json_chars": len(ir.model_dump_json()),
            "used_tokens": int(getattr(ir, "used_tokens", 0)),
            "items": len(items),
            "mode": str(getattr(ir, "mode", "")),
            "compile_ms": ms,
        }
        self.compile_stats.append(card)
        return card

    def stats(self) -> dict[str, Any]:
        usage = self.hms.usage()
        files = {
            self.db_path.name + suffix: _path_bytes(Path(str(self.db_path) + suffix))
            for suffix in ("", "-wal", "-shm")
            if Path(str(self.db_path) + suffix).exists()
        }
        size = sum(files.values())
        return {
            "embedding_calls": usage.embedding_calls,
            "embedded_texts": usage.embedded_texts,
            "llm_calls": usage.llm_calls,
            "llm_tokens": {
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "total": usage.total_tokens,
                "estimated": usage.estimated,
            },
            "stored_count": self._stored_count(),
            "db_bytes": size,
            "db_files": files,
            "retain_ms_total": self.retain_ms_total,
            "extra": {
                "scoring": self.config.scoring.model_dump(),
                "infer": self.infer,
                "index": self.config.vector_store.provider,
                "llm": self.config.llm.provider,
                # Which extraction contract the infer arm ran under (the default flipped to 'lines' on 2026-09-18), so an
                # offline result dir can be told apart from the full-contract EGOLIFE_BENCH dirs.
                # infer 臂在哪种抽取契约下运行（2026-09-18 默认翻转为 'lines'），使离线结果目录能与完整契约的
                # EGOLIFE_BENCH 目录区分开。
                "extraction_output": self.config.ingest.extraction_output,
                "extraction_mode": self.config.ingest.extraction_mode,
            },
        }

    def _stored_count(self) -> int:
        if self.closed:
            return self._final_count
        rows = self.hms.get_all(filters={"user_id": BANK}, top_k=1_000_000).get("results", [])
        return len(rows)

    def close(self) -> None:
        if self.closed:
            return
        self._final_count = self._stored_count()
        self.hms.close()
        self.closed = True


class EliteMatchedArm(EliteArm):
    """Semantic-only weights (w_keyword = w_entity = w_recency = w_support = 0), infer=False.
    仅语义权重（w_keyword = w_entity = w_recency = w_support = 0），infer=False。
    """

    name = "elite_matched"
    infer = False
    scoring = {"w_keyword": 0.0, "w_entity": 0.0, "w_recency": 0.0, "w_support": 0.0}


class EliteFullArm(EliteArm):
    """Default fused weights, infer=False.
    默认融合权重，infer=False。
    """

    name = "elite_full"
    infer = False
    scoring = {}


class EliteInferArm(EliteArm):
    """Default weights, infer=True through the offline rule LLM.
    默认权重，经离线规则 LLM 的 infer=True。
    """

    name = "elite_infer"
    infer = True
    scoring = {}


ARMS: dict[str, type[Arm]] = {
    Mem0FaissArm.name: Mem0FaissArm,
    Mem0MilvusArm.name: Mem0MilvusArm,
    EliteMatchedArm.name: EliteMatchedArm,
    EliteFullArm.name: EliteFullArm,
    EliteInferArm.name: EliteInferArm,
}


def make_arm(name: str, root: Path) -> Arm:
    """Instantiate an arm in <root>/<name>/.
    在 <root>/<name>/ 下实例化一个臂。

    Input / 输入：
        name: One of ARM_NAMES.
        name：ARM_NAMES 之一。

        root: Parent directory for the arm working dirs.
        root：各臂工作目录的父目录。

    Output / 输出：
        The Arm.
        该 Arm。

    State effect / 状态影响：
        Creates the directory and the engine files.
        创建目录与引擎文件。
    """
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; choose from {ARM_NAMES}")
    return ARMS[name](root / name)


__all__ = [
    "ARMS",
    "ARM_NAMES",
    "BANK",
    "EMBEDDING_DIMS",
    "THRESHOLD",
    "TOP_K",
    "Arm",
    "EliteArm",
    "EliteFullArm",
    "EliteInferArm",
    "EliteMatchedArm",
    "Mem0Arm",
    "Mem0FaissArm",
    "Mem0MilvusArm",
    "SharedHashEmbedding",
    "StubLLM",
    "embedding_identity_check",
    "make_arm",
    "shared_hash_embedder",
]
