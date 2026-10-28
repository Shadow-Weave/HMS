"""CJK tokenizer switch (HMSConfig.text.cjk_tokens): bigram default byte-identical, 'none' as the English-centric baseline, index / query symmetry, FTS rebuild on a mode switch, the facade scope list.
CJK 分词开关（HMSConfig.text.cjk_tokens）：bigram 默认逐字节不变、'none' 为英文中心基线、索引 / 查询对称、模式切换时重建 FTS、门面作用域方法表。
"""

# Purpose (EN): Pin CONTRACT §9.6 (a): both tokenizer modes on both sides of the keyword channel, the caches, the meta row and the wrapped facade methods.
# 目的（中）：钉住 CONTRACT §9.6 (a)：关键词通道两侧的两种分词模式、缓存、meta 行与被包装的门面方法。

# Input (EN): In-memory and file-backed HMS facades with the hash embedder; no network.
# 输入（中）：使用 hash 嵌入器的内存与文件 HMS 门面；无网络。

# Output (EN): pytest assertions.
# 输出（中）：pytest 断言。

# State effect (EN): Writes only under tmp_path.
# 状态影响（中）：只写 tmp_path。

from __future__ import annotations

import inspect
import pathlib

import pytest

from hms_elite import HMS
from hms_elite.configs.base import HMSConfig, TextConfig
from hms_elite.exceptions import ConfigError
from hms_elite.memory import main as main_module
from hms_elite.memory.schema import fts_tokens
from hms_elite.memory.storage import StateStore
from hms_elite.models import Scope
from hms_elite.utils import scoring
from hms_elite.utils import text as utext
from tests.conftest import FrozenClock, SequentialIds

SENTENCE = "杰克说他喜欢吃饺子。"
ENGLISH = "User likes oat-milk lattes in the morning; patch retry.ts and billing_e2e"


def _facade(mode: str, path: str = ":memory:") -> HMS:
    """An offline HMS (memory index for ':memory:', sqlite index for a file) in one CJK tokenizer mode.
    某种 CJK 分词模式下的离线 HMS（':memory:' 用 memory 索引，文件用 sqlite 索引）。
    """
    store = StateStore(path, now=FrozenClock().now, id_factory=SequentialIds())
    config = HMSConfig(
        vector_store={"provider": "sqlite" if path != ":memory:" else "memory"},
        store={"path": path},
        text={"cjk_tokens": mode},
    )
    return HMS(config, store=store)


def _fts_rows(hms: HMS) -> list[str]:
    """The atom_fts tokens column, one string per row, in atom order.
    atom_fts 的 tokens 列，每行一个字符串，按原子顺序。
    """
    return [row[0] for row in hms.store._conn.execute("SELECT tokens FROM atom_fts ORDER BY atom_id").fetchall()]


def test_bigram_default_and_none_mode_tokenization():
    """The default tokenizer output is the historical one (bigrams), 'none' emits one token per CJK run on both tokenize and content_tokens, English tokens are identical in both modes, stem never touches a CJK token, and cjk_tokens_scope drives the default argument.
    默认分词输出即历史行为（二元组），'none' 下 tokenize 与 content_tokens 都对每个 CJK 连续串给出一个词元，英文词元两种模式相同，stem 从不改动 CJK 词元，cjk_tokens_scope 决定默认参数。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on utils.text.
        对 utils.text 的断言。

    State effect / 状态影响：
        None (the scope is reset on exit).
        无（作用域退出时恢复）。
    """
    assert utext.cjk_tokens_mode() == "bigram" and utext.CJK_TOKEN_MODES == ("bigram", "none")
    assert utext.tokenize("我喜欢") == ["我喜", "喜欢"] and utext.tokenize("茶") == ["茶"]
    assert utext.tokenize("我喜欢黑咖啡", cjk="none") == ["我喜欢黑咖啡"] and utext.tokenize("茶", cjk="none") == ["茶"]
    assert utext.tokenize("我喜欢 coffee 和茶", cjk="none") == ["我喜欢", "coffee", "和茶"]
    assert utext.content_tokens("我喜欢黑咖啡 retry.ts likes", cjk="none") == ["我喜欢黑咖啡", "retry.ts", "like"]
    assert utext.content_tokens("我喜欢黑咖啡") == ["我喜", "喜欢", "欢黑", "黑咖", "咖啡"]
    assert utext.tokenize(ENGLISH) == utext.tokenize(ENGLISH, cjk="none")
    assert utext.content_tokens(ENGLISH) == utext.content_tokens(ENGLISH, cjk="none")
    assert utext.stem("饺子") == "饺子" and utext.stem("lattes") == "latte"
    with utext.cjk_tokens_scope("none"):
        assert utext.cjk_tokens_mode() == "none"
        assert utext.tokenize(SENTENCE) == utext.tokenize(SENTENCE, cjk="none") == ["杰克说他喜欢吃饺子"]
        with utext.cjk_tokens_scope("bigram"):
            assert utext.tokenize("我喜欢") == ["我喜", "喜欢"]
        assert utext.cjk_tokens_mode() == "none"
    assert utext.cjk_tokens_mode() == "bigram"
    with pytest.raises(ValueError):
        with utext.cjk_tokens_scope("bigrams"):
            pass


def test_config_round_trip_and_defaults():
    """TextConfig defaults to bigram, from_dict accepts {'text': {'cjk_tokens': 'none'}} and rejects 'bigrams' (CONFIG_001), and TextConfig is exported by hms_elite.configs.
    TextConfig 默认为 bigram，from_dict 接受 {'text': {'cjk_tokens': 'none'}} 并拒绝 'bigrams'（CONFIG_001），TextConfig 由 hms_elite.configs 导出。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the config classes.
        对配置类的断言。

    State effect / 状态影响：
        None.
        无。
    """
    from hms_elite.configs import TextConfig as exported

    assert exported is TextConfig and TextConfig().cjk_tokens == "bigram" and HMSConfig().text.cjk_tokens == "bigram"
    cfg = HMSConfig.from_dict({"text": {"cjk_tokens": "none"}, "ingest": {"zh_rules": True}})
    assert cfg.text.cjk_tokens == "none" and cfg.ingest.zh_rules is True
    assert HMSConfig.from_dict(cfg.model_dump()).text.cjk_tokens == "none"
    with pytest.raises(ConfigError) as info:
        HMSConfig.from_dict({"text": {"cjk_tokens": "bigrams"}})
    assert info.value.error_code == "CONFIG_001"


def test_index_and_query_symmetry_per_mode():
    """Under bigram a Chinese two-character query hits the keyword channel (bm25_score > 0, keyword_candidates non-empty); under 'none' the same store and query leave the keyword channel empty (bm25_score 0) while the row is still returned through the gate; atom_fts rows equal fts_tokens() and atom_keyword_tokens() in both modes; the default store gains no meta row.
    bigram 下两字中文查询命中关键词通道（bm25_score > 0，keyword_candidates 非空）；'none' 下同一存储与查询让关键词通道为空（bm25_score 为 0），该行仍经门控返回；两种模式下 atom_fts 行都等于 fts_tokens() 与 atom_keyword_tokens()；默认存储不多出 meta 行。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on search explain details, the store and the scoring helpers.
        对搜索 explain 细节、存储与打分辅助函数的断言。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    scope = Scope(user_id="u")
    seen: dict[str, str] = {}
    for mode in ("bigram", "none"):
        hms = _facade(mode)
        row = hms.add(SENTENCE, user_id="u", infer=False)["results"][0]
        atom = hms.store.get_atoms([row["atom_id"]])[0]
        hit = hms.search("饺子", filters={"user_id": "u"}, explain=True, threshold=0.0)["results"]
        assert [h["memory"] for h in hit] == [SENTENCE]
        bm25 = hit[0]["score_details"]["bm25_score"]
        with utext.cjk_tokens_scope(mode):
            expected = fts_tokens(atom.text, atom.topic_tags, atom.entity_ids)
            assert scoring.atom_keyword_tokens(atom) == set(expected.split())
            candidates = hms.store.keyword_candidates(scope, utext.content_tokens("饺子"), limit=10)
        assert _fts_rows(hms) == [expected]
        seen[mode] = expected
        if mode == "bigram":
            assert bm25 > 0 and [c[0] for c in candidates] == [row["atom_id"]]
            assert "饺子" in expected.split()
            assert hms.store.get_meta(main_module.TEXT_CJK_META_KEY) is None
        else:
            assert bm25 == 0 and candidates == []
            assert expected.split()[0] == "杰克说他喜欢吃饺子"
            assert hms.store.get_meta(main_module.TEXT_CJK_META_KEY) == "none"
        # Two-character words: a bigram store matches them, a 'none' store only matches the whole run.
        # 两字词：bigram 存储能匹配，'none' 存储只匹配整串。
        with utext.cjk_tokens_scope(mode):
            two = hms.store.keyword_candidates(scope, utext.content_tokens("喜欢"), limit=10)
            whole = hms.store.keyword_candidates(scope, utext.content_tokens("杰克说他喜欢吃饺子"), limit=10)
        assert (len(two), len(whole)) == ((1, 1) if mode == "bigram" else (0, 1))
        hms.close()
    assert seen["bigram"] != seen["none"]
    assert scoring._content_vocabulary(SENTENCE, "bigram") != scoring._content_vocabulary(SENTENCE, "none")
    assert scoring._key_tokens("杰克说他", "none") == ("杰克说他", "杰克说他")


def test_fts_rebuilt_when_a_store_is_reopened_under_another_mode(tmp_path: pathlib.Path):
    """A file store written under bigram and reopened under 'none' gets its atom_fts rebuilt once (rows equal fts_tokens under 'none', meta 'text_cjk_tokens' = none, scope write versions untouched); reopening under bigram rebuilds it back and keeps the meta key.
    以 bigram 写入、以 'none' 重新打开的文件存储一次性重建 atom_fts（行等于 'none' 下的 fts_tokens，meta 'text_cjk_tokens' = none，范围写入版本不动）；再以 bigram 打开时重建回来并保留该 meta 键。

    Input / 输入：
        tmp_path: pytest temporary directory.
        tmp_path：pytest 临时目录。

    Output / 输出：
        Assertions on atom_fts, meta and write versions across three opens.
        对三次打开之间 atom_fts、meta 与写入版本的断言。

    State effect / 状态影响：
        Writes one SQLite file under tmp_path.
        在 tmp_path 下写一个 SQLite 文件。
    """
    path = str(tmp_path / "cjk.db")
    first = _facade("bigram", path)
    first.add(SENTENCE, user_id="u", infer=False)
    first.add("User likes oat-milk lattes.", user_id="u", infer=False)
    bigram_rows = _fts_rows(first)
    assert first.store.get_meta(main_module.TEXT_CJK_META_KEY) is None
    first.close()

    second = _facade("none", path)
    assert second.store.get_meta(main_module.TEXT_CJK_META_KEY) == "none"
    generation_after_open = second.store.write_generation
    atoms = second.store.get_atoms([r["atom_id"] for r in second.get_all(filters={"user_id": "u"})["results"]])
    with utext.cjk_tokens_scope("none"):
        expected = sorted(fts_tokens(a.text, a.topic_tags, a.entity_ids) for a in atoms)
    assert sorted(_fts_rows(second)) == expected and sorted(_fts_rows(second)) != sorted(bigram_rows)
    assert second.store.write_generation == generation_after_open
    second.close()

    third = _facade("bigram", path)
    assert third.store.get_meta(main_module.TEXT_CJK_META_KEY) == "bigram"
    assert sorted(_fts_rows(third)) == sorted(bigram_rows)
    third.close()


def test_every_public_facade_method_is_text_scoped():
    """TEXT_SCOPED_METHODS is exactly the public HMS method surface minus the documented exclusions, each wrapper keeps the original signature and docstring, and a scoped method sees its facade's mode while an unscoped call sees the default.
    TEXT_SCOPED_METHODS 恰好是 HMS 公共方法面减去文档列出的排除项，每个包装保留原签名与文档字符串，作用域内的方法看到其门面的模式而作用域外的调用看到默认值。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the HMS class.
        对 HMS 类的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    excluded = {"from_config", "project", "state", "close", "usage", "trace", "start_daemon", "stop_daemon", "history", "vacuum", "get"}
    public = {
        name
        for name, member in vars(HMS).items()
        if not name.startswith("_") and (inspect.isfunction(member) or isinstance(member, (classmethod, property)))
    }
    assert set(main_module.TEXT_SCOPED_METHODS) == public - excluded
    assert list(main_module.TEXT_SCOPED_METHODS) == sorted(main_module.TEXT_SCOPED_METHODS)
    for name in main_module.TEXT_SCOPED_METHODS:
        method = getattr(HMS, name)
        assert hasattr(method, "__wrapped__") and inspect.signature(method) == inspect.signature(method.__wrapped__)
        assert method.__doc__ == method.__wrapped__.__doc__ and method.__name__ == name
    observed: list[str] = []
    hms = _facade("none")
    original = hms.compile.__wrapped__

    def probe(self, *args, **kwargs):
        observed.append(utext.cjk_tokens_mode())
        return original(self, *args, **kwargs)

    hms.compile = utext.text_scoped(probe).__get__(hms, HMS)
    hms.add("User likes tea.", user_id="u", infer=False)
    hms.compile("tea", filters={"user_id": "u"})
    assert observed == ["none"] and utext.cjk_tokens_mode() == "bigram"
    hms.close()
