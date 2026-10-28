"""Deterministic four-choice reader shared by every arm; no model, no hidden fields.
所有臂共用的确定性四选一阅读器；无模型，不用隐藏字段。
"""

# Purpose (EN): Turn (question, options, evidence texts) into one letter with a fixed lexical rule so the only
#               variable between arms is which evidence they recalled. Also expose the gold-window oracle input.
# 目的（中）：用固定的词法规则把（题干、选项、证据文本）变成一个字母，使各臂之间唯一的变量是各自召回的证据。
#            另暴露 gold 窗口的 oracle 输入。

# Input (EN): Public English question and choices, evidence texts in recall order (first 4000 chars), IDF over the
#             windows observed so far (the retained prefix of the stream, never a future window).
# 输入（中）：公开的英文题干与选项、按召回顺序的证据文本（前 4000 字符）、到目前为止已观察窗口（流的已 retain
#            前缀，绝不含未来窗口）上的 IDF。

# Output (EN): Reader (read / evidence_from_windows / oracle_evidence) and ReadResult; never reads
#              answer / keywords / reason / target_time.
# 输出（中）：Reader（read / evidence_from_windows / oracle_evidence）与 ReadResult；绝不读取
#            answer / keywords / reason / target_time。

# State effect (EN): None.
# 状态影响（中）：无。

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from loader import Question
from timeline import GOLD_HALF_WIDTH_CS, gold_window_ids
from windows import Window

EVIDENCE_CHARS = 4000
COOCCUR_WEIGHT = 0.5
_TOKEN = re.compile(r"[a-z0-9]+")
# Small fixed stoplist for question tokens (option tokens are never filtered; IDF already down-weights them).
# 题干 token 的固定小停用表（选项 token 从不过滤；IDF 已经把它们降权）。
STOPWORDS = frozenset(
    "a an the of to in on at for from by with and or but is are was were be been being do does did "
    "what which who whom whose when where why how i me my we our you your he she it they them his her its "
    "their this that these those there here am will would can could should shall may might must have has had "
    "not no yes if then than so as about into over under after before while during up down out off again "
    "s t".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case ASCII word tokens (the transcripts are English; Chinese is not in the evidence).
    小写 ASCII 词 token（转写为英文；证据里没有中文）。

    Input / 输入：
        text: Any string.
        text：任意字符串。

    Output / 输出：
        Token list in order.
        按序的 token 列表。

    State effect / 状态影响：
        None.
        无。
    """
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class ReadResult:
    """Reader output.
    阅读器输出。
    """

    pred: str
    scores: dict[str, float]
    invalid_tie: bool
    evidence_chars: int


class Reader:
    """Option scoring = sum over option tokens of count-in-evidence x IDF, plus a same-window co-occurrence bonus
    with question tokens; IDF is computed over the windows observed so far (observe() follows the stream).
    选项得分 = 选项 token 在证据中的出现次数 × IDF 之和，再加与题干 token 同窗共现的加分；IDF 在到目前为止已
    观察的窗口上计算（observe() 跟随流）。
    """

    def __init__(self, windows: list[Window] | None = None) -> None:
        """Start with an optional initial corpus (the runner starts empty and calls observe() per step).
        以可选的初始语料开始（运行器从空开始，逐步调用 observe()）。

        Input / 输入：
            windows: Windows already visible (None -> empty corpus).
            windows：已可见的窗口（None -> 空语料）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Stores df, N, the observed window list and a window_id -> text map for the oracle.
            保存 df、N、已观察窗口列表以及供 oracle 使用的 window_id -> 文本映射。
        """
        self._n = 0
        self._df: dict[str, int] = {}
        self._text_by_id: dict[str, str] = {}
        self._windows: list[Window] = []
        if windows:
            self.observe(windows)

    def observe(self, windows: list[Window]) -> None:
        """Add newly retained windows to the IDF corpus (called with each step's new_windows, same for every arm).
        把新 retain 的窗口加入 IDF 语料（每步以 new_windows 调用，所有臂相同）。

        Input / 输入：
            windows: Windows released by the timeline at this step.
            windows：时间线在本步释放的窗口。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            N += len(windows); df updated; windows appended.
            N += len(windows)；更新 df；追加窗口。
        """
        for w in windows:
            self._n += 1
            self._windows.append(w)
            self._text_by_id[w.window_id] = w.text
            for tok in set(tokenize(w.text)):
                self._df[tok] = self._df.get(tok, 0) + 1

    @property
    def observed(self) -> int:
        """Number of windows in the IDF corpus so far.
        目前 IDF 语料中的窗口数。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Integer.
            整数。

        State effect / 状态影响：
            None.
            无。
        """
        return self._n

    def idf(self, token: str) -> float:
        """Smoothed IDF: log((N + 1) / (df + 1)) + 1.
        平滑 IDF：log((N + 1) / (df + 1)) + 1。

        Input / 输入：
            token: Lower-case token.
            token：小写 token。

        Output / 输出：
            Positive float.
            正浮点数。

        State effect / 状态影响：
            None.
            无。
        """
        return math.log((self._n + 1) / (self._df.get(token, 0) + 1)) + 1.0

    @staticmethod
    def truncate(evidence_texts: list[str], limit: int = EVIDENCE_CHARS) -> list[str]:
        """Keep evidence in recall order up to `limit` characters (last text cut).
        按召回顺序保留证据直到 `limit` 个字符（最后一段截断）。

        Input / 输入：
            evidence_texts: Texts in recall order.
            evidence_texts：按召回顺序的文本。

            limit: Character budget (4000).
            limit：字符预算（4000）。

        Output / 输出：
            Truncated list (may be empty).
            截断后的列表（可能为空）。

        State effect / 状态影响：
            None.
            无。
        """
        out: list[str] = []
        used = 0
        for text in evidence_texts:
            if used >= limit:
                break
            room = limit - used
            piece = text if len(text) <= room else text[:room]
            out.append(piece)
            used += len(piece) + 1
        return out

    def read(self, question: str, options: dict[str, str], evidence_texts: list[str]) -> ReadResult:
        """Pick one letter from the public question, the four options and the evidence.
        只用公开题干、四个选项与证据选出一个字母。

        Input / 输入：
            question: English question text.
            question：英文题干。

            options: {'A': text, ..., 'D': text}.
            options：{'A': 文本, ..., 'D': 文本}。

            evidence_texts: Recalled texts in rank order (truncated here to 4000 chars).
            evidence_texts：按排名顺序的召回文本（此处截断到 4000 字符）。

        Output / 输出：
            ReadResult; pred = 'A' with invalid_tie=True when every score is 0 or the top is tied.
            ReadResult；全 0 或并列时 pred = 'A' 且 invalid_tie=True。

        State effect / 状态影响：
            None.
            无。
        """
        pieces = self.truncate(evidence_texts)
        q_tokens = {t for t in tokenize(question) if t not in STOPWORDS}
        opt_tokens = {letter: tokenize(text) for letter, text in options.items()}
        all_opt = {t for toks in opt_tokens.values() for t in toks}
        q_only = q_tokens - all_opt
        scores: dict[str, float] = {letter: 0.0 for letter in sorted(options)}
        for piece in pieces:
            counts: dict[str, int] = {}
            for t in tokenize(piece):
                counts[t] = counts.get(t, 0) + 1
            q_present = sum(self.idf(t) for t in q_only if t in counts)
            for letter, toks in opt_tokens.items():
                hit = 0.0
                for t in dict.fromkeys(toks):
                    c = counts.get(t, 0)
                    if c:
                        hit += c * self.idf(t)
                if hit > 0.0:
                    scores[letter] += hit + COOCCUR_WEIGHT * q_present
        best = max(scores.values()) if scores else 0.0
        winners = [letter for letter, s in scores.items() if s == best]
        if best <= 0.0 or len(winners) != 1:
            return ReadResult(pred="A", scores=scores, invalid_tie=True, evidence_chars=sum(len(p) for p in pieces))
        return ReadResult(pred=winners[0], scores=scores, invalid_tie=False, evidence_chars=sum(len(p) for p in pieces))

    def oracle_evidence(self, question: Question, half_width_cs: int = GOLD_HALF_WIDTH_CS) -> list[str]:
        """Gold-window texts among the windows observed so far, in time order (the reader's upper bound input;
        uses target_time by design).
        到目前为止已观察窗口中的 gold 窗口文本，按时间顺序（阅读器上限的输入；按设计使用 target_time）。

        Input / 输入：
            question: With hidden target fields.
            question：带隐藏 target 字段。

            half_width_cs: Gold definition (0 = containment, GOLD_NEIGHBOURHOOD_CS = ±60 s).
            half_width_cs：gold 定义（0 = 包含，GOLD_NEIGHBOURHOOD_CS = ±60 s）。

        Output / 输出：
            Text list (empty when no gold window is visible yet).
            文本列表（尚无可见 gold 窗口时为空）。

        State effect / 状态影响：
            None.
            无。
        """
        return [self._text_by_id[wid] for wid in gold_window_ids(question, self._windows, half_width_cs)]


__all__ = ["COOCCUR_WEIGHT", "EVIDENCE_CHARS", "STOPWORDS", "ReadResult", "Reader", "tokenize"]
