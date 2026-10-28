"""Streaming timeline: questions in (day, query_time) order, windows released by their cutoff rule.
流式时间线：题目按（日，query_time）排序，窗口按对方的截止规则释放。
"""

# Purpose (EN): Decide, before each question, which not-yet-retained windows become visible (same day: end <=
#               query_time; earlier days: whole day, i.e. their "23595999" cutoff), and mark questions whose gold
#               evidence lies after the question (excluded_future) or is otherwise not yet retained. Gold is the
#               window containing target_time (primary); the ±60 s neighbourhood is kept as a secondary set.
# 目的（中）：在每题之前决定哪些尚未 retain 的窗口变为可见（同日：end ≤ query_time；前几日：全天，即对方的
#            "23595999" 截止），并标记证据在提问之后（excluded_future）或此刻尚未 retain 的题目。gold 是包含
#            target_time 的窗口（主指标）；±60 s 邻域作为次要集合保留。

# Input (EN): Question list (loader) and Window list (windows).
# 输入（中）：Question 列表（loader）与 Window 列表（windows）。

# Output (EN): Step records: ordered questions, per-step new windows, retained_count, gold window ids, flags.
# 输出（中）：步骤记录：有序题目、每步新增窗口、retained_count、gold 窗口 id、标记。

# State effect (EN): None.
# 状态影响（中）：无。

from __future__ import annotations

from dataclasses import dataclass, field

from loader import END_OF_DAY_HHMMSSCC, Question, hhmmsscc_to_centiseconds
from windows import Window

GOLD_HALF_WIDTH_CS = 0  # primary gold = window containing target_time (start <= t <= end) / 主 gold = 包含 target_time 的窗口
GOLD_NEIGHBOURHOOD_CS = 60 * 100  # secondary: windows overlapping [t - 60 s, t + 60 s] / 次要：与 ±60 s 重叠的窗口


def cutoff_for(question_day: int, question_cs: int, window_day: int) -> int | None:
    """Their build_streaming_deltas cutoff, verbatim in spirit: later days invisible, same day -> query_time,
    earlier days -> "23595999".
    对方 build_streaming_deltas 的截止规则：更晚的日子不可见，同日 -> query_time，前几日 -> "23595999"。

    Input / 输入：
        question_day, question_cs: The question's query_time.
        question_day、question_cs：题目的 query_time。

        window_day: The window's day.
        window_day：窗口所在日。

    Output / 输出：
        Cutoff in centiseconds, or None when the day is not visible at all.
        厘秒截止值；该日完全不可见时为 None。

    State effect / 状态影响：
        None.
        无。
    """
    if window_day > question_day:
        return None
    if window_day == question_day:
        return question_cs
    return hhmmsscc_to_centiseconds(END_OF_DAY_HHMMSSCC)


def gold_window_ids(question: Question, windows: list[Window], half_width_cs: int = GOLD_HALF_WIDTH_CS) -> list[str]:
    """Windows overlapping [t - h, t + h] on the same day for any target instant (day, t); h = 0 is containment.
    与任一 target 时刻（日，t）同日且与 [t − h, t + h] 有重叠的窗口；h = 0 即包含。

    Input / 输入：
        question: Carries the hidden target fields (used only here and in the oracle).
        question：携带隐藏的 target 字段（只在此处与 oracle 使用）。

        windows: The shared window list.
        windows：共享窗口列表。

        half_width_cs: 0 (primary, containment) or GOLD_NEIGHBOURHOOD_CS (secondary, ±60 s).
        half_width_cs：0（主指标，包含）或 GOLD_NEIGHBOURHOOD_CS（次要，±60 s）。

    Output / 输出：
        window_id list in time order.
        按时间顺序的 window_id 列表。

    State effect / 状态影响：
        None.
        无。
    """
    out: list[str] = []
    for w in windows:
        for day, t in question.targets:
            if w.day == day and w.start_cs <= t + half_width_cs and w.end_cs >= t - half_width_cs:
                out.append(w.window_id)
                break
    return out


@dataclass
class Step:
    """One scheduled question with the windows released just before it.
    一个调度的题目及其之前刚释放的窗口。
    """

    position: int  # 0-based order in the schedule / 调度中的 0 起序号
    question: Question
    new_windows: list[Window]  # retained just before this question / 本题之前新 retain 的窗口
    retained_count: int  # cumulative after retaining new_windows / retain 之后的累计数
    gold_ids: list[str]  # primary: windows containing a target instant / 主：包含 target 时刻的窗口
    gold_retained_ids: list[str]  # gold windows already retained at this step / 此刻已 retain 的 gold 窗口
    target_after_query: bool  # literal rule: some target_time later than query_time / 字面规则
    excluded_future: bool  # no gold window retained when asked -> dropped from the hit denominator / 从分母剔除
    gold_wide_ids: list[str] = field(default_factory=list)  # secondary: ±60 s neighbourhood / 次：±60 s 邻域
    gold_wide_retained_ids: list[str] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)


def schedule(questions: list[Question], windows: list[Window]) -> list[Step]:
    """Sort questions by (day, query_time, original order) and release windows monotonically.
    题目按（日，query_time，原始顺序）排序并单调释放窗口。

    Input / 输入：
        questions: All questions (any order).
        questions：全部题目（任意顺序）。

        windows: The shared window list.
        windows：共享窗口列表。

    Output / 输出：
        Steps in schedule order; a window appears in exactly one step's new_windows or in none.
        按调度顺序的步骤；每个窗口最多出现在一个步骤的 new_windows 中。

    State effect / 状态影响：
        None.
        无。
    """
    ordered = sorted(questions, key=lambda q: (q.query_day, q.query_cs, q.order))
    pending = sorted(windows, key=lambda w: (w.day, w.end_cs, w.window_id))
    retained: set[str] = set()
    steps: list[Step] = []
    for position, q in enumerate(ordered):
        new: list[Window] = []
        rest: list[Window] = []
        for w in pending:
            cutoff = cutoff_for(q.query_day, q.query_cs, w.day)
            if cutoff is not None and w.end_cs <= cutoff:
                new.append(w)
            else:
                rest.append(w)
        pending = rest
        retained.update(w.window_id for w in new)
        gold = gold_window_ids(q, windows)
        gold_retained = [g for g in gold if g in retained]
        gold_wide = gold_window_ids(q, windows, GOLD_NEIGHBOURHOOD_CS)
        gold_wide_retained = [g for g in gold_wide if g in retained]
        target_after = any(day > q.query_day or (day == q.query_day and t > q.query_cs) for day, t in q.targets)
        steps.append(
            Step(
                position=position,
                question=q,
                new_windows=new,
                retained_count=len(retained),
                gold_ids=gold,
                gold_retained_ids=gold_retained,
                target_after_query=target_after,
                excluded_future=not gold_retained,
                gold_wide_ids=gold_wide,
                gold_wide_retained_ids=gold_wide_retained,
                flags={"no_gold_window": not gold},
            )
        )
    return steps


__all__ = ["GOLD_HALF_WIDTH_CS", "GOLD_NEIGHBOURHOOD_CS", "Step", "cutoff_for", "gold_window_ids", "schedule"]
