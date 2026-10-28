"""Canonical UTC instants: one clock format for every table.
规范 UTC 时间：所有表使用同一格式。
"""

# Purpose (EN): Produce and normalize fixed-width UTC ISO instants and date windows.
# 目的（中）：生成并归一化定宽 UTC ISO 时刻与日期窗口。

# Input (EN): ISO strings, datetimes, dates, epoch seconds.
# 输入（中）：ISO 字符串、datetime、date、epoch 秒。

# Output (EN): 32-char 'YYYY-MM-DDTHH:MM:SS.ffffff+00:00' strings and (gte, lte) windows.
# 输出（中）：32 字符 'YYYY-MM-DDTHH:MM:SS.ffffff+00:00' 字符串与 (gte, lte) 窗口。

# State effect (EN): None; now_iso() is the only wall-clock read in the library.
# 状态影响（中）：无；now_iso() 是库中唯一读取墙钟的位置。

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

from hms_elite.exceptions import ValidationError

ISO_WIDTH = 32
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FRACTION_RE = re.compile(r"(\.\d+)")
_OFFSET_NO_COLON_RE = re.compile(r"([+-]\d{2})(\d{2})$")
# The canonical 32-char form itself; such strings re-enter to_utc_iso on every write path (about 16 times per add).
# 规范的 32 字符形式本身；这类字符串在每条写路径上都会重新进入 to_utc_iso（每次 add 约 16 次）。
_CANONICAL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")


def _format(dt: datetime) -> str:
    """Format an aware datetime as the 32-char UTC form.
    将带时区的 datetime 格式化为 32 字符 UTC 形式。
    """
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def now_iso() -> str:
    """Return the current UTC instant, 32-char fixed width.
    返回当前 UTC 时刻（32 字符定宽）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        'YYYY-MM-DDTHH:MM:SS.ffffff+00:00'.
        'YYYY-MM-DDTHH:MM:SS.ffffff+00:00'。

    State effect / 状态影响：
        None (reads the wall clock).
        无（读取墙钟）。
    """
    return _format(datetime.now(timezone.utc))


def to_utc_iso(value: str | datetime | date | int | float) -> str:
    """Normalize any supported instant to the fixed-width UTC ISO form.
    将任意受支持的时刻归一化为定宽 UTC ISO 形式。

    Input / 输入：
        value: ISO string ('Z', '±HH:MM', naive → UTC, date-only → T00:00:00), datetime/date, or epoch seconds.
        value：ISO 字符串（'Z'、'±HH:MM'、无时区视为 UTC、仅日期补 T00:00:00）、datetime/date 或 epoch 秒。

    Output / 输出：
        32-char string; garbage raises ValidationError(VALIDATION_004).
        32 字符字符串；无法解析时抛出 ValidationError(VALIDATION_004)。

    State effect / 状态影响：
        None.
        无。
    """
    if isinstance(value, bool):
        raise ValidationError(f"Invalid timestamp/date: {value!r}", "VALIDATION_004", details={"value": value})
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return _format(dt)
    if isinstance(value, date):
        return _format(datetime(value.year, value.month, value.day, tzinfo=timezone.utc))
    if isinstance(value, (int, float)):
        try:
            return _format(datetime.fromtimestamp(float(value), tz=timezone.utc))
        except (OverflowError, OSError, ValueError) as exc:
            raise ValidationError(
                f"Invalid epoch timestamp: {value!r}", "VALIDATION_004", details={"value": value}
            ) from exc
    if isinstance(value, str):
        # Fast path: an already canonical string is returned unchanged once fromisoformat accepts its fields, so the
        # strip / regex rewrites / re-format below only run for foreign forms (about 6x cheaper per call).
        # 快速路径：已是规范形式的字符串在 fromisoformat 接受其字段后原样返回，下面的 strip / 正则改写 / 重新格式化
        # 只对非规范形式运行（单次调用便宜约 6 倍）。
        if len(value) == ISO_WIDTH and _CANONICAL_RE.match(value):
            try:
                datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValidationError(
                    f"Invalid timestamp/date: {value!r}", "VALIDATION_004", details={"value": value}
                ) from exc
            return value
        text = value.strip()
        if text:
            if text[-1] in ("Z", "z"):
                text = text[:-1] + "+00:00"
            if _DATE_ONLY_RE.match(text):
                text += "T00:00:00"
            text = _OFFSET_NO_COLON_RE.sub(r"\1:\2", text)
            text = _FRACTION_RE.sub(lambda m: (m.group(1) + "000000")[:7], text, count=1)
            try:
                dt = datetime.fromisoformat(text)
            except ValueError as exc:
                raise ValidationError(
                    f"Invalid timestamp/date: {value!r}", "VALIDATION_004", details={"value": value}
                ) from exc
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return _format(dt)
    raise ValidationError(f"Invalid timestamp/date: {value!r}", "VALIDATION_004", details={"value": value})


def parse_iso(value: str) -> datetime:
    """Parse an instant into an aware UTC datetime.
    将时刻解析为带 UTC 时区的 datetime。

    Input / 输入：
        value: Any input accepted by to_utc_iso.
        value：to_utc_iso 接受的任意输入。

    Output / 输出：
        Timezone-aware datetime in UTC.
        UTC 时区的 datetime。

    State effect / 状态影响：
        None.
        无。
    """
    return datetime.fromisoformat(to_utc_iso(value))


def date_of(value: str) -> str:
    """Return the 'YYYY-MM-DD' date of an instant.
    返回时刻对应的 'YYYY-MM-DD' 日期。

    Input / 输入：
        value: Any input accepted by to_utc_iso.
        value：to_utc_iso 接受的任意输入。

    Output / 输出：
        Date string.
        日期字符串。

    State effect / 状态影响：
        None.
        无。
    """
    return to_utc_iso(value)[:10]


def age_days(t: str, as_of: str) -> float:
    """Return max(0, (as_of - t)) in days.
    返回 max(0, as_of - t) 的天数。

    Input / 输入：
        t: The earlier instant.
        t：较早的时刻。

        as_of: The reference instant.
        as_of：参考时刻。

    Output / 输出：
        Non-negative float days.
        非负浮点天数。

    State effect / 状态影响：
        None.
        无。
    """
    delta = parse_iso(as_of) - parse_iso(t)
    return max(0.0, delta.total_seconds() / 86400.0)


def day_window(date_str: str) -> tuple[str, str]:
    """Return the (T00:00:00.000000, T23:59:59.999999) window of a calendar day.
    返回某日的 (T00:00:00.000000, T23:59:59.999999) 窗口。

    Input / 输入：
        date_str: 'YYYY-MM-DD' (any instant is reduced to its UTC date).
        date_str：'YYYY-MM-DD'（任意时刻会取其 UTC 日期）。

    Output / 输出：
        (gte, lte) fixed-width instants.
        (gte, lte) 定宽时刻。

    State effect / 状态影响：
        None.
        无。
    """
    day = date_of(date_str)
    return f"{day}T00:00:00.000000+00:00", f"{day}T23:59:59.999999+00:00"


def month_window(year: int, month: int) -> tuple[str, str]:
    """Return the window covering a whole month.
    返回覆盖整月的窗口。

    Input / 输入：
        year: Four-digit year.
        year：四位年份。

        month: 1..12.
        month：1..12。

    Output / 输出：
        (first day T00:00:00.000000, last day T23:59:59.999999).
        (首日 T00:00:00.000000, 末日 T23:59:59.999999)。

    State effect / 状态影响：
        None.
        无。
    """
    first = date(year, month, 1)
    last = (date(year + (month // 12), (month % 12) + 1, 1) - timedelta(days=1)) if month < 12 else date(year, 12, 31)
    return day_window(first.isoformat())[0], day_window(last.isoformat())[1]


def week_window(date_str: str) -> tuple[str, str]:
    """Return the Monday..Sunday window containing a date.
    返回包含某日的周一到周日窗口。

    Input / 输入：
        date_str: 'YYYY-MM-DD'.
        date_str：'YYYY-MM-DD'。

    Output / 输出：
        (Monday T00:00:00.000000, Sunday T23:59:59.999999).
        (周一 T00:00:00.000000, 周日 T23:59:59.999999)。

    State effect / 状态影响：
        None.
        无。
    """
    day = date.fromisoformat(date_of(date_str))
    monday = day - timedelta(days=day.weekday())
    sunday = monday + timedelta(days=6)
    return day_window(monday.isoformat())[0], day_window(sunday.isoformat())[1]


def year_window(year: int) -> tuple[str, str]:
    """Return the window covering a whole calendar year.
    返回覆盖整个自然年的窗口。

    Input / 输入：
        year: Four-digit year.
        year：四位年份。

    Output / 输出：
        (Jan 1 T00:00:00.000000, Dec 31 T23:59:59.999999).
        (1 月 1 日 T00:00:00.000000, 12 月 31 日 T23:59:59.999999)。

    State effect / 状态影响：
        None.
        无。
    """
    return month_window(year, 1)[0], month_window(year, 12)[1]


def next_day_midnight(date_str: str) -> str:
    """Convert an expiration_date into its valid_to boundary (next day, 00:00 UTC).
    将 expiration_date 转换为 valid_to 边界（次日 UTC 零点）。

    Input / 输入：
        date_str: 'YYYY-MM-DD'.
        date_str：'YYYY-MM-DD'。

    Output / 输出：
        '<date+1>T00:00:00.000000+00:00'.
        '<date+1>T00:00:00.000000+00:00'。

    State effect / 状态影响：
        None.
        无。
    """
    return f"{shift_days(date_str, 1)}T00:00:00.000000+00:00"


def shift_days(date_str: str, n: int) -> str:
    """Shift a date by n days.
    将日期平移 n 天。

    Input / 输入：
        date_str: 'YYYY-MM-DD'.
        date_str：'YYYY-MM-DD'。

        n: Signed number of days.
        n：带符号的天数。

    Output / 输出：
        'YYYY-MM-DD'.
        'YYYY-MM-DD'。

    State effect / 状态影响：
        None.
        无。
    """
    return (date.fromisoformat(date_of(date_str)) + timedelta(days=n)).isoformat()


__all__ = [
    "ISO_WIDTH",
    "age_days",
    "date_of",
    "day_window",
    "month_window",
    "next_day_midnight",
    "now_iso",
    "parse_iso",
    "shift_days",
    "to_utc_iso",
    "week_window",
    "year_window",
]
