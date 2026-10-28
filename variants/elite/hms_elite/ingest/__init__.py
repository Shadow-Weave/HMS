"""Ingest plane: SourceEvent → Anchor → routing → extraction → atoms, with lock discipline Txn A → unlocked → Txn B.
摄取平面：事件→锚点→分流→抽取→原子；事务纪律 A→无锁→B。
"""

# Purpose (EN): Re-export the public ingest API (events, router, extraction, closure, atoms, pipeline).
# 目的（中）：再导出摄取平面的公开 API（events、router、extraction、closure、atoms、pipeline）。

# Input (EN): Nothing at import time.
# 输入（中）：导入时无输入。

# Output (EN): The re-exported names.
# 输出（中）：再导出的名称。

# State effect (EN): None.
# 状态影响（中）：无。

from hms_elite.ingest.atoms import (
    EntityKey,
    atom_from_entity,
    atom_from_ep,
    atom_from_event,
    atom_from_message,
    atom_from_observation,
    conflict_atom,
    create_atom,
    link_atom_axes,
    revise_atom,
)
from hms_elite.ingest.closure import (
    ClosureBatch,
    action_text,
    build_ep_from_closure,
    close_structured_events,
    derived_constraint,
    event_atom_text,
    outcome_text,
)
from hms_elite.ingest.derive import expand_compact_item, expand_lines_ep
from hms_elite.ingest.events import (
    anchor_events,
    default_raw_ref,
    events_from_dicts,
    events_from_messages,
    merge_image_descriptions,
    system_event,
    validate_payload,
)
from hms_elite.ingest.extraction import (
    EXTRACTION_OUTPUTS,
    ExtractionBundle,
    decide_revisions,
    extract_all,
    extract_eps,
    extract_observations,
    neighbourhood,
    procedural_summary,
)
from hms_elite.ingest.lines import NO_SECTION_MARK, parse_lines
from hms_elite.ingest.pipeline import EXTRACTION_MODES, Ingestor, IngestResult
from hms_elite.ingest.router import has_action_closure, route_event

__all__ = [
    "EXTRACTION_MODES",
    "EXTRACTION_OUTPUTS",
    "ClosureBatch",
    "EntityKey",
    "ExtractionBundle",
    "IngestResult",
    "Ingestor",
    "NO_SECTION_MARK",
    "action_text",
    "anchor_events",
    "atom_from_entity",
    "atom_from_ep",
    "atom_from_event",
    "atom_from_message",
    "atom_from_observation",
    "build_ep_from_closure",
    "close_structured_events",
    "conflict_atom",
    "create_atom",
    "decide_revisions",
    "default_raw_ref",
    "derived_constraint",
    "event_atom_text",
    "events_from_dicts",
    "events_from_messages",
    "expand_compact_item",
    "expand_lines_ep",
    "extract_all",
    "extract_eps",
    "extract_observations",
    "has_action_closure",
    "link_atom_axes",
    "merge_image_descriptions",
    "neighbourhood",
    "outcome_text",
    "parse_lines",
    "procedural_summary",
    "revise_atom",
    "route_event",
    "system_event",
    "validate_payload",
]
