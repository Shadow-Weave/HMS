"""Deterministic rendering of grounded media descriptions into HMS text.

The canonical document is also the input to HMS ``chunks`` retain mode.  Every
semantic atom is therefore rendered as an independently grounded paragraph,
separated from the next atom by a blank line.  The generic HMS splitter uses
blank lines as its first boundary, so an atom that fits under
``DEFAULT_CANONICAL_ATOM_MAX_CHARS`` can be packed with neighbouring atoms but
cannot have its evidence/time envelope separated from its body.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from .models import (
    GroundedEntity,
    GroundedStatement,
    NormalizedMultimodalDescription,
    NormalizedTemporalSegment,
    VisualObservation,
)

CANONICAL_CHUNK_CONTRACT = "provenance-closed-v1"
DEFAULT_CANONICAL_ATOM_MAX_CHARS = 2_400
_PART_PLACEHOLDER = "99999/99999"


def format_timecode(milliseconds: int) -> str:
    """Format a non-negative millisecond offset as HH:MM:SS.mmm."""

    if milliseconds < 0:
        raise ValueError("milliseconds must be non-negative")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _evidence_ref(evidence_ids: list[str]) -> str:
    return ", ".join(sorted(evidence_ids))


def _untrusted_text(text: str) -> str:
    """Render generated/OCR text as a quoted JSON scalar, not Markdown control text."""

    return json.dumps(text, ensure_ascii=False)


def _statement_line(statement: GroundedStatement, *, text: str | None = None, prefix: str = "-") -> str:
    rendered_text = statement.text if text is None else text
    return (
        f"{prefix} [evidence: {_evidence_ref(statement.evidence_ids)}] "
        f"[uncertainty: {statement.uncertainty}] {_untrusted_text(rendered_text)}"
    )


def _entity_line(entity: GroundedEntity, *, text: str | None = None) -> str:
    rendered_text = entity.name if text is None else text
    return (
        f"- [evidence: {_evidence_ref(entity.evidence_ids)}] "
        f"[uncertainty: {entity.uncertainty}] {_untrusted_text(rendered_text)}"
    )


def _observation_line(observation: VisualObservation, *, text: str | None = None) -> str:
    rendered_text = observation.text if text is None else text
    return (
        f"- [kind: {observation.kind}] [evidence: {_evidence_ref(observation.evidence_ids)}] "
        f"[uncertainty: {observation.uncertainty}] {_untrusted_text(rendered_text)}"
    )


def _atom_id(
    *,
    asset_sha256: str,
    section: str,
    text: str,
    evidence_ids: list[str],
    uncertainty: str,
    segment: NormalizedTemporalSegment | None,
    kind: str | None,
) -> str:
    """Return a stable identity for all fragments of one semantic atom."""

    identity = {
        "asset_sha256": asset_sha256,
        "evidence_ids": sorted(evidence_ids),
        "kind": kind,
        "section": section,
        "segment_id": segment.segment_id if segment is not None else None,
        "start_ms": segment.start_ms if segment is not None else None,
        "end_ms": segment.end_ms if segment is not None else None,
        "text": text,
        "uncertainty": uncertainty,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atom_envelope(
    description: NormalizedMultimodalDescription,
    *,
    section: str,
    atom_id: str,
    part: str,
    segment: NormalizedTemporalSegment | None,
) -> list[str]:
    lines = [
        f"[canonical-atom: {CANONICAL_CHUNK_CONTRACT}]",
        f"Asset: sha256:{description.asset.sha256}",
        f"Media kind: {description.asset.media_kind}",
        f"Section: {section}",
    ]
    if segment is not None:
        start = format_timecode(segment.start_ms)
        end = format_timecode(segment.end_ms)
        # Preserve the existing machine-readable time/segment spelling while
        # making it part of every timeline atom rather than a separate header.
        lines.extend(
            [
                "## Timeline",
                f"Time: [{start}–{end}] segment={segment.segment_id}",
            ]
        )
    lines.append(f"Atom: sha256:{atom_id} part={part}")
    return lines


def _max_prefix_that_fits(
    text: str,
    *,
    render: Callable[[str, str], str],
    max_chars: int,
) -> int:
    """Find the longest non-empty Unicode prefix fitting the conservative envelope."""

    low, high = 1, len(text)
    best = 0
    while low <= high:
        midpoint = (low + high) // 2
        if len(render(text[:midpoint], _PART_PLACEHOLDER)) <= max_chars:
            best = midpoint
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def _render_fragmented_atom(
    description: NormalizedMultimodalDescription,
    *,
    section: str,
    text: str,
    evidence_ids: list[str],
    uncertainty: str,
    render_semantic_line: Callable[[str], str],
    max_chars: int,
    segment: NormalizedTemporalSegment | None = None,
    kind: str | None = None,
) -> list[str]:
    """Render one semantic atom into bounded, independently grounded paragraphs."""

    atom_id = _atom_id(
        asset_sha256=description.asset.sha256,
        section=section,
        text=text,
        evidence_ids=evidence_ids,
        uncertainty=uncertainty,
        segment=segment,
        kind=kind,
    )

    def render(fragment: str, part: str) -> str:
        lines = _atom_envelope(
            description,
            section=section,
            atom_id=atom_id,
            part=part,
            segment=segment,
        )
        lines.append(render_semantic_line(fragment))
        return "\n".join(lines)

    if not text:
        raise ValueError("canonical semantic atoms require non-empty text")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    fragments: list[str] = []
    remaining = text
    while remaining:
        prefix_length = _max_prefix_that_fits(remaining, render=render, max_chars=max_chars)
        if prefix_length == 0:
            raise ValueError("canonical atom envelope exceeds max_chars")
        fragments.append(remaining[:prefix_length])
        remaining = remaining[prefix_length:]

    total = len(fragments)
    rendered = [render(fragment, f"{index}/{total}") for index, fragment in enumerate(fragments, start=1)]
    if any(len(block) > max_chars for block in rendered):
        raise AssertionError("canonical atom exceeded its conservative render budget")
    return rendered


def _statement_atoms(
    description: NormalizedMultimodalDescription,
    statements: list[GroundedStatement],
    *,
    section: str,
    max_chars: int,
    segment: NormalizedTemporalSegment | None = None,
    visible_text: bool = False,
) -> list[str]:
    blocks: list[str] = []
    for statement in statements:

        def render_line(fragment: str, item: GroundedStatement = statement) -> str:
            if visible_text:
                return f"- [visible-text] {_statement_line(item, text=fragment, prefix='').lstrip()}"
            return _statement_line(item, text=fragment)

        blocks.extend(
            _render_fragmented_atom(
                description,
                section=section,
                text=statement.text,
                evidence_ids=statement.evidence_ids,
                uncertainty=statement.uncertainty,
                render_semantic_line=render_line,
                max_chars=max_chars,
                segment=segment,
                kind="visible-text" if visible_text else "statement",
            )
        )
    return blocks


def _observation_atoms(
    description: NormalizedMultimodalDescription,
    observations: list[VisualObservation],
    *,
    section: str,
    max_chars: int,
    segment: NormalizedTemporalSegment | None = None,
) -> list[str]:
    blocks: list[str] = []
    for observation in observations:
        blocks.extend(
            _render_fragmented_atom(
                description,
                section=section,
                text=observation.text,
                evidence_ids=observation.evidence_ids,
                uncertainty=observation.uncertainty,
                render_semantic_line=lambda fragment, item=observation: _observation_line(item, text=fragment),
                max_chars=max_chars,
                segment=segment,
                kind=observation.kind,
            )
        )
    return blocks


def _entity_atoms(
    description: NormalizedMultimodalDescription,
    entities: list[GroundedEntity],
    *,
    max_chars: int,
) -> list[str]:
    blocks: list[str] = []
    for entity in entities:
        blocks.extend(
            _render_fragmented_atom(
                description,
                section="visible-entities",
                text=entity.name,
                evidence_ids=entity.evidence_ids,
                uncertainty=entity.uncertainty,
                render_semantic_line=lambda fragment, item=entity: _entity_line(item, text=fragment),
                max_chars=max_chars,
                kind="entity",
            )
        )
    return blocks


def _render_segment_atoms(
    description: NormalizedMultimodalDescription,
    segment: NormalizedTemporalSegment,
    *,
    max_chars: int,
) -> list[str]:
    """Render a segment without ever emitting a header-only timeline block."""

    blocks = _statement_atoms(
        description,
        segment.description.summary,
        section="timeline-summary",
        max_chars=max_chars,
        segment=segment,
    )
    blocks.extend(
        _observation_atoms(
            description,
            segment.description.observations,
            section="timeline-observation",
            max_chars=max_chars,
            segment=segment,
        )
    )
    blocks.extend(
        _statement_atoms(
            description,
            segment.description.visible_text,
            section="timeline-visible-text",
            max_chars=max_chars,
            segment=segment,
            visible_text=True,
        )
    )
    if not blocks:
        # Normalized ModelTemporalSegment already rejects this, but keep the
        # serialization invariant local and explicit.
        raise ValueError("timeline segments require at least one semantic atom")
    return blocks


def render_canonical_markdown(
    description: NormalizedMultimodalDescription,
    *,
    max_atom_chars: int = DEFAULT_CANONICAL_ATOM_MAX_CHARS,
) -> str:
    """Render one normalized asset into provenance-closed canonical Markdown.

    ``max_atom_chars`` bounds each blank-line-delimited semantic paragraph.  It
    must remain no larger than the effective HMS retain chunk size; the trusted
    bridge will enforce that deployment invariant before this contract is
    enabled end to end.
    """

    asset = description.asset
    provenance = description.provenance
    source_scope = "visual-only; audio not processed" if asset.media_kind == "video" else "visual"
    manifest = "\n".join(
        [
            "# Media memory",
            f"Canonical chunk contract: {CANONICAL_CHUNK_CONTRACT}",
            f"Canonical atom max chars: {max_atom_chars}",
            f"Asset: sha256:{asset.sha256}",
            f"Media kind: {asset.media_kind}",
            f"Detected MIME: {asset.detected_mime}",
            f"Descriptor: {provenance.provider}/{provenance.configured_model}",
            f"Pipeline version: {provenance.pipeline_version}",
            f"Pipeline fingerprint: {provenance.pipeline_fingerprint}",
            f"Processing scope: {source_scope}",
        ]
    )

    blocks = [manifest]
    blocks.extend(
        _statement_atoms(
            description,
            description.model_output.summary,
            section="summary",
            max_chars=max_atom_chars,
        )
    )
    blocks.extend(_entity_atoms(description, description.model_output.entities, max_chars=max_atom_chars))
    blocks.extend(
        _observation_atoms(
            description,
            description.model_output.observations,
            section="visual-observations",
            max_chars=max_atom_chars,
        )
    )
    blocks.extend(
        _statement_atoms(
            description,
            description.model_output.visible_text,
            section="visible-text",
            max_chars=max_atom_chars,
            visible_text=True,
        )
    )
    for segment in description.temporal_segments:
        blocks.extend(_render_segment_atoms(description, segment, max_chars=max_atom_chars))
    blocks.extend(
        _statement_atoms(
            description,
            description.model_output.limitations,
            section="limitations",
            max_chars=max_atom_chars,
        )
    )

    return "\n\n".join(blocks).rstrip() + "\n"


def flat_provenance_metadata(
    description: NormalizedMultimodalDescription,
    *,
    source_available: bool,
) -> dict[str, str]:
    """Return bounded flat strings compatible with existing fact metadata."""

    asset = description.asset
    provenance = description.provenance
    return {
        "media_asset_sha256": asset.sha256,
        "media_kind": asset.media_kind,
        "media_detected_mime": asset.detected_mime,
        "media_descriptor_provider": provenance.provider,
        "media_descriptor_model": provenance.configured_model,
        "media_pipeline_version": provenance.pipeline_version,
        "media_prompt_version": provenance.prompt_version,
        "media_schema_version": provenance.schema_version,
        "media_sampling_version": provenance.sampling_version,
        "media_pipeline_fingerprint": provenance.pipeline_fingerprint,
        "media_audio_presence": asset.audio_presence,
        "media_audio_processing": asset.audio_processing,
        "media_source_available": str(source_available).lower(),
    }
