"""Validated affect signals produced during Retain fact extraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Sentiment = Literal["positive", "negative", "neutral"]
Emotion = Literal["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


class AffectAnnotation(BaseModel):
    """Strict provider-facing schema embedded in the fact extraction response."""

    model_config = ConfigDict(extra="forbid")

    sentiment: Sentiment = Field(description="Overall affective polarity expressed in the fact")
    emotion: Emotion = Field(description="Primary expressed emotion, or neutral when none is evidenced")
    intensity: float = Field(
        ge=0.0,
        le=1.0,
        description="Strength of the expressed emotion from 0.0 (none) to 1.0 (very strong)",
    )


@dataclass(frozen=True, slots=True)
class AffectSignals:
    """Provider-neutral, versioned affect attached to one memory fact."""

    sentiment: Sentiment
    emotion: Emotion
    intensity: float
    version: str

    def __post_init__(self) -> None:
        if self.sentiment not in {"positive", "negative", "neutral"}:
            raise ValueError("sentiment must be positive, negative, or neutral")
        if self.emotion not in {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}:
            raise ValueError("emotion is not supported by the affect-v1 taxonomy")
        if isinstance(self.intensity, bool) or not isinstance(self.intensity, (int, float)):
            raise TypeError("intensity must be a number")
        if not 0.0 <= float(self.intensity) <= 1.0:
            raise ValueError("intensity must be between 0.0 and 1.0")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")

        object.__setattr__(self, "intensity", float(self.intensity))
        object.__setattr__(self, "version", self.version.strip())

    def to_json(self) -> dict[str, str | float]:
        """Return the JSON object persisted with the memory unit."""

        return asdict(self)


def parse_affect(value: Any, *, version: str) -> AffectSignals | None:
    """Parse untrusted model output without making affect fatal to Retain."""

    if isinstance(value, AffectSignals):
        return value
    if isinstance(value, AffectAnnotation):
        annotation = value
    elif isinstance(value, Mapping):
        try:
            annotation = AffectAnnotation.model_validate(dict(value))
        except (TypeError, ValueError):
            return None
    else:
        return None

    try:
        return AffectSignals(
            sentiment=annotation.sentiment,
            emotion=annotation.emotion,
            intensity=annotation.intensity,
            version=version,
        )
    except (TypeError, ValueError):
        return None
