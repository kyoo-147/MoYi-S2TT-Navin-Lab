"""Direction-aware interfaces for offline ASR, MT, and direct-S2TT teachers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import Field, model_validator

from ..config import StrictModel
from ..data.manifest import LanguageCode
from ..directions import DIRECTION_KEYS

TeacherTask = Literal["asr", "mt", "s2tt"]
TeacherStatus = Literal["candidate", "approved", "hold"]
JsonScalar = str | int | float | bool | None


class TeacherSpec(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]+$")
    task: TeacherTask
    status: TeacherStatus
    revision: str | None = None
    license: str
    research_only: bool
    directions: tuple[str, ...]
    notes: str = ""

    @model_validator(mode="after")
    def validate_spec(self) -> TeacherSpec:
        unknown = set(self.directions) - DIRECTION_KEYS
        if unknown:
            raise ValueError(f"unsupported teacher directions: {sorted(unknown)}")
        if self.status == "approved" and not self.revision:
            raise ValueError("approved teachers require a pinned revision")
        return self


class GenerationConfig(StrictModel):
    seed: int = 147
    batch_size: int = Field(default=1, gt=0)
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class TeacherInput(StrictModel):
    source_id: str
    src_lang: LanguageCode
    tgt_lang: LanguageCode
    source_text: str | None = None
    audio_path: str | None = None
    audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_input(self) -> TeacherInput:
        if not self.source_text and not self.audio_path:
            raise ValueError("teacher input requires source_text or audio_path")
        if self.src_lang == self.tgt_lang:
            raise ValueError("teacher input languages must differ")
        return self


class TeacherPrediction(StrictModel):
    source_id: str
    text: str = Field(min_length=1)
    confidence_proxy: float | None = None
    metadata: dict[str, JsonScalar] = Field(default_factory=dict)


class Teacher(Protocol):
    @property
    def spec(self) -> TeacherSpec: ...

    def generate(
        self, inputs: Sequence[TeacherInput], config: GenerationConfig
    ) -> Sequence[TeacherPrediction]: ...
