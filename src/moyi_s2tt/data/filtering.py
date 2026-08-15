"""Safety-aware deterministic filters for teacher-generated targets."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from ..config import StrictModel
from ..directions import DIRECTION_KEYS
from ..distillation.cache import CachedPrediction
from .manifest import LanguageCode, ManifestRecord

_TOKEN = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?!\w)")
_UNITS = re.compile(r"(?<!\w)(?:mm|cm|km|kg|mg|bar|psi|rpm|hz|khz|kw|mw|°c|°f|v|a|%)(?!\w)", re.I)
_NEGATION_MARKERS: dict[LanguageCode, tuple[str, ...]] = {
    "vi": ("không", "chưa", "đừng", "chẳng"),
    "en": ("not", "no", "never", "don't", "do not", "cannot", "can't"),
    "zh": ("不", "没", "不要", "不能"),
    "ko": ("안", "못", "않", "말다"),
}


class FilterThresholds(StrictModel):
    max_repeated_token_fraction: float = Field(default=0.5, ge=0, le=1)
    min_confidence_proxy: float | None = None
    min_lexical_agreement: float | None = Field(default=None, ge=0, le=1)
    require_language_id: bool = False


class FilterPolicy(FilterThresholds):
    direction: str
    notes: tuple[str, ...] = ()


def load_filter_policy(path: Path) -> FilterPolicy:
    with path.open(encoding="utf-8") as handle:
        value: Any = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected filtering YAML object in {path}")
    policy = FilterPolicy.model_validate(value)
    if policy.direction not in DIRECTION_KEYS:
        raise ValueError(f"unsupported filtering direction: {policy.direction}")
    return policy


class FilterDecision(StrictModel):
    keep: bool
    reject_reasons: tuple[str, ...]
    audit_flags: tuple[str, ...]
    numbers_preserved: bool
    units_preserved: bool
    negation_preserved: bool
    language_id_valid: bool | None
    repeated_token_fraction: float
    lexical_agreement: float | None


def _normalized_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN.findall(normalized)


def _numbers(text: str) -> set[str]:
    return {value.replace(",", ".") for value in _NUMBER.findall(text)}


def _units(text: str) -> set[str]:
    return {value.casefold() for value in _UNITS.findall(unicodedata.normalize("NFKC", text))}


def _contains_negation(text: str, language: LanguageCode) -> bool:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if language in {"zh", "ko"}:
        return any(marker in normalized for marker in _NEGATION_MARKERS[language])
    return any(
        re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", normalized) is not None
        for marker in _NEGATION_MARKERS[language]
    )


def repeated_token_fraction(text: str) -> float:
    tokens = _normalized_tokens(text)
    if not tokens:
        return 1.0
    return max(Counter(tokens).values()) / len(tokens)


def lexical_agreement(left: str, right: str) -> float:
    """Jaccard token overlap proxy; this is not a semantic metric."""

    left_tokens = set(_normalized_tokens(left))
    right_tokens = set(_normalized_tokens(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def filter_prediction(
    source: ManifestRecord,
    prediction: CachedPrediction,
    *,
    thresholds: FilterThresholds | None = None,
    secondary_text: str | None = None,
    language_id_valid: bool | None = None,
) -> FilterDecision:
    if source.id != prediction.source_id:
        raise ValueError("prediction source ID does not match manifest record")
    thresholds = thresholds or FilterThresholds()

    reasons: list[str] = []
    audit_flags: list[str] = []
    numbers_preserved = _numbers(source.src_text) == _numbers(prediction.text)
    units_preserved = _units(source.src_text).issubset(_units(prediction.text))
    source_has_negation = _contains_negation(source.src_text, source.src_lang)
    target_has_negation = _contains_negation(prediction.text, source.tgt_lang)
    negation_preserved = not source_has_negation or target_has_negation
    repetition = repeated_token_fraction(prediction.text)
    agreement = lexical_agreement(prediction.text, secondary_text) if secondary_text else None

    if not numbers_preserved:
        reasons.append("numbers_not_preserved")
    if not units_preserved:
        reasons.append("units_not_preserved")
    if not negation_preserved:
        reasons.append("negation_not_preserved")
    if repetition > thresholds.max_repeated_token_fraction:
        reasons.append("excessive_token_repetition")
    if thresholds.min_confidence_proxy is not None:
        if prediction.confidence_proxy is None:
            reasons.append("confidence_unavailable")
        elif prediction.confidence_proxy < thresholds.min_confidence_proxy:
            reasons.append("confidence_below_threshold")
    if thresholds.min_lexical_agreement is not None:
        if agreement is None:
            reasons.append("agreement_unavailable")
        elif agreement < thresholds.min_lexical_agreement:
            reasons.append("lexical_agreement_below_threshold")
    if language_id_valid is False:
        reasons.append("target_language_invalid")
    elif language_id_valid is None:
        if thresholds.require_language_id:
            reasons.append("language_id_unavailable")
        else:
            audit_flags.append("language_id_unverified")

    return FilterDecision(
        keep=not reasons,
        reject_reasons=tuple(sorted(set(reasons))),
        audit_flags=tuple(sorted(set(audit_flags))),
        numbers_preserved=numbers_preserved,
        units_preserved=units_preserved,
        negation_preserved=negation_preserved,
        language_id_valid=language_id_valid,
        repeated_token_fraction=repetition,
        lexical_agreement=agreement,
    )


def apply_filter_decision(record: ManifestRecord, decision: FilterDecision) -> ManifestRecord:
    if record.target_kind != "teacher":
        raise ValueError("filter decisions apply only to teacher targets")
    flags = set(record.quality_flags)
    flags.discard("teacher_label_unfiltered")
    flags.update(decision.audit_flags)
    flags.add("teacher_label_accepted" if decision.keep else "teacher_label_rejected")
    values = record.model_dump(mode="python")
    values.update(
        {
            "filter_decision": "keep" if decision.keep else "reject",
            "filter_reasons": decision.reject_reasons,
            "quality_flags": tuple(sorted(flags)),
        }
    )
    return ManifestRecord.model_validate(values)
