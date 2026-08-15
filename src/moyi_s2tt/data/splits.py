"""Leakage checks shared by every dataset adapter."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, TypeVar

_SPACE = re.compile(r"\s+")
T = TypeVar("T", bound="LeakageRecord")


class LeakageRecord(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def source_dataset(self) -> str: ...

    @property
    def source_item_id(self) -> str: ...

    @property
    def semantic_group_id(self) -> str: ...

    @property
    def src_text(self) -> str: ...

    @property
    def split(self) -> str: ...

    @property
    def audio_sha256(self) -> str | None: ...

    @property
    def speaker_id(self) -> str | None: ...


@dataclass(frozen=True)
class LeakageIssue:
    key_type: str
    key: str
    splits: tuple[str, ...]


def normalized_text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = _SPACE.sub(" ", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def find_split_leakage(records: Iterable[LeakageRecord]) -> list[LeakageIssue]:
    """Find speaker, source, audio, or normalized-text split crossings."""

    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    issues: list[LeakageIssue] = []
    for record in records:
        if record.id in seen_ids:
            issues.append(LeakageIssue("record_id", record.id, (record.split,)))
        seen_ids.add(record.id)
        source_key = f"{record.source_dataset}:{record.source_item_id}"
        groups[("source_item", source_key)].add(record.split)
        semantic_key = f"{record.source_dataset}:{record.semantic_group_id}"
        groups[("semantic_group", semantic_key)].add(record.split)
        groups[("source_text", normalized_text_hash(record.src_text))].add(record.split)
        if record.speaker_id:
            speaker_key = f"{record.source_dataset}:{record.speaker_id}"
            groups[("speaker", speaker_key)].add(record.split)
        if record.audio_sha256:
            groups[("audio_sha256", record.audio_sha256)].add(record.split)

    for (key_type, key), splits in sorted(groups.items()):
        if len(splits) > 1:
            issues.append(LeakageIssue(key_type, key, tuple(sorted(splits))))
    return issues


def require_leakage_safe(records: Iterable[T]) -> list[T]:
    materialized = list(records)
    issues = find_split_leakage(materialized)
    if issues:
        summary = "; ".join(
            f"{issue.key_type}:{issue.key}={','.join(issue.splits)}" for issue in issues[:10]
        )
        raise ValueError(f"split leakage detected ({len(issues)} issue(s)): {summary}")
    return materialized
