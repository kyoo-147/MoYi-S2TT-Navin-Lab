"""Convert private multi-teacher candidates and summarize deterministic filtering."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from ..data.filtering import FilterPolicy, apply_filter_decision, filter_prediction
from ..data.manifest import ManifestRecord
from ..evaluation.contracts import FrozenEvaluationSet, require_evaluation_clean
from .candidates import FilteringAggregate, PseudoLabelCandidate


def candidate_to_manifest(candidate: PseudoLabelCandidate) -> ManifestRecord:
    if candidate.failure_stage or not candidate.mt:
        raise ValueError("failed pseudo-label candidates cannot become manifest rows")
    source = candidate.source
    prediction = candidate.mt
    return ManifestRecord(
        id=source.id,
        source_item_id=source.source_item_id,
        semantic_group_id=source.semantic_group_id,
        source_locale=source.source_locale,
        source_audio_ref=source.source_audio_ref,
        audio_path=source.audio_path,
        audio_sha256=source.audio_sha256,
        duration_s=source.duration_s,
        sample_rate=source.source_sample_rate or 16_000,
        speaker_id=source.speaker_id,
        src_lang=source.src_lang,
        tgt_lang="en",
        src_text=source.src_text,
        tgt_text=prediction.text,
        target_kind="teacher",
        domain=source.domain,
        split=source.split,
        source_dataset=source.source_dataset,
        source_revision=source.source_revision,
        source_license=source.source_license,
        teacher_id=prediction.teacher_id,
        teacher_revision=prediction.teacher_revision,
        teacher_license=prediction.teacher_license,
        generation_config_sha256=prediction.generation_config_sha256,
        teacher_generated_at=prediction.generated_at,
        teacher_chain_sha256=candidate.teacher_chain_sha256,
        materialization_status=source.materialization_status,
        quality_flags=("teacher_label_unfiltered",),
    )


def filter_candidates(
    candidates: Iterable[PseudoLabelCandidate],
    policy: FilterPolicy,
    frozen: FrozenEvaluationSet,
) -> tuple[list[ManifestRecord], FilteringAggregate]:
    successful = [candidate for candidate in candidates if not candidate.failure_stage]
    records = [candidate_to_manifest(candidate) for candidate in successful]
    require_evaluation_clean(records, frozen)
    filtered: list[ManifestRecord] = []
    reasons: Counter[str] = Counter()
    domains: dict[str, Counter[str]] = defaultdict(Counter)
    durations: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_hours = 0.0
    rejected_hours = 0.0
    for candidate, record in zip(successful, records, strict=True):
        assert candidate.mt and candidate.asr and candidate.direct_s2tt
        decision = filter_prediction(
            record,
            candidate.mt,
            thresholds=policy,
            secondary_text=candidate.direct_s2tt.text,
            asr_transcript=candidate.asr.text,
        )
        result = apply_filter_decision(record, decision)
        filtered.append(result)
        status = "accepted" if decision.keep else "rejected"
        domains[record.domain][status] += 1
        bucket = (
            "0-5s"
            if record.duration_s <= 5
            else "5-10s"
            if record.duration_s <= 10
            else "10-20s"
            if record.duration_s <= 20
            else "20-30s"
        )
        durations[bucket][status] += 1
        if decision.keep:
            accepted_hours += record.duration_s / 3600
        else:
            rejected_hours += record.duration_s / 3600
            reasons.update(decision.reject_reasons)
    accepted = sum(record.filter_decision == "keep" for record in filtered)
    report = FilteringAggregate(
        status="MEASURED_PRIVATE_LABELS_HUMAN_AUDIT_PENDING",
        total_rows=len(filtered),
        accepted_rows=accepted,
        rejected_rows=len(filtered) - accepted,
        accepted_hours=round(accepted_hours, 6),
        rejected_hours=round(rejected_hours, 6),
        reject_reasons=dict(sorted(reasons.items())),
        rows_by_domain={key: dict(sorted(value.items())) for key, value in sorted(domains.items())},
        rows_by_duration_bucket={
            key: dict(sorted(value.items())) for key, value in sorted(durations.items())
        },
        limitations=(
            "Thresholds are deterministic but not calibrated until human "
            "false-accept/reject review.",
            "Lexical overlap is not a semantic metric; semantic gating is disabled "
            "when unavailable.",
            "Teacher labels remain teacher labels and are not gold evaluation references.",
        ),
    )
    return filtered, report
