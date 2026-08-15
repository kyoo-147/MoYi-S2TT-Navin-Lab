import hashlib

from moyi_s2tt.data.filtering import FilterPolicy
from moyi_s2tt.data.source import SourceRecord
from moyi_s2tt.distillation.cache import CachedPrediction
from moyi_s2tt.distillation.candidates import PseudoLabelCandidate
from moyi_s2tt.distillation.filter_report import filter_candidates
from moyi_s2tt.evaluation.contracts import FrozenEvaluationSet


def cached(source_id: str, text: str, teacher: str) -> CachedPrediction:
    return CachedPrediction(
        cache_key=hashlib.sha256(teacher.encode()).hexdigest(),
        source_id=source_id,
        text=text,
        teacher_id=f"fixture/{teacher}",
        teacher_revision="revision",
        teacher_license="fixture-only",
        generation_config_sha256="f" * 64,
        generated_at="2026-08-15T00:00:00+00:00",
    )


def candidate(target: str) -> PseudoLabelCandidate:
    source = SourceRecord(
        id="source-1",
        source_item_id="clip.wav",
        semantic_group_id="train-prompt",
        source_locale="vi_vn",
        source_audio_ref="audio/clip.wav",
        audio_path="audio/clip.wav",
        audio_sha256="a" * 64,
        duration_s=2,
        source_sample_rate=16_000,
        source_channels=1,
        source_codec="wav-pcm16",
        src_lang="vi",
        src_text="không chạy máy ở 50 rpm",
        domain="industrial",
        split="train",
        source_dataset="fixture",
        source_revision="v1",
        source_license="fixture-only",
        materialization_status="ready",
    )
    return PseudoLabelCandidate(
        source=source,
        asr=cached(source.id, source.src_text, "asr"),
        mt=cached(source.id, target, "mt"),
        direct_s2tt=cached(source.id, target, "direct"),
    )


def test_filter_report_counts_rows_hours_and_reasons() -> None:
    frozen = FrozenEvaluationSet(
        id="fixture",
        direction="vi-en",
        source_dataset="fixture",
        source_revision="v1",
        source_locale="vi",
        target_locale="en",
        validation_semantic_ids=(),
        test_semantic_ids=(),
        training_exclusion_semantic_ids=(),
    )
    policy = FilterPolicy(direction="vi-en", max_repeated_token_fraction=0.5)
    records, report = filter_candidates(
        [candidate("Do not run the machine at 50 rpm")], policy, frozen
    )
    assert records[0].filter_decision == "keep"
    assert records[0].target_kind == "teacher"
    assert records[0].teacher_chain_sha256
    assert report.accepted_rows == 1
    assert report.accepted_hours == round(2 / 3600, 6)
