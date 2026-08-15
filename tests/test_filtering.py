from pathlib import Path

from moyi_s2tt.data.filtering import (
    FilterThresholds,
    apply_filter_decision,
    filter_prediction,
    lexical_agreement,
    load_filter_policy,
)
from moyi_s2tt.data.manifest import ManifestRecord
from moyi_s2tt.distillation.cache import CachedPrediction


def source(text: str = "Không chạy máy ở 50 rpm.") -> ManifestRecord:
    return ManifestRecord(
        id="source-row",
        source_item_id="item",
        semantic_group_id="prompt",
        source_locale="vi_vn",
        source_audio_ref="audio.wav",
        duration_s=1.0,
        src_lang="vi",
        tgt_lang="en",
        src_text=text,
        tgt_text="reference",
        target_kind="gold",
        domain="industrial",
        split="train",
        source_dataset="fixture",
        source_revision="v1",
        source_license="fixture-only",
    )


def prediction(text: str) -> CachedPrediction:
    return CachedPrediction(
        cache_key="a" * 64,
        source_id="source-row",
        text=text,
        confidence_proxy=0.8,
        teacher_id="fixture/teacher",
        teacher_revision="v1",
        teacher_license="fixture-only",
        generation_config_sha256="b" * 64,
        generated_at="2026-08-15T00:00:00+00:00",
    )


def teacher_record(text: str) -> ManifestRecord:
    base = source().model_dump(mode="python")
    base.update(
        {
            "id": "teacher-row",
            "tgt_text": text,
            "target_kind": "teacher",
            "teacher_id": "fixture/teacher",
            "teacher_revision": "v1",
            "teacher_license": "fixture-only",
            "generation_config_sha256": "b" * 64,
            "teacher_generated_at": "2026-08-15T00:00:00+00:00",
            "quality_flags": ("teacher_label_unfiltered",),
        }
    )
    return ManifestRecord.model_validate(base)


def test_critical_tokens_pass_with_unverified_language_audit_flag() -> None:
    decision = filter_prediction(source(), prediction("Do not run the machine at 50 rpm."))
    assert decision.keep is True
    assert decision.reject_reasons == ()
    assert decision.audit_flags == ("language_id_unverified",)


def test_number_unit_and_negation_failures_are_explicit() -> None:
    decision = filter_prediction(source(), prediction("Run the machine at 60 Hz."))
    assert decision.keep is False
    assert set(decision.reject_reasons) == {
        "negation_not_preserved",
        "numbers_not_preserved",
        "units_not_preserved",
    }


def test_repetition_and_language_id_gates() -> None:
    decision = filter_prediction(
        source("xin chào"),
        prediction("machine machine machine now"),
        thresholds=FilterThresholds(max_repeated_token_fraction=0.5),
        language_id_valid=False,
    )
    assert "excessive_token_repetition" in decision.reject_reasons
    assert "target_language_invalid" in decision.reject_reasons


def test_optional_agreement_is_lexical_and_thresholded() -> None:
    assert lexical_agreement("close the safety valve", "close safety valve") == 0.75
    decision = filter_prediction(
        source("đóng van"),
        prediction("close the safety valve"),
        thresholds=FilterThresholds(min_lexical_agreement=0.8),
        secondary_text="shut the valve",
    )
    assert decision.keep is False
    assert "lexical_agreement_below_threshold" in decision.reject_reasons


def test_filter_decision_updates_teacher_lineage() -> None:
    record = teacher_record("Do not run the machine at 50 rpm.")
    decision = filter_prediction(source(), prediction(record.tgt_text))
    accepted = apply_filter_decision(record, decision)
    assert accepted.filter_decision == "keep"
    assert "teacher_label_accepted" in accepted.quality_flags
    assert "teacher_label_unfiltered" not in accepted.quality_flags


def test_negation_matching_does_not_accept_substrings() -> None:
    decision = filter_prediction(source("Không chạy máy."), prediction("Normal operation."))
    assert "negation_not_preserved" in decision.reject_reasons


def test_repository_filter_policy_is_validated() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = load_filter_policy(root / "configs" / "filtering" / "vi-en.yaml")
    assert policy.direction == "vi-en"
    assert policy.min_confidence_proxy is None


def test_transcript_and_duration_sanity_gates() -> None:
    decision = filter_prediction(
        source("đóng van an toàn"),
        prediction("close the valve with an excessively verbose explanation"),
        thresholds=FilterThresholds(max_transcript_wer=0.2, max_chars_per_second=10),
        asr_transcript="mở cửa hoàn toàn",
    )
    assert "asr_transcript_wer_above_threshold" in decision.reject_reasons
    assert "target_too_long_for_duration" in decision.reject_reasons
    assert decision.transcript_wer == 0.75
