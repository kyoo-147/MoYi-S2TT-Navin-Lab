from pathlib import Path

import pytest

from moyi_s2tt.teachers.base import GenerationConfig, TeacherInput, TeacherSpec
from moyi_s2tt.teachers.catalog import load_teacher_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_public_teacher_catalog_is_declarative_and_unapproved() -> None:
    specs = load_teacher_catalog(ROOT / "configs" / "teachers")
    assert len(specs) == 5
    assert {spec.task for spec in specs} == {"asr", "mt", "s2tt"}
    assert not [spec for spec in specs if spec.status == "approved"]
    assert len([spec for spec in specs if spec.status == "pinned"]) == 3
    assert all(spec.revision for spec in specs if spec.status == "pinned")
    assert next(spec for spec in specs if spec.id == "openai/whisper-large-v3-turbo").revision == (
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    )
    seamless = next(spec for spec in specs if spec.id == "facebook/seamless-m4t-v2-large")
    assert seamless.research_only is True


def test_approved_teacher_requires_pinned_revision() -> None:
    with pytest.raises(ValueError, match="immutable revision"):
        TeacherSpec(
            id="fixture/teacher",
            task="mt",
            status="approved",
            license="fixture-only",
            research_only=True,
            directions=("vi-en",),
        )


def test_teacher_input_requires_audio_or_text() -> None:
    with pytest.raises(ValueError, match="source_text or audio_path"):
        TeacherInput(source_id="row-1", src_lang="vi", tgt_lang="en")


def test_generation_hash_is_order_independent() -> None:
    left = GenerationConfig(parameters={"temperature": 0.0, "beams": 4})
    right = GenerationConfig(parameters={"beams": 4, "temperature": 0.0})
    assert left.sha256 == right.sha256
