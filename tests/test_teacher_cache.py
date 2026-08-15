from collections.abc import Sequence
from pathlib import Path

import pytest

from moyi_s2tt.data.manifest import ManifestRecord
from moyi_s2tt.distillation.cache import PredictionCache, generate_cached
from moyi_s2tt.distillation.pseudo_labels import apply_teacher_prediction
from moyi_s2tt.teachers.base import (
    GenerationConfig,
    TeacherInput,
    TeacherPrediction,
    TeacherSpec,
)


class FakeTeacher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._spec = TeacherSpec(
            id="fixture/teacher",
            task="mt",
            status="approved",
            revision="fixture-revision-1",
            license="fixture-only",
            research_only=True,
            directions=("vi-en",),
        )

    @property
    def spec(self) -> TeacherSpec:
        return self._spec

    def generate(
        self, inputs: Sequence[TeacherInput], config: GenerationConfig
    ) -> Sequence[TeacherPrediction]:
        self.calls.append(tuple(item.source_id for item in inputs))
        return [
            TeacherPrediction(
                source_id=item.source_id,
                text=f"translated: {item.source_text}",
                confidence_proxy=0.75,
                metadata={"fixture": True},
            )
            for item in inputs
        ]


def source_record() -> ManifestRecord:
    return ManifestRecord(
        id="source-row-1",
        source_item_id="item-1",
        semantic_group_id="prompt-1",
        source_locale="vi_vn",
        source_audio_ref="audio/one.wav",
        duration_s=1.0,
        src_lang="vi",
        tgt_lang="en",
        src_text="xin chào",
        tgt_text="source reference only",
        target_kind="gold",
        domain="conversation",
        split="train",
        source_dataset="fixture",
        source_revision="fixture-v1",
        source_license="fixture-only",
    )


def test_cache_skips_repeat_teacher_inference(tmp_path: Path) -> None:
    teacher = FakeTeacher()
    item = TeacherInput(
        source_id="source-row-1", src_lang="vi", tgt_lang="en", source_text="xin chào"
    )
    config = GenerationConfig(batch_size=4, parameters={"beams": 2})
    with PredictionCache(tmp_path / "labels.sqlite3") as cache:
        first = generate_cached(teacher, [item], config, cache)
        second = generate_cached(teacher, [item], config, cache)

    assert teacher.calls == [("source-row-1",)]
    assert first == second
    assert first[0].teacher_revision == "fixture-revision-1"
    assert first[0].generation_config_sha256 == config.sha256


def test_generation_config_change_is_a_cache_miss(tmp_path: Path) -> None:
    teacher = FakeTeacher()
    item = TeacherInput(
        source_id="source-row-1", src_lang="vi", tgt_lang="en", source_text="xin chào"
    )
    with PredictionCache(tmp_path / "labels.sqlite3") as cache:
        generate_cached(teacher, [item], GenerationConfig(parameters={"beams": 1}), cache)
        generate_cached(teacher, [item], GenerationConfig(parameters={"beams": 2}), cache)
    assert len(teacher.calls) == 2


def test_cached_output_becomes_unfiltered_teacher_manifest(tmp_path: Path) -> None:
    teacher = FakeTeacher()
    source = source_record()
    item = TeacherInput(
        source_id=source.id,
        src_lang=source.src_lang,
        tgt_lang=source.tgt_lang,
        source_text=source.src_text,
    )
    with PredictionCache(tmp_path / "labels.sqlite3") as cache:
        prediction = generate_cached(teacher, [item], GenerationConfig(), cache)[0]
    pseudo = apply_teacher_prediction(source, prediction)
    assert pseudo.target_kind == "teacher"
    assert pseudo.tgt_text == "translated: xin chào"
    assert pseudo.teacher_id == teacher.spec.id
    assert pseudo.teacher_license == "fixture-only"
    assert "teacher_label_unfiltered" in pseudo.quality_flags


def test_teacher_rejects_unsupported_direction(tmp_path: Path) -> None:
    teacher = FakeTeacher()
    item = TeacherInput(source_id="row", src_lang="en", tgt_lang="vi", source_text="hello")
    with (
        PredictionCache(tmp_path / "labels.sqlite3") as cache,
        pytest.raises(ValueError, match="does not support"),
    ):
        generate_cached(teacher, [item], GenerationConfig(), cache)
