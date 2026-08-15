from pathlib import Path

import pytest

from moyi_s2tt.data.common_voice import (
    CommonVoiceAdapter,
    load_common_voice_split,
    read_clip_durations,
)

FIXTURES = Path(__file__).parent / "fixtures" / "common_voice"


def test_clip_duration_header_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "durations.tsv"
    path.write_text("clip\tduration\nclip.mp3\t1000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected clip-duration header"):
        read_clip_durations(path)


def test_common_voice_required_columns_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "train.tsv"
    path.write_text("path\tsentence\nclip.mp3\thello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing Common Voice columns"):
        load_common_voice_split(
            path,
            FIXTURES / "clip_durations.tsv",
            adapter=CommonVoiceAdapter("vi"),
            split="train",
        )


def test_audio_probe_duration_mismatch_rejects_materialization(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "clip-dev.mp3").write_bytes(b"not-real-audio")

    def mismatched_probe(_path: Path) -> object:
        class Metadata:
            duration_s = 9.0
            sample_rate = 16_000
            channels = 1
            codec = "mp3"

        return Metadata()

    batch = load_common_voice_split(
        FIXTURES / "dev.tsv",
        FIXTURES / "clip_durations.tsv",
        adapter=CommonVoiceAdapter("vi"),
        split="validation",
        clips_root=clips,
        audio_probe=mismatched_probe,  # type: ignore[arg-type]
    )
    assert batch.records == ()
    assert [issue.reason for issue in batch.issues] == ["duration_probe_mismatch"]
