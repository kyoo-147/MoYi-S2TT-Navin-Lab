import json
from pathlib import Path

from moyi_s2tt.cli import main
from moyi_s2tt.data.fleurs import align_fleurs_records, read_fleurs_tsv
from moyi_s2tt.data.manifest import write_manifest
from moyi_s2tt.data.source import SourceRecord, write_source_manifest


def test_list_directions(capsys: object) -> None:
    assert main(["list-directions"]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output == ["vi-en", "en-vi", "vi-zh", "zh-vi", "vi-ko", "ko-vi"]


def test_validate_manifest_command(tmp_path: Path, capsys: object) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "fleurs"
    records = align_fleurs_records(
        read_fleurs_tsv(fixtures / "vi_vn-train.tsv"),
        read_fleurs_tsv(fixtures / "en_us-train.tsv"),
        src_lang="vi",
        tgt_lang="en",
        split="train",
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, records)
    assert main(["validate-manifest", str(manifest)]) == 0
    assert json.loads(capsys.readouterr().out) == {  # type: ignore[attr-defined]
        "rows": 3,
        "split_issues": 0,
    }


def test_validate_teacher_catalog(capsys: object) -> None:
    root = Path(__file__).resolve().parents[1]
    assert main(["validate-teachers", "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out) == {  # type: ignore[attr-defined]
        "approved": 0,
        "teachers": 5,
    }


def test_validate_source_manifest_command(tmp_path: Path, capsys: object) -> None:
    record = SourceRecord(
        id="source-fixture-row",
        source_item_id="clip.mp3",
        semantic_group_id="sentence",
        source_locale="vi",
        source_audio_ref="clips/clip.mp3",
        duration_s=1.5,
        speaker_id="speaker",
        src_lang="vi",
        src_text="xin chào",
        domain="general_read",
        split="train",
        source_dataset="fixture",
        source_revision="v1",
        source_license="fixture-only",
    )
    manifest = tmp_path / "source.jsonl"
    write_source_manifest(manifest, [record])
    assert main(["validate-source-manifest", str(manifest)]) == 0
    assert json.loads(capsys.readouterr().out) == {  # type: ignore[attr-defined]
        "rows": 1,
        "split_issues": 0,
    }
