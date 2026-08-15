import json
from pathlib import Path

from moyi_s2tt.cli import main
from moyi_s2tt.data.fleurs import align_fleurs_records, read_fleurs_tsv
from moyi_s2tt.data.manifest import write_manifest


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
