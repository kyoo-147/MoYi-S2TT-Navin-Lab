from pathlib import Path

from tools.check_public_boundary import check_paths


def test_private_and_model_paths_are_rejected(tmp_path: Path) -> None:
    private = tmp_path / "docs" / "proposal.md"
    private.parent.mkdir()
    private.write_text("private", encoding="utf-8")
    model = tmp_path / "models" / "student.onnx"
    model.parent.mkdir()
    model.write_bytes(b"model")

    errors = check_paths(tmp_path, [private, model])

    assert any("private path" in error for error in errors)
    assert any("model suffix" in error for error in errors)


def test_approved_source_path_passes(tmp_path: Path) -> None:
    source = tmp_path / "src" / "moyi_s2tt" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    assert check_paths(tmp_path, [source]) == []
