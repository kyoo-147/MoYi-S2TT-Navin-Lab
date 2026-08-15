from pathlib import Path

from moyi_s2tt.config import load_direction, validate_config_tree

ROOT = Path(__file__).resolve().parents[1]


def test_repository_config_tree_is_complete() -> None:
    assert validate_config_tree(ROOT) == {
        "directions": 6,
        "student_tiers": 4,
        "experiments": 2,
    }


def test_only_vi_en_is_enabled() -> None:
    configs = [load_direction(path) for path in sorted((ROOT / "configs/languages").glob("*.yaml"))]
    assert [config.key for config in configs if config.enabled] == ["vi-en"]
