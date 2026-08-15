from pathlib import Path

from moyi_s2tt.teachers.audit import load_audit_config
from scripts.run_teacher_audit import word_error_rate

ROOT = Path(__file__).resolve().parents[1]


def test_vi_en_audit_contract_is_small_deterministic_and_pinned() -> None:
    config = load_audit_config(ROOT / "configs" / "inference" / "vi-en-teacher-audit.yaml")
    assert config.direction == "vi-en"
    assert config.max_rows == 100
    assert config.max_audio_seconds == 30
    assert config.asr.generation(config.seed).sha256 == (
        "dbc0614046ee4535ae87ea5a666a854977e5595925d04906c9f37d723ef8d251"
    )
    assert config.seamless_m4t.status == "hold"


def test_word_error_rate_uses_real_edit_distance() -> None:
    assert word_error_rate("mot hai ba", "mot hai ba") == 0
    assert word_error_rate("mot hai ba", "mot ba") == 1 / 3
    assert word_error_rate("", "extra") == 1
