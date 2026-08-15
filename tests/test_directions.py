import pytest

from moyi_s2tt.directions import ALL_DIRECTIONS, DIRECTION_KEYS, Direction


def test_all_vietnamese_centered_directions_are_reserved() -> None:
    assert {"vi-en", "en-vi", "vi-zh", "zh-vi", "vi-ko", "ko-vi"} == DIRECTION_KEYS
    assert len(ALL_DIRECTIONS) == 6


def test_parse_normalizes_case() -> None:
    assert Direction.parse("VI-EN") == Direction("vi", "en")


@pytest.mark.parametrize("value", ["en-zh", "ko-zh", "vi-vi", "fr-vi", "invalid"])
def test_invalid_directions_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        Direction.parse(value)
