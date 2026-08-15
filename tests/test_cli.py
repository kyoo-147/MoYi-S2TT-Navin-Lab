import json

from moyi_s2tt.cli import main


def test_list_directions(capsys: object) -> None:
    assert main(["list-directions"]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output == ["vi-en", "en-vi", "vi-zh", "zh-vi", "vi-ko", "ko-vi"]
