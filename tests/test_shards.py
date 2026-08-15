import json
from pathlib import Path

import pytest

from moyi_s2tt.distillation.shards import ShardLedger, ShardRunner, plan_shards


def test_shard_plan_is_sorted_and_content_addressed() -> None:
    shards = plan_shards(["c", "a", "b"], 2)
    assert shards[0].item_ids == ("a", "b")
    assert shards[1].item_ids == ("c",)
    assert len(shards[0].input_sha256) == 64


def test_interrupted_shards_resume_without_reprocessing_complete_shard(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    fail_once = True

    def process(items: list[str]) -> list[dict[str, object]]:
        nonlocal fail_once
        calls.append(tuple(items))
        if items == ["c", "d"] and fail_once:
            fail_once = False
            raise RuntimeError("simulated interruption")
        return [{"id": item, "target_kind": "teacher"} for item in items]

    with ShardLedger(tmp_path / "ledger.sqlite3") as ledger:
        runner = ShardRunner[str](tmp_path / "shards", ledger)
        with pytest.raises(RuntimeError, match="interruption"):
            runner.run(
                ["d", "b", "a", "c"],
                get_id=lambda item: item,
                shard_size=2,
                process=process,
            )
        results = runner.run(
            ["d", "b", "a", "c"],
            get_id=lambda item: item,
            shard_size=2,
            process=process,
        )

    assert calls == [("a", "b"), ("c", "d"), ("c", "d")]
    assert [result.rows for result in results] == [2, 2]
    rows = [
        json.loads(line)
        for line in (tmp_path / "shards" / "shard-00000.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in rows] == ["a", "b"]
