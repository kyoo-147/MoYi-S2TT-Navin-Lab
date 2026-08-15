"""Deterministic, atomic, resumable pseudo-label shard execution."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import Field

from ..config import StrictModel

Item = TypeVar("Item")


class ShardSpec(StrictModel):
    index: int = Field(ge=0)
    item_ids: tuple[str, ...]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ShardResult(StrictModel):
    index: int
    rows: int
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str


def plan_shards(item_ids: Sequence[str], shard_size: int) -> tuple[ShardSpec, ...]:
    if shard_size <= 0:
        raise ValueError("shard size must be positive")
    ordered = tuple(sorted(item_ids))
    if len(ordered) != len(set(ordered)):
        raise ValueError("pseudo-label source IDs must be unique")
    return tuple(
        ShardSpec(
            index=index,
            item_ids=ordered[offset : offset + shard_size],
            input_sha256=hashlib.sha256(
                json.dumps(
                    ordered[offset : offset + shard_size], separators=(",", ":")
                ).encode()
            ).hexdigest(),
        )
        for index, offset in enumerate(range(0, len(ordered), shard_size))
    )


class ShardLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS shards (
                shard_index INTEGER PRIMARY KEY,
                input_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                output_sha256 TEXT,
                rows INTEGER
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ShardLedger:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def is_complete(self, shard: ShardSpec, output: Path) -> bool:
        row = self.connection.execute(
            "SELECT input_sha256, status, output_sha256 FROM shards WHERE shard_index = ?",
            (shard.index,),
        ).fetchone()
        if not row:
            return False
        if row[0] != shard.input_sha256:
            raise ValueError(f"shard {shard.index} input hash changed")
        if row[1] != "complete":
            return False
        if not output.is_file() or hashlib.sha256(output.read_bytes()).hexdigest() != row[2]:
            raise ValueError(f"completed shard {shard.index} output integrity failure")
        return True

    def begin(self, shard: ShardSpec) -> None:
        row = self.connection.execute(
            "SELECT input_sha256, attempts FROM shards WHERE shard_index = ?", (shard.index,)
        ).fetchone()
        if row and row[0] != shard.input_sha256:
            raise ValueError(f"shard {shard.index} input hash changed")
        attempts = int(row[1]) + 1 if row else 1
        self.connection.execute(
            """
            INSERT OR REPLACE INTO shards
            (shard_index, input_sha256, status, attempts, output_sha256, rows)
            VALUES (?, ?, 'running', ?, NULL, NULL)
            """,
            (shard.index, shard.input_sha256, attempts),
        )
        self.connection.commit()

    def complete(self, shard: ShardSpec, output_sha256: str, rows: int) -> None:
        self.connection.execute(
            """
            UPDATE shards SET status = 'complete', output_sha256 = ?, rows = ?
            WHERE shard_index = ? AND input_sha256 = ?
            """,
            (output_sha256, rows, shard.index, shard.input_sha256),
        )
        self.connection.commit()


class ShardRunner(Generic[Item]):
    def __init__(self, output_root: Path, ledger: ShardLedger) -> None:
        self.output_root = output_root
        self.ledger = ledger
        output_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        items: Sequence[Item],
        *,
        get_id: Callable[[Item], str],
        shard_size: int,
        process: Callable[[Sequence[Item]], Sequence[dict[str, object]]],
    ) -> list[ShardResult]:
        by_id = {get_id(item): item for item in items}
        shards = plan_shards(tuple(by_id), shard_size)
        results: list[ShardResult] = []
        for shard in shards:
            destination = self.output_root / f"shard-{shard.index:05d}.jsonl"
            if not self.ledger.is_complete(shard, destination):
                self.ledger.begin(shard)
                rows = process([by_id[item_id] for item_id in shard.item_ids])
                if len(rows) != len(shard.item_ids):
                    raise ValueError(f"shard {shard.index} output count mismatch")
                temporary = destination.with_suffix(".jsonl.tmp")
                with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                os.replace(temporary, destination)
                self.ledger.complete(shard, digest, len(rows))
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            results.append(
                ShardResult(
                    index=shard.index,
                    rows=len(shard.item_ids),
                    output_sha256=digest,
                    path=str(destination),
                )
            )
        return results
