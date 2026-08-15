"""Create a deterministic human-review sheet from a metadata-only manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from moyi_s2tt.data.manifest import iter_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", default="147")
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    records = list(iter_manifest(args.manifest))
    ranked = sorted(
        records,
        key=lambda row: hashlib.sha256(f"{args.seed}:{row.id}".encode()).hexdigest(),
    )[: args.size]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "id",
        "source_item_id",
        "src_text",
        "tgt_text",
        "review_decision",
        "review_notes",
    )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in ranked:
            writer.writerow(
                {
                    "id": row.id,
                    "source_item_id": row.source_item_id,
                    "src_text": row.src_text,
                    "tgt_text": row.tgt_text,
                    "review_decision": "",
                    "review_notes": "",
                }
            )
    print(f"wrote {len(ranked)} audit rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
