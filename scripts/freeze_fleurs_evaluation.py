"""Freeze leakage-safe VI→EN FLEURS validation/test semantic IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

from moyi_s2tt.data.fleurs import FLEURS_REVISION, read_fleurs_tsv
from moyi_s2tt.evaluation.contracts import FrozenEvaluationSet, write_frozen_evaluation


def shared_ids(source: Path, target: Path) -> tuple[str, ...]:
    source_ids = {record.sentence_id for record in read_fleurs_tsv(source)}
    target_ids = {record.sentence_id for record in read_fleurs_tsv(target)}
    return tuple(sorted(source_ids & target_ids, key=int))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/vi-en-fleurs-v1.json"),
    )
    args = parser.parse_args()
    validation = shared_ids(args.source_root / "dev.tsv", args.target_root / "dev.tsv")
    test = shared_ids(args.source_root / "test.tsv", args.target_root / "test.tsv")
    exclusions = tuple(sorted(set(validation) | set(test), key=int))
    frozen = FrozenEvaluationSet(
        id="vi-en-fleurs-v1",
        direction="vi-en",
        source_dataset="google/fleurs",
        source_revision=FLEURS_REVISION,
        source_locale="vi_vn",
        target_locale="en_us",
        validation_semantic_ids=validation,
        test_semantic_ids=test,
        training_exclusion_semantic_ids=exclusions,
    )
    write_frozen_evaluation(args.output, frozen)
    print(f"{args.output}: validation={len(validation)} test={len(test)} sha256={frozen.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
