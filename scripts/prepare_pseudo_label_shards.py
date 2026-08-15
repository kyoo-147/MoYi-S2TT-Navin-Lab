"""Shard, filter, and report private cached teacher-chain predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moyi_s2tt.data.filtering import load_filter_policy
from moyi_s2tt.data.manifest import write_manifest
from moyi_s2tt.data.source import iter_source_manifest
from moyi_s2tt.distillation.cache import CachedPrediction
from moyi_s2tt.distillation.candidates import PseudoLabelCandidate
from moyi_s2tt.distillation.filter_report import filter_candidates
from moyi_s2tt.distillation.shards import ShardLedger, ShardRunner
from moyi_s2tt.evaluation.contracts import load_frozen_evaluation
from moyi_s2tt.runtime.checkpointing import require_private_root


def require_under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"private pseudo-label path must remain under MOYI_PRIVATE_ROOT: {path}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    private = require_private_root(repo)
    source_path = require_under(args.source_manifest, private)
    predictions_path = require_under(args.teacher_predictions, private)
    output = require_under(args.output, private)
    output.mkdir(parents=True, exist_ok=True)
    sources = {record.id: record for record in iter_source_manifest(source_path)}
    candidates: list[PseudoLabelCandidate] = []
    with predictions_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            source_id = str(value["source_id"])
            if source_id not in sources:
                raise ValueError(f"prediction row {line_number} has unknown source ID {source_id}")
            candidates.append(
                PseudoLabelCandidate(
                    source=sources[source_id],
                    asr=CachedPrediction.model_validate(value["asr_teacher"]),
                    mt=CachedPrediction.model_validate(value["mt_teacher"]),
                    direct_s2tt=CachedPrediction.model_validate(value["direct_s2tt_teacher"]),
                )
            )
    with ShardLedger(output / "shards.sqlite3") as ledger:
        runner = ShardRunner[PseudoLabelCandidate](output / "candidates", ledger)
        runner.run(
            candidates,
            get_id=lambda candidate: candidate.source.id,
            shard_size=args.shard_size,
            process=lambda rows: [row.model_dump(mode="json") for row in rows],
        )

    policy = load_filter_policy(repo / "configs/filtering/vi-en.yaml")
    frozen = load_frozen_evaluation(repo / "data/evaluation/vi-en-fleurs-v1.json")
    records, report = filter_candidates(candidates, policy, frozen)
    accepted = [record for record in records if record.filter_decision == "keep"]
    rejected = [record for record in records if record.filter_decision == "reject"]
    write_manifest(output / "accepted.jsonl", accepted)
    write_manifest(output / "rejected.jsonl", rejected)
    sample_ids = tuple(record.id for record in (accepted[:10] + rejected[:10]))
    report = report.model_copy(update={"human_audit_sample_ids": sample_ids})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
