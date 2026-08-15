"""Run pinned Whisper Tiny VI→EN overfit/smoke training in private storage."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from moyi_s2tt.data.manifest import ManifestRecord, iter_manifest
from moyi_s2tt.evaluation.contracts import load_frozen_evaluation, require_evaluation_clean
from moyi_s2tt.runtime.checkpointing import require_private_root
from moyi_s2tt.runtime.colab import environment_report
from moyi_s2tt.training.contracts import (
    load_training_config,
    make_run_contract,
    require_compatible_run,
    require_trainer_checkpoint,
    select_training_rows,
    write_run_contract,
)


class WhisperRows:
    def __init__(self, rows: list[ManifestRecord], processor: Any, audio_root: Path) -> None:
        self.rows = rows
        self.processor = processor
        self.audio_root = audio_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        soundfile = importlib.import_module("soundfile")
        row = self.rows[index]
        if not row.audio_path or not row.audio_sha256:
            raise ValueError(f"row {row.id} is not materialized")
        path = Path(row.audio_path)
        path = path if path.is_absolute() else self.audio_root / path
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row.audio_sha256:
            raise ValueError(f"audio hash mismatch for {row.id}")
        audio, rate = soundfile.read(path, dtype="float32")
        features = self.processor.feature_extractor(
            audio, sampling_rate=rate, return_tensors="pt"
        ).input_features[0]
        labels = self.processor.tokenizer(row.tgt_text).input_ids
        return {"input_features": features, "labels": labels}


class WhisperCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        torch = importlib.import_module("torch")
        inputs = self.processor.feature_extractor.pad(
            [{"input_features": item["input_features"]} for item in features],
            return_tensors="pt",
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in features], return_tensors="pt"
        )
        values = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (values[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            values = values[:, 1:]
        inputs["labels"] = torch.as_tensor(values)
        return dict(inputs)


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    private = require_private_root(repo)
    for path in (args.manifest.resolve(), args.audio_root.resolve()):
        if path != private and private not in path.parents:
            raise ValueError("training inputs must remain under MOYI_PRIVATE_ROOT")
    config = load_training_config(args.config)
    rows = list(iter_manifest(args.manifest))
    selected = list(select_training_rows(rows, config))
    frozen = load_frozen_evaluation(repo / "data/evaluation/vi-en-fleurs-v1.json")
    require_evaluation_clean(selected, frozen)
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    contract = make_run_contract(config, selected, git_commit)
    output = private / "training" / config.id
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "run-contract.json"
    if contract_path.exists():
        require_compatible_run(contract_path, contract)
    else:
        write_run_contract(contract_path, contract)

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    processor = transformers.AutoProcessor.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        language=config.source_language,
        task=config.task,
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config.model_id, revision=config.model_revision, use_safetensors=True
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    lower, upper = config.expected_parameter_range_millions
    if not lower * 1_000_000 <= parameter_count <= upper * 1_000_000:
        raise ValueError(f"model parameter count outside declared tier: {parameter_count}")
    validation_count = int(len(selected) * config.validation_fraction)
    validation = selected[-validation_count:] if validation_count else []
    training = selected[:-validation_count] if validation_count else selected
    common: dict[str, Any] = {
        "output_dir": str(output / "trainer"),
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "save_steps": config.save_steps,
        "logging_steps": 5,
        "save_total_limit": 3,
        "seed": config.seed,
        "data_seed": config.seed,
        "fp16": config.mixed_precision == "fp16",
        "bf16": config.mixed_precision == "bf16",
        "report_to": [],
        "remove_unused_columns": False,
        "predict_with_generate": True,
    }
    if validation:
        common.update({"eval_strategy": "steps", "eval_steps": config.eval_steps})
    arguments = transformers.Seq2SeqTrainingArguments(**common)
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=WhisperRows(training, processor, args.audio_root),
        eval_dataset=WhisperRows(validation, processor, args.audio_root) if validation else None,
        data_collator=WhisperCollator(processor),
        processing_class=processor,
    )
    checkpoints = sorted((output / "trainer").glob("checkpoint-*"), key=lambda p: int(p.name[11:]))
    resume = checkpoints[-1] if checkpoints else None
    if resume:
        require_trainer_checkpoint(resume)
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(output / "trainer" / "final")
    trainer.save_state()
    report = environment_report()
    evidence = {
        "schema_version": 1,
        "status": "VERIFIED_REAL_TRAINING",
        "run_id": config.id,
        "mode": config.mode,
        "config_sha256": config.sha256,
        "data_sha256": contract.data_sha256,
        "rows": len(selected),
        "parameter_count": parameter_count,
        "global_step": trainer.state.global_step,
        "metrics": result.metrics,
        "accelerator": report.accelerator,
        "package_versions": report.packages,
        "private_output_sha256": tree_sha256(output / "trainer"),
        "claim": "pipeline_gate_only_not_product_quality",
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
