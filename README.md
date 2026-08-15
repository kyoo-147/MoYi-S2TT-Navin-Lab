# MoYi S2TT Navin Lab

Training, distillation, evaluation, and mobile export tooling for compact multilingual speech-to-text translation (S2TT) models.

## Scope

The shared code is direction-agnostic and reserves six Vietnamese-centered directions:

- `vi-en`, `en-vi`
- `vi-zh`, `zh-vi`
- `vi-ko`, `ko-vi`

The first active research slice is **Vietnamese to English (`vi-en`)**. Other directions remain disabled until the same data, quality, and deployment gates can be applied truthfully.

Model experiments use four evidence-driven tiers:

| Tier | Intended role |
|---|---|
| `tiny` | 30–50M pipeline smoke proof |
| `mobile` | 70–90M low-latency candidate |
| `base` | 120–160M primary quality candidate |
| `plus` | 200–250M optional measured upper bound |

Parameter ranges are targets, not released model claims.

## Repository boundary

This repository tracks source code, tests, clean notebooks, configuration, dataset provenance metadata, and tiny approved fixtures. It does **not** track raw datasets, bulk pseudo-labels, checkpoints, model weights, private recordings, credentials, or internal competition/business material.

Dataset and model licenses remain asset-specific. No dataset or checkpoint is redistributed merely because a downloader or manifest exists here.

## Development

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --all-groups
uv run ruff check .
uv run mypy src
uv run pytest
uv run python tools/check_public_boundary.py
uv run python tools/check_notebooks.py
```

Validate the six direction contracts:

```bash
uv run moyi-s2tt validate-configs
uv run moyi-s2tt list-directions
```

## Licensed source adapters

FLEURS and Common Voice share a canonical source-audio contract. Common Voice Scripted Speech 26.0 is pinned for VI/EN/ZH-CN/KO metadata, with VI first. Its content is listed as CC0-1.0, while current access terms require Mozilla Data Collective authentication and terms acceptance and restrict mirroring. No Common Voice archive or row is redistributed here, and the source remains unaccepted until a real archive checksum, TSV parse, audio probe, and cross-split leakage report pass.

## Frozen evaluation

`vi-en-fleurs-v1` freezes 149 aligned validation and 347 aligned test semantic IDs at the pinned FLEURS revision. Their 496-ID union is a mandatory exclusion set for training and pseudo-label generation. Conversation and industrial human evaluation remain explicitly unavailable; FLEURS read speech is not used to claim those domains.

```bash
uv run moyi-s2tt validate-evaluation data/evaluation/vi-en-fleurs-v1.json
```

## Offline teacher boundary

Teacher candidates for Whisper, NLLB, OPUS-MT, and Seamless are declarative and validated with:

```bash
uv run moyi-s2tt validate-teachers
```

No teacher is approved or pinned yet, so production inference intentionally refuses to run. The shared interface and SQLite cache key every prediction by input content, teacher revision, and generation configuration. Cached labels retain teacher/license provenance and remain explicitly unfiltered until later quality gates accept or reject them. Bulk labels and cache databases are ignored by Git.

## Label filtering

Teacher labels remain non-gold. The first deterministic VI→EN gates reject changed numbers, abbreviated units, missing negation, excessive repetition, invalid target-language checks, and optional low lexical teacher agreement. Confidence thresholds remain unset until each teacher proxy is calibrated. Lexical overlap is explicitly a triage proxy, not semantic equivalence; accepted rows still require downstream audit and frozen evaluation.

## Status

The repository foundation and canonical manifest contracts are implemented. Pinned FLEURS metadata tooling can produce leakage-checked, metadata-only VI→EN rows; no FLEURS audio has been accepted or redistributed. No model-quality, accepted dataset-scale, mobile-latency, or accelerator-placement result is claimed until a versioned evidence artifact is produced.

## License

No license has been granted yet. Source is visible for review; reuse rights remain reserved until a license is selected.
