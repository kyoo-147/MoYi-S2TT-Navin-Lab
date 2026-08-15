# Data boundary

Only schemas, source registries, checksums, manifests, and tiny redistribution-approved fixtures belong in Git.

Raw audio, derived datasets, pseudo-label corpora, private recordings, and model artifacts must remain in ignored local storage or separately approved remote storage. A downloader does not grant redistribution rights. Every accepted row must preserve its source revision, license, split, and teacher provenance.

## Pinned FLEURS metadata smoke

Download the VI/EN TSV metadata from the pinned public revision (audio is not downloaded):

```bash
uv run python scripts/download_fleurs.py
```

Prepare one metadata-only aligned manifest after mapping FLEURS `dev.tsv` to the canonical `validation` split:

```bash
uv run python scripts/prepare_fleurs_manifest.py \
  --source-tsv data/downloads/fleurs/<revision>/vi_vn/train.tsv \
  --target-tsv data/downloads/fleurs/<revision>/en_us/train.tsv \
  --source-language vi --target-language en --split train \
  --output data/manifests/vi-en-fleurs-train.jsonl
uv run moyi-s2tt validate-manifest data/manifests/vi-en-fleurs-train.jsonl
uv run python scripts/sample_alignment_audit.py \
  data/manifests/vi-en-fleurs-train.jsonl \
  --output data/downloads/vi-en-alignment-audit.tsv --size 100
```

Generated rows remain `metadata_only`: they intentionally have no audio hash or speaker ID and carry a manual-alignment-audit flag. They are not training-ready until audio is lawfully materialized, hashed, audited, and the manifest passes cross-split checks.
