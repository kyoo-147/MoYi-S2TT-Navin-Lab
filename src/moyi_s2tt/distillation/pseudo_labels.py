"""Convert cached teacher outputs into provenance-complete manifest rows."""

from __future__ import annotations

import hashlib

from ..data.manifest import ManifestRecord
from .cache import CachedPrediction


def apply_teacher_prediction(
    source: ManifestRecord, prediction: CachedPrediction
) -> ManifestRecord:
    if source.id != prediction.source_id:
        raise ValueError("prediction source ID does not match manifest record")
    identifier = hashlib.sha256(f"{source.id}\0{prediction.cache_key}".encode()).hexdigest()
    values = source.model_dump(mode="python")
    values.update(
        {
            "id": f"teacher-{identifier}",
            "tgt_text": prediction.text,
            "target_kind": "teacher",
            "teacher_id": prediction.teacher_id,
            "teacher_revision": prediction.teacher_revision,
            "teacher_license": prediction.teacher_license,
            "generation_config_sha256": prediction.generation_config_sha256,
            "quality_flags": tuple(
                sorted(set(source.quality_flags) | {"teacher_label_unfiltered"})
            ),
        }
    )
    return ManifestRecord.model_validate(values)
