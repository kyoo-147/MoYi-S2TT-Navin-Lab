"""Pinned Hugging Face teacher implementations loaded only in private inference jobs."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from .base import GenerationConfig, TeacherInput, TeacherPrediction, TeacherSpec


def _modules() -> tuple[Any, Any]:
    try:
        return importlib.import_module("torch"), importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError("teacher inference requires the pinned Colab packages") from error


class HuggingFaceWhisperTeacher:
    def __init__(self, spec: TeacherSpec) -> None:
        if not spec.revision:
            raise ValueError("Whisper teacher revision must be pinned")
        if spec.task not in {"asr", "s2tt"}:
            raise ValueError("Whisper adapter only supports ASR or S2TT specs")
        self._spec = spec
        torch, transformers = _modules()
        use_cuda = bool(torch.cuda.is_available())
        dtype = torch.float16 if use_cuda else torch.float32
        model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            spec.id,
            revision=spec.revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        processor = transformers.AutoProcessor.from_pretrained(spec.id, revision=spec.revision)
        self._pipeline = transformers.pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=0 if use_cuda else -1,
        )

    @property
    def spec(self) -> TeacherSpec:
        return self._spec

    def generate(
        self, inputs: Sequence[TeacherInput], config: GenerationConfig
    ) -> Sequence[TeacherPrediction]:
        if any(not item.audio_path for item in inputs):
            raise ValueError("Whisper teacher requires materialized audio paths")
        generate_kwargs = dict(config.parameters)
        results = self._pipeline(
            [item.audio_path for item in inputs],
            batch_size=config.batch_size,
            generate_kwargs=generate_kwargs,
        )
        if isinstance(results, dict):
            results = [results]
        return [
            TeacherPrediction(
                source_id=item.source_id,
                text=str(result["text"]).strip(),
                metadata={"adapter": "transformers.pipeline"},
            )
            for item, result in zip(inputs, results, strict=True)
        ]


class HuggingFaceNllbTeacher:
    def __init__(self, spec: TeacherSpec, source_token: str, target_token: str) -> None:
        if not spec.revision:
            raise ValueError("NLLB teacher revision must be pinned")
        if spec.task != "mt":
            raise ValueError("NLLB adapter requires an MT spec")
        self._spec = spec
        self._source_token = source_token
        self._target_token = target_token
        torch, transformers = _modules()
        self._torch = torch
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            spec.id, revision=spec.revision, src_lang=source_token
        )
        self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            spec.id, revision=spec.revision, use_safetensors=True
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    @property
    def spec(self) -> TeacherSpec:
        return self._spec

    def generate(
        self, inputs: Sequence[TeacherInput], config: GenerationConfig
    ) -> Sequence[TeacherPrediction]:
        texts = [item.source_text for item in inputs]
        if any(text is None for text in texts):
            raise ValueError("NLLB teacher requires source text")
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)
        parameters = {
            key: value
            for key, value in config.parameters.items()
            if key not in {"source_language_token", "target_language_token"}
        }
        forced_bos = self._tokenizer.convert_tokens_to_ids(self._target_token)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                **parameters,
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [
            TeacherPrediction(
                source_id=item.source_id,
                text=text.strip(),
                metadata={
                    "adapter": "transformers.nllb",
                    "source_token": self._source_token,
                    "target_token": self._target_token,
                },
            )
            for item, text in zip(inputs, decoded, strict=True)
        ]
