"""Canonical language-direction contracts shared by every pipeline stage."""

from dataclasses import dataclass

SUPPORTED_LANGUAGES = frozenset({"vi", "en", "zh", "ko"})


@dataclass(frozen=True, slots=True)
class Direction:
    """A supported source-to-target translation direction."""

    source: str
    target: str

    def __post_init__(self) -> None:
        if self.source not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported source language: {self.source}")
        if self.target not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported target language: {self.target}")
        if self.source == self.target:
            raise ValueError("source and target languages must differ")
        if "vi" not in {self.source, self.target}:
            raise ValueError("MoYi directions must be Vietnamese-centered")

    @property
    def key(self) -> str:
        return f"{self.source}-{self.target}"

    @classmethod
    def parse(cls, value: str) -> "Direction":
        parts = value.lower().split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid direction: {value}")
        return cls(source=parts[0], target=parts[1])


ALL_DIRECTIONS = (
    Direction("vi", "en"),
    Direction("en", "vi"),
    Direction("vi", "zh"),
    Direction("zh", "vi"),
    Direction("vi", "ko"),
    Direction("ko", "vi"),
)
DIRECTION_KEYS = frozenset(direction.key for direction in ALL_DIRECTIONS)
