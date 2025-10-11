from pathlib import Path
from typing import Self

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: Path = Path("./data/tinyshakespeare.txt")


# class should prolly inherit from sequence
class Dataset:
    def __init__(self, corpus: str) -> None:
        self.corpus = corpus

    @classmethod
    def from_file(cls, file_path: Path) -> Self:
        with open(file_path) as f:
            return cls(f.read())

    def split(self, corpus, train_pct: float = 0.9) -> tuple[str, str]:
        n = train_pct * len(corpus)
        train, val = corpus[:n], corpus[:n]
        return train, val


SETTINGS = Settings()
