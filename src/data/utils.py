from pathlib import Path
from typing import Self

import torch
from pydantic_settings import BaseSettings

from src.tokenizer import Tokenizer


class Settings(BaseSettings):
    data_path: Path = Path("./src/data/tinyshakespeare.txt")


# class should prolly inherit from sequence
class Dataset:
    def __init__(self, corpus: str) -> None:
        self.corpus = corpus

    @classmethod
    def from_file(cls, file_path: Path) -> Self:
        with open(file_path) as f:
            return cls(f.read())

    def split(self, train_pct: float = 0.9) -> tuple[str, str]:
        n = train_pct * len(self.corpus)
        train, val = self.corpus[:n], self.corpus[:n]
        return train, val


# TODO(marcalph) define hparams class
class Dataloader:
    def __init__(
        self, dataset: Dataset, tokenizer: Tokenizer, block_sz: int = 8, batch_sz: int = 4
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_sz = batch_sz
        self.block_sz = block_sz  # context size
        torch.manual_seed(1337)

    def _consume(self, split: torch.LongTensor):
        x = split[: self.block_sz]
        y = split[1 : self.block_sz + 1]
        for t in range(self.block_sz):
            context = x[: t + 1]
            target = y[t]
        return context, target

    def get_batch(self, split):
        split = self.tokenizer.encode(split)
        ix = torch.randint(len(split) - self.block_sz, (self.batch_sz,))
        xb = torch.stack([split[i : i + self.block_sz] for i in ix])
        yb = torch.stack([split[i + 1 : i + self.block_sz + 1] for i in ix])
        return xb, yb


SETTINGS = Settings()
