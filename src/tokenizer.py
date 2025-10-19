from abc import ABC, abstractmethod

import torch
from loguru import logger


class Tokenizer(ABC):
    def __init__(self, corpus: str):
        self.vocab: list[str] = []
        self.toktoi: dict[str, int] = {}
        self.itotok: dict[int, str] = {}
        self.vocab_sz: int = 0
        self._build_vocab(corpus)

    @abstractmethod
    def _build_vocab(self, corpus: str) -> None: ...

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.toktoi[token] for token in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join([self.itotok[tokenid] for tokenid in tokens.tolist()])


# TODO(marcalph): implement BPE/SentencePiece tokenizers
class CharTokenizer(Tokenizer):
    def __init__(self, corpus):
        super().__init__(corpus)

    def _build_vocab(self, corpus) -> None:
        self.vocab = sorted(set(corpus))
        self.toktoi = {c: i for i, c in enumerate(self.vocab)}
        self.itotok = {i: c for c, i in self.toktoi.items()}
        self.vocab_sz = len(self.toktoi)
        logger.debug(f"Built vocabulary of sz {self.vocab_sz} for {self}.")


if __name__ == "__main__":
    from src.data.utils import Dataset
    from src.lm.utils import SETTINGS

    ds = Dataset.from_file(file_path=SETTINGS.data_path)
    tok = CharTokenizer(corpus=ds.corpus)
