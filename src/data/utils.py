from pathlib import Path
from typing import Self

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import Tokenizer


class TextDataset(Dataset):
    """PyTorch Dataset for character-level language modeling."""

    def __init__(self, data: torch.Tensor, block_sz: int) -> None:
        self.data = data
        self.block_sz = block_sz

    def __len__(self) -> int:
        return len(self.data) - self.block_sz

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_sz]
        y = self.data[idx + 1 : idx + self.block_sz + 1]
        return x, y


class Corpus:
    """Handles loading and splitting text corpus."""

    def __init__(self, text: str) -> None:
        self.text = text

    @classmethod
    def from_file(cls, file_path: Path) -> Self:
        with open(file_path) as f:
            return cls(f.read())

    def split(self, train_pct: float = 0.9) -> tuple[str, str]:
        n = int(train_pct * len(self.text))
        return self.text[:n], self.text[n:]


def create_dataloaders(
    corpus: Corpus,
    tokenizer: Tokenizer,
    block_sz: int,
    batch_sz: int,
    train_pct: float = 0.9,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from corpus."""
    train_text, val_text = corpus.split(train_pct)

    train_data = tokenizer.encode(train_text)
    val_data = tokenizer.encode(val_text)

    train_dataset = TextDataset(train_data, block_sz)
    val_dataset = TextDataset(val_data, block_sz)

    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)

    return train_loader, val_loader
