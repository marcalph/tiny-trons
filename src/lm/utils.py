from pathlib import Path

import torch
from pydantic_settings import BaseSettings


class Hparams(BaseSettings):
    batch_sz: int = 32  # B: number of sequences per batch (larger = better GPU utilization)
    block_sz: int = 256  # T: context length (larger = more context for model)
    max_iters: int = 10000  # total training steps
    eval_iters: int = 200  # steps between loss evaluation
    eval_interval: int = 500  # unused currently
    lr: float = 1e-2  # learning rate for AdamW (higher for simple bigram model)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Tensor shapes: input (B, T), embeddings (B, T, C), logits (B, T, vocab_sz)


class Settings(BaseSettings):
    data_path: Path = Path("./src/data/tinyshakespeare.txt")
    hparams: Hparams = Hparams()


HPARAMS = Hparams()
SETTINGS = Settings()
