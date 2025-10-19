from pathlib import Path

import torch
from pydantic_settings import BaseSettings


class Hparams(BaseSettings):
    batch_sz: int = 32
    block_sz: int = 8
    max_iters: int = 3000
    eval_iters: int = 200  # for loss estimation
    eval_interval: int = 300
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Settings(BaseSettings):
    data_path: Path = Path("./src/data/tinyshakespeare.txt")
    hparams: Hparams = Hparams()


HPARAMS = Hparams()
SETTINGS = Settings()
