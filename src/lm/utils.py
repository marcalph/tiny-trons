from pathlib import Path

import torch
from pydantic_settings import BaseSettings


class Hparams(BaseSettings):
    batch_sz = 32
    block_sz = 8
    max_iters = 3000
    eval_iters = 200  # for loss estimation
    eval_interval = 300
    lr = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"


class Settings(BaseSettings):
    data_path: Path = Path("./src/data/tinyshakespeare.txt")
    hparams = Hparams


HPARAMS = Hparams()
SETTINGS = Settings()
