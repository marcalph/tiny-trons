from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: Path = Path("./data/tinyshakespeare.txt")


def read_corpus(file_path: Path) -> str:
    with open(file_path) as f:
        return f.read()


SETTINGS = Settings()
