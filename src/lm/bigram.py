import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F

from src.data.utils import Dataloader, Dataset
from src.lm.utils import SETTINGS
from src.tokenizer import CharTokenizer

torch.manual_seed(1337)


class BigramLM(nn.Module):
    """Simple Bigram model."""

    def __init__(self, vocab_sz):
        super().__init__()
        self.embeddings = nn.Embedding(
            vocab_sz, vocab_sz
        )  # Emb_d is num_emb because no project layer

    def forward(self, idx, targets=None):
        # idx and targets are (B, T)
        logits = self.embeddings(idx)  # (B, T, C=nEmb=vocab_sz)
        # logits = self.lm_head(logits)

        if targets is None:
            loss = None
        else:
            B, T, nEmb = logits.shape
            logits = logits.view(B * T, nEmb)
            targets = targets.view(B * T)  # i.e. .view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens) -> torch.Tensor:
        # idx is (B, T), logits is (b, T, C) not flatten because no targets are provided
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).long()  # (B, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx.long()


if __name__ == "__main__":
    ds = Dataset.from_file(SETTINGS.data_path)
    tok = CharTokenizer(corpus=ds.corpus)
    dl = Dataloader(ds, tok, 8, 4)
    xb, yb = dl.get_batch(ds.corpus)
    m = BigramLM(tok.vocab_sz)
    logits, loss = m(xb, yb)
    logger.debug(f"logits.shape -> {logits.shape}")
    logger.debug(loss)
    print(-np.log(1 / tok.vocab_sz), loss)

    print(tok.decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0]))
