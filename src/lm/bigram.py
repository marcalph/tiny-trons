import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F

from src.data.utils import SETTINGS, Dataloader, Dataset
from src.tokenizer import CharTokenizer

torch.manual_seed(1337)


class BigramLM(nn.Module):
    """Simple Bigram model."""

    def __init__(self, vocab_sz):
        super().__init__()
        self.embeddings = nn.Embedding(
            vocab_sz, vocab_sz
        )  # Emb_d is num_emb because no project layer

    def forward(self, idx, targets):
        # idx and targets are (B, T)
        logits = self.embeddings(idx)  # (B, T, nEmb)
        # logits = self.lm_head(logits)
        B, T, Emb_d = logits.shape
        logger.debug(logits.shape)
        logits = logits.view(B * T, Emb_d)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss


if __name__ == "__main__":
    ds = Dataset.from_file(SETTINGS.data_path)
    tok = CharTokenizer(corpus=ds.corpus)
    dl = Dataloader(ds, tok, 8, 4)
    xb, yb = dl.get_batch(ds.corpus)
    m = BigramLM(tok.vocab_sz)
    logits, loss = m(xb, yb)
    logger.debug(logits.shape)
    logger.debug(loss)
    print(-np.log(1 / tok.vocab_sz), loss)
