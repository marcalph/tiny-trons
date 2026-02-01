import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F

from src.data.utils import Corpus, create_dataloaders
from src.lm.utils import SETTINGS
from src.tokenizer import CharTokenizer
from src.lm.attention import AttenBlock
torch.manual_seed(1337)


class AttenBigramLM(nn.Module):
    """Simple Bigram model."""

    def __init__(self, vocab_sz, emb_d, block_sz, n_heads):
        super().__init__()
        self.block_sz = block_sz
        self.embeddings = nn.Embedding(
            vocab_sz, emb_d
        )
        self.position_embeddings = nn.Embedding(block_sz, emb_d)
        self.blocks = nn.Sequential(
            AttenBlock(emb_d, n_heads=n_heads, block_sz=block_sz),
            AttenBlock(emb_d, n_heads=n_heads, block_sz=block_sz),
            AttenBlock(emb_d, n_heads=n_heads, block_sz=block_sz),
            nn.LayerNorm(emb_d)
        )
        self.lm_head = nn.Linear(emb_d, vocab_sz)


    def forward(self, idx, targets=None):
        # idx and targets are (B, T)
        B, T = idx.shape

        tok_emb = self.embeddings(idx)  # (B, T, C=emb_d)
        pos_emb = self.position_embeddings(torch.arange(T))  # (T, C=emb_d)
        x = tok_emb + pos_emb  # (B, T, C) bc broadcasting
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B, T, vocab_sz)

        if targets is None:
            loss = None
        else:
            B, T, nEmb = logits.shape
            logits = logits.view(B * T, nEmb)
            targets = targets.view(B * T)  # i.e. .view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens) -> torch.Tensor:
        # idx is (B, T), logits is (B, T, C) not flattened because no targets provided
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_sz:]  # crop to last block_sz tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).long()  # (B, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx.long()


if __name__ == "__main__":
    corpus = Corpus.from_file(SETTINGS.data_path)
    tok = CharTokenizer(corpus=corpus.text)
    train_loader, _ = create_dataloaders(corpus, tok, block_sz=8, batch_sz=4)

    xb, yb = next(iter(train_loader))
    m = AttenBigramLM(tok.vocab_sz, block_sz=12, emb_d=32, head_sz=32)
    logits, loss = m(xb, yb)
    logger.debug(f"logits.shape -> {logits.shape}")
    logger.debug(loss)
    print(f"Expected loss: {-np.log(1 / tok.vocab_sz):.4f}, Actual: {loss:.4f}")

    logger.info(
        tok.decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=90)[0])
    )
