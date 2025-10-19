import torch
from bigram import BigramLM
from loguru import logger
from torch.optim.adamw import AdamW

from src.data.utils import SETTINGS, Dataloader, Dataset
from src.tokenizer import CharTokenizer

if __name__ == "__main__":
    batch_sz = 32
    ds = Dataset.from_file(SETTINGS.data_path)
    tok = CharTokenizer(corpus=ds.corpus)
    dl = Dataloader(ds, tok, 8, batch_sz)
    xb, yb = dl.get_batch(ds.corpus)
    m = BigramLM(tok.vocab_sz)
    logits, loss = m(xb, yb)
    logger.debug(f"logits.shape -> {logits.shape}")
    logger.debug(loss)
    opt = AdamW(m.parameters(), lr=1e-3)

    # train loop
    for _ in range(1000):
        xb, yb = dl.get_batch(ds.corpus)
        logits, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)  # set.grad to None to save mem
        loss.backward()
        opt.step()
        if _ % 100 == 0:
            logger.debug(
                tok.decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0])
            )
            logger.info(f"loss: {loss}@step{_}")
