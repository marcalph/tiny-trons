import torch
from loguru import logger
from torch.optim.adamw import AdamW

from src.data.utils import Dataloader, Dataset
from src.lm.bigram import BigramLM
from src.lm.utils import HPARAMS, SETTINGS
from src.tokenizer import CharTokenizer


@torch.no_grad
def estimate_loss(dl, model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(HPARAMS.eval_iters)
        for k in range(HPARAMS.max_iters):
            X, Y = dl.get_batch(ds.corpus)
            _, loss = model(X, Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


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
