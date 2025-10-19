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
    for name, split in {"train": train, "val": val}.items():
        losses = torch.zeros(HPARAMS.eval_iters)
        for k in range(HPARAMS.eval_iters):
            X, Y = dl.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
            out[name] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    batch_sz = 32
    ds = Dataset.from_file(SETTINGS.data_path)
    tok = CharTokenizer(corpus=ds.corpus)
    dl = Dataloader(ds, tok, HPARAMS.block_sz, HPARAMS.batch_sz)
    train, val = ds.split()
    xb, yb = dl.get_batch(train)
    m = BigramLM(tok.vocab_sz)
    logits, loss = m(xb, yb)
    opt = AdamW(m.parameters(), lr=1e-3)

    # train loop
    for _ in range(HPARAMS.max_iters):
        xb, yb = dl.get_batch(ds.corpus)
        logits, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)  # set.grad to None to save mem
        loss.backward()
        opt.step()
        if _ % HPARAMS.eval_iters == 0:
            losses = estimate_loss(dl, m)
            logger.debug(
                tok.decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0])
            )
            logger.info(f"@step{_}: train loss {losses['train']}, val loss {losses['val']}")
