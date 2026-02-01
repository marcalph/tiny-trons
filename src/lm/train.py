import torch
from loguru import logger
from torch.optim.adamw import AdamW

from src.data.utils import Corpus, create_dataloaders
from src.lm.attenbigram import AttenBigramLM
from src.lm.utils import HPARAMS, SETTINGS
from src.tokenizer import CharTokenizer


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters: int, device: str):
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        losses = torch.zeros(eval_iters)
        loader_iter = iter(loader)
        for k in range(eval_iters):
            try:
                xb, yb = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                xb, yb = next(loader_iter)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean()

    model.train()
    return out


if __name__ == "__main__":
    torch.manual_seed(1337)

    corpus = Corpus.from_file(SETTINGS.data_path)
    tok = CharTokenizer(corpus=corpus.text)

    use_cuda = HPARAMS.device == "cuda"
    train_loader, val_loader = create_dataloaders(
        corpus,
        tok,
        HPARAMS.block_sz,
        HPARAMS.batch_sz,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    model = AttenBigramLM(tok.vocab_sz, emb_d=HPARAMS.emb_d, block_sz=HPARAMS.block_sz, n_heads=4)
    model = model.to(HPARAMS.device)
    logger.info(f"Using device: {HPARAMS.device}")

    opt = AdamW(model.parameters(), lr=HPARAMS.lr)

    # Training loop
    train_iter = iter(train_loader)
    for step in range(HPARAMS.max_iters):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(HPARAMS.device), yb.to(HPARAMS.device)
        _, loss = model(xb, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % HPARAMS.eval_iters == 0:
            losses = estimate_loss(
                model, train_loader, val_loader, HPARAMS.eval_iters, HPARAMS.device
            )
            logger.info(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            # Generate sample
            prompt = torch.zeros((1, 1), dtype=torch.long, device=HPARAMS.device)
            generated = model.generate(prompt, max_new_tokens=200)[0]
            logger.debug(tok.decode(generated))
