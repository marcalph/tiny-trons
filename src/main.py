# import logging
# from pathlib import Path

# import torch
# from model.architecturing import (
#     batch_sz,
#     context_sz,
#     d_embd,
#     d_head,
#     device,
#     eval_interval,
#     eval_steps,
#     max_steps,
#     n_layers,
#     smolTRF,
# )
# from model.tokenizing import CharTokenizer
# from model.training import Splitter, estimate_loss, make_batches

# from data.utils import read_corpus

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# if __name__ == "__main__":
#     corpus = read_corpus(Path("./data/tinyshakespeare.txt"))
#     tokenizer = CharTokenizer()
#     tokenizer.read(corpus)
#     tokenized_corpus = tokenizer.encode(corpus)

#     splitter = Splitter()
#     train_data, val_data = splitter.sequential_split(tokenized_corpus).values()

#     m = smolTRF(n_layers, tokenizer.vocab_sz, d_embd, d_head)
#     m = m.to(device)

#     optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)  # 3e-4
#     for step in range(max_steps):
#         xb, yb = next(make_batches(train_data, device, batch_sz, context_sz))
#         logits, loss = m(xb, yb)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#         if step % eval_interval == 0:
#             train_loss = estimate_loss(m, train_data, eval_steps, batch_sz, context_sz)
#             val_loss = estimate_loss(m, val_data, eval_steps, batch_sz, context_sz)
#             print(f"Step: {step}, Train loss: {train_loss}, Val loss: {val_loss}")
#             print(
#                 tokenizer.decode(
#                     m.generate(torch.zeros((1, 1), dtype=torch.long).to(device), 100)[0]
#                 )
#             )
#     print(xb.shape)
