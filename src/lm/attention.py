import torch
torch.manual_seed(1337)
import torch.nn as nn
import torch.nn.functional as F

B, T, C = 4, 8 , 32
x = torch.randn(B, T, C)


head_sz = 16


class AttentionHead(nn.Module):
    tril : torch.Tensor

    def __init__(self, head_sz, emb_d, block_sz):
        super().__init__()
        self.head_sz = head_sz
        self.emb_d = emb_d
        self.block_sz = block_sz

        self.key = nn.Linear(self.emb_d, self.head_sz, bias=False)
        self.query = nn.Linear(self.emb_d, head_sz, bias=False)
        self.value = nn.Linear(self.emb_d, head_sz, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(self.block_sz, self.block_sz)))
    
    def forward(self, x):
        B, T, C = x.shape       # C is emb_d

        k = self.key(x)         # (B,  T, C=head_sz)
        q = self.query(x)       # (B,  T, head_sz)
        v = self.value(x)       # (B,  T, head_sz)


        # attention score
        wei = q @ k.transpose(-2, -1) * self.head_sz ** -0.5  # (B, T, head_sz) @ (B, head_sz, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei@v

        return out