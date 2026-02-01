import torch
torch.manual_seed(1337)
import torch.nn as nn
import torch.nn.functional as F

B, T, C = 4, 8 , 32
x = torch.randn(B, T, C)


head_sz = 16


class AttenBlock(nn.Module):
    def __init__(self, emb_d, n_heads, block_sz, dropout=0.2):
        super().__init__()
        head_sz = emb_d // n_heads
        self.sa = MultiHeadAttention(n_heads, head_sz, emb_d, block_sz, dropout=dropout)
        self.ffwd = FFN(emb_d, dropout=dropout)
        self.ln1 = nn.LayerNorm(emb_d)
        self.ln2 = nn.LayerNorm(emb_d)
    
    def forward(self, x):
        x = self.sa(self.ln1(x)) + x 
        x = self.ffwd(self.ln2(x)) + x
        return x




class FFN(nn.Module):
    def __init__(self, emb_d, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_d,  4 * emb_d),
            nn.ReLU(),
            nn.Linear(4* emb_d,   emb_d),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AttentionHead(nn.Module):
    tril : torch.Tensor

    def __init__(self, head_sz, emb_d, block_sz, dropout=0.2):
        super().__init__()
        self.head_sz = head_sz
        self.emb_d = emb_d
        self.block_sz = block_sz

        self.key = nn.Linear(self.emb_d, self.head_sz, bias=False)
        self.query = nn.Linear(self.emb_d, head_sz, bias=False)
        self.value = nn.Linear(self.emb_d, head_sz, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.block_sz, self.block_sz)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape       # C is emb_d

        k = self.key(x)         # (B,  T, C=head_sz)
        q = self.query(x)       # (B,  T, head_sz)
        v = self.value(x)       # (B,  T, head_sz)


        # attention score
        wei = q @ k.transpose(-2, -1) * self.head_sz ** -0.5  # (B, T, head_sz) @ (B, head_sz, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei@v

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_sz, emb_d, block_sz, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_sz, emb_d, block_sz, dropout=dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(emb_d, emb_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out