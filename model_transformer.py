import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        # SwiGLU uses 3 linear layers
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def rope_tables(seq_len, dim, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, dh); rotate interleaved pairs (0,1), (2,3), ...
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, rope=False):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.rope = rope
        if rope:
            assert self.d_head % 2 == 0
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if self.rope:
            cos, sin = rope_tables(T, self.d_head, x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, rope):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, rope=rope)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, d_model * 4)  # 4x expansion

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BaselineTransformer(nn.Module):
    """pos='learned': absolute learned embeddings (cannot extrapolate past the
    lengths seen with real tokens). pos='rope': rotary, no position parameters."""

    def __init__(self, vocab_size, d_model=256, n_head=4, n_layer=1,
                 pos='learned', max_len=256):
        super().__init__()
        assert pos in ('learned', 'rope')
        self.pos = pos
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model) if pos == 'learned' else None
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, rope=(pos == 'rope')) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
