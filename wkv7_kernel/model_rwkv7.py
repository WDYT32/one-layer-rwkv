import torch
import torch.nn as nn
import torch.nn.functional as F

from model_transformer import SwiGLU

try:  # optional C++ CPU kernel (see wkv7_cpu.py); the PyTorch loop below is the fallback
    from wkv7_cpu import wkv7_cpu, available as cpu_kernel_available
except Exception:  # pragma: no cover
    wkv7_cpu = None

    def cpu_kernel_available():
        return False

USE_CPU_KERNEL = True  # set to False to force the reference PyTorch loop (used by test_wkv7_cpu.py)

DECAY_SCALE = 0.6065306597126334  # e^{-0.5}: decay w_t = exp(-DECAY_SCALE * sigmoid(.)) in (0.545, 1)


class LoRA(nn.Module):
    """x -> up(act(down(x))), as used for w, a, g in RWKV-7."""

    def __init__(self, d_model, rank, act='none', bias_init=None, up_init='zeros'):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=bias_init is not None)
        if up_init == 'zeros':
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.orthogonal_(self.up.weight, gain=0.1)
        if bias_init is not None:
            with torch.no_grad():
                self.up.bias.copy_(bias_init)
        self.act = act

    def forward(self, x):
        h = self.down(x)
        if self.act == 'tanh':
            h = torch.tanh(h)
        elif self.act == 'sigmoid':
            h = torch.sigmoid(h)
        return self.up(h)


class RWKV7TimeMix(nn.Module):
    """RWKV-7 time mixing with the generalized delta rule.

    Per head, with state S in R^{V x K}:
        S_t = S_{t-1} diag(w_t) + (S_{t-1} (-kk_t)) (kk_t * a_t)^T + v_t k~_t^T
        y_t = S_t r_t
    kk_t is the L2-normalised removal key, a_t the in-context learning rate,
    w_t a data-dependent per-channel decay.

    Simplifications vs. the official implementation: no value-residual
    (v_first) across layers, simple init, and a naive Python loop over time
    (fine for T <= ~64) unless the optional C++ CPU kernel in wkv7_cpu.py is available.
    """

    def __init__(self, d_model, n_head, lora_dim=32):
        super().__init__()
        assert d_model % n_head == 0
        self.C, self.H, self.S = d_model, n_head, d_model // n_head

        # token-shift mixing coefficients
        for name in ('r', 'w', 'k', 'v', 'a', 'g'):
            setattr(self, f'mu_{name}', nn.Parameter(torch.full((d_model,), 0.5)))

        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # spread initial decays: some channels remember long, some short
        w_bias = torch.linspace(-6.0, 1.0, self.S).repeat(n_head)
        self.lora_w = LoRA(d_model, lora_dim, act='tanh', bias_init=w_bias)
        self.lora_a = LoRA(d_model, lora_dim, act='none', bias_init=torch.zeros(d_model))
        self.lora_g = LoRA(d_model, lora_dim, act='sigmoid', up_init='ortho')

        self.k_k = nn.Parameter(torch.full((d_model,), 0.7))
        self.k_a = nn.Parameter(torch.ones(d_model))
        self.r_k = nn.Parameter(torch.zeros(n_head, self.S))
        self.ln_x = nn.GroupNorm(n_head, d_model, eps=64e-5)

    def forward(self, x):
        B, T, C = x.shape
        H, S = self.H, self.S

        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        xx = prev - x
        xr = x + xx * self.mu_r
        xw = x + xx * self.mu_w
        xk = x + xx * self.mu_k
        xv = x + xx * self.mu_v
        xa = x + xx * self.mu_a
        xg = x + xx * self.mu_g

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        decay = torch.exp(-DECAY_SCALE * torch.sigmoid(self.lora_w(xw)))
        a = torch.sigmoid(self.lora_a(xa))
        g = self.lora_g(xg)

        kk = F.normalize((k * self.k_k).view(B, T, H, S), dim=-1)
        k = k * (1 + (a - 1) * self.k_a)

        r = r.view(B, T, H, S).float()
        k = k.view(B, T, H, S).float()
        v = v.view(B, T, H, S).float()
        decay = decay.view(B, T, H, S).float()
        a_vec = -kk.float()                                  # "a" in the (r,w,k,v,a,b) notation
        b_vec = kk.float() * a.view(B, T, H, S).float()      # "b"

        if USE_CPU_KERNEL and x.device.type == 'cpu' and cpu_kernel_available():
            y = wkv7_cpu(r, k, v, decay, a_vec, b_vec)        # (B, T, H, S); C++ kernel, own backward
        else:
            state = x.new_zeros(B, H, S, S, dtype=torch.float32)  # (V, K)
            outs = []
            for t in range(T):
                sa = torch.einsum('bhvk,bhk->bhv', state, a_vec[:, t])
                state = (state * decay[:, t].unsqueeze(-2)
                         + sa.unsqueeze(-1) * b_vec[:, t].unsqueeze(-2)
                         + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2))
                outs.append(torch.einsum('bhvk,bhk->bhv', state, r[:, t]))
            y = torch.stack(outs, dim=1)                      # (B, T, H, S)

        bonus = (r * k * self.r_k).sum(-1, keepdim=True) * v  # (B, T, H, S)
        y = self.ln_x(y.reshape(B * T, C)).view(B, T, C) + bonus.reshape(B, T, C)
        return self.output(y.to(x.dtype) * g)


class RWKV7Block(nn.Module):
    def __init__(self, d_model, n_head, lora_dim):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.tmix = RWKV7TimeMix(d_model, n_head, lora_dim)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * 4)  # same FFN as the transformer

    def forward(self, x):
        x = x + self.tmix(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class RWKV7Model(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, n_layer=1, lora_dim=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.ln_0 = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [RWKV7Block(d_model, n_head, lora_dim) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.ln_0(self.token_emb(idx))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
