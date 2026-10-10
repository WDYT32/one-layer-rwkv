import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_transformer import SwiGLU

try:
    from fla.ops.rwkv7 import chunk_rwkv7
    HAS_FLA = True
except ImportError:
    HAS_FLA = False


DECAY_SCALE = 0.6065306597126334  # e^{-0.5}: decay w_t = exp(-DECAY_SCALE * sigmoid(.)) in (0.545, 1)


class LoRA(nn.Module):
    """x -> up(act(down(x))), as used for w, a, g in RWKV-7."""

    def __init__(self, d_model, rank, act='none', bias_init=None, up_init='zeros', out_dim=None):
        super().__init__()
        out_dim = d_model if out_dim is None else out_dim
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=bias_init is not None)
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

    def __init__(self, d_model, n_head, lora_dim=32, dim_att=None):
        super().__init__()
        self.dim_att = dim_att if dim_att is not None else d_model
        assert self.dim_att % n_head == 0
        self.C, self.H, self.S = d_model, n_head, self.dim_att // n_head

        # token-shift mixing coefficients (застосовуються до входу x, тому залишаються d_model)
        for name in ('r', 'w', 'k', 'v', 'a', 'g'):
            setattr(self, f'mu_{name}', nn.Parameter(torch.full((d_model,), 0.5)))

        self.receptance = nn.Linear(d_model, self.dim_att, bias=False)
        self.key = nn.Linear(d_model, self.dim_att, bias=False)
        self.value = nn.Linear(d_model, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, d_model, bias=False)

        # spread initial decays: some channels remember long, some short
        w_bias = torch.linspace(-6.0, 1.0, self.S).repeat(n_head)
        self.lora_w = LoRA(d_model, lora_dim, act='tanh', bias_init=w_bias, out_dim=self.dim_att)
        self.lora_a = LoRA(d_model, lora_dim, act='none', bias_init=torch.zeros(self.dim_att), out_dim=self.dim_att)
        self.lora_g = LoRA(d_model, lora_dim, act='sigmoid', up_init='ortho', out_dim=self.dim_att)

        self.k_k = nn.Parameter(torch.full((self.dim_att,), 0.7))
        self.k_a = nn.Parameter(torch.ones(self.dim_att))
        self.r_k = nn.Parameter(torch.zeros(n_head, self.S))
        self.ln_x = nn.GroupNorm(n_head, self.dim_att, eps=64e-5)

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
        log_decay = -DECAY_SCALE * torch.sigmoid(self.lora_w(xw))
        decay = torch.exp(log_decay)
        a = torch.sigmoid(self.lora_a(xa))
        g = self.lora_g(xg)

        kk = F.normalize((k * self.k_k).view(B, T, H, S), dim=-1)
        k = k * (1 + (a - 1) * self.k_a)

        r = r.view(B, T, H, S).float()
        k = k.view(B, T, H, S).float()
        v = v.view(B, T, H, S).float()
        log_decay = log_decay.view(B, T, H, S).float()
        decay = decay.view(B, T, H, S).float()
        a_vec = -kk.float()                                  # "a" in the (r,w,k,v,a,b) notation
        b_vec = kk.float() * a.view(B, T, H, S).float()      # "b"

        if HAS_FLA and x.device.type == 'cuda':
            # fla's chunk_rwkv7 expects bfloat16 or float16 for Triton kernels
            y, _ = chunk_rwkv7(
                r.bfloat16(),
                k.bfloat16(),
                v.bfloat16(),
                a_vec.bfloat16(),
                b_vec.bfloat16(),
                log_w=log_decay.bfloat16()
            )
            y = y.float()
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
        y = self.ln_x(y.reshape(B * T, self.dim_att)).view(B, T, self.dim_att) + bonus.reshape(B, T, self.dim_att)
        return self.output(y.to(x.dtype) * g)


class RWKV7ChannelMix(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        # У RWKV-7 залишився лише один параметр зсуву (x_k)
        self.x_k = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        
        self.key = nn.Linear(d_model, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # Зміщення послідовності на 1 крок у минуле мінус поточний стан
        xx = self.time_shift(x) - x
        
        # Token Shift: x + (x_shifted - x) * x_k
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        
        return self.value(k)


class RWKV7ChannelMixMoE(nn.Module):
    """Top-1 MoE ChannelMix with fatigue (refractory) hard masking.

    Routing: deterministic argmax over fatigue-masked router logits (no Gumbel noise);
    exploration comes only from fatigue. The fatigue recurrence is sequential in t
    (choice at t depends on choices before it), so it is one cheap loop over T on
    (B, E) tensors; everything heavy is loop-free.

    Compute: only the selected expert runs for each token. Tokens are sorted by expert,
    packed into an (E, cap, D) buffer and processed with two bmm calls
    (cap = largest group, so padding waste is small when routing is balanced).

    Gradient (train): out *= 1 + p_sel - p_sel.detach()  (forward factor is exactly 1,
    so train == eval). The router learns through the selected expert's probability;
    unselected experts get no gradient at that token and are reached via fatigue.
    dense_ste=True restores the dense variant where every expert gets a gradient.
    Fatigue is causal and restarts at every forward, so right-padding cannot affect real tokens.
    """

    def __init__(self, d_model, hidden_dim, num_experts=4, gamma=0.9, kappa=0.2, theta=0.5, dense_ste=False):
        super().__init__()
        self.num_experts = num_experts
        self.gamma, self.kappa, self.theta = gamma, kappa, theta
        self.dense_ste = dense_ste
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.router = nn.Linear(d_model, num_experts, bias=False)
        # per-expert token-shift vector and stacked FFN weights (nn.Linear-style init)
        self.x_k = nn.Parameter(torch.full((num_experts, d_model), 0.5))
        self.w_key = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        self.w_value = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model))
        nn.init.uniform_(self.w_key, -1 / math.sqrt(d_model), 1 / math.sqrt(d_model))
        nn.init.uniform_(self.w_value, -1 / math.sqrt(hidden_dim), 1 / math.sqrt(hidden_dim))

        self.last_indices = None  # (B, T) chosen expert per token, for logging

    @torch.no_grad()
    def _route(self, logits_all):
        B, T, E = logits_all.shape
        lg = logits_all.detach().float()
        F_e = lg.new_zeros(B, E)  # fp32 even under autocast
        inc = lg.new_full((B, 1), self.kappa)
        masks = torch.empty(T, B, E, dtype=torch.bool, device=lg.device)
        idxs = torch.empty(T, B, dtype=torch.long, device=lg.device)
        for t in range(T):
            mask = F_e > self.theta
            mask = mask & ~mask.all(dim=-1, keepdim=True)  # all fatigued -> unmask all (no NaN)
            idx = lg[:, t].masked_fill(mask, float('-inf')).argmax(dim=-1)
            F_e = (F_e * self.gamma).scatter_add_(1, idx.unsqueeze(1), inc)
            masks[t] = mask
            idxs[t] = idx
        return masks.transpose(0, 1), idxs.transpose(0, 1)

    def _grouped_ffn(self, xin, idx):
        """xin (N, D), idx (N,) -> (N, D); each row goes through its own expert only."""
        E, N = self.num_experts, idx.numel()
        order = idx.argsort(stable=True)
        sidx = idx[order]
        counts = torch.bincount(idx, minlength=E)
        pos = torch.arange(N, device=idx.device) - (counts.cumsum(0) - counts)[sidx]
        cap = int(counts.max())
        buf = xin.new_zeros(E, cap, xin.size(-1))
        buf[sidx, pos] = xin[order]
        h = torch.relu(torch.bmm(buf, self.w_key)) ** 2
        o = torch.bmm(h, self.w_value)
        return o[sidx, pos][order.argsort()]

    def forward(self, x):
        B, T, C = x.shape
        logits_all = self.router(x)
        mask_all, indices = self._route(logits_all)
        self.last_indices = indices
        xx = self.time_shift(x) - x

        if self.training and self.dense_ste:
            probs = F.softmax(logits_all.float().masked_fill(mask_all, float('-inf')), dim=-1).to(x.dtype)
            hard = F.one_hot(indices, self.num_experts).to(x.dtype)
            w = hard + probs - probs.detach()
            xin = x.unsqueeze(0) + xx.unsqueeze(0) * self.x_k[:, None, None, :]
            h = torch.relu(torch.einsum('ebtd,edh->ebth', xin, self.w_key)) ** 2
            v = torch.einsum('ebth,ehd->ebtd', h, self.w_value)
            return torch.einsum('ebtd,bte->btd', v, w)

        idxf = indices.reshape(-1)
        xin = x.reshape(-1, C) + xx.reshape(-1, C) * self.x_k[idxf]
        out = self._grouped_ffn(xin, idxf).view(B, T, C)
        if self.training:
            probs = F.softmax(logits_all.float().masked_fill(mask_all, float('-inf')), dim=-1)
            p_sel = probs.gather(-1, indices.unsqueeze(-1))
            out = out * (1 + p_sel - p_sel.detach()).to(out.dtype)
        return out


class RWKV7Block(nn.Module):
    def __init__(self, d_model, n_head, lora_dim, dim_att=None, ffn_expand=4, moe_experts=0, moe_kwargs=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.tmix = RWKV7TimeMix(d_model, n_head, lora_dim, dim_att=dim_att)
        self.ln_2 = nn.LayerNorm(d_model)
        if moe_experts > 0:
            self.ffn = RWKV7ChannelMixMoE(d_model, d_model * ffn_expand, num_experts=moe_experts, **(moe_kwargs or {}))
        else:
            self.ffn = RWKV7ChannelMix(d_model, d_model * ffn_expand)

    def forward(self, x):
        x = x + self.tmix(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class RWKV7Model(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, n_layer=1, lora_dim=32, dim_att=None, ffn_expand=4, moe_experts=0, moe_kwargs=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.ln_0 = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [RWKV7Block(d_model, n_head, lora_dim, dim_att=dim_att, ffn_expand=ffn_expand, moe_experts=moe_experts, moe_kwargs=moe_kwargs) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.ln_0(self.token_emb(idx))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
