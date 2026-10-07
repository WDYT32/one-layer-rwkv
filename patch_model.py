import re

with open('model_rwkv7.py', 'r') as f:
    content = f.read()

# 1. Remove the custom kernel imports
pattern_imports = r"(try:\s*# optional C\+\+ CPU kernel[\s\S]*?USE_CUDA_KERNEL = True\s*(?:#.*)?\n)"
replacement_imports = """try:
    from fla.ops.rwkv7 import chunk_rwkv7
    HAS_FLA = True
except ImportError:
    HAS_FLA = False

"""
content = re.sub(pattern_imports, replacement_imports, content)

# 2. Modify the forward pass logic
pattern_forward = r"(        decay = torch\.exp\(-DECAY_SCALE \* torch\.sigmoid\(self\.lora_w\(xw\)\)\)[\s\S]*?)y = torch\.stack\(outs, dim=1\)                      # \(B, T, H, S\)"
replacement_forward = """        log_decay = -DECAY_SCALE * torch.sigmoid(self.lora_w(xw))
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
            y = torch.stack(outs, dim=1)                      # (B, T, H, S)"""

content = re.sub(pattern_forward, replacement_forward, content)

with open('model_rwkv7.py', 'w') as f:
    f.write(content)
