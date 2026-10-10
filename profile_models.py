"""Where does the time go? Usage: python profile_models.py [--bs 64] [--T 250]
Times fwd+bwd of rwkv / rwkv-moe (and the transformer if importable) and prints the
top ops of the RWKV-MoE step from torch.profiler."""
import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

import model_rwkv7
from model_rwkv7 import RWKV7Model

p = argparse.ArgumentParser()
p.add_argument('--bs', type=int, default=64)
p.add_argument('--T', type=int, default=250)
p.add_argument('--iters', type=int, default=3)
a = p.parse_args()

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V, D, H = 21, 256, 4
print(f'device={dev} fla={model_rwkv7.HAS_FLA} threads={torch.get_num_threads()} B={a.bs} T={a.T}')
print('WKV path:', 'fla chunk kernel' if (model_rwkv7.HAS_FLA and dev.type == 'cuda') else 'NAIVE python loop over T')

x = torch.randint(0, V, (a.bs, a.T), device=dev)


def sync():
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def step(m):
    m(x).float().sum().backward()
    m.zero_grad(set_to_none=True)


def bench(name, m):
    m.to(dev).train()
    step(m); sync()
    t0 = time.time()
    for _ in range(a.iters):
        step(m)
    sync()
    print(f'{name:14s} {(time.time() - t0) / a.iters:8.3f} s / step')


models = {
    'rwkv': RWKV7Model(V, D, H, 1),
    'rwkv-moe': RWKV7Model(V, D, H * 4, 1, dim_att=D * 4, ffn_expand=8, moe_experts=4),
}
try:
    from model_transformer import BaselineTransformer
    models['transformer'] = BaselineTransformer(V, D, H, 1, pos='learned', max_len=max(320, a.T))
except Exception as e:  # noqa: BLE001
    print('transformer baseline not importable:', e)

for n, m in models.items():
    bench(n, m)

m = models['rwkv-moe']
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == 'cuda' else [])
with profile(activities=acts) as prof:
    step(m); sync()
key = 'self_cuda_time_total' if dev.type == 'cuda' else 'self_cpu_time_total'
print(prof.key_averages().table(sort_by=key, row_limit=12, max_name_column_width=48))
