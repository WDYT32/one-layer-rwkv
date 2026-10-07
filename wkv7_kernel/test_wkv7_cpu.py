"""Checks the C++ CPU kernel against the reference PyTorch loop inside RWKV7TimeMix
(forward output and ALL gradients), then times both. Run:  python test_wkv7_cpu.py"""
import time
import torch
import model_rwkv7 as M

torch.manual_seed(0)
B, T, C, H = 4, 40, 128, 2
layer = M.RWKV7TimeMix(C, H, 16)
with torch.no_grad():  # LoRA 'up' layers start at zero -> perturb so the test is non-trivial
    for p in layer.parameters():
        p.add_(0.05 * torch.randn_like(p))
x = torch.randn(B, T, C, requires_grad=True)
wgt = torch.linspace(-1, 1, B * T * C).view(B, T, C)


def run(flag):
    M.USE_CPU_KERNEL = flag
    layer.zero_grad(); x.grad = None
    y = layer(x)
    (y * wgt).sum().backward()
    return y.detach(), x.grad.clone(), {n: p.grad.clone() for n, p in layer.named_parameters()}


assert M.cpu_kernel_available(), "kernel did not build (need a C++ compiler and `pip install ninja`)"
y0, gx0, gp0 = run(False)
y1, gx1, gp1 = run(True)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()


print(f"output          rel err {rel(y1, y0):.2e}")
print(f"input grad      rel err {rel(gx1, gx0):.2e}")
worst = max(rel(gp1[n], gp0[n]) for n in gp0)
print(f"param grads     worst rel err {worst:.2e}  ({len(gp0)} tensors)")
assert rel(y1, y0) < 1e-3 and rel(gx1, gx0) < 1e-3 and worst < 1e-3, "MISMATCH"
print("OK: kernel matches the reference loop")

# timing at a size close to the real run
B, T, C, H = 16, 250, 256, 4
layer = M.RWKV7TimeMix(C, H, 32)
x = torch.randn(B, T, C, requires_grad=True)
for flag in (False, True):
    M.USE_CPU_KERNEL = flag
    layer(x).sum().backward()  # warm-up
    t0 = time.time()
    for _ in range(3):
        layer.zero_grad(); x.grad = None
        layer(x).sum().backward()
    print(f"{'C++ kernel' if flag else 'PyTorch loop'}: {(time.time() - t0) / 3:.2f} s per fwd+bwd "
          f"(B={B}, T={T}, threads={torch.get_num_threads()})")
