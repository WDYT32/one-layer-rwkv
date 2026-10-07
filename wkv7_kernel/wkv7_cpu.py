"""CPU kernel for the RWKV-7 recurrence (C++ with a hand-written backward).

Compiled on first use with torch.utils.cpp_extension (needs a C++ compiler and
`pip install ninja`; the result is cached). If compilation fails, available()
returns False and model_rwkv7.py silently keeps its pure-PyTorch loop.

The kernel runs in float32 and parallelises over (batch x head) with PyTorch's
own thread pool, so torch.set_num_threads(n) controls it.

Recurrence (per head, state S[v][k]), identical to the loop in model_rwkv7.py:
    sa_t = S_{t-1} a_t
    S_t  = S_{t-1} diag(d_t) + sa_t b_t^T + v_t k_t^T
    y_t  = S_t r_t
"""
import os
import sys
import warnings

import torch

_ext = None
_failed = False


def _load():
    global _ext, _failed
    if _ext is not None or _failed:
        return _ext
    try:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        flags = ['/O2'] if sys.platform == 'win32' else ['-O3', '-march=native']
        print("[wkv7_cpu] compiling the CPU kernel (first use only, ~30-90 s)...", flush=True)
        _ext = load(name='wkv7_cpu_ext', sources=[os.path.join(here, 'wkv7_cpu.cpp')],
                    extra_include_paths=[here], extra_cflags=flags, verbose=False)
    except Exception as e:  # compiler / ninja missing, etc.
        _failed = True
        warnings.warn(f"wkv7_cpu: could not build the C++ kernel ({type(e).__name__}: {e}); "
                      f"falling back to the slow PyTorch loop. Try: pip install ninja")
    return _ext


def available():
    return _load() is not None


class _WKV7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, d, a, b):
        y, states = _ext.forward(r, k, v, d, a, b)
        ctx.save_for_backward(r, k, v, d, a, b)
        ctx.states = states  # intermediate (neither input nor output): kept as an attribute
        return y

    @staticmethod
    def backward(ctx, dy):
        r, k, v, d, a, b = ctx.saved_tensors
        grads = _ext.backward(r, k, v, d, a, b, ctx.states, dy.contiguous())
        ctx.states = None
        return tuple(grads)


def wkv7_cpu(r, k, v, d, a, b):
    """All inputs (B, T, H, S); d is the decay itself (not its log), a = -kk, b = kk * a_lr.
    Returns y of shape (B, T, H, S) in float32."""
    if _load() is None:
        raise RuntimeError("wkv7_cpu kernel is not available")
    args = [t.contiguous().float() for t in (r, k, v, d, a, b)]
    return _WKV7.apply(*args)
