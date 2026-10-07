// Torch glue for the RWKV-7 CPU kernel. Core math: wkv7_core.h (unit-tested separately).
// Tensors: float32, contiguous, shape (B, T, H, S).
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>
#include "wkv7_core.h"

static void check4d(const torch::Tensor& x, const char* name, const torch::Tensor& ref) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.dim() == 4, name, " must have shape (B,T,H,S)");
    TORCH_CHECK(x.sizes() == ref.sizes(), name, " must have the same shape as r");
}

std::vector<torch::Tensor> wkv7_forward(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                                        torch::Tensor d, torch::Tensor a, torch::Tensor b) {
    check4d(r, "r", r); check4d(k, "k", r); check4d(v, "v", r);
    check4d(d, "d", r); check4d(a, "a", r); check4d(b, "b", r);
    const int64_t B = r.size(0), T = r.size(1), H = r.size(2), S = r.size(3);
    auto y = torch::empty_like(r);
    auto states = torch::empty({B, H, T, S, S}, r.options());
    const float *rp = r.data_ptr<float>(), *kp = k.data_ptr<float>(), *vp = v.data_ptr<float>();
    const float *dp = d.data_ptr<float>(), *ap = a.data_ptr<float>(), *bp = b.data_ptr<float>();
    float* yp = y.data_ptr<float>();
    float* sp = states.data_ptr<float>();
    at::parallel_for(0, B * H, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t bi = i / H, hi = i % H;
            const int64_t off = (bi * T * H + hi) * S;
            wkv7::fwd_one<float>((int)T, (int)S, H * S, rp + off, kp + off, vp + off, dp + off,
                                 ap + off, bp + off, yp + off, sp + i * T * S * S);
        }
    });
    return {y, states};
}

std::vector<torch::Tensor> wkv7_backward(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                                         torch::Tensor d, torch::Tensor a, torch::Tensor b,
                                         torch::Tensor states, torch::Tensor dy) {
    check4d(r, "r", r); check4d(k, "k", r); check4d(v, "v", r);
    check4d(d, "d", r); check4d(a, "a", r); check4d(b, "b", r); check4d(dy, "dy", r);
    TORCH_CHECK(states.is_contiguous() && states.scalar_type() == torch::kFloat32, "bad states");
    const int64_t B = r.size(0), T = r.size(1), H = r.size(2), S = r.size(3);
    auto dr = torch::empty_like(r), dk = torch::empty_like(r), dv = torch::empty_like(r);
    auto dd = torch::empty_like(r), da = torch::empty_like(r), db = torch::empty_like(r);
    const float *rp = r.data_ptr<float>(), *kp = k.data_ptr<float>(), *vp = v.data_ptr<float>();
    const float *dp = d.data_ptr<float>(), *ap = a.data_ptr<float>(), *bp = b.data_ptr<float>();
    const float *sp = states.data_ptr<float>(), *dyp = dy.data_ptr<float>();
    float *drp = dr.data_ptr<float>(), *dkp = dk.data_ptr<float>(), *dvp = dv.data_ptr<float>();
    float *ddp = dd.data_ptr<float>(), *dap = da.data_ptr<float>(), *dbp = db.data_ptr<float>();
    at::parallel_for(0, B * H, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> G((size_t)S * S), dsa((size_t)S);
        for (int64_t i = begin; i < end; ++i) {
            const int64_t bi = i / H, hi = i % H;
            const int64_t off = (bi * T * H + hi) * S;
            wkv7::bwd_one<float>((int)T, (int)S, H * S, rp + off, kp + off, vp + off, dp + off,
                                 ap + off, bp + off, sp + i * T * S * S, dyp + off,
                                 drp + off, dkp + off, dvp + off, ddp + off, dap + off, dbp + off,
                                 G.data(), dsa.data());
        }
    });
    return {dr, dk, dv, dd, da, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &wkv7_forward, "RWKV-7 recurrence forward (CPU)");
    m.def("backward", &wkv7_backward, "RWKV-7 recurrence backward (CPU)");
}
