// Pure C++ core of the RWKV-7 recurrence for ONE (batch, head) pair.
// No torch dependency, so it can be unit-tested on its own.
//
//   sa_t[v]   = sum_k S_{t-1}[v][k] * a_t[k]
//   S_t[v][k] = S_{t-1}[v][k] * d_t[k] + sa_t[v] * b_t[k] + v_t[v] * k_t[k]
//   y_t[v]    = sum_k S_t[v][k] * r_t[k]          (S_{-1} = 0)
//
// Per-step vectors r,k,v,d,a,b,y are laid out with a time stride `st`
// (st = H*S for a (B,T,H,S) tensor). `states` holds every S_t, contiguous [T,S,S].
#pragma once
#include <cstdint>
#include <cstring>

namespace wkv7 {

template <typename T>
void fwd_one(int Tn, int S, int64_t st,
             const T* r, const T* k, const T* v, const T* d, const T* a, const T* b,
             T* y, T* states) {
    for (int t = 0; t < Tn; ++t) {
        const T* Sp = t > 0 ? states + (int64_t)(t - 1) * S * S : nullptr;
        T* Sc = states + (int64_t)t * S * S;
        const T *rt = r + t * st, *kt = k + t * st, *vt = v + t * st;
        const T *dt = d + t * st, *at = a + t * st, *bt = b + t * st;
        T* yt = y + t * st;
        for (int i = 0; i < S; ++i) {
            T sa = 0;
            if (Sp) {
                const T* row = Sp + (int64_t)i * S;
                for (int j = 0; j < S; ++j) sa += row[j] * at[j];
            }
            const T vi = vt[i];
            T acc = 0;
            T* out = Sc + (int64_t)i * S;
            if (Sp) {
                const T* row = Sp + (int64_t)i * S;
                for (int j = 0; j < S; ++j) {
                    T s = row[j] * dt[j] + sa * bt[j] + vi * kt[j];
                    out[j] = s;
                    acc += s * rt[j];
                }
            } else {
                for (int j = 0; j < S; ++j) {
                    T s = vi * kt[j];
                    out[j] = s;
                    acc += s * rt[j];
                }
            }
            yt[i] = acc;
        }
    }
}

// G: scratch [S*S], dsa: scratch [S]. Every element of dr..db is written.
template <typename T>
void bwd_one(int Tn, int S, int64_t st,
             const T* r, const T* k, const T* v, const T* d, const T* a, const T* b,
             const T* states, const T* dy,
             T* dr, T* dk, T* dv, T* dd, T* da, T* db,
             T* G, T* dsa) {
    std::memset(G, 0, sizeof(T) * (size_t)S * S);
    for (int t = Tn - 1; t >= 0; --t) {
        const T* Sc = states + (int64_t)t * S * S;
        const T* Sp = t > 0 ? states + (int64_t)(t - 1) * S * S : nullptr;
        const T *rt = r + t * st, *kt = k + t * st, *vt = v + t * st;
        const T *dt = d + t * st, *at = a + t * st, *bt = b + t * st;
        const T* dyt = dy + t * st;
        T *drt = dr + t * st, *dkt = dk + t * st, *dvt = dv + t * st;
        T *ddt = dd + t * st, *dat = da + t * st, *dbt = db + t * st;
        for (int j = 0; j < S; ++j) { drt[j] = 0; dkt[j] = 0; ddt[j] = 0; dat[j] = 0; dbt[j] = 0; }

        for (int i = 0; i < S; ++i) {
            T* g = G + (int64_t)i * S;
            const T dyi = dyt[i];
            const T* sc = Sc + (int64_t)i * S;
            const T* sp = Sp ? Sp + (int64_t)i * S : nullptr;
            T sa = 0;
            if (sp) for (int j = 0; j < S; ++j) sa += sp[j] * at[j];
            T dvv = 0, dsav = 0;
            const T vi = vt[i];
            for (int j = 0; j < S; ++j) {
                g[j] += dyi * rt[j];          // y_t = S_t r_t
                drt[j] += sc[j] * dyi;
                dvv += g[j] * kt[j];
                dsav += g[j] * bt[j];
                dkt[j] += g[j] * vi;
                dbt[j] += g[j] * sa;
                if (sp) ddt[j] += g[j] * sp[j];
            }
            dvt[i] = dvv;
            dsa[i] = dsav;
        }
        // propagate to S_{t-1}: G <- G * d + dsa (x) a ; and da
        for (int i = 0; i < S; ++i) {
            T* g = G + (int64_t)i * S;
            const T ds = dsa[i];
            const T* sp = Sp ? Sp + (int64_t)i * S : nullptr;
            if (sp) for (int j = 0; j < S; ++j) dat[j] += ds * sp[j];
            for (int j = 0; j < S; ++j) g[j] = g[j] * dt[j] + ds * at[j];
        }
    }
}

}  // namespace wkv7
