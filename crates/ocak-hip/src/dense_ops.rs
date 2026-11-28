//! Dense transformer HIP compute kernels for `HipExec`:
//! `embed`, `add`, `add_bias`, `add_rms_norm`, `rms_norm_heads`, `rope`,
//! paged FP16 `kv_write`, `attention_decode`, `attention_prefill`, `swiglu`, and 2-pass `argmax`.
use crate::ffi::HipError;
use crate::kernels::HipWType;
use crate::runtime::{HipBuffer, HipContext, HipFunction, HipModule, HipStream};
use std::ffi::c_void;
use std::sync::Arc;

pub const DENSE_OPS_SRC: &str = r#"
typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;
typedef int                i32;
typedef signed char        i8;

struct __align__(16) f32x4 { float x, y, z, w; };

__device__ __forceinline__ float f16_to_f32(u16 h) {
    union { u16 u; _Float16 f; } v;
    v.u = h;
    return (float)v.f;
}

__device__ __forceinline__ u16 f32_to_f16(float x) {
    union { u16 u; _Float16 f; } v;
    v.f = (_Float16)x;
    return v.u;
}

__device__ __forceinline__ float f16at(const u8* __restrict__ w, u64 off) {
    u16 h = (u16)w[off] | ((u16)w[off + 1] << 8);
    return f16_to_f32(h);
}

__device__ __forceinline__ void lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// Helper: quantize one 32-element block held across 32 lanes (1 float per lane) into xq/xsc.
__device__ __forceinline__ void warp_quant_q8_block(
    float v, u32 lane, u32 blk, u32* __restrict__ xq, f32x4* __restrict__ xsc
) {
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    float dx = amax * (1.0f / 127.0f);
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    i32 q = min(127, max(-127, __float2int_rn(v * inv)));

    // Pack 4 consecutive lanes into 1 u32 via shuffle-down; lane % 4 == 0 writes.
    i32 q1 = __shfl_down(q, 1, 32);
    i32 q2 = __shfl_down(q, 2, 32);
    i32 q3 = __shfl_down(q, 3, 32);
    if ((lane & 3u) == 0u) {
        xq[blk * 8u + (lane >> 2)] = (u32)(q & 0xFF)
            | ((u32)(q1 & 0xFF) << 8)
            | ((u32)(q2 & 0xFF) << 16)
            | ((u32)(q3 & 0xFF) << 24);
    }

    i32 s0 = (lane < 16u) ? q : 0;
    i32 s1 = (lane < 16u) ? 0 : q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s0 += __shfl_xor(s0, off, 32);
        s1 += __shfl_xor(s1, off, 32);
    }
    if (lane == 0u) {
        xsc[blk] = { dx, dx * (float)s0, dx * (float)s1, 0.0f };
    }
}

// ---------------------------------------------------------------------------
// 1. Embedding lookup & dequantization (single token or batched token array)
//    wtype: 0=F32, 1=F16, 2=Q4_0, 3=Q8_0, 4=Q4_K, 5=Q5_K, 6=Q6_K
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_elem(
    const u8* __restrict__ row, u32 col, u32 wtype
) {
    if (wtype == 0u) {
        return ((const float*)row)[col];
    } else if (wtype == 1u) {
        return f16at(row, (u64)col * 2u);
    } else if (wtype == 2u) {
        // Q4_0: 18 bytes / 32 elems
        u32 b = (col >> 5) * 18u;
        u32 l = col & 31u;
        float d = f16at(row, b);
        u8 q = row[b + 2u + (l & 15u)];
        i32 v = (l < 16u) ? (i32)(q & 0xFu) : (i32)(q >> 4);
        return d * (float)(v - 8);
    } else if (wtype == 3u) {
        // Q8_0: 34 bytes / 32 elems
        u32 b = (col >> 5) * 34u;
        float d = f16at(row, b);
        return d * (float)((i8)row[b + 2u + (col & 31u)]);
    } else if (wtype == 4u) {
        // Q4_K: 144 bytes / 256 elems
        u32 b = (col >> 8) * 144u;
        u32 j = (col >> 5) & 7u;
        u32 l = col & 31u;
        u32 sb = b + 4u;
        u32 sc, mn;
        if (j < 4u) {
            sc = row[sb + j] & 63u;
            mn = row[sb + j + 4u] & 63u;
        } else {
            sc = (row[sb + j + 4u] & 0xFu) | ((row[sb + j - 4u] >> 6) << 4);
            mn = (row[sb + j + 4u] >> 4)   | ((row[sb + j] >> 6) << 4);
        }
        float d = f16at(row, b) * (float)sc;
        float m = f16at(row, b + 2u) * (float)mn;
        u8 qb = row[b + 16u + (j >> 1) * 32u + l];
        u32 q = (j & 1u) ? (qb >> 4) : (qb & 0xFu);
        return d * (float)q - m;
    } else if (wtype == 5u) {
        // Q5_K: 176 bytes / 256 elems
        u32 b = (col >> 8) * 176u;
        u32 j = (col >> 5) & 7u;
        u32 l = col & 31u;
        u32 sb = b + 4u;
        u32 sc, mn;
        if (j < 4u) {
            sc = row[sb + j] & 63u;
            mn = row[sb + j + 4u] & 63u;
        } else {
            sc = (row[sb + j + 4u] & 0xFu) | ((row[sb + j - 4u] >> 6) << 4);
            mn = (row[sb + j + 4u] >> 4)   | ((row[sb + j] >> 6) << 4);
        }
        float d = f16at(row, b) * (float)sc;
        float m = f16at(row, b + 2u) * (float)mn;
        u8 qb = row[b + 48u + (j >> 1) * 32u + l];
        u32 qlo = (j & 1u) ? (qb >> 4) : (qb & 0xFu);
        u32 qhi = (row[b + 16u + l] >> j) & 1u;
        return d * (float)(qlo | (qhi << 4)) - m;
    } else {
        // Q6_K: 210 bytes / 256 elems
        u32 b = (col >> 8) * 210u;
        u32 n = col & 255u;
        u32 half_ = n >> 7;
        u32 g = (n >> 5) & 3u;
        u32 l = n & 31u;
        u8 lo = row[b + half_ * 64u + (g & 1u) * 32u + l];
        u8 hi = row[b + 128u + half_ * 32u + l];
        u32 qlo = (lo >> ((g >> 1) * 4u)) & 0xFu;
        u32 qhi = (hi >> (g * 2u)) & 3u;
        i32 q = (i32)(qlo | (qhi << 4)) - 32;
        float d = f16at(row, b + 208u) * (float)((i8)row[b + 192u + half_ * 8u + (l >> 4) + 2u * g]);
        return d * (float)q;
    }
}

extern "C" __global__ __launch_bounds__(256)
void embed_token(
    const u8* __restrict__ w,
    float* __restrict__ out,
    u32 token,
    u32 n_embd,
    u32 row_bytes,
    u32 wtype
) {
    u32 i = blockIdx.x * 256u + threadIdx.x;
    if (i >= n_embd) return;
    out[i] = dequant_elem(w + (u64)token * row_bytes, i, wtype);
}

// Device-meta variant for HIP Graph decode: reads token from step_meta[0].
extern "C" __global__ __launch_bounds__(256)
void embed_token_meta(
    const u8* __restrict__ w,
    const u32* __restrict__ step_meta,
    float* __restrict__ out,
    u32 n_embd,
    u32 row_bytes,
    u32 wtype
) {
    u32 i = blockIdx.x * 256u + threadIdx.x;
    if (i >= n_embd) return;
    u32 token = step_meta[0];
    out[i] = dequant_elem(w + (u64)token * row_bytes, i, wtype);
}

extern "C" __global__ __launch_bounds__(256)
void embed_tokens_batch(
    const u8* __restrict__ w,
    const u32* __restrict__ tokens,
    float* __restrict__ out,
    u32 n_embd,
    u32 row_bytes,
    u32 wtype,
    u32 batch
) {
    u32 i = blockIdx.x * 256u + threadIdx.x;
    u32 t = blockIdx.y;
    if (i >= n_embd || t >= batch) return;
    u32 tok = tokens[t];
    out[(u64)t * n_embd + i] = dequant_elem(w + (u64)tok * row_bytes, i, wtype);
}

// ---------------------------------------------------------------------------
// 2. Elementwise add, bias add, fused add+rms_norm, per-head rms_norm, swiglu
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    u32 n
) {
    u32 i = blockIdx.x * 256u + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

extern "C" __global__ __launch_bounds__(256)
void add_bias_f32(
    float* __restrict__ y,
    const float* __restrict__ bias,
    u32 dim,
    u32 batch
) {
    u32 i = blockIdx.x * 256u + threadIdx.x;
    u32 t = blockIdx.y;
    if (i < dim && t < batch) y[(u64)t * dim + i] += bias[i];
}

// Fused residual add + RMSNorm: x[row] += r[row]; y[row] = rms_norm(x[row], w, eps)
extern "C" __global__ __launch_bounds__(256)
void add_rms_norm_f32(
    float* __restrict__ x,
    const float* __restrict__ r,
    const float* __restrict__ w,
    float* __restrict__ y,
    u32 n,
    float eps
) {
    __shared__ float smem[8];
    u32 lid = threadIdx.x;
    u32 row = blockIdx.x;
    float* xr = x + (u64)row * n;
    const float* rr = r + (u64)row * n;
    float* yr = y + (u64)row * n;

    float ss = 0.0f;
    for (u32 i = lid; i < n; i += 256u) {
        float v = xr[i] + rr[i];
        xr[i] = v;
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if ((lid & 31u) == 0u) smem[lid >> 5] = ss;
    lds_barrier();
    if (lid < 8u) {
        float v = smem[lid];
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor(v, off, 8);
        smem[lid] = v;
    }
    lds_barrier();
    float scale = rsqrtf(smem[0] / (float)n + eps);
    for (u32 i = lid; i < n; i += 256u) {
        yr[i] = xr[i] * scale * w[i];
    }
}

extern "C" __global__ __launch_bounds__(64)
void rms_norm_heads_f32(
    float* __restrict__ x,
    const float* __restrict__ w,
    u32 head_dim,
    u32 n_heads,
    float eps
) {
    __shared__ float smem[2];
    u32 lid = threadIdx.x;
    u32 h = blockIdx.x; // covers n_heads * batch
    float* xr = x + (u64)h * head_dim;

    float ss = 0.0f;
    for (u32 i = lid; i < head_dim; i += 64u) {
        float v = xr[i];
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if ((lid & 31u) == 0u) smem[lid >> 5] = ss;
    lds_barrier();
    float scale = rsqrtf((smem[0] + smem[1]) / (float)head_dim + eps);
    for (u32 i = lid; i < head_dim; i += 64u) {
        xr[i] = xr[i] * scale * w[i];
    }
}

extern "C" __global__ __launch_bounds__(256)
void swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    u32 n
) {
    u32 i = blockIdx.x * 256u + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + __expf(-g))) * up[i];
    }
}

// Fused RMSNorm + Q8 activation quantize (single row, n % 32 == 0).
extern "C" __global__ __launch_bounds__(256)
void rms_norm_q8_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 n,
    float eps
) {
    __shared__ float smem[8];
    u32 lid = threadIdx.x;
    u32 warp = lid >> 5;
    u32 lane = lid & 31u;
    float ss = 0.0f;
    for (u32 i = lid; i < n; i += 256u) {
        float v = x[i];
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if (lane == 0u) smem[warp] = ss;
    lds_barrier();
    if (lid < 8u) {
        float v = smem[lid];
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor(v, off, 8);
        smem[lid] = v;
    }
    lds_barrier();
    float scale = rsqrtf(smem[0] / (float)n + eps);
    u32 n_blocks = n >> 5;
    for (u32 blk = warp; blk < n_blocks; blk += 8u) {
        u32 idx = blk * 32u + lane;
        float v = x[idx] * scale * w[idx];
        y[idx] = v;
        warp_quant_q8_block(v, lane, blk, xq, xsc);
    }
}

// Fused residual add + RMSNorm + Q8 activation quantize (single row, n % 32 == 0).
extern "C" __global__ __launch_bounds__(256)
void add_rms_norm_q8_f32(
    float* __restrict__ x,
    const float* __restrict__ r,
    const float* __restrict__ w,
    float* __restrict__ y,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 n,
    float eps
) {
    __shared__ float smem[8];
    u32 lid = threadIdx.x;
    u32 warp = lid >> 5;
    u32 lane = lid & 31u;
    float ss = 0.0f;
    for (u32 i = lid; i < n; i += 256u) {
        float v = x[i] + r[i];
        x[i] = v;
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if (lane == 0u) smem[warp] = ss;
    lds_barrier();
    if (lid < 8u) {
        float v = smem[lid];
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor(v, off, 8);
        smem[lid] = v;
    }
    lds_barrier();
    float scale = rsqrtf(smem[0] / (float)n + eps);
    u32 n_blocks = n >> 5;
    for (u32 blk = warp; blk < n_blocks; blk += 8u) {
        u32 idx = blk * 32u + lane;
        float v = x[idx] * scale * w[idx];
        y[idx] = v;
        warp_quant_q8_block(v, lane, blk, xq, xsc);
    }
}

// Fused SwiGLU + Q8 activation quantize (1 warp per 32-element block, 8 blocks/WG).
extern "C" __global__ __launch_bounds__(256)
void swiglu_q8_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 n
) {
    u32 lid = threadIdx.x;
    u32 warp = lid >> 5;
    u32 lane = lid & 31u;
    u32 blk = blockIdx.x * 8u + warp;
    if (blk >= (n >> 5)) return;
    u32 idx = blk * 32u + lane;
    float g = gate[idx];
    float v = (g / (1.0f + __expf(-g))) * up[idx];
    out[idx] = v;
    warp_quant_q8_block(v, lane, blk, xq, xsc);
}

// Fused RoPE(Q) + RoPE(K) + paged FP16 KV write for 1-token decode (reads pos from step_meta[1]).
// Grid: (n_heads + n_kv_heads, 1, 1), Block: (128, 1, 1) (supports head_dim <= 256).
extern "C" __global__ __launch_bounds__(128)
void rope_qk_kv_write_meta(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    u32* __restrict__ kv_layer,
    const u32* __restrict__ page_table,
    const u32* __restrict__ step_meta,
    u32 n_heads,
    u32 n_kv_heads,
    u32 head_dim,
    u32 n_rot,
    float rope_base,
    u32 mode
) {
    u32 bid = blockIdx.x;
    u32 lid = threadIdx.x;
    u32 pos = step_meta[1];
    u32 half_rot = n_rot >> 1;
    u32 half_hd = head_dim >> 1;

    if (bid < n_heads) {
        // Apply RoPE to Q head `bid`
        if (lid < half_rot) {
            float freq = powf(rope_base, -(2.0f * (float)lid) / (float)n_rot);
            float c, s;
            sincosf((float)pos * freq, &s, &c);
            float* base = q + (u64)bid * head_dim;
            u32 i0 = (mode == 0u) ? (2u * lid) : lid;
            u32 i1 = (mode == 0u) ? (2u * lid + 1u) : (lid + half_rot);
            float x0 = base[i0], x1 = base[i1];
            base[i0] = x0 * c - x1 * s;
            base[i1] = x0 * s + x1 * c;
        }
    } else {
        // Apply RoPE to K head `kvh` AND immediately write K, V head `kvh` to FP16 KV page
        u32 kvh = bid - n_heads;
        if (kvh >= n_kv_heads) return;
        float* kbase = k + (u64)kvh * head_dim;
        if (lid < half_rot) {
            float freq = powf(rope_base, -(2.0f * (float)lid) / (float)n_rot);
            float c, s;
            sincosf((float)pos * freq, &s, &c);
            u32 i0 = (mode == 0u) ? (2u * lid) : lid;
            u32 i1 = (mode == 0u) ? (2u * lid + 1u) : (lid + half_rot);
            float x0 = kbase[i0], x1 = kbase[i1];
            kbase[i0] = x0 * c - x1 * s;
            kbase[i1] = x0 * s + x1 * c;
        }
        lds_barrier();
        if (lid < half_hd) {
            u32 page = page_table[pos >> 4];
            u32 slot = pos & 15u;
            u32 half_kv = n_kv_heads * half_hd;
            u64 k_off = ((u64)page * 32u + slot) * half_kv + (u64)kvh * half_hd + lid;
            u64 v_off = ((u64)page * 32u + 16u + slot) * half_kv + (u64)kvh * half_hd + lid;
            const float* vbase = v + (u64)kvh * head_dim;
            u32 pk = (u32)f32_to_f16(kbase[2u * lid]) | ((u32)f32_to_f16(kbase[2u * lid + 1u]) << 16);
            u32 pv = (u32)f32_to_f16(vbase[2u * lid]) | ((u32)f32_to_f16(vbase[2u * lid + 1u]) << 16);
            kv_layer[k_off] = pk;
            kv_layer[v_off] = pv;
        }
    }
}

// ---------------------------------------------------------------------------
// 3. RoPE (single or batched, mode 0 = adjacent pairs, mode 1 = NeoX split halves)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void rope_f32(
    float* __restrict__ x,
    u32 n_heads,
    u32 head_dim,
    u32 n_rot,
    u32 pos0,
    float rope_base,
    u32 mode,
    u32 batch
) {
    u32 half_rot = n_rot >> 1;
    u32 idx = blockIdx.x * 256u + threadIdx.x;
    u32 total_per_tok = n_heads * half_rot;
    u32 t = blockIdx.y;
    if (idx >= total_per_tok || t >= batch) return;

    u32 h = idx / half_rot;
    u32 j = idx % half_rot;
    float pos = (float)(pos0 + t);
    float freq = powf(rope_base, -(2.0f * (float)j) / (float)n_rot);
    float ang = pos * freq;
    float c, s;
    sincosf(ang, &s, &c);

    float* base = x + ((u64)t * n_heads + h) * head_dim;
    u32 i0 = (mode == 0u) ? (2u * j) : j;
    u32 i1 = (mode == 0u) ? (2u * j + 1u) : (j + half_rot);
    float x0 = base[i0];
    float x1 = base[i1];
    base[i0] = x0 * c - x1 * s;
    base[i1] = x0 * s + x1 * c;
}

extern "C" __global__ __launch_bounds__(256)
void rope_f32_meta(
    float* __restrict__ x,
    const u32* __restrict__ step_meta,
    u32 n_heads,
    u32 head_dim,
    u32 n_rot,
    float rope_base,
    u32 mode
) {
    u32 half_rot = n_rot >> 1;
    u32 idx = blockIdx.x * 256u + threadIdx.x;
    u32 total = n_heads * half_rot;
    if (idx >= total) return;

    u32 h = idx / half_rot;
    u32 j = idx % half_rot;
    float pos = (float)step_meta[1];
    float freq = powf(rope_base, -(2.0f * (float)j) / (float)n_rot);
    float ang = pos * freq;
    float c, s;
    sincosf(ang, &s, &c);

    float* base = x + (u64)h * head_dim;
    u32 i0 = (mode == 0u) ? (2u * j) : j;
    u32 i1 = (mode == 0u) ? (2u * j + 1u) : (j + half_rot);
    float x0 = base[i0];
    float x1 = base[i1];
    base[i0] = x0 * c - x1 * s;
    base[i1] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------------------
// 4. Paged FP16 KV write (16 tokens/page; packed u16 pairs in u32)
//    Layout matches ocak-vk: kv[layer_off + page * 2 * 16 * half_kv + is_v * 16 * half_kv + slot * half_kv + i]
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void kv_write_f16(
    const float* __restrict__ k,
    const float* __restrict__ v,
    u32* __restrict__ kv_layer,
    const u32* __restrict__ page_table,
    u32 pos0,
    u32 kv_dim,
    u32 batch
) {
    u32 half_kv = kv_dim >> 1;
    u32 i = blockIdx.x * 256u + threadIdx.x;
    u32 t = blockIdx.y;
    if (i >= half_kv || t >= batch) return;

    u32 pos = pos0 + t;
    u32 page = page_table[pos >> 4];
    u32 slot = pos & 15u;
    u64 k_off = ((u64)page * 32u + slot) * half_kv + i;
    u64 v_off = ((u64)page * 32u + 16u + slot) * half_kv + i;

    const float* kt = k + (u64)t * kv_dim;
    const float* vt = v + (u64)t * kv_dim;
    u32 pk = (u32)f32_to_f16(kt[2u * i]) | ((u32)f32_to_f16(kt[2u * i + 1u]) << 16);
    u32 pv = (u32)f32_to_f16(vt[2u * i]) | ((u32)f32_to_f16(vt[2u * i + 1u]) << 16);
    kv_layer[k_off] = pk;
    kv_layer[v_off] = pv;
}

extern "C" __global__ __launch_bounds__(256)
void kv_write_f16_meta(
    const float* __restrict__ k,
    const float* __restrict__ v,
    u32* __restrict__ kv_layer,
    const u32* __restrict__ page_table,
    const u32* __restrict__ step_meta,
    u32 kv_dim
) {
    u32 half_kv = kv_dim >> 1;
    u32 i = blockIdx.x * 256u + threadIdx.x;
    if (i >= half_kv) return;

    u32 pos = step_meta[1];
    u32 page = page_table[pos >> 4];
    u32 slot = pos & 15u;
    u64 k_off = ((u64)page * 32u + slot) * half_kv + i;
    u64 v_off = ((u64)page * 32u + 16u + slot) * half_kv + i;

    u32 pk = (u32)f32_to_f16(k[2u * i]) | ((u32)f32_to_f16(k[2u * i + 1u]) << 16);
    u32 pv = (u32)f32_to_f16(v[2u * i]) | ((u32)f32_to_f16(v[2u * i + 1u]) << 16);
    kv_layer[k_off] = pk;
    kv_layer[v_off] = pv;
}

// ---------------------------------------------------------------------------
// 5a. Split-K Decode Attention (0 bytes LDS, 1 Wave32 per (head, split))
//     Reads pos from step_meta[1] (HIP Graph compatible).
//     Grid: (n_heads, n_splits, 1), Block: (32, 1, 1).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(32)
void attention_split_f16_meta(
    const float* __restrict__ q,
    const u32* __restrict__ kv_layer,
    const u32* __restrict__ page_table,
    const u32* __restrict__ step_meta,
    float* __restrict__ part_out,  // [n_heads, n_splits, head_dim]
    float* __restrict__ part_ml,   // [n_heads, n_splits, 2] (max, sum)
    u32 n_heads,
    u32 n_kv_heads,
    u32 head_dim,
    u32 sliding_window,
    u32 n_splits
) {
    u32 h = blockIdx.x;
    u32 sp = blockIdx.y;
    u32 lane = threadIdx.x;
    if (h >= n_heads) return;

    u32 pos = step_meta[1];
    u32 k_start = (sliding_window > 0u && pos + 1u > sliding_window) ? (pos + 1u - sliding_window) : 0u;
    u32 n_keys = pos + 1u - k_start;
    u32 chunk = (n_keys + n_splits - 1u) / n_splits;
    u32 kb = k_start + sp * chunk;
    u32 ke = min(pos + 1u, kb + chunk);

    u32 part_idx = h * n_splits + sp;
    if (kb >= ke) {
        if (lane == 0u) {
            part_ml[part_idx * 2u] = -1e30f;
            part_ml[part_idx * 2u + 1u] = 0.0f;
        }
        return;
    }

    u32 kvh = (h * n_kv_heads) / n_heads;
    u32 half_kv = (n_kv_heads * head_dim) >> 1;
    u32 half_hd = head_dim >> 1;
    float scale = rsqrtf((float)head_dim);
    const float* qr = q + (u64)h * head_dim;

    float q0[4], q1[4], acc0[4], acc1[4];
    u32 n_pairs = (half_hd + 31u) >> 5;
    for (u32 p = 0; p < n_pairs; p++) {
        u32 idx = lane + p * 32u;
        if (idx < half_hd) {
            q0[p] = qr[2u * idx] * scale;
            q1[p] = qr[2u * idx + 1u] * scale;
        } else {
            q0[p] = 0.0f;
            q1[p] = 0.0f;
        }
        acc0[p] = 0.0f;
        acc1[p] = 0.0f;
    }

    float m_val = -1e30f;
    float l_val = 0.0f;
    u32 cur_pg_idx = 0xFFFFFFFFu;
    u32 page = 0u;

    for (u32 kt = kb; kt < ke; kt++) {
        u32 pg_idx = kt >> 4;
        if (pg_idx != cur_pg_idx) {
            cur_pg_idx = pg_idx;
            page = page_table[pg_idx];
        }
        u32 slot = kt & 15u;
        u64 k_base = ((u64)page * 32u + slot) * half_kv + (u64)kvh * half_hd;
        u64 v_base = ((u64)page * 32u + 16u + slot) * half_kv + (u64)kvh * half_hd;

        float dot = 0.0f;
        for (u32 p = 0; p < n_pairs; p++) {
            u32 idx = lane + p * 32u;
            if (idx < half_hd) {
                u32 pk = kv_layer[k_base + idx];
                dot = __fmaf_rn(q0[p], f16_to_f32((u16)(pk & 0xFFFFu)), dot);
                dot = __fmaf_rn(q1[p], f16_to_f32((u16)(pk >> 16)), dot);
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor(dot, off, 32);

        float m_new = fmaxf(m_val, dot);
        float alpha = __expf(m_val - m_new);
        float w_k = __expf(dot - m_new);
        l_val = l_val * alpha + w_k;
        m_val = m_new;

        for (u32 p = 0; p < n_pairs; p++) {
            u32 idx = lane + p * 32u;
            if (idx < half_hd) {
                u32 pv = kv_layer[v_base + idx];
                acc0[p] = __fmaf_rn(w_k, f16_to_f32((u16)(pv & 0xFFFFu)), acc0[p] * alpha);
                acc1[p] = __fmaf_rn(w_k, f16_to_f32((u16)(pv >> 16)), acc1[p] * alpha);
            }
        }
    }

    if (lane == 0u) {
        part_ml[part_idx * 2u] = m_val;
        part_ml[part_idx * 2u + 1u] = l_val;
    }
    float* pout = part_out + (u64)part_idx * head_dim;
    for (u32 p = 0; p < n_pairs; p++) {
        u32 idx = lane + p * 32u;
        if (idx < half_hd) {
            pout[2u * idx] = acc0[p];
            pout[2u * idx + 1u] = acc1[p];
        }
    }
}

extern "C" __global__ __launch_bounds__(64)
void attention_merge_f16(
    const float* __restrict__ part_out,
    const float* __restrict__ part_ml,
    float* __restrict__ out,
    u32 head_dim,
    u32 n_splits
) {
    u32 h = blockIdx.x;
    u32 lid = threadIdx.x;
    float mg = -1e30f;
    for (u32 sp = 0; sp < n_splits; sp++) {
        float m = part_ml[(h * n_splits + sp) * 2u];
        mg = fmaxf(mg, m);
    }
    float ltot = 0.0f;
    float sc[8];
    for (u32 sp = 0; sp < n_splits; sp++) {
        float l = part_ml[(h * n_splits + sp) * 2u + 1u];
        float m = part_ml[(h * n_splits + sp) * 2u];
        float w = (l > 0.0f) ? __expf(m - mg) : 0.0f;
        sc[sp] = w;
        ltot += l * w;
    }
    float inv_l = (ltot > 0.0f) ? (1.0f / ltot) : 0.0f;
    float* orow = out + (u64)h * head_dim;
    for (u32 d = lid; d < head_dim; d += 64u) {
        float acc = 0.0f;
        for (u32 sp = 0; sp < n_splits; sp++) {
            if (sc[sp] > 0.0f) {
                acc = __fmaf_rn(part_out[(u64)(h * n_splits + sp) * head_dim + d], sc[sp], acc);
            }
        }
        orow[d] = acc * inv_l;
    }
}

// Fused Split-K attention merge + Q8 quantize (head_dim % 32 == 0, 1 WG of 64 threads = 2 warps per head).
extern "C" __global__ __launch_bounds__(64)
void attention_merge_q8_f16(
    const float* __restrict__ part_out,
    const float* __restrict__ part_ml,
    float* __restrict__ out,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 head_dim,
    u32 n_splits
) {
    u32 h = blockIdx.x;
    u32 lid = threadIdx.x;
    u32 warp = lid >> 5;
    u32 lane = lid & 31u;

    float mg = -1e30f;
    for (u32 sp = 0; sp < n_splits; sp++) {
        mg = fmaxf(mg, part_ml[(h * n_splits + sp) * 2u]);
    }
    float ltot = 0.0f;
    float sc[8];
    for (u32 sp = 0; sp < n_splits; sp++) {
        float l = part_ml[(h * n_splits + sp) * 2u + 1u];
        float m = part_ml[(h * n_splits + sp) * 2u];
        float w = (l > 0.0f) ? __expf(m - mg) : 0.0f;
        sc[sp] = w;
        ltot += l * w;
    }
    float inv_l = (ltot > 0.0f) ? (1.0f / ltot) : 0.0f;
    float* orow = out + (u64)h * head_dim;
    u32 blocks_per_head = head_dim >> 5;
    for (u32 b = warp; b < blocks_per_head; b += 2u) {
        u32 d = b * 32u + lane;
        float acc = 0.0f;
        for (u32 sp = 0; sp < n_splits; sp++) {
            if (sc[sp] > 0.0f) {
                acc = __fmaf_rn(part_out[(u64)(h * n_splits + sp) * head_dim + d], sc[sp], acc);
            }
        }
        float v = acc * inv_l;
        orow[d] = v;
        warp_quant_q8_block(v, lane, h * blocks_per_head + b, xq, xsc);
    }
}

// ---------------------------------------------------------------------------
// 5b. Flash-Tiled Prefill Attention (8 query rows x 16-key KV page tile in LDS)
//     Grid: (n_heads, batch.div_ceil(8), 1), Block: (256, 1, 1) = 8 Wave32s (1 wave/query row).
//     Each 16-key tile is one physical KV page: staged once in LDS and reused 8x!
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void attention_prefill_f16(
    const float* __restrict__ q,
    const u32* __restrict__ kv_layer,
    const u32* __restrict__ page_table,
    float* __restrict__ out,
    u32 n_heads,
    u32 n_kv_heads,
    u32 head_dim,
    u32 pos0,
    u32 sliding_window,
    u32 batch
) {
    // Max head_dim = 256 -> half_hd = 128 u32s -> 16 * 128 = 2048 u32s = 8 KB for K + 8 KB for V = 16 KB LDS.
    __shared__ u32 sk[16 * 128];
    __shared__ u32 sv[16 * 128];

    u32 h = blockIdx.x;
    u32 t0 = blockIdx.y * 8u;
    u32 lid = threadIdx.x;
    u32 warp = lid >> 5;   // 0..7 = query row within tile
    u32 lane = lid & 31u;
    u32 t = t0 + warp;
    bool live = (t < batch);

    u32 kvh = (h * n_kv_heads) / n_heads;
    u32 half_kv = (n_kv_heads * head_dim) >> 1;
    u32 half_hd = head_dim >> 1;
    u32 n_pairs = (half_hd + 31u) >> 5;
    float scale = rsqrtf((float)head_dim);

    u32 pos = pos0 + t;
    u32 max_pos = pos0 + min(t0 + 7u, batch - 1u);
    u32 min_k_start = (sliding_window > 0u && (pos0 + t0 + 1u) > sliding_window)
        ? (pos0 + t0 + 1u - sliding_window) : 0u;
    u32 my_k_start = (sliding_window > 0u && (pos + 1u) > sliding_window)
        ? (pos + 1u - sliding_window) : 0u;

    float q0[4], q1[4], acc0[4], acc1[4];
    const float* qr = q + ((u64)min(t, batch - 1u) * n_heads + h) * head_dim;
    for (u32 p = 0; p < n_pairs; p++) {
        u32 idx = lane + p * 32u;
        if (live && idx < half_hd) {
            q0[p] = qr[2u * idx] * scale;
            q1[p] = qr[2u * idx + 1u] * scale;
        } else {
            q0[p] = 0.0f;
            q1[p] = 0.0f;
        }
        acc0[p] = 0.0f;
        acc1[p] = 0.0f;
    }

    float m_val = -1e30f;
    float l_val = 0.0f;

    u32 pg_beg = min_k_start >> 4;
    u32 pg_end = (max_pos >> 4) + 1u;
    u32 tile_words = 16u * half_hd;

    for (u32 pg_idx = pg_beg; pg_idx < pg_end; pg_idx++) {
        u32 page = page_table[pg_idx];
        u64 k_pg_base = (u64)page * 32u * half_kv + (u64)kvh * half_hd;
        u64 v_pg_base = ((u64)page * 32u + 16u) * half_kv + (u64)kvh * half_hd;

        // Cooperative coalesced load of this 16-token KV page into LDS
        for (u32 w = lid; w < tile_words; w += 256u) {
            u32 slot = w / half_hd;
            u32 col = w - slot * half_hd;
            sk[w] = kv_layer[k_pg_base + (u64)slot * half_kv + col];
            sv[w] = kv_layer[v_pg_base + (u64)slot * half_kv + col];
        }
        lds_barrier();

        if (live) {
            u32 kt0 = pg_idx << 4;
            u32 s_beg = (kt0 < my_k_start) ? (my_k_start - kt0) : 0u;
            u32 s_end = (kt0 + 16u <= pos + 1u) ? 16u : ((pos + 1u > kt0) ? (pos + 1u - kt0) : 0u);
            for (u32 slot = s_beg; slot < s_end; slot++) {
                u32 base = slot * half_hd;
                float dot = 0.0f;
                for (u32 p = 0; p < n_pairs; p++) {
                    u32 idx = lane + p * 32u;
                    if (idx < half_hd) {
                        u32 pk = sk[base + idx];
                        dot = __fmaf_rn(q0[p], f16_to_f32((u16)(pk & 0xFFFFu)), dot);
                        dot = __fmaf_rn(q1[p], f16_to_f32((u16)(pk >> 16)), dot);
                    }
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor(dot, off, 32);

                float m_new = fmaxf(m_val, dot);
                float alpha = __expf(m_val - m_new);
                float w_k = __expf(dot - m_new);
                l_val = l_val * alpha + w_k;
                m_val = m_new;

                for (u32 p = 0; p < n_pairs; p++) {
                    u32 idx = lane + p * 32u;
                    if (idx < half_hd) {
                        u32 pv = sv[base + idx];
                        acc0[p] = __fmaf_rn(w_k, f16_to_f32((u16)(pv & 0xFFFFu)), acc0[p] * alpha);
                        acc1[p] = __fmaf_rn(w_k, f16_to_f32((u16)(pv >> 16)), acc1[p] * alpha);
                    }
                }
            }
        }
        lds_barrier();
    }

    if (live) {
        float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        float* orow = out + ((u64)t * n_heads + h) * head_dim;
        for (u32 p = 0; p < n_pairs; p++) {
            u32 idx = lane + p * 32u;
            if (idx < half_hd) {
                orow[2u * idx] = acc0[p] * inv_l;
                orow[2u * idx + 1u] = acc1[p] * inv_l;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Paged FP16 GQA Attention (Decode + Causal Prefill)
//    1 workgroup (256 threads = 8 warps) per (query_token, query_head).
//    Each warp processes keys strided by 8 with online softmax, then merges across 8 warps.
//    Supports head_dim <= 256 (max 8 elems/lane in registers) and sliding_window.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void attention_f16(
    const float* __restrict__ q,
    const u32* __restrict__ kv_layer,
    const u32* __restrict__ page_table,
    float* __restrict__ out,
    u32 n_heads,
    u32 n_kv_heads,
    u32 head_dim,
    u32 pos0,
    u32 sliding_window,
    u32 batch
) {
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8 * 256];

    u32 h = blockIdx.x;
    u32 t = blockIdx.y;
    if (h >= n_heads || t >= batch) return;

    u32 lid = threadIdx.x;
    u32 warp = lid >> 5;
    u32 lane = lid & 31u;

    u32 pos = pos0 + t;
    u32 k_start = (sliding_window > 0u && pos + 1u > sliding_window) ? (pos + 1u - sliding_window) : 0u;
    u32 kvh = (h * n_kv_heads) / n_heads;
    u32 half_kv = (n_kv_heads * head_dim) >> 1;
    u32 half_hd = head_dim >> 1;
    float scale = rsqrtf((float)head_dim);

    const float* qr = q + ((u64)t * n_heads + h) * head_dim;
    // Load Q into registers: each lane holds pairs at indices `lane, lane+32, ...` up to half_hd (<= 128 -> <= 4 pairs = 8 floats).
    float q0[4], q1[4];
    float acc0[4], acc1[4];
    u32 n_pairs = (half_hd + 31u) >> 5; // <= 4 for head_dim <= 256
    #pragma unroll
    for (u32 p = 0; p < 4u; p++) {
        u32 idx = lane + p * 32u;
        if (idx < half_hd) {
            q0[p] = qr[2u * idx] * scale;
            q1[p] = qr[2u * idx + 1u] * scale;
        } else {
            q0[p] = 0.0f;
            q1[p] = 0.0f;
        }
        acc0[p] = 0.0f;
        acc1[p] = 0.0f;
    }

    float m_val = -1e30f;
    float l_val = 0.0f;

    for (u32 kt = k_start + warp; kt <= pos; kt += 8u) {
        u32 page = page_table[kt >> 4];
        u32 slot = kt & 15u;
        u64 k_base = ((u64)page * 32u + slot) * half_kv + (u64)kvh * half_hd;
        u64 v_base = ((u64)page * 32u + 16u + slot) * half_kv + (u64)kvh * half_hd;

        float dot = 0.0f;
        #pragma unroll
        for (u32 p = 0; p < 4u; p++) {
            u32 idx = lane + p * 32u;
            if (idx < half_hd) {
                u32 pk = kv_layer[k_base + idx];
                float k0 = f16_to_f32((u16)(pk & 0xFFFFu));
                float k1 = f16_to_f32((u16)(pk >> 16));
                dot = __fmaf_rn(q0[p], k0, dot);
                dot = __fmaf_rn(q1[p], k1, dot);
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor(dot, off, 32);

        float m_new = fmaxf(m_val, dot);
        float alpha = __expf(m_val - m_new);
        float w_k = __expf(dot - m_new);
        l_val = l_val * alpha + w_k;
        m_val = m_new;

        #pragma unroll
        for (u32 p = 0; p < 4u; p++) {
            u32 idx = lane + p * 32u;
            if (idx < half_hd) {
                u32 pv = kv_layer[v_base + idx];
                float v0 = f16_to_f32((u16)(pv & 0xFFFFu));
                float v1 = f16_to_f32((u16)(pv >> 16));
                acc0[p] = __fmaf_rn(w_k, v0, acc0[p] * alpha);
                acc1[p] = __fmaf_rn(w_k, v1, acc1[p] * alpha);
            }
        }
    }

    if (lane == 0u) {
        s_max[warp] = m_val;
        s_sum[warp] = l_val;
    }
    #pragma unroll
    for (u32 p = 0; p < 4u; p++) {
        u32 idx = lane + p * 32u;
        if (idx < half_hd) {
            s_acc[warp * 256u + 2u * idx] = acc0[p];
            s_acc[warp * 256u + 2u * idx + 1u] = acc1[p];
        }
    }
    lds_barrier();

    // Merge across the 8 warps: each thread lid < head_dim computes one output element.
    if (lid < head_dim) {
        float mg = s_max[0];
        #pragma unroll
        for (u32 w = 1; w < 8u; w++) mg = fmaxf(mg, s_max[w]);
        float ltot = 0.0f;
        float val = 0.0f;
        #pragma unroll
        for (u32 w = 0; w < 8u; w++) {
            if (s_sum[w] > 0.0f) {
                float sc = __expf(s_max[w] - mg);
                ltot += s_sum[w] * sc;
                val += s_acc[w * 256u + lid] * sc;
            }
        }
        float* orow = out + ((u64)t * n_heads + h) * head_dim;
        orow[lid] = (ltot > 0.0f) ? (val / ltot) : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// 6. Two-pass GPU argmax (4-byte greedy readback)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void argmax_p1(
    const float* __restrict__ x,
    float* __restrict__ part_v,
    u32* __restrict__ part_i,
    u32 n
) {
    __shared__ float sv[8];
    __shared__ u32   si[8];
    u32 lid = threadIdx.x;
    u32 bid = blockIdx.x;
    u32 stride = gridDim.x * 256u;

    float bv = -1e38f;
    u32 bi = 0u;
    for (u32 i = bid * 256u + lid; i < n; i += stride) {
        float v = x[i];
        if (v > bv) { bv = v; bi = i; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor(bv, off, 32);
        u32 oi = __shfl_xor(bi, off, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((lid & 31u) == 0u) {
        sv[lid >> 5] = bv;
        si[lid >> 5] = bi;
    }
    lds_barrier();
    if (lid == 0u) {
        float mv = sv[0];
        u32 mi = si[0];
        for (u32 w = 1; w < 8u; w++) {
            if (sv[w] > mv || (sv[w] == mv && si[w] < mi)) { mv = sv[w]; mi = si[w]; }
        }
        part_v[bid] = mv;
        part_i[bid] = mi;
    }
}

extern "C" __global__ __launch_bounds__(256)
void argmax_p2(
    const float* __restrict__ part_v,
    const u32* __restrict__ part_i,
    u32* __restrict__ out_idx,
    u32 n_parts
) {
    __shared__ float sv[8];
    __shared__ u32   si[8];
    u32 lid = threadIdx.x;

    float bv = -1e38f;
    u32 bi = 0u;
    for (u32 i = lid; i < n_parts; i += 256u) {
        float v = part_v[i];
        u32 idx = part_i[i];
        if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor(bv, off, 32);
        u32 oi = __shfl_xor(bi, off, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((lid & 31u) == 0u) {
        sv[lid >> 5] = bv;
        si[lid >> 5] = bi;
    }
    lds_barrier();
    if (lid == 0u) {
        float mv = sv[0];
        u32 mi = si[0];
        for (u32 w = 1; w < 8u; w++) {
            if (sv[w] > mv || (sv[w] == mv && si[w] < mi)) { mv = sv[w]; mi = si[w]; }
        }
        out_idx[0] = mi;
    }
}
"#;

/// Compiled dense transformer op bundle for `HipExec`.
pub struct HipDenseOps {
    _module: Arc<HipModule>,
    embed_token: HipFunction,
    embed_token_meta: HipFunction,
    embed_tokens_batch: HipFunction,
    add_f32: HipFunction,
    add_bias_f32: HipFunction,
    add_rms_norm_f32: HipFunction,
    rms_norm_q8_f32: HipFunction,
    add_rms_norm_q8_f32: HipFunction,
    rms_norm_heads_f32: HipFunction,
    swiglu_f32: HipFunction,
    swiglu_q8_f32: HipFunction,
    rope_f32: HipFunction,
    rope_f32_meta: HipFunction,
    rope_qk_kv_write_meta: HipFunction,
    kv_write_f16: HipFunction,
    kv_write_f16_meta: HipFunction,
    attention_f16: HipFunction,
    attention_split_f16_meta: HipFunction,
    attention_merge_f16: HipFunction,
    attention_merge_q8_f16: HipFunction,
    attention_prefill_f16: HipFunction,
    argmax_p1: HipFunction,
    argmax_p2: HipFunction,
}

impl HipDenseOps {
    pub fn compile(ctx: &Arc<HipContext>) -> Result<Self, HipError> {
        let module = Arc::new(ctx.compile_module(DENSE_OPS_SRC, "ocak_hip_dense_ops.hip", &[])?);
        Ok(Self {
            embed_token: module.function("embed_token")?,
            embed_token_meta: module.function("embed_token_meta")?,
            embed_tokens_batch: module.function("embed_tokens_batch")?,
            add_f32: module.function("add_f32")?,
            add_bias_f32: module.function("add_bias_f32")?,
            add_rms_norm_f32: module.function("add_rms_norm_f32")?,
            rms_norm_q8_f32: module.function("rms_norm_q8_f32")?,
            add_rms_norm_q8_f32: module.function("add_rms_norm_q8_f32")?,
            rms_norm_heads_f32: module.function("rms_norm_heads_f32")?,
            swiglu_f32: module.function("swiglu_f32")?,
            swiglu_q8_f32: module.function("swiglu_q8_f32")?,
            rope_f32: module.function("rope_f32")?,
            rope_f32_meta: module.function("rope_f32_meta")?,
            rope_qk_kv_write_meta: module.function("rope_qk_kv_write_meta")?,
            kv_write_f16: module.function("kv_write_f16")?,
            kv_write_f16_meta: module.function("kv_write_f16_meta")?,
            attention_f16: module.function("attention_f16")?,
            attention_split_f16_meta: module.function("attention_split_f16_meta")?,
            attention_merge_f16: module.function("attention_merge_f16")?,
            attention_merge_q8_f16: module.function("attention_merge_q8_f16")?,
            attention_prefill_f16: module.function("attention_prefill_f16")?,
            argmax_p1: module.function("argmax_p1")?,
            argmax_p2: module.function("argmax_p2")?,
            _module: module,
        })
    }

    pub fn embed_token_meta(
        &self,
        stream: &HipStream,
        w: &HipBuffer,
        ty: HipWType,
        step_meta: &HipBuffer,
        out: &HipBuffer,
        n_embd: u32,
    ) -> Result<(), HipError> {
        let mut wp = w.as_device_ptr();
        let mut mp = step_meta.as_device_ptr();
        let mut op = out.as_device_ptr();
        let mut ne = n_embd;
        let mut rb = ty.row_bytes(n_embd);
        let mut wt = ty.code();
        let mut params: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut wt as *mut _ as *mut c_void,
        ];
        unsafe { self.embed_token_meta.launch(stream, (n_embd.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn embed_token(
        &self,
        stream: &HipStream,
        w: &HipBuffer,
        ty: HipWType,
        out: &HipBuffer,
        token: u32,
        n_embd: u32,
    ) -> Result<(), HipError> {
        let mut wp = w.as_device_ptr();
        let mut op = out.as_device_ptr();
        let mut tok = token;
        let mut ne = n_embd;
        let mut rb = ty.row_bytes(n_embd);
        let mut wt = ty.code();
        let mut params: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tok as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut wt as *mut _ as *mut c_void,
        ];
        unsafe { self.embed_token.launch(stream, (n_embd.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn embed_tokens_batch(
        &self,
        stream: &HipStream,
        w: &HipBuffer,
        ty: HipWType,
        tokens: &HipBuffer,
        out: &HipBuffer,
        n_embd: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut wp = w.as_device_ptr();
        let mut tp = tokens.as_device_ptr();
        let mut op = out.as_device_ptr();
        let mut ne = n_embd;
        let mut rb = ty.row_bytes(n_embd);
        let mut wt = ty.code();
        let mut b = batch;
        let mut params: [*mut c_void; 7] = [
            &mut wp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut wt as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe {
            self.embed_tokens_batch.launch(stream, (n_embd.div_ceil(256), batch, 1), (256, 1, 1), 0, &mut params)
        }
    }

    pub fn add(
        &self,
        stream: &HipStream,
        a: &HipBuffer,
        b: &HipBuffer,
        y: &HipBuffer,
        n: u32,
    ) -> Result<(), HipError> {
        let mut ap = a.as_device_ptr();
        let mut bp = b.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut n = n;
        let mut params: [*mut c_void; 4] = [
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        unsafe { self.add_f32.launch(stream, (n.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn add_bias(
        &self,
        stream: &HipStream,
        y: &HipBuffer,
        bias: &HipBuffer,
        dim: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut yp = y.as_device_ptr();
        let mut bp = bias.as_device_ptr();
        let mut d = dim;
        let mut b = batch;
        let mut params: [*mut c_void; 4] = [
            &mut yp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe { self.add_bias_f32.launch(stream, (dim.div_ceil(256), batch, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn add_rms_norm(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        r: &HipBuffer,
        w: &HipBuffer,
        y: &HipBuffer,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut rp = r.as_device_ptr();
        let mut wp = w.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut n = n;
        let mut eps = eps;
        let mut params: [*mut c_void; 6] = [
            &mut xp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe { self.add_rms_norm_f32.launch(stream, (rows, 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn rms_norm_heads(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        w: &HipBuffer,
        head_dim: u32,
        n_heads_total: u32,
        eps: f32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut wp = w.as_device_ptr();
        let mut hd = head_dim;
        let mut nh = n_heads_total;
        let mut eps = eps;
        let mut params: [*mut c_void; 5] = [
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe { self.rms_norm_heads_f32.launch(stream, (n_heads_total, 1, 1), (64, 1, 1), 0, &mut params) }
    }

    pub fn swiglu(
        &self,
        stream: &HipStream,
        gate: &HipBuffer,
        up: &HipBuffer,
        out: &HipBuffer,
        n: u32,
    ) -> Result<(), HipError> {
        let mut gp = gate.as_device_ptr();
        let mut up_p = up.as_device_ptr();
        let mut op = out.as_device_ptr();
        let mut n = n;
        let mut params: [*mut c_void; 4] = [
            &mut gp as *mut _ as *mut c_void,
            &mut up_p as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        unsafe { self.swiglu_f32.launch(stream, (n.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn rope(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        n_heads: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        rope_base: f32,
        mode: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut p0 = pos0;
        let mut rb = rope_base;
        let mut md = mode;
        let mut b = batch;
        let mut params: [*mut c_void; 8] = [
            &mut xp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut p0 as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut md as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        let total = n_heads * (n_rot / 2);
        unsafe { self.rope_f32.launch(stream, (total.div_ceil(256), batch, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn kv_write(
        &self,
        stream: &HipStream,
        k: &HipBuffer,
        v: &HipBuffer,
        kv: &HipBuffer,
        kv_byte_offset: usize,
        page_table: &HipBuffer,
        pos0: u32,
        kv_dim: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut kp = k.as_device_ptr();
        let mut vp = v.as_device_ptr();
        let mut kvp = kv.device_ptr_at(kv_byte_offset)?;
        let mut ptp = page_table.as_device_ptr();
        let mut p0 = pos0;
        let mut kd = kv_dim;
        let mut b = batch;
        let mut params: [*mut c_void; 7] = [
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut kvp as *mut _ as *mut c_void,
            &mut ptp as *mut _ as *mut c_void,
            &mut p0 as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        let half_kv = kv_dim / 2;
        unsafe { self.kv_write_f16.launch(stream, (half_kv.div_ceil(256), batch, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn attention(
        &self,
        stream: &HipStream,
        q: &HipBuffer,
        kv: &HipBuffer,
        kv_byte_offset: usize,
        page_table: &HipBuffer,
        out: &HipBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        pos0: u32,
        sliding_window: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut qp = q.as_device_ptr();
        let mut kvp = kv.device_ptr_at(kv_byte_offset)?;
        let mut ptp = page_table.as_device_ptr();
        let mut op = out.as_device_ptr();
        let mut nh = n_heads;
        let mut nkv = n_kv_heads;
        let mut hd = head_dim;
        let mut p0 = pos0;
        let mut sw = sliding_window;
        let mut b = batch;
        let mut params: [*mut c_void; 10] = [
            &mut qp as *mut _ as *mut c_void,
            &mut kvp as *mut _ as *mut c_void,
            &mut ptp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut p0 as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe { self.attention_f16.launch(stream, (n_heads, batch, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn rope_meta(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        step_meta: &HipBuffer,
        n_heads: u32,
        head_dim: u32,
        n_rot: u32,
        rope_base: f32,
        mode: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut mp = step_meta.as_device_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut rb = rope_base;
        let mut md = mode;
        let mut params: [*mut c_void; 7] = [
            &mut xp as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut md as *mut _ as *mut c_void,
        ];
        let total = n_heads * (n_rot / 2);
        unsafe { self.rope_f32_meta.launch(stream, (total.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn kv_write_meta(
        &self,
        stream: &HipStream,
        k: &HipBuffer,
        v: &HipBuffer,
        kv: &HipBuffer,
        kv_byte_offset: usize,
        page_table: &HipBuffer,
        step_meta: &HipBuffer,
        kv_dim: u32,
    ) -> Result<(), HipError> {
        let mut kp = k.as_device_ptr();
        let mut vp = v.as_device_ptr();
        let mut kvp = kv.device_ptr_at(kv_byte_offset)?;
        let mut ptp = page_table.as_device_ptr();
        let mut mp = step_meta.as_device_ptr();
        let mut kd = kv_dim;
        let mut params: [*mut c_void; 6] = [
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut kvp as *mut _ as *mut c_void,
            &mut ptp as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void,
        ];
        let half_kv = kv_dim / 2;
        unsafe { self.kv_write_f16_meta.launch(stream, (half_kv.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn attention_decode_split_meta(
        &self,
        stream: &HipStream,
        q: &HipBuffer,
        kv: &HipBuffer,
        kv_byte_offset: usize,
        page_table: &HipBuffer,
        step_meta: &HipBuffer,
        part_out: &HipBuffer,
        part_ml: &HipBuffer,
        out: &HipBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        sliding_window: u32,
        n_splits: u32,
    ) -> Result<(), HipError> {
        let mut qp = q.as_device_ptr();
        let mut kvp = kv.device_ptr_at(kv_byte_offset)?;
        let mut ptp = page_table.as_device_ptr();
        let mut mp = step_meta.as_device_ptr();
        let mut pop = part_out.as_device_ptr();
        let mut pmlp = part_ml.as_device_ptr();
        let mut nh = n_heads;
        let mut nkv = n_kv_heads;
        let mut hd = head_dim;
        let mut sw = sliding_window;
        let mut ns = n_splits;
        let mut p1: [*mut c_void; 11] = [
            &mut qp as *mut _ as *mut c_void,
            &mut kvp as *mut _ as *mut c_void,
            &mut ptp as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut pop as *mut _ as *mut c_void,
            &mut pmlp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
        ];
        unsafe {
            self.attention_split_f16_meta
                .launch(stream, (n_heads, n_splits, 1), (32, 1, 1), 0, &mut p1)
        }?;

        let mut pop2 = part_out.as_device_ptr();
        let mut pmlp2 = part_ml.as_device_ptr();
        let mut op = out.as_device_ptr();
        let mut hd2 = head_dim;
        let mut ns2 = n_splits;
        let mut p2: [*mut c_void; 5] = [
            &mut pop2 as *mut _ as *mut c_void,
            &mut pmlp2 as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut hd2 as *mut _ as *mut c_void,
            &mut ns2 as *mut _ as *mut c_void,
        ];
        unsafe { self.attention_merge_f16.launch(stream, (n_heads, 1, 1), (64, 1, 1), 0, &mut p2) }
    }

    pub fn rms_norm_q8(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        w: &HipBuffer,
        y: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n: u32,
        eps: f32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut wp = w.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut n = n;
        let mut eps = eps;
        let mut params: [*mut c_void; 7] = [
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe { self.rms_norm_q8_f32.launch(stream, (1, 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn add_rms_norm_q8(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        r: &HipBuffer,
        w: &HipBuffer,
        y: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n: u32,
        eps: f32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut rp = r.as_device_ptr();
        let mut wp = w.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut n = n;
        let mut eps = eps;
        let mut params: [*mut c_void; 8] = [
            &mut xp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe { self.add_rms_norm_q8_f32.launch(stream, (1, 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn swiglu_q8(
        &self,
        stream: &HipStream,
        gate: &HipBuffer,
        up: &HipBuffer,
        out: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n: u32,
    ) -> Result<(), HipError> {
        let mut gp = gate.as_device_ptr();
        let mut up_p = up.as_device_ptr();
        let mut op = out.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut n = n;
        let mut params: [*mut c_void; 6] = [
            &mut gp as *mut _ as *mut c_void,
            &mut up_p as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let n_blocks = n / 32;
        unsafe { self.swiglu_q8_f32.launch(stream, (n_blocks.div_ceil(8), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn rope_qk_kv_write_meta(
        &self,
        stream: &HipStream,
        q: &HipBuffer,
        k: &HipBuffer,
        v: &HipBuffer,
        kv: &HipBuffer,
        kv_byte_offset: usize,
        page_table: &HipBuffer,
        step_meta: &HipBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_rot: u32,
        rope_base: f32,
        mode: u32,
    ) -> Result<(), HipError> {
        let mut qp = q.as_device_ptr();
        let mut kp = k.as_device_ptr();
        let mut vp = v.as_device_ptr();
        let mut kvp = kv.device_ptr_at(kv_byte_offset)?;
        let mut ptp = page_table.as_device_ptr();
        let mut mp = step_meta.as_device_ptr();
        let mut nh = n_heads;
        let mut nkv = n_kv_heads;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut rb = rope_base;
        let mut md = mode;
        let mut params: [*mut c_void; 12] = [
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut kvp as *mut _ as *mut c_void,
            &mut ptp as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut md as *mut _ as *mut c_void,
        ];
        unsafe {
            self.rope_qk_kv_write_meta
                .launch(stream, (n_heads + n_kv_heads, 1, 1), (128, 1, 1), 0, &mut params)
        }
    }

    pub fn attention_decode_split_q8_meta(
        &self,
        stream: &HipStream,
        q: &HipBuffer,
        kv: &HipBuffer,
        kv_byte_offset: usize,
        page_table: &HipBuffer,
        step_meta: &HipBuffer,
        part_out: &HipBuffer,
        part_ml: &HipBuffer,
        out: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        sliding_window: u32,
        n_splits: u32,
    ) -> Result<(), HipError> {
        let mut qp = q.as_device_ptr();
        let mut kvp = kv.device_ptr_at(kv_byte_offset)?;
        let mut ptp = page_table.as_device_ptr();
        let mut mp = step_meta.as_device_ptr();
        let mut pop = part_out.as_device_ptr();
        let mut pmlp = part_ml.as_device_ptr();
        let mut nh = n_heads;
        let mut nkv = n_kv_heads;
        let mut hd = head_dim;
        let mut sw = sliding_window;
        let mut ns = n_splits;
        let mut p1: [*mut c_void; 11] = [
            &mut qp as *mut _ as *mut c_void,
            &mut kvp as *mut _ as *mut c_void,
            &mut ptp as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut pop as *mut _ as *mut c_void,
            &mut pmlp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
        ];
        unsafe {
            self.attention_split_f16_meta
                .launch(stream, (n_heads, n_splits, 1), (32, 1, 1), 0, &mut p1)
        }?;

        let mut pop2 = part_out.as_device_ptr();
        let mut pmlp2 = part_ml.as_device_ptr();
        let mut op = out.as_device_ptr();
        let mut xqp = xq.as_device_ptr();
        let mut xscp = xsc.as_device_ptr();
        let mut hd2 = head_dim;
        let mut ns2 = n_splits;
        let mut p2: [*mut c_void; 7] = [
            &mut pop2 as *mut _ as *mut c_void,
            &mut pmlp2 as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut xqp as *mut _ as *mut c_void,
            &mut xscp as *mut _ as *mut c_void,
            &mut hd2 as *mut _ as *mut c_void,
            &mut ns2 as *mut _ as *mut c_void,
        ];
        unsafe { self.attention_merge_q8_f16.launch(stream, (n_heads, 1, 1), (64, 1, 1), 0, &mut p2) }
    }

    pub fn attention_prefill(
        &self,
        stream: &HipStream,
        q: &HipBuffer,
        kv: &HipBuffer,
        kv_byte_offset: usize,
        page_table: &HipBuffer,
        out: &HipBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        pos0: u32,
        sliding_window: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut qp = q.as_device_ptr();
        let mut kvp = kv.device_ptr_at(kv_byte_offset)?;
        let mut ptp = page_table.as_device_ptr();
        let mut op = out.as_device_ptr();
        let mut nh = n_heads;
        let mut nkv = n_kv_heads;
        let mut hd = head_dim;
        let mut p0 = pos0;
        let mut sw = sliding_window;
        let mut b = batch;
        let mut params: [*mut c_void; 10] = [
            &mut qp as *mut _ as *mut c_void,
            &mut kvp as *mut _ as *mut c_void,
            &mut ptp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut p0 as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe {
            self.attention_prefill_f16
                .launch(stream, (n_heads, batch.div_ceil(8), 1), (256, 1, 1), 0, &mut params)
        }
    }

    pub fn argmax(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        part_v: &HipBuffer,
        part_i: &HipBuffer,
        out_idx: &HipBuffer,
        n: u32,
    ) -> Result<(), HipError> {
        let n_parts = n.div_ceil(1024).clamp(1, 256);
        let mut xp = x.as_device_ptr();
        let mut pvp = part_v.as_device_ptr();
        let mut pip = part_i.as_device_ptr();
        let mut n_mut = n;
        let mut p1: [*mut c_void; 4] = [
            &mut xp as *mut _ as *mut c_void,
            &mut pvp as *mut _ as *mut c_void,
            &mut pip as *mut _ as *mut c_void,
            &mut n_mut as *mut _ as *mut c_void,
        ];
        unsafe { self.argmax_p1.launch(stream, (n_parts, 1, 1), (256, 1, 1), 0, &mut p1) }?;

        let mut pvp2 = part_v.as_device_ptr();
        let mut pip2 = part_i.as_device_ptr();
        let mut op = out_idx.as_device_ptr();
        let mut np = n_parts;
        let mut p2: [*mut c_void; 4] = [
            &mut pvp2 as *mut _ as *mut c_void,
            &mut pip2 as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
        ];
        unsafe { self.argmax_p2.launch(stream, (1, 1, 1), (256, 1, 1), 0, &mut p2) }
    }
}
