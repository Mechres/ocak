//! HIP C++ matvec/matmul/norm/quant kernels compiled at runtime via `libhiprtc`.
//! Supports `F32`, `F16`, `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K` for both decode (`qmv`)
//! and tiled int8-dot prefill (`qmm`).
use crate::ffi::HipError;
use crate::runtime::{HipBuffer, HipContext, HipFunction, HipModule, HipStream};
use std::ffi::c_void;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipWType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Q2K,
    Q3K,
}

impl HipWType {
    pub fn code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 3,
            Self::Q4K => 4,
            Self::Q5K => 5,
            Self::Q6K => 6,
            Self::Q2K => 7,
            Self::Q3K => 8,
        }
    }

    pub fn row_bytes(self, k: u32) -> u32 {
        match self {
            Self::F32 => k * 4,
            Self::F16 => k * 2,
            Self::Q4_0 => (k / 32) * 18,
            Self::Q8_0 => (k / 32) * 34,
            Self::Q4K => (k / 256) * 144,
            Self::Q5K => (k / 256) * 176,
            Self::Q6K => (k / 256) * 210,
            Self::Q2K => (k / 256) * 84,
            Self::Q3K => (k / 256) * 110,
        }
    }

    pub fn uses_q8_act(self) -> bool {
        !matches!(self, Self::F32 | Self::F16)
    }
}

pub const KERNELS_SRC: &str = r#"
typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;
typedef int                i32;
typedef signed char        i8;

struct __align__(16) u32x4 { u32 x, y, z, w; };
struct __align__(16) f32x4 { float x, y, z, w; };

__device__ __forceinline__ float f16_bits_to_f32(u16 h) {
    union { u16 u; _Float16 f; } v;
    v.u = h;
    return (float)v.f;
}

__device__ __forceinline__ float f16at(const u32* __restrict__ w32, u32 byte_off) {
    u32 word = w32[byte_off >> 2];
    u16 h = (u16)(word >> ((byte_off & 2u) << 3));
    return f16_bits_to_f32(h);
}

// 4 bytes at any offset via two aligned 32-bit loads (buffers have 16-byte tail pad).
__device__ __forceinline__ u32 bytes4(const u32* __restrict__ w32, u32 byte_off) {
    u32 a = byte_off >> 2;
    u32 sh = (byte_off & 3u) << 3;
    return sh == 0u ? w32[a] : ((w32[a] >> sh) | (w32[a + 1u] << (32u - sh)));
}

// Per-byte (b - c) for packed bytes with b in [0, 127], c <= 0x7F: no cross-byte borrows.
__device__ __forceinline__ u32 sub_bytes(u32 x, u32 c) {
    return ((x | 0x80808080u) - c) ^ 0x80808080u;
}

__device__ __forceinline__ i32 sdot4(u32 a, u32 b, i32 acc) {
    return __builtin_amdgcn_sdot4((i32)a, (i32)b, acc, false);
}

// VOP3P v_dot4_i32_i8 with inline literal 0: avoids `v_mov_b32 vdst, 0` + `v_dot4c`.
__device__ __forceinline__ i32 sdot4_zero(u32 a, u32 b) {
    i32 dst;
    asm("v_dot4_i32_i8 %0, %1, %2, 0" : "=v"(dst) : "v"(a), "v"(b));
    return dst;
}

// Workgroup LDS-only barrier: avoids `buffer_gl0_inv` L0 cache flush from __syncthreads().
__device__ __forceinline__ void lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// ---------------------------------------------------------------------------
// 1. RMSNorm: y[row] = rms_norm(x[row], w, eps)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void rms_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    u32 n,
    float eps
) {
    __shared__ float smem[8];
    u32 lid = threadIdx.x;
    u32 row = blockIdx.x;
    const float* xr = x + (u64)row * n;
    float* yr = y + (u64)row * n;

    float ss = 0.0f;
    for (u32 i = lid; i < n; i += 256u) {
        float v = xr[i];
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

// ---------------------------------------------------------------------------
// 2. Quantize activations to int8 blocks of 32 (GROUP32 and opt-in GROUP256)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void quant_q8(
    const f32x4* __restrict__ x4,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 n_blocks
) {
    u32 gid = blockIdx.x * 256u + threadIdx.x;
    u32 blk = gid >> 3;
    u32 lane = gid & 7u;
    if (blk >= n_blocks) return;

    f32x4 v = x4[gid];
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor(amax, 4, 8));
    amax = fmaxf(amax, __shfl_xor(amax, 2, 8));
    amax = fmaxf(amax, __shfl_xor(amax, 1, 8));

    float dx = amax * (1.0f / 127.0f);
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    i32 q0 = min(127, max(-127, __float2int_rn(v.x * inv)));
    i32 q1 = min(127, max(-127, __float2int_rn(v.y * inv)));
    i32 q2 = min(127, max(-127, __float2int_rn(v.z * inv)));
    i32 q3 = min(127, max(-127, __float2int_rn(v.w * inv)));
    xq[gid] = (u32)(q0 & 0xFF) | ((u32)(q1 & 0xFF) << 8)
            | ((u32)(q2 & 0xFF) << 16) | ((u32)(q3 & 0xFF) << 24);

    i32 s = q0 + q1 + q2 + q3;
    i32 s0 = (lane < 4u) ? s : 0;
    i32 s1 = (lane < 4u) ? 0 : s;
    s0 += __shfl_xor(s0, 4, 8); s0 += __shfl_xor(s0, 2, 8); s0 += __shfl_xor(s0, 1, 8);
    s1 += __shfl_xor(s1, 4, 8); s1 += __shfl_xor(s1, 2, 8); s1 += __shfl_xor(s1, 1, 8);
    if (lane == 0u) {
        xsc[blk] = { dx, dx * (float)s0, dx * (float)s1, 0.0f };
    }
}

extern "C" __global__ __launch_bounds__(256)
void quant_q8_g256(
    const f32x4* __restrict__ x4,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 n_blocks
) {
    __shared__ float gmax[8];
    u32 gid = blockIdx.x * 256u + threadIdx.x;
    u32 lid = threadIdx.x;
    u32 blk = gid >> 3;
    u32 lane = gid & 7u;
    bool live = blk < n_blocks;

    f32x4 v = live ? x4[gid] : f32x4{0.0f, 0.0f, 0.0f, 0.0f};
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    if ((lid & 31u) == 0u) gmax[lid >> 5] = amax;
    lds_barrier();
    u32 grp = lid >> 6;
    amax = fmaxf(gmax[grp * 2u], gmax[grp * 2u + 1u]);
    if (!live) return;

    float dx = amax * (1.0f / 127.0f);
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    i32 q0 = min(127, max(-127, __float2int_rn(v.x * inv)));
    i32 q1 = min(127, max(-127, __float2int_rn(v.y * inv)));
    i32 q2 = min(127, max(-127, __float2int_rn(v.z * inv)));
    i32 q3 = min(127, max(-127, __float2int_rn(v.w * inv)));
    xq[gid] = (u32)(q0 & 0xFF) | ((u32)(q1 & 0xFF) << 8)
            | ((u32)(q2 & 0xFF) << 16) | ((u32)(q3 & 0xFF) << 24);

    i32 s = q0 + q1 + q2 + q3;
    s += __shfl_xor(s, 4, 8); s += __shfl_xor(s, 2, 8); s += __shfl_xor(s, 1, 8);
    if (lane == 0u) {
        xsc[blk] = { dx, __int_as_float(s), 0.0f, 0.0f };
    }
}

// ---------------------------------------------------------------------------
// 3. Block-level (32-weight) unpackers for decode qmv
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dqi32_q4_0(
    const u32* __restrict__ w32, u32 rb, u32 blk, u32 q[8], float& d
) {
    u32 b = rb + blk * 18u;
    d = f16at(w32, b);
    u32 o = b + 2u;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u32 v = bytes4(w32, o + i * 4u);
        q[i]     = sub_bytes(v & 0x0F0F0F0Fu, 0x08080808u);
        q[i + 4] = sub_bytes((v >> 4) & 0x0F0F0F0Fu, 0x08080808u);
    }
}

__device__ __forceinline__ void dqi32_q8_0(
    const u32* __restrict__ w32, u32 rb, u32 blk, u32 q[8], float& d
) {
    u32 b = rb + blk * 34u;
    d = f16at(w32, b);
    u32 o = b + 2u;
    u32 a = o >> 2;
    if ((o & 2u) == 0u) {
        #pragma unroll
        for (int i = 0; i < 8; i++) q[i] = w32[a + i];
    } else {
        u32 prev = w32[a];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            u32 cur = w32[a + 1 + i];
            q[i] = (prev >> 16) | (cur << 16);
            prev = cur;
        }
    }
}

__device__ __forceinline__ void dqi32_q4_k(
    const u32* __restrict__ w32, u32 rb, u32 blk, u32 q[8], float& d, float& mn
) {
    u32 b = rb + (blk >> 3) * 144u;
    u32 j = blk & 7u;
    u32 dd = w32[b >> 2];
    float d0 = f16_bits_to_f32((u16)(dd & 0xFFFFu));
    float dm = f16_bits_to_f32((u16)(dd >> 16));
    u32 s0 = w32[(b >> 2) + 1u];
    u32 s1 = w32[(b >> 2) + 2u];
    u32 s2 = w32[(b >> 2) + 3u];
    u32 sc, m;
    if (j < 4u) {
        u32 sh = j * 8u;
        sc = (s0 >> sh) & 63u;
        m  = (s1 >> sh) & 63u;
    } else {
        u32 sh = (j - 4u) * 8u;
        u32 b2 = (s2 >> sh) & 0xFFu;
        sc = (b2 & 0xFu) | (((s0 >> (sh + 6u)) & 3u) << 4);
        m  = (b2 >> 4)   | (((s1 >> (sh + 6u)) & 3u) << 4);
    }
    d = d0 * (float)sc;
    mn = dm * (float)m;
    u32 sh = (j & 1u) * 4u;
    u32 qo = (b + 16u + (j >> 1) * 32u) >> 2;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        q[i] = (w32[qo + i] >> sh) & 0x0F0F0F0Fu;
    }
}

__device__ __forceinline__ void dqi32_q5_k(
    const u32* __restrict__ w32, u32 rb, u32 blk, u32 q[8], float& d, float& mn
) {
    u32 b = rb + (blk >> 3) * 176u;
    u32 j = blk & 7u;
    u32 dd = w32[b >> 2];
    float d0 = f16_bits_to_f32((u16)(dd & 0xFFFFu));
    float dm = f16_bits_to_f32((u16)(dd >> 16));
    u32 s0 = w32[(b >> 2) + 1u];
    u32 s1 = w32[(b >> 2) + 2u];
    u32 s2 = w32[(b >> 2) + 3u];
    u32 sc, m;
    if (j < 4u) {
        u32 sh = j * 8u;
        sc = (s0 >> sh) & 63u;
        m  = (s1 >> sh) & 63u;
    } else {
        u32 sh = (j - 4u) * 8u;
        u32 b2 = (s2 >> sh) & 0xFFu;
        sc = (b2 & 0xFu) | (((s0 >> (sh + 6u)) & 3u) << 4);
        m  = (b2 >> 4)   | (((s1 >> (sh + 6u)) & 3u) << 4);
    }
    d = d0 * (float)sc;
    mn = dm * (float)m;
    u32 sh = (j & 1u) * 4u;
    u32 qo = (b + 48u + (j >> 1) * 32u) >> 2;
    u32 ho = (b + 16u) >> 2;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        q[i] = ((w32[qo + i] >> sh) & 0x0F0F0F0Fu) | (((w32[ho + i] >> j) & 0x01010101u) << 4);
    }
}

__device__ __forceinline__ void dqi32_q6_k(
    const u32* __restrict__ w32, u32 rb, u32 blk, u32 q[8], float& d0_out, float& d1_out
) {
    u32 b = rb + (blk >> 3) * 210u;
    u32 j = blk & 7u;
    u32 half_ = j >> 2;
    u32 g = j & 3u;
    u32 qlo = b + half_ * 64u + (g & 1u) * 32u;
    u32 qho = b + 128u + half_ * 32u;
    u32 sl = (g >> 1) * 4u;
    u32 sh = g * 2u;
    float dbase = f16at(w32, b + 208u);
    u32 sc_off = b + 192u + half_ * 8u + 2u * g;
    u32 sc2 = (w32[sc_off >> 2] >> ((sc_off & 2u) << 3)) & 0xFFFFu;
    d0_out = dbase * (float)(i8)(sc2 & 0xFFu);
    d1_out = dbase * (float)(i8)(sc2 >> 8);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u32 lo = bytes4(w32, qlo + i * 4u);
        u32 hi = bytes4(w32, qho + i * 4u);
        u32 u = ((lo >> sl) & 0x0F0F0F0Fu) | (((hi >> sh) & 0x03030303u) << 4);
        q[i] = sub_bytes(u, 0x20202020u);
    }
}

__device__ __forceinline__ void dqi32_q2_k(
    const u32* __restrict__ w32, u32 rb, u32 blk, u32 q[8], float& d_out, float& m0_out, float& m1_out
) {
    u32 b = rb + (blk >> 3) * 84u;
    u32 ib = blk & 7u;
    u32 half_ = ib >> 2;
    u32 g = ib & 3u;
    u32 s_off = b + half_ * 8u + 2u * g;
    u32 s01 = (w32[s_off >> 2] >> ((s_off & 2u) << 3)) & 0xFFFFu;
    u32 s0 = s01 & 0xFFu, s1 = s01 >> 8;
    u32 qo = (b + 16u + half_ * 32u) >> 2;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        q[i] = ((w32[qo + i] >> (2u * g)) & 0x03030303u) * ((i < 4) ? (s0 & 0xFu) : (s1 & 0xFu));
    }
    u32 dd = w32[(b + 80u) >> 2];
    d_out = f16_bits_to_f32((u16)(dd & 0xFFFFu));
    float dmin = f16_bits_to_f32((u16)(dd >> 16));
    m0_out = dmin * (float)(s0 >> 4);
    m1_out = dmin * (float)(s1 >> 4);
}

__device__ __forceinline__ void dqi32_q3_k(
    const u32* __restrict__ w32, u32 rb, u32 blk, u32 q[8], float& d0_out, float& d1_out
) {
    u32 b = rb + (blk >> 3) * 110u;
    u32 ib = blk & 7u;
    u32 half_ = ib >> 2;
    u32 g = ib & 3u;
    u32 qo = b + 32u + half_ * 32u;
    u32 hs = half_ * 4u + g;
    u32 sh = (b & 3u) << 3;
    u32 qa = qo >> 2, ha = b >> 2;
    u32 prev = ((w32[qa] >> (2u * g)) & 0x03030303u) | (((w32[ha] >> hs) & 0x01010101u) << 2);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u32 next = ((w32[qa + i + 1] >> (2u * g)) & 0x03030303u) | (((w32[ha + i + 1] >> hs) & 0x01010101u) << 2);
        u32 aligned = (sh == 0u) ? prev : ((prev >> sh) | (next << (32u - sh)));
        q[i] = sub_bytes(aligned, 0x04040404u);
        prev = next;
    }
    float d = f16at(w32, b + 108u);
    u32 s01 = bytes4(w32, b + 96u + 4u * (g >> 1));
    u32 s2w = bytes4(w32, b + 104u);
    u32 is0 = half_ * 8u + 2u * g;
    u32 lo0 = (s01 >> (8u * (is0 & 3u) + 4u * half_)) & 0xFu;
    u32 hi0 = (s2w >> (8u * (is0 & 3u) + 2u * (is0 >> 2))) & 3u;
    d0_out = d * (float)((i32)(lo0 | (hi0 << 4)) - 32);
    u32 is1 = is0 + 1u;
    u32 lo1 = (s01 >> (8u * (is1 & 3u) + 4u * half_)) & 0xFu;
    u32 hi1 = (s2w >> (8u * (is1 & 3u) + 2u * (is1 >> 2))) & 3u;
    d1_out = d * (float)((i32)(lo1 | (hi1 << 4)) - 32);
}

// ---------------------------------------------------------------------------
// 4. Decode matvec (batch = 1..8): 32 lanes/row, 4 rows/workgroup
// ---------------------------------------------------------------------------
#define MAXB 8u

extern "C" __global__ __launch_bounds__(128)
void qmv_f32(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    u32 k, u32 m, u32 batch
) {
    u32 tid = threadIdx.x;
    u32 row = blockIdx.x * 4u + threadIdx.y;
    float acc[MAXB];
    #pragma unroll
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f;
    if (row < m) {
        const float* wr = w + (u64)row * k;
        for (u32 i = tid; i < k; i += 32u) {
            float wv = wr[i];
            for (u32 t = 0; t < batch; t++) acc[t] = __fmaf_rn(wv, x[(u64)t * k + i], acc[t]);
        }
    }
    for (u32 t = 0; t < batch; t++) {
        float v = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
        if (tid == 0u && row < m) y[(u64)t * m + row] = v;
    }
}

extern "C" __global__ __launch_bounds__(128)
void qmv_f16(
    const u16* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    u32 k, u32 m, u32 batch
) {
    u32 tid = threadIdx.x;
    u32 row = blockIdx.x * 4u + threadIdx.y;
    float acc[MAXB];
    #pragma unroll
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f;
    if (row < m) {
        const u16* wr = w + (u64)row * k;
        for (u32 i = tid; i < k; i += 32u) {
            float wv = f16_bits_to_f32(wr[i]);
            for (u32 t = 0; t < batch; t++) acc[t] = __fmaf_rn(wv, x[(u64)t * k + i], acc[t]);
        }
    }
    for (u32 t = 0; t < batch; t++) {
        float v = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
        if (tid == 0u && row < m) y[(u64)t * m + row] = v;
    }
}

#define DEFINE_QMV_NOMIN(NAME, UNPACK) \
extern "C" __global__ __launch_bounds__(128) \
void NAME( \
    const u32* __restrict__ w32, \
    const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, \
    float* __restrict__ y, \
    u32 k, u32 m, u32 row_bytes, u32 batch \
) { \
    u32 tid = threadIdx.x; \
    u32 row = blockIdx.x * 4u + threadIdx.y; \
    float acc[MAXB]; \
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d; \
            UNPACK(w32, rb, blk, q, d); \
            for (u32 t = 0; t < batch; t++) { \
                u32 xb = t * kblocks + blk; \
                u32x4 xa = xq4[xb * 2u]; \
                u32x4 xc = xq4[xb * 2u + 1u]; \
                i32 s = sdot4_zero(q[0], xa.x); \
                s = sdot4(q[1], xa.y, s); s = sdot4(q[2], xa.z, s); s = sdot4(q[3], xa.w, s); \
                s = sdot4(q[4], xc.x, s); s = sdot4(q[5], xc.y, s); s = sdot4(q[6], xc.z, s); s = sdot4(q[7], xc.w, s); \
                acc[t] = __fmaf_rn(d * xsc[xb].x, (float)s, acc[t]); \
            } \
        } \
    } \
    for (u32 t = 0; t < batch; t++) { \
        float v = acc[t]; \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32); \
        if (tid == 0u && row < m) y[(u64)t * m + row] = v; \
    } \
}

#define DEFINE_QMV_MIN(NAME, UNPACK) \
extern "C" __global__ __launch_bounds__(128) \
void NAME( \
    const u32* __restrict__ w32, \
    const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, \
    float* __restrict__ y, \
    u32 k, u32 m, u32 row_bytes, u32 batch \
) { \
    u32 tid = threadIdx.x; \
    u32 row = blockIdx.x * 4u + threadIdx.y; \
    float acc[MAXB]; \
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d, mn; \
            UNPACK(w32, rb, blk, q, d, mn); \
            for (u32 t = 0; t < batch; t++) { \
                u32 xb = t * kblocks + blk; \
                u32x4 xa = xq4[xb * 2u]; \
                u32x4 xc = xq4[xb * 2u + 1u]; \
                i32 s = sdot4_zero(q[0], xa.x); \
                s = sdot4(q[1], xa.y, s); s = sdot4(q[2], xa.z, s); s = sdot4(q[3], xa.w, s); \
                s = sdot4(q[4], xc.x, s); s = sdot4(q[5], xc.y, s); s = sdot4(q[6], xc.z, s); s = sdot4(q[7], xc.w, s); \
                f32x4 xs = xsc[xb]; \
                acc[t] = __fmaf_rn(d * xs.x, (float)s, acc[t]); \
                acc[t] = __fmaf_rn(-mn, xs.y + xs.z, acc[t]); \
            } \
        } \
    } \
    for (u32 t = 0; t < batch; t++) { \
        float v = acc[t]; \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32); \
        if (tid == 0u && row < m) y[(u64)t * m + row] = v; \
    } \
}

DEFINE_QMV_NOMIN(qmv_q4_0, dqi32_q4_0)
DEFINE_QMV_NOMIN(qmv_q8_0, dqi32_q8_0)
DEFINE_QMV_MIN(qmv_q4_k, dqi32_q4_k)
DEFINE_QMV_MIN(qmv_q5_k, dqi32_q5_k)

// Specialized batch == 1 decode matvec (scalar acc, no MAXB=8 array/loop)
#define DEFINE_QMV1_NOMIN(NAME, UNPACK) \
extern "C" __global__ __launch_bounds__(128) \
void NAME( \
    const u32* __restrict__ w32, \
    const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, \
    float* __restrict__ y, \
    u32 k, u32 m, u32 row_bytes \
) { \
    u32 tid = threadIdx.x; \
    u32 row = blockIdx.x * 4u + threadIdx.y; \
    float acc = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d; \
            UNPACK(w32, rb, blk, q, d); \
            u32x4 xa = xq4[blk * 2u]; \
            u32x4 xc = xq4[blk * 2u + 1u]; \
            i32 s = sdot4_zero(q[0], xa.x); \
            s = sdot4(q[1], xa.y, s); s = sdot4(q[2], xa.z, s); s = sdot4(q[3], xa.w, s); \
            s = sdot4(q[4], xc.x, s); s = sdot4(q[5], xc.y, s); s = sdot4(q[6], xc.z, s); s = sdot4(q[7], xc.w, s); \
            acc = __fmaf_rn(d * xsc[blk].x, (float)s, acc); \
        } \
    } \
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32); \
    if (tid == 0u && row < m) y[row] = acc; \
}

#define DEFINE_QMV1_MIN(NAME, UNPACK) \
extern "C" __global__ __launch_bounds__(128) \
void NAME( \
    const u32* __restrict__ w32, \
    const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, \
    float* __restrict__ y, \
    u32 k, u32 m, u32 row_bytes \
) { \
    u32 tid = threadIdx.x; \
    u32 row = blockIdx.x * 4u + threadIdx.y; \
    float acc = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d, mn; \
            UNPACK(w32, rb, blk, q, d, mn); \
            u32x4 xa = xq4[blk * 2u]; \
            u32x4 xc = xq4[blk * 2u + 1u]; \
            i32 s = sdot4_zero(q[0], xa.x); \
            s = sdot4(q[1], xa.y, s); s = sdot4(q[2], xa.z, s); s = sdot4(q[3], xa.w, s); \
            s = sdot4(q[4], xc.x, s); s = sdot4(q[5], xc.y, s); s = sdot4(q[6], xc.z, s); s = sdot4(q[7], xc.w, s); \
            f32x4 xs = xsc[blk]; \
            acc = __fmaf_rn(d * xs.x, (float)s, acc); \
            acc = __fmaf_rn(-mn, xs.y + xs.z, acc); \
        } \
    } \
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32); \
    if (tid == 0u && row < m) y[row] = acc; \
}

DEFINE_QMV1_NOMIN(qmv1_q4_0, dqi32_q4_0)
DEFINE_QMV1_NOMIN(qmv1_q8_0, dqi32_q8_0)
DEFINE_QMV1_MIN(qmv1_q4_k, dqi32_q4_k)
DEFINE_QMV1_MIN(qmv1_q5_k, dqi32_q5_k)

extern "C" __global__ __launch_bounds__(128)
void qmv1_q6_k(
    const u32* __restrict__ w32,
    const u32x4* __restrict__ xq4,
    const f32x4* __restrict__ xsc,
    float* __restrict__ y,
    u32 k, u32 m, u32 row_bytes
) {
    u32 tid = threadIdx.x;
    u32 row = blockIdx.x * 4u + threadIdx.y;
    float acc = 0.0f;
    if (row < m) {
        u32 rb = row * row_bytes;
        u32 kblocks = k >> 5;
        for (u32 blk = tid; blk < kblocks; blk += 32u) {
            u32 q[8]; float d0, d1;
            dqi32_q6_k(w32, rb, blk, q, d0, d1);
            u32x4 xa = xq4[blk * 2u];
            u32x4 xc = xq4[blk * 2u + 1u];
            i32 i0 = sdot4_zero(q[0], xa.x);
            i0 = sdot4(q[1], xa.y, i0); i0 = sdot4(q[2], xa.z, i0); i0 = sdot4(q[3], xa.w, i0);
            i32 i1 = sdot4_zero(q[4], xc.x);
            i1 = sdot4(q[5], xc.y, i1); i1 = sdot4(q[6], xc.z, i1); i1 = sdot4(q[7], xc.w, i1);
            float dx = xsc[blk].x;
            acc = __fmaf_rn(dx, __fmaf_rn(d0, (float)i0, d1 * (float)i1), acc);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (tid == 0u && row < m) y[row] = acc;
}

extern "C" __global__ __launch_bounds__(128)
void qmv_q6_k(
    const u32* __restrict__ w32,
    const u32x4* __restrict__ xq4,
    const f32x4* __restrict__ xsc,
    float* __restrict__ y,
    u32 k, u32 m, u32 row_bytes, u32 batch
) {
    u32 tid = threadIdx.x;
    u32 row = blockIdx.x * 4u + threadIdx.y;
    float acc[MAXB];
    #pragma unroll
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f;
    if (row < m) {
        u32 rb = row * row_bytes;
        u32 kblocks = k >> 5;
        for (u32 blk = tid; blk < kblocks; blk += 32u) {
            u32 q[8];
            float d0, d1;
            dqi32_q6_k(w32, rb, blk, q, d0, d1);
            for (u32 t = 0; t < batch; t++) {
                u32 xb = t * kblocks + blk;
                u32x4 xa = xq4[xb * 2u];
                u32x4 xc = xq4[xb * 2u + 1u];
                i32 i0 = sdot4_zero(q[0], xa.x);
                i0 = sdot4(q[1], xa.y, i0); i0 = sdot4(q[2], xa.z, i0); i0 = sdot4(q[3], xa.w, i0);
                i32 i1 = sdot4_zero(q[4], xc.x);
                i1 = sdot4(q[5], xc.y, i1); i1 = sdot4(q[6], xc.z, i1); i1 = sdot4(q[7], xc.w, i1);
                float dx = xsc[xb].x;
                acc[t] = __fmaf_rn(dx, __fmaf_rn(d0, (float)i0, d1 * (float)i1), acc[t]);
            }
        }
    }
    for (u32 t = 0; t < batch; t++) {
        float v = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
        if (tid == 0u && row < m) y[(u64)t * m + row] = v;
    }
}

extern "C" __global__ __launch_bounds__(128)
void qmv_q2_k(
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, const f32x4* __restrict__ xsc,
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes, u32 batch
) {
    u32 tid = threadIdx.x;
    u32 row = blockIdx.x * 4u + threadIdx.y;
    float acc[MAXB];
    #pragma unroll
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f;
    if (row < m) {
        u32 rb = row * row_bytes;
        u32 kblocks = k >> 5;
        for (u32 blk = tid; blk < kblocks; blk += 32u) {
            u32 q[8]; float d, m0, m1;
            dqi32_q2_k(w32, rb, blk, q, d, m0, m1);
            for (u32 t = 0; t < batch; t++) {
                u32 xb = t * kblocks + blk;
                u32x4 xa = xq4[xb * 2u], xc = xq4[xb * 2u + 1u];
                i32 s = sdot4_zero(q[0], xa.x);
                s = sdot4(q[1], xa.y, s); s = sdot4(q[2], xa.z, s); s = sdot4(q[3], xa.w, s);
                s = sdot4(q[4], xc.x, s); s = sdot4(q[5], xc.y, s); s = sdot4(q[6], xc.z, s); s = sdot4(q[7], xc.w, s);
                f32x4 xs = xsc[xb];
                acc[t] = __fmaf_rn(d * xs.x, (float)s, acc[t]);
                acc[t] = __fmaf_rn(-m0, xs.y, __fmaf_rn(-m1, xs.z, acc[t]));
            }
        }
    }
    for (u32 t = 0; t < batch; t++) {
        float v = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
        if (tid == 0u && row < m) y[(u64)t * m + row] = v;
    }
}

extern "C" __global__ __launch_bounds__(128)
void qmv1_q2_k(
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, const f32x4* __restrict__ xsc,
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes
) {
    u32 tid = threadIdx.x;
    u32 row = blockIdx.x * 4u + threadIdx.y;
    float acc = 0.0f;
    if (row < m) {
        u32 rb = row * row_bytes;
        u32 kblocks = k >> 5;
        for (u32 blk = tid; blk < kblocks; blk += 32u) {
            u32 q[8]; float d, m0, m1;
            dqi32_q2_k(w32, rb, blk, q, d, m0, m1);
            u32x4 xa = xq4[blk * 2u], xc = xq4[blk * 2u + 1u];
            i32 s = sdot4_zero(q[0], xa.x);
            s = sdot4(q[1], xa.y, s); s = sdot4(q[2], xa.z, s); s = sdot4(q[3], xa.w, s);
            s = sdot4(q[4], xc.x, s); s = sdot4(q[5], xc.y, s); s = sdot4(q[6], xc.z, s); s = sdot4(q[7], xc.w, s);
            f32x4 xs = xsc[blk];
            acc = __fmaf_rn(d * xs.x, (float)s, acc);
            acc = __fmaf_rn(-m0, xs.y, __fmaf_rn(-m1, xs.z, acc));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (tid == 0u && row < m) y[row] = acc;
}

extern "C" __global__ __launch_bounds__(128)
void qmv_q3_k(
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, const f32x4* __restrict__ xsc,
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes, u32 batch
) {
    u32 tid = threadIdx.x;
    u32 row = blockIdx.x * 4u + threadIdx.y;
    float acc[MAXB];
    #pragma unroll
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f;
    if (row < m) {
        u32 rb = row * row_bytes;
        u32 kblocks = k >> 5;
        for (u32 blk = tid; blk < kblocks; blk += 32u) {
            u32 q[8]; float d0, d1;
            dqi32_q3_k(w32, rb, blk, q, d0, d1);
            for (u32 t = 0; t < batch; t++) {
                u32 xb = t * kblocks + blk;
                u32x4 xa = xq4[xb * 2u], xc = xq4[xb * 2u + 1u];
                i32 i0 = sdot4_zero(q[0], xa.x);
                i0 = sdot4(q[1], xa.y, i0); i0 = sdot4(q[2], xa.z, i0); i0 = sdot4(q[3], xa.w, i0);
                i32 i1 = sdot4_zero(q[4], xc.x);
                i1 = sdot4(q[5], xc.y, i1); i1 = sdot4(q[6], xc.z, i1); i1 = sdot4(q[7], xc.w, i1);
                acc[t] = __fmaf_rn(xsc[xb].x, __fmaf_rn(d0, (float)i0, d1 * (float)i1), acc[t]);
            }
        }
    }
    for (u32 t = 0; t < batch; t++) {
        float v = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
        if (tid == 0u && row < m) y[(u64)t * m + row] = v;
    }
}

extern "C" __global__ __launch_bounds__(128)
void qmv1_q3_k(
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, const f32x4* __restrict__ xsc,
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes
) {
    u32 tid = threadIdx.x;
    u32 row = blockIdx.x * 4u + threadIdx.y;
    float acc = 0.0f;
    if (row < m) {
        u32 rb = row * row_bytes;
        u32 kblocks = k >> 5;
        for (u32 blk = tid; blk < kblocks; blk += 32u) {
            u32 q[8]; float d0, d1;
            dqi32_q3_k(w32, rb, blk, q, d0, d1);
            u32x4 xa = xq4[blk * 2u], xc = xq4[blk * 2u + 1u];
            i32 i0 = sdot4_zero(q[0], xa.x);
            i0 = sdot4(q[1], xa.y, i0); i0 = sdot4(q[2], xa.z, i0); i0 = sdot4(q[3], xa.w, i0);
            i32 i1 = sdot4_zero(q[4], xc.x);
            i1 = sdot4(q[5], xc.y, i1); i1 = sdot4(q[6], xc.z, i1); i1 = sdot4(q[7], xc.w, i1);
            acc = __fmaf_rn(xsc[blk].x, __fmaf_rn(d0, (float)i0, d1 * (float)i1), acc);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (tid == 0u && row < m) y[row] = acc;
}

// ---------------------------------------------------------------------------
// 5. Prefill tiled int8-dot matmul: BM=64, BN=128, BK=32, TM=4, TN=8, 256 threads
// ---------------------------------------------------------------------------
#define BM 64u
#define BN 128u
#define TM 4u
#define TN 8u

__device__ __forceinline__ void dqi8_q4_0(
    const u32* __restrict__ w32, u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 b = rb + (k0 >> 5) * 18u;
    u32 p = k0 & 31u;
    d = f16at(w32, b);
    u32 o = b + 2u + (p & 15u);
    u32 sh = (p >> 4) * 4u;
    q0 = sub_bytes((bytes4(w32, o) >> sh) & 0x0F0F0F0Fu, 0x08080808u);
    q1 = sub_bytes((bytes4(w32, o + 4u) >> sh) & 0x0F0F0F0Fu, 0x08080808u);
}

__device__ __forceinline__ void dqi8_q8_0(
    const u32* __restrict__ w32, u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 b = rb + (k0 >> 5) * 34u;
    u32 o = b + 2u + (k0 & 31u);
    d = f16at(w32, b);
    q0 = bytes4(w32, o);
    q1 = bytes4(w32, o + 4u);
}

__device__ __forceinline__ void dqi8_q4_k(
    const u32* __restrict__ w32, u32 rb, u32 k0, u32& q0, u32& q1, float& d, float& mn
) {
    u32 b = rb + (k0 >> 8) * 144u;
    u32 j = (k0 >> 5) & 7u;
    u32 l = k0 & 31u;
    u32 dd = w32[b >> 2];
    float d0 = f16_bits_to_f32((u16)(dd & 0xFFFFu));
    float dm = f16_bits_to_f32((u16)(dd >> 16));
    u32 s0 = w32[(b >> 2) + 1u];
    u32 s1 = w32[(b >> 2) + 2u];
    u32 s2 = w32[(b >> 2) + 3u];
    u32 sc, m;
    if (j < 4u) {
        u32 sh = j * 8u;
        sc = (s0 >> sh) & 63u;
        m  = (s1 >> sh) & 63u;
    } else {
        u32 sh = (j - 4u) * 8u;
        u32 b2 = (s2 >> sh) & 0xFFu;
        sc = (b2 & 0xFu) | (((s0 >> (sh + 6u)) & 3u) << 4);
        m  = (b2 >> 4)   | (((s1 >> (sh + 6u)) & 3u) << 4);
    }
    d = d0 * (float)sc;
    mn = dm * (float)m;
    u32 sh = (j & 1u) * 4u;
    u32 qo = (b + 16u + (j >> 1) * 32u + l) >> 2;
    q0 = (w32[qo] >> sh) & 0x0F0F0F0Fu;
    q1 = (w32[qo + 1u] >> sh) & 0x0F0F0F0Fu;
}

__device__ __forceinline__ void dqi8_q5_k(
    const u32* __restrict__ w32, u32 rb, u32 k0, u32& q0, u32& q1, float& d, float& mn
) {
    u32 b = rb + (k0 >> 8) * 176u;
    u32 j = (k0 >> 5) & 7u;
    u32 l = k0 & 31u;
    u32 dd = w32[b >> 2];
    float d0 = f16_bits_to_f32((u16)(dd & 0xFFFFu));
    float dm = f16_bits_to_f32((u16)(dd >> 16));
    u32 s0 = w32[(b >> 2) + 1u];
    u32 s1 = w32[(b >> 2) + 2u];
    u32 s2 = w32[(b >> 2) + 3u];
    u32 sc, m;
    if (j < 4u) {
        u32 sh = j * 8u;
        sc = (s0 >> sh) & 63u;
        m  = (s1 >> sh) & 63u;
    } else {
        u32 sh = (j - 4u) * 8u;
        u32 b2 = (s2 >> sh) & 0xFFu;
        sc = (b2 & 0xFu) | (((s0 >> (sh + 6u)) & 3u) << 4);
        m  = (b2 >> 4)   | (((s1 >> (sh + 6u)) & 3u) << 4);
    }
    d = d0 * (float)sc;
    mn = dm * (float)m;
    u32 sh = (j & 1u) * 4u;
    u32 qo = (b + 48u + (j >> 1) * 32u + l) >> 2;
    u32 ho = (b + 16u + l) >> 2;
    q0 = ((w32[qo] >> sh) & 0x0F0F0F0Fu) | (((w32[ho] >> j) & 0x01010101u) << 4);
    q1 = ((w32[qo + 1u] >> sh) & 0x0F0F0F0Fu) | (((w32[ho + 1u] >> j) & 0x01010101u) << 4);
}

__device__ __forceinline__ void dqi8_q6_k(
    const u32* __restrict__ w32, u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 b = rb + (k0 >> 8) * 210u;
    u32 n = k0 & 255u;
    u32 half_ = n >> 7;
    u32 g = (n >> 5) & 3u;
    u32 l = n & 31u;
    u32 qlo = b + half_ * 64u + (g & 1u) * 32u + l;
    u32 qho = b + 128u + half_ * 32u + l;
    u32 sl = (g >> 1) * 4u;
    u32 sh = g * 2u;
    u32 sc_off = b + 192u + half_ * 8u + (l >> 4) + 2u * g;
    u32 sc_byte = (w32[sc_off >> 2] >> ((sc_off & 3u) << 3)) & 0xFFu;
    d = f16at(w32, b + 208u) * (float)(i8)sc_byte;
    u32 lo0 = bytes4(w32, qlo), lo1 = bytes4(w32, qlo + 4u);
    u32 hi0 = bytes4(w32, qho), hi1 = bytes4(w32, qho + 4u);
    u32 u0 = ((lo0 >> sl) & 0x0F0F0F0Fu) | (((hi0 >> sh) & 0x03030303u) << 4);
    u32 u1 = ((lo1 >> sl) & 0x0F0F0F0Fu) | (((hi1 >> sh) & 0x03030303u) << 4);
    q0 = sub_bytes(u0, 0x20202020u);
    q1 = sub_bytes(u1, 0x20202020u);
}

__device__ __forceinline__ void dqi8_q2_k(
    const u32* __restrict__ w32, u32 rb, u32 k0, u32& q0, u32& q1, float& d, float& mn
) {
    u32 b = rb + (k0 >> 8) * 84u;
    u32 n = k0 & 255u;
    u32 half_ = n >> 7;
    u32 g = (n >> 5) & 3u;
    u32 l = n & 31u;
    u32 sc_off = b + half_ * 8u + 2u * g + (l >> 4);
    u32 sc = (w32[sc_off >> 2] >> ((sc_off & 3u) << 3)) & 0xFFu;
    u32 dd = w32[(b + 80u) >> 2];
    d = f16_bits_to_f32((u16)(dd & 0xFFFFu));
    mn = f16_bits_to_f32((u16)(dd >> 16)) * (float)(sc >> 4);
    u32 qo = (b + 16u + half_ * 32u + l) >> 2;
    q0 = ((w32[qo] >> (2u * g)) & 0x03030303u) * (sc & 0xFu);
    q1 = ((w32[qo + 1u] >> (2u * g)) & 0x03030303u) * (sc & 0xFu);
}

__device__ __forceinline__ void dqi8_q3_k(
    const u32* __restrict__ w32, u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 b = rb + (k0 >> 8) * 110u;
    u32 n = k0 & 255u;
    u32 half_ = n >> 7;
    u32 g = (n >> 5) & 3u;
    u32 l = n & 31u;
    u32 is = half_ * 8u + 2u * g + (l >> 4);
    u32 s01 = bytes4(w32, b + 96u + 4u * (g >> 1));
    u32 s2w = bytes4(w32, b + 104u);
    u32 lo4 = (s01 >> (8u * (is & 3u) + 4u * half_)) & 0xFu;
    u32 hi2 = (s2w >> (8u * (is & 3u) + 2u * (is >> 2))) & 3u;
    d = f16at(w32, b + 108u) * (float)((i32)(lo4 | (hi2 << 4)) - 32);
    u32 qo = b + 32u + half_ * 32u + l;
    u32 hs = half_ * 4u + g;
    u32 lo0 = (bytes4(w32, qo) >> (2u * g)) & 0x03030303u;
    u32 lo1 = (bytes4(w32, qo + 4u) >> (2u * g)) & 0x03030303u;
    u32 hm0 = (bytes4(w32, b + l) >> hs) & 0x01010101u;
    u32 hm1 = (bytes4(w32, b + l + 4u) >> hs) & 0x01010101u;
    q0 = sub_bytes(lo0 | (hm0 << 2), 0x04040404u);
    q1 = sub_bytes(lo1 | (hm1 << 2), 0x04040404u);
}

#define DEFINE_QMM_NOMIN(NAME, UNPACK) \
extern "C" __global__ __launch_bounds__(256, 2) \
void NAME( \
    const u32* __restrict__ w32, const u32* __restrict__ xq, const f32x4* __restrict__ xsc, \
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes, u32 batch, u32 kbchunk, u32 split_stride \
) { \
    __shared__ u32x4 ws[8u * (BM / 4u)]; \
    __shared__ u32x4 xs[8u * (BN / 4u)]; \
    __shared__ float wsd[BM]; \
    __shared__ float xsd[BN]; \
    u32 lid = threadIdx.x; u32 row0 = blockIdx.x * BM; u32 tok0 = blockIdx.y * BN; \
    u32 ld_row = lid >> 2; u32 ld_part = lid & 3u; \
    u32 rb = min(row0 + ld_row, m - 1u) * row_bytes; \
    u32 x_col = lid >> 1; u32 x_part = lid & 1u; \
    u32 col = min(tok0 + x_col, batch - 1u); \
    u32 kblocks = k >> 5; u32 xblk0 = col * kblocks; \
    u32 kb_beg = blockIdx.z * kbchunk; \
    u32 kb_end = min(kblocks, kb_beg + kbchunk); \
    if (kb_beg >= kb_end) return; \
    float* ybase = y + (u64)blockIdx.z * split_stride; \
    u32 tx = lid & 15u; u32 ty = lid >> 4; \
    float acc[TN][TM]; \
    for (u32 j = 0; j < TN; j++) for (u32 i = 0; i < TM; i++) acc[j][i] = 0.0f; \
    u32 pq0, pq1; float pd; u32 pxv[4]; float pxs; \
    UNPACK(w32, rb, (kb_beg << 5) + ld_part * 8u, pq0, pq1, pd); \
    for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb_beg) * 8u + x_part * 4u + c]; \
    pxs = xsc[xblk0 + kb_beg].x; \
    for (u32 kb = kb_beg; kb < kb_end; kb++) { \
        u32 kw = ld_part * 2u; \
        ((u32*)&ws[kw * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq0; \
        ((u32*)&ws[(kw + 1u) * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq1; \
        if (ld_part == 0u) wsd[ld_row] = pd; \
        for (u32 c = 0; c < 4u; c++) \
            ((u32*)&xs[(x_part * 4u + c) * (BN / 4u) + (x_col >> 2)])[x_col & 3u] = pxv[c]; \
        if (x_part == 0u) xsd[x_col] = pxs; \
        lds_barrier(); \
        if (kb + 1u < kb_end) { \
            UNPACK(w32, rb, ((kb + 1u) << 5) + ld_part * 8u, pq0, pq1, pd); \
            for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb + 1u) * 8u + x_part * 4u + c]; \
            pxs = xsc[xblk0 + kb + 1u].x; \
        } \
        i32 ia[TN][TM]; \
        const u32x4* wp = &ws[tx]; const u32x4* xp = &xs[ty * 2u]; \
        for (u32 kk = 0; kk < 8u; kk++) { \
            u32x4 wv = wp[kk * (BM / 4u)]; \
            u32x4 xv0 = xp[kk * (BN / 4u)]; u32x4 xv1 = xp[kk * (BN / 4u) + 1u]; \
            u32 xv[8] = { xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w }; \
            for (u32 j = 0; j < TN; j++) { \
                if (kk == 0u) { \
                    ia[j][0] = sdot4_zero(wv.x, xv[j]); ia[j][1] = sdot4_zero(wv.y, xv[j]); \
                    ia[j][2] = sdot4_zero(wv.z, xv[j]); ia[j][3] = sdot4_zero(wv.w, xv[j]); \
                } else { \
                    ia[j][0] = sdot4(wv.x, xv[j], ia[j][0]); ia[j][1] = sdot4(wv.y, xv[j], ia[j][1]); \
                    ia[j][2] = sdot4(wv.z, xv[j], ia[j][2]); ia[j][3] = sdot4(wv.w, xv[j], ia[j][3]); \
                } \
            } \
        } \
        float wd[4]; for (u32 i = 0; i < 4u; i++) wd[i] = wsd[tx * 4u + i]; \
        for (u32 j = 0; j < TN; j++) { \
            float dx = xsd[ty * TN + j]; \
            for (u32 i = 0; i < 4u; i++) acc[j][i] = __fmaf_rn(wd[i] * dx, (float)ia[j][i], acc[j][i]); \
        } \
        lds_barrier(); \
    } \
    for (u32 j = 0; j < TN; j++) { \
        u32 c = tok0 + ty * TN + j; \
        if (c < batch) for (u32 i = 0; i < TM; i++) { \
            u32 r = row0 + tx * 4u + i; \
            if (r < m) ybase[(u64)c * m + r] = acc[j][i]; \
        } \
    } \
}

#define DEFINE_QMM_MIN(NAME, UNPACK) \
extern "C" __global__ __launch_bounds__(256, 2) \
void NAME( \
    const u32* __restrict__ w32, const u32* __restrict__ xq, const f32x4* __restrict__ xsc, \
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes, u32 batch, u32 kbchunk, u32 split_stride \
) { \
    __shared__ u32x4 ws[8u * (BM / 4u)]; \
    __shared__ u32x4 xs[8u * (BN / 4u)]; \
    __shared__ float wsd[BM]; __shared__ float wsm[BM]; \
    __shared__ float xsd[BN]; __shared__ float xsm[BN]; \
    u32 lid = threadIdx.x; u32 row0 = blockIdx.x * BM; u32 tok0 = blockIdx.y * BN; \
    u32 ld_row = lid >> 2; u32 ld_part = lid & 3u; \
    u32 rb = min(row0 + ld_row, m - 1u) * row_bytes; \
    u32 x_col = lid >> 1; u32 x_part = lid & 1u; \
    u32 col = min(tok0 + x_col, batch - 1u); \
    u32 kblocks = k >> 5; u32 xblk0 = col * kblocks; \
    u32 kb_beg = blockIdx.z * kbchunk; \
    u32 kb_end = min(kblocks, kb_beg + kbchunk); \
    if (kb_beg >= kb_end) return; \
    float* ybase = y + (u64)blockIdx.z * split_stride; \
    u32 tx = lid & 15u; u32 ty = lid >> 4; \
    float acc[TN][TM]; \
    for (u32 j = 0; j < TN; j++) for (u32 i = 0; i < TM; i++) acc[j][i] = 0.0f; \
    u32 pq0, pq1; float pd, pm; u32 pxv[4]; f32x4 pxs; \
    UNPACK(w32, rb, (kb_beg << 5) + ld_part * 8u, pq0, pq1, pd, pm); \
    for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb_beg) * 8u + x_part * 4u + c]; \
    pxs = xsc[xblk0 + kb_beg]; \
    for (u32 kb = kb_beg; kb < kb_end; kb++) { \
        u32 kw = ld_part * 2u; \
        ((u32*)&ws[kw * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq0; \
        ((u32*)&ws[(kw + 1u) * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq1; \
        if (ld_part == 0u) { wsd[ld_row] = pd; wsm[ld_row] = pm; } \
        for (u32 c = 0; c < 4u; c++) \
            ((u32*)&xs[(x_part * 4u + c) * (BN / 4u) + (x_col >> 2)])[x_col & 3u] = pxv[c]; \
        if (x_part == 0u) { xsd[x_col] = pxs.x; xsm[x_col] = -(pxs.y + pxs.z); } \
        lds_barrier(); \
        if (kb + 1u < kb_end) { \
            UNPACK(w32, rb, ((kb + 1u) << 5) + ld_part * 8u, pq0, pq1, pd, pm); \
            for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb + 1u) * 8u + x_part * 4u + c]; \
            pxs = xsc[xblk0 + kb + 1u]; \
        } \
        i32 ia[TN][TM]; \
        const u32x4* wp = &ws[tx]; const u32x4* xp = &xs[ty * 2u]; \
        for (u32 kk = 0; kk < 8u; kk++) { \
            u32x4 wv = wp[kk * (BM / 4u)]; \
            u32x4 xv0 = xp[kk * (BN / 4u)]; u32x4 xv1 = xp[kk * (BN / 4u) + 1u]; \
            u32 xv[8] = { xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w }; \
            for (u32 j = 0; j < TN; j++) { \
                if (kk == 0u) { \
                    ia[j][0] = sdot4_zero(wv.x, xv[j]); ia[j][1] = sdot4_zero(wv.y, xv[j]); \
                    ia[j][2] = sdot4_zero(wv.z, xv[j]); ia[j][3] = sdot4_zero(wv.w, xv[j]); \
                } else { \
                    ia[j][0] = sdot4(wv.x, xv[j], ia[j][0]); ia[j][1] = sdot4(wv.y, xv[j], ia[j][1]); \
                    ia[j][2] = sdot4(wv.z, xv[j], ia[j][2]); ia[j][3] = sdot4(wv.w, xv[j], ia[j][3]); \
                } \
            } \
        } \
        float wd[4], wm[4]; \
        for (u32 i = 0; i < 4u; i++) { wd[i] = wsd[tx * 4u + i]; wm[i] = wsm[tx * 4u + i]; } \
        for (u32 j = 0; j < TN; j++) { \
            float dx = xsd[ty * TN + j]; float sx = xsm[ty * TN + j]; \
            for (u32 i = 0; i < 4u; i++) { \
                acc[j][i] = __fmaf_rn(wd[i] * dx, (float)ia[j][i], acc[j][i]); \
                acc[j][i] = __fmaf_rn(wm[i], sx, acc[j][i]); \
            } \
        } \
        lds_barrier(); \
    } \
    for (u32 j = 0; j < TN; j++) { \
        u32 c = tok0 + ty * TN + j; \
        if (c < batch) for (u32 i = 0; i < TM; i++) { \
            u32 r = row0 + tx * 4u + i; \
            if (r < m) ybase[(u64)c * m + r] = acc[j][i]; \
        } \
    } \
}

DEFINE_QMM_NOMIN(qmm_q4_0, dqi8_q4_0)
DEFINE_QMM_NOMIN(qmm_q8_0, dqi8_q8_0)
DEFINE_QMM_MIN(qmm_q4_k, dqi8_q4_k)
DEFINE_QMM_MIN(qmm_q5_k, dqi8_q5_k)

// Q6_K has a per-16 scale (two halves per 32-block).
extern "C" __global__ __launch_bounds__(256, 2)
void qmm_q6_k(
    const u32* __restrict__ w32, const u32* __restrict__ xq, const f32x4* __restrict__ xsc,
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes, u32 batch, u32 kbchunk, u32 split_stride
) {
    __shared__ u32x4 ws[8u * (BM / 4u)];
    __shared__ u32x4 xs[8u * (BN / 4u)];
    __shared__ float wsd[2u * BM];
    __shared__ float xsd[BN];

    u32 lid = threadIdx.x;
    u32 row0 = blockIdx.x * BM;
    u32 tok0 = blockIdx.y * BN;
    u32 ld_row = lid >> 2;
    u32 ld_part = lid & 3u;
    u32 rb = min(row0 + ld_row, m - 1u) * row_bytes;
    u32 x_col = lid >> 1;
    u32 x_part = lid & 1u;
    u32 col = min(tok0 + x_col, batch - 1u);
    u32 kblocks = k >> 5;
    u32 xblk0 = col * kblocks;
    u32 kb_beg = blockIdx.z * kbchunk;
    u32 kb_end = min(kblocks, kb_beg + kbchunk);
    if (kb_beg >= kb_end) return;
    float* ybase = y + (u64)blockIdx.z * split_stride;
    u32 tx = lid & 15u;
    u32 ty = lid >> 4;
    float acc[TN][TM];
    #pragma unroll
    for (u32 j = 0; j < TN; j++)
        #pragma unroll
        for (u32 i = 0; i < TM; i++) acc[j][i] = 0.0f;

    u32 pq0, pq1; float pd; u32 pxv[4]; float pxs;
    dqi8_q6_k(w32, rb, (kb_beg << 5) + ld_part * 8u, pq0, pq1, pd);
    #pragma unroll
    for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb_beg) * 8u + x_part * 4u + c];
    pxs = xsc[xblk0 + kb_beg].x;

    for (u32 kb = kb_beg; kb < kb_end; kb++) {
        u32 kw = ld_part * 2u;
        ((u32*)&ws[kw * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq0;
        ((u32*)&ws[(kw + 1u) * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq1;
        if ((ld_part & 1u) == 0u) wsd[(ld_part >> 1) * BM + ld_row] = pd;
        #pragma unroll
        for (u32 c = 0; c < 4u; c++)
            ((u32*)&xs[(x_part * 4u + c) * (BN / 4u) + (x_col >> 2)])[x_col & 3u] = pxv[c];
        if (x_part == 0u) xsd[x_col] = pxs;
        lds_barrier();

        if (kb + 1u < kb_end) {
            dqi8_q6_k(w32, rb, ((kb + 1u) << 5) + ld_part * 8u, pq0, pq1, pd);
            #pragma unroll
            for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb + 1u) * 8u + x_part * 4u + c];
            pxs = xsc[xblk0 + kb + 1u].x;
        }

        const u32x4* wp = &ws[tx];
        const u32x4* xp = &xs[ty * 2u];
        #pragma unroll
        for (u32 half_ = 0; half_ < 2u; half_++) {
            i32 ia[TN][TM];
            #pragma unroll
            for (u32 k2 = 0; k2 < 4u; k2++) {
                u32 kk = half_ * 4u + k2;
                u32x4 wv = wp[kk * (BM / 4u)];
                u32x4 xv0 = xp[kk * (BN / 4u)];
                u32x4 xv1 = xp[kk * (BN / 4u) + 1u];
                u32 xv[8] = { xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w };
                #pragma unroll
                for (u32 j = 0; j < TN; j++) {
                    if (k2 == 0u) {
                        ia[j][0] = sdot4_zero(wv.x, xv[j]); ia[j][1] = sdot4_zero(wv.y, xv[j]);
                        ia[j][2] = sdot4_zero(wv.z, xv[j]); ia[j][3] = sdot4_zero(wv.w, xv[j]);
                    } else {
                        ia[j][0] = sdot4(wv.x, xv[j], ia[j][0]); ia[j][1] = sdot4(wv.y, xv[j], ia[j][1]);
                        ia[j][2] = sdot4(wv.z, xv[j], ia[j][2]); ia[j][3] = sdot4(wv.w, xv[j], ia[j][3]);
                    }
                }
            }
            float wd[4];
            #pragma unroll
            for (u32 i = 0; i < 4u; i++) wd[i] = wsd[half_ * BM + tx * 4u + i];
            #pragma unroll
            for (u32 j = 0; j < TN; j++) {
                float dx = xsd[ty * TN + j];
                #pragma unroll
                for (u32 i = 0; i < 4u; i++) acc[j][i] = __fmaf_rn(wd[i] * dx, (float)ia[j][i], acc[j][i]);
            }
        }
        lds_barrier();
    }
    #pragma unroll
    for (u32 j = 0; j < TN; j++) {
        u32 c = tok0 + ty * TN + j;
        if (c < batch)
            #pragma unroll
            for (u32 i = 0; i < TM; i++) {
                u32 r = row0 + tx * 4u + i;
                if (r < m) ybase[(u64)c * m + r] = acc[j][i];
            }
    }
}

extern "C" __global__ __launch_bounds__(256)
void sum_splits_f32(
    const float* __restrict__ partials,
    float* __restrict__ out,
    u32 n,
    u32 n_splits
) {
    u32 i = blockIdx.x * 256u + threadIdx.x;
    if (i >= n) return;
    float s = partials[i];
    for (u32 sp = 1u; sp < n_splits; sp++) s += partials[(u64)sp * n + i];
    out[i] = s;
}

// Q2_K: one weight scale per 32 (sub-scales folded into int8 weights), two mins per 32 (per 16).
extern "C" __global__ __launch_bounds__(256, 2)
void qmm_q2_k(
    const u32* __restrict__ w32, const u32* __restrict__ xq, const f32x4* __restrict__ xsc,
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes, u32 batch, u32 kbchunk, u32 split_stride
) {
    __shared__ u32x4 ws[8u * (BM / 4u)];
    __shared__ u32x4 xs[8u * (BN / 4u)];
    __shared__ float wsd[BM];
    __shared__ float wsm[2u * BM];
    __shared__ float xsd[BN];
    __shared__ float xsm[2u * BN];
    u32 lid = threadIdx.x; u32 row0 = blockIdx.x * BM; u32 tok0 = blockIdx.y * BN;
    u32 ld_row = lid >> 2; u32 ld_part = lid & 3u;
    u32 rb = min(row0 + ld_row, m - 1u) * row_bytes;
    u32 x_col = lid >> 1; u32 x_part = lid & 1u;
    u32 col = min(tok0 + x_col, batch - 1u);
    u32 kblocks = k >> 5; u32 xblk0 = col * kblocks;
    u32 kb_beg = blockIdx.z * kbchunk;
    u32 kb_end = min(kblocks, kb_beg + kbchunk);
    if (kb_beg >= kb_end) return;
    float* ybase = y + (u64)blockIdx.z * split_stride;
    u32 tx = lid & 15u; u32 ty = lid >> 4;
    float acc[TN][TM];
    #pragma unroll
    for (u32 j = 0; j < TN; j++) for (u32 i = 0; i < TM; i++) acc[j][i] = 0.0f;
    u32 pq0, pq1; float pd, pm; u32 pxv[4]; f32x4 pxs;
    dqi8_q2_k(w32, rb, (kb_beg << 5) + ld_part * 8u, pq0, pq1, pd, pm);
    #pragma unroll
    for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb_beg) * 8u + x_part * 4u + c];
    pxs = xsc[xblk0 + kb_beg];
    for (u32 kb = kb_beg; kb < kb_end; kb++) {
        u32 kw = ld_part * 2u;
        ((u32*)&ws[kw * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq0;
        ((u32*)&ws[(kw + 1u) * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq1;
        if (ld_part == 0u) wsd[ld_row] = pd;
        if ((ld_part & 1u) == 0u) wsm[(ld_part >> 1) * BM + ld_row] = pm;
        #pragma unroll
        for (u32 c = 0; c < 4u; c++)
            ((u32*)&xs[(x_part * 4u + c) * (BN / 4u) + (x_col >> 2)])[x_col & 3u] = pxv[c];
        if (x_part == 0u) { xsd[x_col] = pxs.x; xsm[x_col] = -pxs.y; xsm[BN + x_col] = -pxs.z; }
        lds_barrier();
        if (kb + 1u < kb_end) {
            dqi8_q2_k(w32, rb, ((kb + 1u) << 5) + ld_part * 8u, pq0, pq1, pd, pm);
            #pragma unroll
            for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb + 1u) * 8u + x_part * 4u + c];
            pxs = xsc[xblk0 + kb + 1u];
        }
        i32 ia[TN][TM];
        const u32x4* wp = &ws[tx]; const u32x4* xp = &xs[ty * 2u];
        #pragma unroll
        for (u32 kk = 0; kk < 8u; kk++) {
            u32x4 wv = wp[kk * (BM / 4u)];
            u32x4 xv0 = xp[kk * (BN / 4u)], xv1 = xp[kk * (BN / 4u) + 1u];
            u32 xv[8] = { xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w };
            #pragma unroll
            for (u32 j = 0; j < TN; j++) {
                if (kk == 0u) {
                    ia[j][0] = sdot4_zero(wv.x, xv[j]); ia[j][1] = sdot4_zero(wv.y, xv[j]);
                    ia[j][2] = sdot4_zero(wv.z, xv[j]); ia[j][3] = sdot4_zero(wv.w, xv[j]);
                } else {
                    ia[j][0] = sdot4(wv.x, xv[j], ia[j][0]); ia[j][1] = sdot4(wv.y, xv[j], ia[j][1]);
                    ia[j][2] = sdot4(wv.z, xv[j], ia[j][2]); ia[j][3] = sdot4(wv.w, xv[j], ia[j][3]);
                }
            }
        }
        float wd[4], wm0[4], wm1[4];
        #pragma unroll
        for (u32 i = 0; i < 4u; i++) {
            wd[i] = wsd[tx * 4u + i];
            wm0[i] = wsm[tx * 4u + i];
            wm1[i] = wsm[BM + tx * 4u + i];
        }
        #pragma unroll
        for (u32 j = 0; j < TN; j++) {
            float dx = xsd[ty * TN + j];
            float sx0 = xsm[ty * TN + j], sx1 = xsm[BN + ty * TN + j];
            #pragma unroll
            for (u32 i = 0; i < 4u; i++) {
                acc[j][i] = __fmaf_rn(wd[i] * dx, (float)ia[j][i], acc[j][i]);
                acc[j][i] = __fmaf_rn(wm0[i], sx0, __fmaf_rn(wm1[i], sx1, acc[j][i]));
            }
        }
        lds_barrier();
    }
    #pragma unroll
    for (u32 j = 0; j < TN; j++) {
        u32 c = tok0 + ty * TN + j;
        if (c < batch) for (u32 i = 0; i < TM; i++) {
            u32 r = row0 + tx * 4u + i;
            if (r < m) ybase[(u64)c * m + r] = acc[j][i];
        }
    }
}

// Q3_K: per-16 scale (two halves per 32-block, no min).
extern "C" __global__ __launch_bounds__(256, 2)
void qmm_q3_k(
    const u32* __restrict__ w32, const u32* __restrict__ xq, const f32x4* __restrict__ xsc,
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes, u32 batch, u32 kbchunk, u32 split_stride
) {
    __shared__ u32x4 ws[8u * (BM / 4u)];
    __shared__ u32x4 xs[8u * (BN / 4u)];
    __shared__ float wsd[2u * BM];
    __shared__ float xsd[BN];
    u32 lid = threadIdx.x; u32 row0 = blockIdx.x * BM; u32 tok0 = blockIdx.y * BN;
    u32 ld_row = lid >> 2; u32 ld_part = lid & 3u;
    u32 rb = min(row0 + ld_row, m - 1u) * row_bytes;
    u32 x_col = lid >> 1; u32 x_part = lid & 1u;
    u32 col = min(tok0 + x_col, batch - 1u);
    u32 kblocks = k >> 5; u32 xblk0 = col * kblocks;
    u32 kb_beg = blockIdx.z * kbchunk;
    u32 kb_end = min(kblocks, kb_beg + kbchunk);
    if (kb_beg >= kb_end) return;
    float* ybase = y + (u64)blockIdx.z * split_stride;
    u32 tx = lid & 15u; u32 ty = lid >> 4;
    float acc[TN][TM];
    #pragma unroll
    for (u32 j = 0; j < TN; j++) for (u32 i = 0; i < TM; i++) acc[j][i] = 0.0f;
    u32 pq0, pq1; float pd; u32 pxv[4]; float pxs;
    dqi8_q3_k(w32, rb, (kb_beg << 5) + ld_part * 8u, pq0, pq1, pd);
    #pragma unroll
    for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb_beg) * 8u + x_part * 4u + c];
    pxs = xsc[xblk0 + kb_beg].x;
    for (u32 kb = kb_beg; kb < kb_end; kb++) {
        u32 kw = ld_part * 2u;
        ((u32*)&ws[kw * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq0;
        ((u32*)&ws[(kw + 1u) * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq1;
        if ((ld_part & 1u) == 0u) wsd[(ld_part >> 1) * BM + ld_row] = pd;
        #pragma unroll
        for (u32 c = 0; c < 4u; c++)
            ((u32*)&xs[(x_part * 4u + c) * (BN / 4u) + (x_col >> 2)])[x_col & 3u] = pxv[c];
        if (x_part == 0u) xsd[x_col] = pxs;
        lds_barrier();
        if (kb + 1u < kb_end) {
            dqi8_q3_k(w32, rb, ((kb + 1u) << 5) + ld_part * 8u, pq0, pq1, pd);
            #pragma unroll
            for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb + 1u) * 8u + x_part * 4u + c];
            pxs = xsc[xblk0 + kb + 1u].x;
        }
        const u32x4* wp = &ws[tx]; const u32x4* xp = &xs[ty * 2u];
        #pragma unroll
        for (u32 half_ = 0; half_ < 2u; half_++) {
            i32 ia[TN][TM];
            #pragma unroll
            for (u32 k2 = 0; k2 < 4u; k2++) {
                u32 kk = half_ * 4u + k2;
                u32x4 wv = wp[kk * (BM / 4u)];
                u32x4 xv0 = xp[kk * (BN / 4u)], xv1 = xp[kk * (BN / 4u) + 1u];
                u32 xv[8] = { xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w };
                #pragma unroll
                for (u32 j = 0; j < TN; j++) {
                    if (k2 == 0u) {
                        ia[j][0] = sdot4_zero(wv.x, xv[j]); ia[j][1] = sdot4_zero(wv.y, xv[j]);
                        ia[j][2] = sdot4_zero(wv.z, xv[j]); ia[j][3] = sdot4_zero(wv.w, xv[j]);
                    } else {
                        ia[j][0] = sdot4(wv.x, xv[j], ia[j][0]); ia[j][1] = sdot4(wv.y, xv[j], ia[j][1]);
                        ia[j][2] = sdot4(wv.z, xv[j], ia[j][2]); ia[j][3] = sdot4(wv.w, xv[j], ia[j][3]);
                    }
                }
            }
            float wd[4];
            #pragma unroll
            for (u32 i = 0; i < 4u; i++) wd[i] = wsd[half_ * BM + tx * 4u + i];
            #pragma unroll
            for (u32 j = 0; j < TN; j++) {
                float dx = xsd[ty * TN + j];
                #pragma unroll
                for (u32 i = 0; i < 4u; i++) acc[j][i] = __fmaf_rn(wd[i] * dx, (float)ia[j][i], acc[j][i]);
            }
        }
        lds_barrier();
    }
    #pragma unroll
    for (u32 j = 0; j < TN; j++) {
        u32 c = tok0 + ty * TN + j;
        if (c < batch) for (u32 i = 0; i < TM; i++) {
            u32 r = row0 + tx * 4u + i;
            if (r < m) ybase[(u64)c * m + r] = acc[j][i];
        }
    }
}

// Opt-in GROUP256 Q4_K prefill (64x128, TM=4, TN=8): rescales acc once per 256 elements.
extern "C" __global__ __launch_bounds__(256, 2)
void qmm_q4_k_g256(
    const u32* __restrict__ w32, const u32* __restrict__ xq, const f32x4* __restrict__ xsc,
    float* __restrict__ y, u32 k, u32 m, u32 row_bytes, u32 batch
) {
    __shared__ u32x4 ws[8u * (BM / 4u)];
    __shared__ u32x4 xs[8u * (BN / 4u)];
    __shared__ float wsd[BM]; __shared__ float wsm[BM];
    __shared__ float xdx[BN]; __shared__ float xsm[BN];
    u32 lid = threadIdx.x; u32 row0 = blockIdx.x * BM; u32 tok0 = blockIdx.y * BN;
    u32 ld_row = lid >> 2; u32 ld_part = lid & 3u;
    u32 rb = min(row0 + ld_row, m - 1u) * row_bytes;
    u32 x_col = lid >> 1; u32 x_part = lid & 1u;
    u32 col = min(tok0 + x_col, batch - 1u);
    u32 kblocks = k >> 5; u32 xblk0 = col * kblocks;
    u32 tx = lid & 15u; u32 ty = lid >> 4;
    float acc[TN][TM]; float cur[TN];
    #pragma unroll
    for (u32 j = 0; j < TN; j++) {
        cur[j] = 1.0f;
        #pragma unroll
        for (u32 i = 0; i < TM; i++) acc[j][i] = 0.0f;
    }
    u32 pq0, pq1; float pd, pm; u32 pxv[4]; f32x4 pxs;
    dqi8_q4_k(w32, rb, ld_part * 8u, pq0, pq1, pd, pm);
    #pragma unroll
    for (u32 c = 0; c < 4u; c++) pxv[c] = xq[xblk0 * 8u + x_part * 4u + c];
    pxs = xsc[xblk0];
    for (u32 kb = 0; kb < kblocks; kb++) {
        u32 kw = ld_part * 2u;
        ((u32*)&ws[kw * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq0;
        ((u32*)&ws[(kw + 1u) * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq1;
        if (ld_part == 0u) { wsd[ld_row] = pd; wsm[ld_row] = pm; }
        #pragma unroll
        for (u32 c = 0; c < 4u; c++)
            ((u32*)&xs[(x_part * 4u + c) * (BN / 4u) + (x_col >> 2)])[x_col & 3u] = pxv[c];
        if (x_part == 0u) {
            if ((kb & 7u) == 0u) xdx[x_col] = pxs.x;
            xsm[x_col] = -(float)__float_as_int(pxs.y);
        }
        lds_barrier();
        if ((kb & 7u) == 0u) {
            #pragma unroll
            for (u32 j = 0; j < TN; j++) {
                float ndx = xdx[ty * TN + j];
                if (ndx > 0.0f) {
                    float ratio = cur[j] / ndx;
                    #pragma unroll
                    for (u32 i = 0; i < TM; i++) acc[j][i] *= ratio;
                    cur[j] = ndx;
                }
            }
        }
        if (kb + 1u < kblocks) {
            dqi8_q4_k(w32, rb, ((kb + 1u) << 5) + ld_part * 8u, pq0, pq1, pd, pm);
            #pragma unroll
            for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb + 1u) * 8u + x_part * 4u + c];
            pxs = xsc[xblk0 + kb + 1u];
        }
        i32 ia[TN][TM];
        const u32x4* wp = &ws[tx]; const u32x4* xp = &xs[ty * 2u];
        #pragma unroll
        for (u32 kk = 0; kk < 8u; kk++) {
            u32x4 wv = wp[kk * (BM / 4u)];
            u32x4 xv0 = xp[kk * (BN / 4u)]; u32x4 xv1 = xp[kk * (BN / 4u) + 1u];
            u32 xv[8] = { xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w };
            #pragma unroll
            for (u32 j = 0; j < TN; j++) {
                if (kk == 0u) {
                    ia[j][0] = sdot4_zero(wv.x, xv[j]); ia[j][1] = sdot4_zero(wv.y, xv[j]);
                    ia[j][2] = sdot4_zero(wv.z, xv[j]); ia[j][3] = sdot4_zero(wv.w, xv[j]);
                } else {
                    ia[j][0] = sdot4(wv.x, xv[j], ia[j][0]); ia[j][1] = sdot4(wv.y, xv[j], ia[j][1]);
                    ia[j][2] = sdot4(wv.z, xv[j], ia[j][2]); ia[j][3] = sdot4(wv.w, xv[j], ia[j][3]);
                }
            }
        }
        float wd[4], wm[4];
        #pragma unroll
        for (u32 i = 0; i < 4u; i++) { wd[i] = wsd[tx * 4u + i]; wm[i] = wsm[tx * 4u + i]; }
        #pragma unroll
        for (u32 j = 0; j < TN; j++) {
            float sx = xsm[ty * TN + j];
            #pragma unroll
            for (u32 i = 0; i < 4u; i++) {
                acc[j][i] = __fmaf_rn(wd[i], (float)ia[j][i], acc[j][i]);
                acc[j][i] = __fmaf_rn(wm[i], sx, acc[j][i]);
            }
        }
        lds_barrier();
    }
    #pragma unroll
    for (u32 j = 0; j < TN; j++) {
        u32 c = tok0 + ty * TN + j;
        if (c < batch) {
            float s = cur[j];
            #pragma unroll
            for (u32 i = 0; i < TM; i++) {
                u32 r = row0 + tx * 4u + i;
                if (r < m) y[(u64)c * m + r] = acc[j][i] * s;
            }
        }
    }
}

// Simple tiled FP32 / FP16 matmul (32x32 tile, 16x16 threads) for unquantized layers / heads.
extern "C" __global__ __launch_bounds__(256)
void qmm_f32(
    const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
    u32 k, u32 m, u32 batch
) {
    __shared__ float ws[16][32];
    __shared__ float xs[16][32];
    u32 tx = threadIdx.x; u32 ty = threadIdx.y; // 16x16
    u32 row0 = blockIdx.x * 32u; u32 tok0 = blockIdx.y * 32u;
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    for (u32 k0 = 0; k0 < k; k0 += 16u) {
        for (u32 p = 0; p < 2u; p++) {
            u32 r = row0 + ty * 2u + p;
            ws[tx][ty * 2u + p] = (r < m && k0 + tx < k) ? w[(u64)r * k + k0 + tx] : 0.0f;
            u32 c = tok0 + ty * 2u + p;
            xs[tx][ty * 2u + p] = (c < batch && k0 + tx < k) ? x[(u64)c * k + k0 + tx] : 0.0f;
        }
        lds_barrier();
        #pragma unroll
        for (u32 kk = 0; kk < 16u; kk++) {
            float w0 = ws[kk][tx * 2u], w1 = ws[kk][tx * 2u + 1u];
            float x0 = xs[kk][ty * 2u], x1 = xs[kk][ty * 2u + 1u];
            acc[0][0] = __fmaf_rn(w0, x0, acc[0][0]); acc[0][1] = __fmaf_rn(w1, x0, acc[0][1]);
            acc[1][0] = __fmaf_rn(w0, x1, acc[1][0]); acc[1][1] = __fmaf_rn(w1, x1, acc[1][1]);
        }
        lds_barrier();
    }
    for (u32 j = 0; j < 2u; j++) {
        u32 c = tok0 + ty * 2u + j;
        if (c < batch) for (u32 i = 0; i < 2u; i++) {
            u32 r = row0 + tx * 2u + i;
            if (r < m) y[(u64)c * m + r] = acc[j][i];
        }
    }
}

extern "C" __global__ __launch_bounds__(256)
void qmm_f16(
    const u16* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
    u32 k, u32 m, u32 batch
) {
    __shared__ float ws[16][32];
    __shared__ float xs[16][32];
    u32 tx = threadIdx.x; u32 ty = threadIdx.y;
    u32 row0 = blockIdx.x * 32u; u32 tok0 = blockIdx.y * 32u;
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    for (u32 k0 = 0; k0 < k; k0 += 16u) {
        for (u32 p = 0; p < 2u; p++) {
            u32 r = row0 + ty * 2u + p;
            ws[tx][ty * 2u + p] = (r < m && k0 + tx < k) ? f16_bits_to_f32(w[(u64)r * k + k0 + tx]) : 0.0f;
            u32 c = tok0 + ty * 2u + p;
            xs[tx][ty * 2u + p] = (c < batch && k0 + tx < k) ? x[(u64)c * k + k0 + tx] : 0.0f;
        }
        lds_barrier();
        #pragma unroll
        for (u32 kk = 0; kk < 16u; kk++) {
            float w0 = ws[kk][tx * 2u], w1 = ws[kk][tx * 2u + 1u];
            float x0 = xs[kk][ty * 2u], x1 = xs[kk][ty * 2u + 1u];
            acc[0][0] = __fmaf_rn(w0, x0, acc[0][0]); acc[0][1] = __fmaf_rn(w1, x0, acc[0][1]);
            acc[1][0] = __fmaf_rn(w0, x1, acc[1][0]); acc[1][1] = __fmaf_rn(w1, x1, acc[1][1]);
        }
        lds_barrier();
    }
    for (u32 j = 0; j < 2u; j++) {
        u32 c = tok0 + ty * 2u + j;
        if (c < batch) for (u32 i = 0; i < 2u; i++) {
            u32 r = row0 + tx * 2u + i;
            if (r < m) y[(u64)c * m + r] = acc[j][i];
        }
    }
}
"#;

/// Compiled matvec/matmul/quant/norm kernel bundle for `HipContext`.
pub struct HipKernels {
    _module: Arc<HipModule>,
    rms_norm_f32: HipFunction,
    quant_q8: HipFunction,
    quant_q8_g256: HipFunction,
    qmv_f32: HipFunction,
    qmv_f16: HipFunction,
    qmv_q4_0: HipFunction,
    qmv_q8_0: HipFunction,
    qmv_q4_k: HipFunction,
    qmv_q5_k: HipFunction,
    qmv_q6_k: HipFunction,
    qmv_q2_k: HipFunction,
    qmv_q3_k: HipFunction,
    qmv1_q4_0: HipFunction,
    qmv1_q8_0: HipFunction,
    qmv1_q4_k: HipFunction,
    qmv1_q5_k: HipFunction,
    qmv1_q6_k: HipFunction,
    qmv1_q2_k: HipFunction,
    qmv1_q3_k: HipFunction,
    qmm_f32: HipFunction,
    qmm_f16: HipFunction,
    qmm_q4_0: HipFunction,
    qmm_q8_0: HipFunction,
    qmm_q4_k: HipFunction,
    qmm_q5_k: HipFunction,
    qmm_q6_k: HipFunction,
    qmm_q2_k: HipFunction,
    qmm_q3_k: HipFunction,
    qmm_q4_k_g256: HipFunction,
    sum_splits_f32: HipFunction,
}

impl HipKernels {
    pub fn compile(ctx: &Arc<HipContext>) -> Result<Self, HipError> {
        let module = Arc::new(ctx.compile_module(KERNELS_SRC, "ocak_hip_kernels.hip", &[])?);
        Ok(Self {
            rms_norm_f32: module.function("rms_norm_f32")?,
            quant_q8: module.function("quant_q8")?,
            quant_q8_g256: module.function("quant_q8_g256")?,
            qmv_f32: module.function("qmv_f32")?,
            qmv_f16: module.function("qmv_f16")?,
            qmv_q4_0: module.function("qmv_q4_0")?,
            qmv_q8_0: module.function("qmv_q8_0")?,
            qmv_q4_k: module.function("qmv_q4_k")?,
            qmv_q5_k: module.function("qmv_q5_k")?,
            qmv_q6_k: module.function("qmv_q6_k")?,
            qmv_q2_k: module.function("qmv_q2_k")?,
            qmv_q3_k: module.function("qmv_q3_k")?,
            qmv1_q4_0: module.function("qmv1_q4_0")?,
            qmv1_q8_0: module.function("qmv1_q8_0")?,
            qmv1_q4_k: module.function("qmv1_q4_k")?,
            qmv1_q5_k: module.function("qmv1_q5_k")?,
            qmv1_q6_k: module.function("qmv1_q6_k")?,
            qmv1_q2_k: module.function("qmv1_q2_k")?,
            qmv1_q3_k: module.function("qmv1_q3_k")?,
            qmm_f32: module.function("qmm_f32")?,
            qmm_f16: module.function("qmm_f16")?,
            qmm_q4_0: module.function("qmm_q4_0")?,
            qmm_q8_0: module.function("qmm_q8_0")?,
            qmm_q4_k: module.function("qmm_q4_k")?,
            qmm_q5_k: module.function("qmm_q5_k")?,
            qmm_q6_k: module.function("qmm_q6_k")?,
            qmm_q2_k: module.function("qmm_q2_k")?,
            qmm_q3_k: module.function("qmm_q3_k")?,
            qmm_q4_k_g256: module.function("qmm_q4_k_g256")?,
            sum_splits_f32: module.function("sum_splits_f32")?,
            _module: module,
        })
    }

    pub fn rms_norm(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        w: &HipBuffer,
        y: &HipBuffer,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut wp = w.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut n = n;
        let mut eps = eps;
        let mut params: [*mut c_void; 5] = [
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe { self.rms_norm_f32.launch(stream, (rows, 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn quant_q8(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n_blocks: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut nb = n_blocks;
        let mut params: [*mut c_void; 4] = [
            &mut xp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let grid_x = n_blocks.div_ceil(32);
        unsafe { self.quant_q8.launch(stream, (grid_x, 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn quant_q8_g256(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n_blocks: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut nb = n_blocks;
        let mut params: [*mut c_void; 4] = [
            &mut xp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let grid_x = n_blocks.div_ceil(32);
        unsafe { self.quant_q8_g256.launch(stream, (grid_x, 1, 1), (256, 1, 1), 0, &mut params) }
    }

    /// Unified decode matvec (1..=8 tokens). For quantized types, `xq`/`xsc` must hold `quant_q8(x)`.
    pub fn qmv(
        &self,
        ty: HipWType,
        stream: &HipStream,
        w: &HipBuffer,
        x_f32: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        k: u32,
        m: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        if batch == 1 {
            return match ty {
                HipWType::F32 => self.launch_qmv_float(&self.qmv_f32, stream, w, x_f32, y, k, m, 1),
                HipWType::F16 => self.launch_qmv_float(&self.qmv_f16, stream, w, x_f32, y, k, m, 1),
                HipWType::Q4_0 => self.launch_qmv1(&self.qmv1_q4_0, stream, w, xq, xsc, y, k, m, ty.row_bytes(k)),
                HipWType::Q8_0 => self.launch_qmv1(&self.qmv1_q8_0, stream, w, xq, xsc, y, k, m, ty.row_bytes(k)),
                HipWType::Q4K => self.launch_qmv1(&self.qmv1_q4_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k)),
                HipWType::Q5K => self.launch_qmv1(&self.qmv1_q5_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k)),
                HipWType::Q6K => self.launch_qmv1(&self.qmv1_q6_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k)),
                HipWType::Q2K => self.launch_qmv1(&self.qmv1_q2_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k)),
                HipWType::Q3K => self.launch_qmv1(&self.qmv1_q3_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k)),
            };
        }
        match ty {
            HipWType::F32 => self.launch_qmv_float(&self.qmv_f32, stream, w, x_f32, y, k, m, batch),
            HipWType::F16 => self.launch_qmv_float(&self.qmv_f16, stream, w, x_f32, y, k, m, batch),
            HipWType::Q4_0 => self.launch_qmv(&self.qmv_q4_0, stream, w, xq, xsc, y, k, m, ty.row_bytes(k), batch),
            HipWType::Q8_0 => self.launch_qmv(&self.qmv_q8_0, stream, w, xq, xsc, y, k, m, ty.row_bytes(k), batch),
            HipWType::Q4K => self.launch_qmv(&self.qmv_q4_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k), batch),
            HipWType::Q5K => self.launch_qmv(&self.qmv_q5_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k), batch),
            HipWType::Q6K => self.launch_qmv(&self.qmv_q6_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k), batch),
            HipWType::Q2K => self.launch_qmv(&self.qmv_q2_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k), batch),
            HipWType::Q3K => self.launch_qmv(&self.qmv_q3_k, stream, w, xq, xsc, y, k, m, ty.row_bytes(k), batch),
        }
    }

    /// Unified prefill matmul (any batch). Uses `qmv` for small batches (<= 8) and tiled `qmm` above.
    pub fn qmm(
        &self,
        ty: HipWType,
        stream: &HipStream,
        w: &HipBuffer,
        x_f32: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        k: u32,
        m: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        self.qmm_with_split(ty, stream, w, x_f32, xq, xsc, y, None, k, m, batch)
    }

    /// Prefill matmul with optional Split-K scratch buffer for small-`m` matrices.
    pub fn qmm_with_split(
        &self,
        ty: HipWType,
        stream: &HipStream,
        w: &HipBuffer,
        x_f32: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        split_buf: Option<&HipBuffer>,
        k: u32,
        m: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        if batch <= 8 {
            return self.qmv(ty, stream, w, x_f32, xq, xsc, y, k, m, batch);
        }
        let func = match ty {
            HipWType::F32 => return self.launch_qmm_float(&self.qmm_f32, stream, w, x_f32, y, k, m, batch),
            HipWType::F16 => return self.launch_qmm_float(&self.qmm_f16, stream, w, x_f32, y, k, m, batch),
            HipWType::Q4_0 => &self.qmm_q4_0,
            HipWType::Q8_0 => &self.qmm_q8_0,
            HipWType::Q4K => &self.qmm_q4_k,
            HipWType::Q5K => &self.qmm_q5_k,
            HipWType::Q6K => &self.qmm_q6_k,
            HipWType::Q2K => &self.qmm_q2_k,
            HipWType::Q3K => &self.qmm_q3_k,
        };
        let gx = m.div_ceil(64);
        let gy = batch.div_ceil(128);
        let kblocks = k / 32;
        let out_elems = (batch as usize) * (m as usize);
        let n_splits = if let Some(sb) = split_buf {
            if gx * gy < 40 && kblocks >= 16 {
                let target = 64u32.div_ceil(gx * gy).clamp(1, 8);
                let max_by_k = (kblocks / 8).max(1);
                let max_by_buf = (sb.size() / (out_elems * 4).max(1)) as u32;
                target.min(max_by_k).min(max_by_buf).max(1)
            } else {
                1
            }
        } else {
            1
        };
        if n_splits > 1 {
            let sb = split_buf.unwrap();
            self.launch_qmm_split(func, stream, w, xq, xsc, sb, k, m, ty.row_bytes(k), batch, n_splits)?;
            let mut pp = sb.as_device_ptr();
            let mut yp = y.as_device_ptr();
            let mut n = out_elems as u32;
            let mut ns = n_splits;
            let mut params: [*mut c_void; 4] = [
                &mut pp as *mut _ as *mut c_void,
                &mut yp as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut ns as *mut _ as *mut c_void,
            ];
            unsafe { self.sum_splits_f32.launch(stream, (n.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
        } else {
            self.launch_qmm_split(func, stream, w, xq, xsc, y, k, m, ty.row_bytes(k), batch, 1)
        }
    }

    pub fn qmv_q8_0(&self, stream: &HipStream, w: &HipBuffer, xq: &HipBuffer, xsc: &HipBuffer, y: &HipBuffer, k: u32, m: u32, batch: u32) -> Result<(), HipError> {
        self.launch_qmv(&self.qmv_q8_0, stream, w, xq, xsc, y, k, m, HipWType::Q8_0.row_bytes(k), batch)
    }

    pub fn qmv_q4_k(&self, stream: &HipStream, w: &HipBuffer, xq: &HipBuffer, xsc: &HipBuffer, y: &HipBuffer, k: u32, m: u32, batch: u32) -> Result<(), HipError> {
        self.launch_qmv(&self.qmv_q4_k, stream, w, xq, xsc, y, k, m, HipWType::Q4K.row_bytes(k), batch)
    }

    pub fn qmm_q8_0(&self, stream: &HipStream, w: &HipBuffer, xq: &HipBuffer, xsc: &HipBuffer, y: &HipBuffer, k: u32, m: u32, batch: u32) -> Result<(), HipError> {
        self.launch_qmm_split(&self.qmm_q8_0, stream, w, xq, xsc, y, k, m, HipWType::Q8_0.row_bytes(k), batch, 1)
    }

    pub fn qmm_q4_k(&self, stream: &HipStream, w: &HipBuffer, xq: &HipBuffer, xsc: &HipBuffer, y: &HipBuffer, k: u32, m: u32, batch: u32) -> Result<(), HipError> {
        self.launch_qmm_split(&self.qmm_q4_k, stream, w, xq, xsc, y, k, m, HipWType::Q4K.row_bytes(k), batch, 1)
    }

    pub fn qmm_q4_k_g256(&self, stream: &HipStream, w: &HipBuffer, xq: &HipBuffer, xsc: &HipBuffer, y: &HipBuffer, k: u32, m: u32, batch: u32) -> Result<(), HipError> {
        self.launch_qmm_nosplit(&self.qmm_q4_k_g256, stream, w, xq, xsc, y, k, m, HipWType::Q4K.row_bytes(k), batch)
    }

    fn launch_qmv_float(
        &self,
        func: &HipFunction,
        stream: &HipStream,
        w: &HipBuffer,
        x: &HipBuffer,
        y: &HipBuffer,
        k: u32,
        m: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        if batch == 0 || batch > 8 {
            return Err(HipError::InvalidArg(format!("qmv batch {batch} must be in 1..=8")));
        }
        let mut wp = w.as_device_ptr();
        let mut xp = x.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut k = k;
        let mut m = m;
        let mut b = batch;
        let mut params: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe { func.launch(stream, (m.div_ceil(4), 1, 1), (32, 4, 1), 0, &mut params) }
    }

    fn launch_qmm_float(
        &self,
        func: &HipFunction,
        stream: &HipStream,
        w: &HipBuffer,
        x: &HipBuffer,
        y: &HipBuffer,
        k: u32,
        m: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut wp = w.as_device_ptr();
        let mut xp = x.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut k = k;
        let mut m = m;
        let mut b = batch;
        let mut params: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe { func.launch(stream, (m.div_ceil(32), batch.div_ceil(32), 1), (16, 16, 1), 0, &mut params) }
    }

    fn launch_qmv1(
        &self,
        func: &HipFunction,
        stream: &HipStream,
        w: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        k: u32,
        m: u32,
        row_bytes: u32,
    ) -> Result<(), HipError> {
        let mut wp = w.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut k = k;
        let mut m = m;
        let mut rb = row_bytes;
        let mut params: [*mut c_void; 7] = [
            &mut wp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
        ];
        unsafe { func.launch(stream, (m.div_ceil(4), 1, 1), (32, 4, 1), 0, &mut params) }
    }

    fn launch_qmv(
        &self,
        func: &HipFunction,
        stream: &HipStream,
        w: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        k: u32,
        m: u32,
        row_bytes: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        if batch == 0 || batch > 8 {
            return Err(HipError::InvalidArg(format!("qmv batch {batch} must be in 1..=8")));
        }
        let mut wp = w.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut k = k;
        let mut m = m;
        let mut rb = row_bytes;
        let mut b = batch;
        let mut params: [*mut c_void; 8] = [
            &mut wp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe { func.launch(stream, (m.div_ceil(4), 1, 1), (32, 4, 1), 0, &mut params) }
    }

    fn launch_qmm_nosplit(
        &self,
        func: &HipFunction,
        stream: &HipStream,
        w: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        k: u32,
        m: u32,
        row_bytes: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut wp = w.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut k = k;
        let mut m = m;
        let mut rb = row_bytes;
        let mut b = batch;
        let mut params: [*mut c_void; 8] = [
            &mut wp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        let gx = m.div_ceil(64);
        let gy = batch.div_ceil(128);
        unsafe { func.launch(stream, (gx, gy, 1), (256, 1, 1), 0, &mut params) }
    }

    fn launch_qmm_split(
        &self,
        func: &HipFunction,
        stream: &HipStream,
        w: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        k: u32,
        m: u32,
        row_bytes: u32,
        batch: u32,
        n_splits: u32,
    ) -> Result<(), HipError> {
        let kblocks = k / 32;
        let kbchunk = kblocks.div_ceil(n_splits);
        let split_stride = if n_splits > 1 { batch * m } else { 0 };
        let mut wp = w.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut k = k;
        let mut m = m;
        let mut rb = row_bytes;
        let mut b = batch;
        let mut kc = kbchunk;
        let mut ss = split_stride;
        let mut params: [*mut c_void; 10] = [
            &mut wp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut ss as *mut _ as *mut c_void,
        ];
        let gx = m.div_ceil(64);
        let gy = batch.div_ceil(128);
        unsafe { func.launch(stream, (gx, gy, n_splits), (256, 1, 1), 0, &mut params) }
    }
}
