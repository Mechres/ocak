//! HIP C++ matvec/matmul kernels for all 9 GGML `IQ*` lookup-table quantization formats:
//! `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`.
//! Compiled on demand via `libhiprtc` (with on-disk HSACO cache) so non-IQ models pay zero
//! compile/load overhead.
use crate::ffi::HipError;
use crate::kernels::HipWType;
use crate::runtime::{HipBuffer, HipContext, HipFunction, HipModule, HipStream};
use std::ffi::c_void;
use std::sync::Arc;

pub const IQ_KERNELS_SRC: &str = r#"
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

__device__ __forceinline__ u32 u16at(const u32* __restrict__ w32, u32 byte_off) {
    return (w32[byte_off >> 2] >> ((byte_off & 2u) << 3)) & 0xFFFFu;
}

__device__ __forceinline__ u32 byte_at(const u32* __restrict__ w32, u32 byte_off) {
    return (w32[byte_off >> 2] >> ((byte_off & 3u) << 3)) & 0xFFu;
}

__device__ __forceinline__ u32 byte_of(u32 w4, u32 i) {
    return (w4 >> (8u * i)) & 0xFFu;
}

__device__ __forceinline__ u32 bytes4(const u32* __restrict__ w32, u32 byte_off) {
    u32 a = byte_off >> 2;
    u32 sh = (byte_off & 3u) << 3;
    return (w32[a] >> sh) | ((w32[a + 1u] << 1u) << (31u - sh));
}

__device__ __forceinline__ u32 sub_bytes(u32 x, u32 c) {
    return ((x | 0x80808080u) - c) ^ 0x80808080u;
}

__device__ __forceinline__ u32 pack4(i32 a, i32 b, i32 c, i32 e) {
    return (u32)(a & 0xFF) | ((u32)(b & 0xFF) << 8)
         | ((u32)(c & 0xFF) << 16) | ((u32)(e & 0xFF) << 24);
}

__device__ __forceinline__ u32 apply_signs4(u32 g, u32 bits) {
    u32 m = ((bits & 0xFu) * 0x00204081u) & 0x01010101u;
    return (g ^ ((m << 8) - m)) + m;
}

__device__ __forceinline__ u32 neg_bytes(u32 g, u32 m) {
    return (g ^ m) + ((m >> 7) & 0x01010101u);
}

__device__ __forceinline__ i32 sdot4(u32 a, u32 b, i32 acc) {
    return __builtin_amdgcn_sdot4((i32)a, (i32)b, acc, false);
}

__device__ __forceinline__ i32 sdot4_zero(u32 a, u32 b) {
    i32 dst;
    asm("v_dot4_i32_i8 %0, %1, %2, 0" : "=v"(dst) : "v"(a), "v"(b));
    return dst;
}

__device__ __forceinline__ void lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// ---------------------------------------------------------------------------
// 1. Block-level (32-weight) unpackers for decode qmv1 / qmv (LDS-staged tables)
// ---------------------------------------------------------------------------

// IQ1_S: stab = tbl[8592..12688] (4096 words)
__device__ __forceinline__ void dqi32_iq1_s(
    const u32* __restrict__ w32, const u32* __restrict__ stab,
    u32 rb, u32 blk, u32 q[8], float& d
) {
    u32 ib = blk & 7u;
    u32 b = rb + (blk >> 3) * 50u;
    u32 qs = bytes4(w32, b + 2u + ib * 4u);
    u32 qhb = u16at(w32, b + 34u + 2u * ib);
    d = f16at(w32, b) * (float)(2u * ((qhb >> 12) & 7u) + 1u) * 0.125f;
    u32 c = ((qhb & 0x8000u) != 0u) ? 0x09090909u : 0x07070707u;
    #pragma unroll
    for (u32 l = 0u; l < 4u; l++) {
        u32 gidx = byte_of(qs, l) | (((qhb >> (3u * l)) & 7u) << 8);
        q[2u * l]      = sub_bytes(stab[2u * gidx], c);
        q[2u * l + 1u] = sub_bytes(stab[2u * gidx + 1u], c);
    }
}

// IQ1_M: stab = tbl[8592..12688] (4096 words), per-16 scales (d0, d1)
__device__ __forceinline__ void dqi32_iq1_m(
    const u32* __restrict__ w32, const u32* __restrict__ stab,
    u32 rb, u32 blk, u32 q[8], float& d0, float& d1
) {
    u32 ib = blk & 7u;
    u32 b = rb + (blk >> 3) * 56u;
    u32 qs = bytes4(w32, b + ib * 4u);
    u32 sca = w32[(b + 48u) >> 2];
    u32 scb = w32[(b + 52u) >> 2];
    u32 sc0 = sca & 0xFFFFu, sc1 = sca >> 16;
    u32 sc2 = scb & 0xFFFFu, sc3 = scb >> 16;
    u32 dbits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) | ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
    float dsup = f16_bits_to_f32((u16)dbits) * 0.125f;
    u32 s16 = (ib < 2u) ? sc0 : ((ib < 4u) ? sc1 : ((ib < 6u) ? sc2 : sc3));
    u32 ssh = 6u * (ib & 1u);
    u32 qb01 = u16at(w32, b + 32u + ib * 2u);
    u32 qb0 = qb01 & 0xFFu, qb1 = qb01 >> 8;
    #pragma unroll
    for (u32 l = 0u; l < 4u; l++) {
        u32 qb = (l < 2u) ? qb0 : qb1;
        u32 hi = ((l & 1u) == 0u) ? ((qb << 8) & 0x700u) : ((qb << 4) & 0x700u);
        u32 gidx = byte_of(qs, l) | hi;
        bool neg = (qb & (((l & 1u) == 0u) ? 0x08u : 0x80u)) != 0u;
        u32 c = neg ? 0x09090909u : 0x07070707u;
        q[2u * l]      = sub_bytes(stab[2u * gidx], c);
        q[2u * l + 1u] = sub_bytes(stab[2u * gidx + 1u], c);
    }
    d0 = dsup * (float)(2u * ((s16 >> ssh) & 7u) + 1u);
    d1 = dsup * (float)(2u * ((s16 >> (ssh + 3u)) & 7u) + 1u);
}

// IQ2_XXS: stab = tbl[1792..2448] (656 words: KSIGNS at 0..128, IQ2XXS_GRID at 144..656)
__device__ __forceinline__ void dqi32_iq2_xxs(
    const u32* __restrict__ w32, const u32* __restrict__ stab,
    u32 rb, u32 blk, u32 q[8], float& d
) {
    u32 ib = blk & 7u;
    u32 b = rb + (blk >> 3) * 66u;
    u32 o = b + 2u + ib * 8u;
    u32 a0 = bytes4(w32, o), a1 = bytes4(w32, o + 4u);
    d = f16at(w32, b) * (0.5f + (float)(a1 >> 28)) * 0.25f;
    #pragma unroll
    for (u32 l = 0u; l < 4u; l++) {
        u32 gidx = byte_of(a0, l);
        u32 sg = stab[(a1 >> (7u * l)) & 127u];
        q[2u * l]      = apply_signs4(stab[144u + 2u * gidx], sg);
        q[2u * l + 1u] = apply_signs4(stab[144u + 2u * gidx + 1u], sg >> 4);
    }
}

// IQ2_XS: stab = tbl[0..1024] (IQ2XS_GRID), smask = tbl[12688..12944] (256 words)
__device__ __forceinline__ void dqi32_iq2_xs(
    const u32* __restrict__ w32, const u32* __restrict__ stab, const u32* __restrict__ smask,
    u32 rb, u32 blk, u32 q[8], float& d0, float& d1
) {
    u32 ib = blk & 7u;
    u32 b = rb + (blk >> 3) * 74u;
    u32 o = b + 2u + ib * 8u;
    u32 w0 = bytes4(w32, o), w1 = bytes4(w32, o + 4u);
    u32 sc = byte_at(w32, b + 66u + ib);
    float d = f16at(w32, b) * 0.25f;
    #pragma unroll
    for (u32 l = 0u; l < 4u; l++) {
        u32 qq = ((l < 2u) ? (w0 >> (16u * l)) : (w1 >> (16u * (l - 2u)))) & 0xFFFFu;
        u32 gidx = qq & 511u;
        u32 si = qq >> 9;
        q[2u * l]      = neg_bytes(stab[2u * gidx],      smask[2u * si]);
        q[2u * l + 1u] = neg_bytes(stab[2u * gidx + 1u], smask[2u * si + 1u]);
    }
    d0 = d * (0.5f + (float)(sc & 0xFu));
    d1 = d * (0.5f + (float)(sc >> 4));
}

// IQ2_S: stab = tbl[2448..4496] (2048 words), smask = tbl[12944..13456] (512 words)
__device__ __forceinline__ void dqi32_iq2_s(
    const u32* __restrict__ w32, const u32* __restrict__ stab, const u32* __restrict__ smask,
    u32 rb, u32 blk, u32 q[8], float& d0, float& d1
) {
    u32 ib = blk & 7u;
    u32 b = rb + (blk >> 3) * 82u;
    u32 qs = bytes4(w32, b + 2u + ib * 4u);
    u32 sgw = bytes4(w32, b + 34u + ib * 4u);
    u32 qhb = byte_at(w32, b + 66u + ib);
    u32 sc = byte_at(w32, b + 74u + ib);
    float d = f16at(w32, b) * 0.25f;
    #pragma unroll
    for (u32 l = 0u; l < 4u; l++) {
        u32 gidx = byte_of(qs, l) | ((qhb << (8u - 2u * l)) & 0x300u);
        u32 si = byte_of(sgw, l);
        q[2u * l]      = neg_bytes(stab[2u * gidx],      smask[2u * si]);
        q[2u * l + 1u] = neg_bytes(stab[2u * gidx + 1u], smask[2u * si + 1u]);
    }
    d0 = d * (0.5f + (float)(sc & 0xFu));
    d1 = d * (0.5f + (float)(sc >> 4));
}

// IQ3_XXS: stab = tbl[1024..1920] (896 words: IQ3XXS_GRID at 0..256, KSIGNS at 768..896)
__device__ __forceinline__ void dqi32_iq3_xxs(
    const u32* __restrict__ w32, const u32* __restrict__ stab,
    u32 rb, u32 blk, u32 q[8], float& d
) {
    u32 ib = blk & 7u;
    u32 b = rb + (blk >> 3) * 98u;
    u32 o = b + 2u + ib * 8u;
    u32 w0 = bytes4(w32, o), w1 = bytes4(w32, o + 4u);
    u32 aux32 = bytes4(w32, b + 66u + ib * 4u);
    d = f16at(w32, b) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    #pragma unroll
    for (u32 l = 0u; l < 4u; l++) {
        u32 ww = (l < 2u) ? w0 : w1;
        u32 sh = 16u * (l & 1u);
        u32 sg = stab[768u + ((aux32 >> (7u * l)) & 127u)];
        q[2u * l]      = apply_signs4(stab[(ww >> sh) & 0xFFu], sg);
        q[2u * l + 1u] = apply_signs4(stab[(ww >> (sh + 8u)) & 0xFFu], sg >> 4);
    }
}

// IQ3_S: stab = tbl[1280..1792] (512 words: IQ3S_GRID)
__device__ __forceinline__ void dqi32_iq3_s(
    const u32* __restrict__ w32, const u32* __restrict__ stab,
    u32 rb, u32 blk, u32 q[8], float& d
) {
    u32 ib = blk & 7u;
    u32 b = rb + (blk >> 3) * 110u;
    u32 o = b + 2u + ib * 8u;
    u32 w0 = bytes4(w32, o), w1 = bytes4(w32, o + 4u);
    u32 sgw = bytes4(w32, b + 74u + ib * 4u);
    u32 qh = byte_at(w32, b + 66u + ib);
    u32 sc = byte_at(w32, b + 106u + (ib >> 1));
    d = f16at(w32, b) * (float)(1u + 2u * (((ib & 1u) == 0u) ? (sc & 0xFu) : (sc >> 4)));
    #pragma unroll
    for (u32 l = 0u; l < 4u; l++) {
        u32 ww = (l < 2u) ? w0 : w1;
        u32 sh = 16u * (l & 1u);
        u32 i1 = ((ww >> sh) & 0xFFu) | ((qh << (8u - 2u * l)) & 256u);
        u32 i2 = ((ww >> (sh + 8u)) & 0xFFu) | ((qh << (7u - 2u * l)) & 256u);
        u32 sg = byte_of(sgw, l);
        q[2u * l]      = apply_signs4(stab[i1], sg);
        q[2u * l + 1u] = apply_signs4(stab[i2], sg >> 4);
    }
}

// IQ4_NL: stab = tbl[1920..1936] (16 words: KVALUES_IQ4NL)
__device__ __forceinline__ void dqi32_iq4_nl(
    const u32* __restrict__ w32, const u32* __restrict__ stab,
    u32 rb, u32 blk, u32 q[8], float& d
) {
    u32 b = rb + blk * 18u;
    d = f16at(w32, b);
    u32 o = b + 2u;
    #pragma unroll
    for (u32 wv = 0u; wv < 4u; wv++) {
        u32 ww = bytes4(w32, o + wv * 4u);
        q[wv] = pack4(
            (i32)stab[ww & 0xFu], (i32)stab[(ww >> 8) & 0xFu],
            (i32)stab[(ww >> 16) & 0xFu], (i32)stab[(ww >> 24) & 0xFu]
        );
        q[wv + 4u] = pack4(
            (i32)stab[(ww >> 4) & 0xFu], (i32)stab[(ww >> 12) & 0xFu],
            (i32)stab[(ww >> 20) & 0xFu], (i32)stab[ww >> 28]
        );
    }
}

// IQ4_XS: stab = tbl[1920..1936] (16 words: KVALUES_IQ4NL)
__device__ __forceinline__ void dqi32_iq4_xs(
    const u32* __restrict__ w32, const u32* __restrict__ stab,
    u32 rb, u32 blk, u32 q[8], float& d
) {
    u32 ib = blk & 7u;
    u32 b = rb + (blk >> 3) * 136u;
    u32 shb = u16at(w32, b + 2u);
    u32 sl = byte_at(w32, b + 4u + (ib >> 1));
    i32 ls = (i32)((sl >> (4u * (ib & 1u))) & 0xFu) | (i32)(((shb >> (2u * ib)) & 3u) << 4);
    d = f16at(w32, b) * (float)(ls - 32);
    u32 qo = (b + 8u + ib * 16u) >> 2;
    #pragma unroll
    for (u32 wv = 0u; wv < 4u; wv++) {
        u32 ww = w32[qo + wv];
        q[wv] = pack4(
            (i32)stab[ww & 0xFu], (i32)stab[(ww >> 8) & 0xFu],
            (i32)stab[(ww >> 16) & 0xFu], (i32)stab[(ww >> 24) & 0xFu]
        );
        q[wv + 4u] = pack4(
            (i32)stab[(ww >> 4) & 0xFu], (i32)stab[(ww >> 12) & 0xFu],
            (i32)stab[(ww >> 20) & 0xFu], (i32)stab[ww >> 28]
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Decode matvec kernels (RPW = 16, LDS-staged tables, batch == 1 and 1..=8)
// ---------------------------------------------------------------------------
#define MAXB 8u
#define RPW 16u

#define DEFINE_IQ_QMV1_S32(NAME, UNPACK, TBL_LO, TBL_N) \
extern "C" __global__ __launch_bounds__(512) \
void NAME( \
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, float* __restrict__ y, \
    const u32* __restrict__ tbl, u32 k, u32 m, u32 row_bytes \
) { \
    __shared__ u32 stab[TBL_N]; \
    u32 tid = threadIdx.x; \
    u32 lid = threadIdx.y * 32u + tid; \
    for (u32 i = lid; i < (TBL_N); i += 32u * RPW) stab[i] = tbl[(TBL_LO) + i]; \
    lds_barrier(); \
    u32 row = blockIdx.x * RPW + threadIdx.y; \
    float acc = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d; \
            UNPACK(w32, stab, rb, blk, q, d); \
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

#define DEFINE_IQ_QMV_S32(NAME, UNPACK, TBL_LO, TBL_N) \
extern "C" __global__ __launch_bounds__(512) \
void NAME( \
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, float* __restrict__ y, \
    const u32* __restrict__ tbl, u32 k, u32 m, u32 row_bytes, u32 batch \
) { \
    __shared__ u32 stab[TBL_N]; \
    u32 tid = threadIdx.x; \
    u32 lid = threadIdx.y * 32u + tid; \
    for (u32 i = lid; i < (TBL_N); i += 32u * RPW) stab[i] = tbl[(TBL_LO) + i]; \
    lds_barrier(); \
    u32 row = blockIdx.x * RPW + threadIdx.y; \
    float acc[MAXB]; \
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d; \
            UNPACK(w32, stab, rb, blk, q, d); \
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

#define DEFINE_IQ_QMV1_S16(NAME, UNPACK, TBL_LO, TBL_N) \
extern "C" __global__ __launch_bounds__(512) \
void NAME( \
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, float* __restrict__ y, \
    const u32* __restrict__ tbl, u32 k, u32 m, u32 row_bytes \
) { \
    __shared__ u32 stab[TBL_N]; \
    u32 tid = threadIdx.x; \
    u32 lid = threadIdx.y * 32u + tid; \
    for (u32 i = lid; i < (TBL_N); i += 32u * RPW) stab[i] = tbl[(TBL_LO) + i]; \
    lds_barrier(); \
    u32 row = blockIdx.x * RPW + threadIdx.y; \
    float acc = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d0, d1; \
            UNPACK(w32, stab, rb, blk, q, d0, d1); \
            u32x4 xa = xq4[blk * 2u]; \
            u32x4 xc = xq4[blk * 2u + 1u]; \
            i32 i0 = sdot4_zero(q[0], xa.x); \
            i0 = sdot4(q[1], xa.y, i0); i0 = sdot4(q[2], xa.z, i0); i0 = sdot4(q[3], xa.w, i0); \
            i32 i1 = sdot4_zero(q[4], xc.x); \
            i1 = sdot4(q[5], xc.y, i1); i1 = sdot4(q[6], xc.z, i1); i1 = sdot4(q[7], xc.w, i1); \
            acc = __fmaf_rn(xsc[blk].x, __fmaf_rn(d0, (float)i0, d1 * (float)i1), acc); \
        } \
    } \
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32); \
    if (tid == 0u && row < m) y[row] = acc; \
}

#define DEFINE_IQ_QMV_S16(NAME, UNPACK, TBL_LO, TBL_N) \
extern "C" __global__ __launch_bounds__(512) \
void NAME( \
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, float* __restrict__ y, \
    const u32* __restrict__ tbl, u32 k, u32 m, u32 row_bytes, u32 batch \
) { \
    __shared__ u32 stab[TBL_N]; \
    u32 tid = threadIdx.x; \
    u32 lid = threadIdx.y * 32u + tid; \
    for (u32 i = lid; i < (TBL_N); i += 32u * RPW) stab[i] = tbl[(TBL_LO) + i]; \
    lds_barrier(); \
    u32 row = blockIdx.x * RPW + threadIdx.y; \
    float acc[MAXB]; \
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d0, d1; \
            UNPACK(w32, stab, rb, blk, q, d0, d1); \
            for (u32 t = 0; t < batch; t++) { \
                u32 xb = t * kblocks + blk; \
                u32x4 xa = xq4[xb * 2u]; \
                u32x4 xc = xq4[xb * 2u + 1u]; \
                i32 i0 = sdot4_zero(q[0], xa.x); \
                i0 = sdot4(q[1], xa.y, i0); i0 = sdot4(q[2], xa.z, i0); i0 = sdot4(q[3], xa.w, i0); \
                i32 i1 = sdot4_zero(q[4], xc.x); \
                i1 = sdot4(q[5], xc.y, i1); i1 = sdot4(q[6], xc.z, i1); i1 = sdot4(q[7], xc.w, i1); \
                acc[t] = __fmaf_rn(xsc[xb].x, __fmaf_rn(d0, (float)i0, d1 * (float)i1), acc[t]); \
            } \
        } \
    } \
    for (u32 t = 0; t < batch; t++) { \
        float v = acc[t]; \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32); \
        if (tid == 0u && row < m) y[(u64)t * m + row] = v; \
    } \
}

#define DEFINE_IQ_QMV1_S16_SM(NAME, UNPACK, TBL_LO, TBL_N, SM_LO, SM_N) \
extern "C" __global__ __launch_bounds__(512) \
void NAME( \
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, float* __restrict__ y, \
    const u32* __restrict__ tbl, u32 k, u32 m, u32 row_bytes \
) { \
    __shared__ u32 stab[TBL_N]; \
    __shared__ u32 smask[SM_N]; \
    u32 tid = threadIdx.x; \
    u32 lid = threadIdx.y * 32u + tid; \
    for (u32 i = lid; i < (TBL_N); i += 32u * RPW) stab[i] = tbl[(TBL_LO) + i]; \
    for (u32 i = lid; i < (SM_N); i += 32u * RPW) smask[i] = tbl[(SM_LO) + i]; \
    lds_barrier(); \
    u32 row = blockIdx.x * RPW + threadIdx.y; \
    float acc = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d0, d1; \
            UNPACK(w32, stab, smask, rb, blk, q, d0, d1); \
            u32x4 xa = xq4[blk * 2u]; \
            u32x4 xc = xq4[blk * 2u + 1u]; \
            i32 i0 = sdot4_zero(q[0], xa.x); \
            i0 = sdot4(q[1], xa.y, i0); i0 = sdot4(q[2], xa.z, i0); i0 = sdot4(q[3], xa.w, i0); \
            i32 i1 = sdot4_zero(q[4], xc.x); \
            i1 = sdot4(q[5], xc.y, i1); i1 = sdot4(q[6], xc.z, i1); i1 = sdot4(q[7], xc.w, i1); \
            acc = __fmaf_rn(xsc[blk].x, __fmaf_rn(d0, (float)i0, d1 * (float)i1), acc); \
        } \
    } \
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32); \
    if (tid == 0u && row < m) y[row] = acc; \
}

#define DEFINE_IQ_QMV_S16_SM(NAME, UNPACK, TBL_LO, TBL_N, SM_LO, SM_N) \
extern "C" __global__ __launch_bounds__(512) \
void NAME( \
    const u32* __restrict__ w32, const u32x4* __restrict__ xq4, \
    const f32x4* __restrict__ xsc, float* __restrict__ y, \
    const u32* __restrict__ tbl, u32 k, u32 m, u32 row_bytes, u32 batch \
) { \
    __shared__ u32 stab[TBL_N]; \
    __shared__ u32 smask[SM_N]; \
    u32 tid = threadIdx.x; \
    u32 lid = threadIdx.y * 32u + tid; \
    for (u32 i = lid; i < (TBL_N); i += 32u * RPW) stab[i] = tbl[(TBL_LO) + i]; \
    for (u32 i = lid; i < (SM_N); i += 32u * RPW) smask[i] = tbl[(SM_LO) + i]; \
    lds_barrier(); \
    u32 row = blockIdx.x * RPW + threadIdx.y; \
    float acc[MAXB]; \
    for (u32 t = 0; t < MAXB; t++) acc[t] = 0.0f; \
    if (row < m) { \
        u32 rb = row * row_bytes; \
        u32 kblocks = k >> 5; \
        for (u32 blk = tid; blk < kblocks; blk += 32u) { \
            u32 q[8]; float d0, d1; \
            UNPACK(w32, stab, smask, rb, blk, q, d0, d1); \
            for (u32 t = 0; t < batch; t++) { \
                u32 xb = t * kblocks + blk; \
                u32x4 xa = xq4[xb * 2u]; \
                u32x4 xc = xq4[xb * 2u + 1u]; \
                i32 i0 = sdot4_zero(q[0], xa.x); \
                i0 = sdot4(q[1], xa.y, i0); i0 = sdot4(q[2], xa.z, i0); i0 = sdot4(q[3], xa.w, i0); \
                i32 i1 = sdot4_zero(q[4], xc.x); \
                i1 = sdot4(q[5], xc.y, i1); i1 = sdot4(q[6], xc.z, i1); i1 = sdot4(q[7], xc.w, i1); \
                acc[t] = __fmaf_rn(xsc[xb].x, __fmaf_rn(d0, (float)i0, d1 * (float)i1), acc[t]); \
            } \
        } \
    } \
    for (u32 t = 0; t < batch; t++) { \
        float v = acc[t]; \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32); \
        if (tid == 0u && row < m) y[(u64)t * m + row] = v; \
    } \
}

DEFINE_IQ_QMV1_S32(qmv1_iq1_s,   dqi32_iq1_s,   8592u, 4096u)
DEFINE_IQ_QMV_S32(qmv_iq1_s,     dqi32_iq1_s,   8592u, 4096u)
DEFINE_IQ_QMV1_S16(qmv1_iq1_m,   dqi32_iq1_m,   8592u, 4096u)
DEFINE_IQ_QMV_S16(qmv_iq1_m,     dqi32_iq1_m,   8592u, 4096u)
DEFINE_IQ_QMV1_S32(qmv1_iq2_xxs, dqi32_iq2_xxs, 1792u, 656u)
DEFINE_IQ_QMV_S32(qmv_iq2_xxs,   dqi32_iq2_xxs, 1792u, 656u)
DEFINE_IQ_QMV1_S16_SM(qmv1_iq2_xs, dqi32_iq2_xs, 0u,    1024u, 12688u, 256u)
DEFINE_IQ_QMV_S16_SM(qmv_iq2_xs,   dqi32_iq2_xs, 0u,    1024u, 12688u, 256u)
DEFINE_IQ_QMV1_S16_SM(qmv1_iq2_s,  dqi32_iq2_s,  2448u, 2048u, 12944u, 512u)
DEFINE_IQ_QMV_S16_SM(qmv_iq2_s,    dqi32_iq2_s,  2448u, 2048u, 12944u, 512u)
DEFINE_IQ_QMV1_S32(qmv1_iq3_xxs, dqi32_iq3_xxs, 1024u, 896u)
DEFINE_IQ_QMV_S32(qmv_iq3_xxs,   dqi32_iq3_xxs, 1024u, 896u)
DEFINE_IQ_QMV1_S32(qmv1_iq3_s,   dqi32_iq3_s,   1280u, 512u)
DEFINE_IQ_QMV_S32(qmv_iq3_s,     dqi32_iq3_s,   1280u, 512u)
DEFINE_IQ_QMV1_S32(qmv1_iq4_nl,  dqi32_iq4_nl,  1920u, 16u)
DEFINE_IQ_QMV_S32(qmv_iq4_nl,    dqi32_iq4_nl,  1920u, 16u)
DEFINE_IQ_QMV1_S32(qmv1_iq4_xs,  dqi32_iq4_xs,  1920u, 16u)
DEFINE_IQ_QMV_S32(qmv_iq4_xs,    dqi32_iq4_xs,  1920u, 16u)

// ---------------------------------------------------------------------------
// 3. 8-weight unpackers for tiled prefill qmm (64x128, BK=32, Split-K)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dqi8_iq1_s(
    const u32* __restrict__ w32, const u32* __restrict__ tbl,
    u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 ib = (k0 >> 5) & 7u;
    u32 l = (k0 >> 3) & 3u;
    u32 b = rb + (k0 >> 8) * 50u;
    u32 qhb = u16at(w32, b + 34u + 2u * ib);
    d = f16at(w32, b) * (float)(2u * ((qhb >> 12) & 7u) + 1u) * 0.125f;
    bool neg = (qhb & 0x8000u) != 0u;
    u32 gidx = byte_at(w32, b + 2u + ib * 4u + l) | (((qhb >> (3u * l)) & 7u) << 8);
    u32 c = neg ? 0x09090909u : 0x07070707u;
    q0 = sub_bytes(tbl[8592u + 2u * gidx], c);
    q1 = sub_bytes(tbl[8592u + 2u * gidx + 1u], c);
}

__device__ __forceinline__ void dqi8_iq1_m(
    const u32* __restrict__ w32, const u32* __restrict__ tbl,
    u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 ib = (k0 >> 5) & 7u;
    u32 l = (k0 >> 3) & 3u;
    u32 b = rb + (k0 >> 8) * 56u;
    u32 sca = w32[(b + 48u) >> 2];
    u32 scb = w32[(b + 52u) >> 2];
    u32 sc0 = sca & 0xFFFFu, sc1 = sca >> 16;
    u32 sc2 = scb & 0xFFFFu, sc3 = scb >> 16;
    u32 dbits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) | ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
    float dsup = f16_bits_to_f32((u16)dbits);
    u32 s16 = (ib < 2u) ? sc0 : ((ib < 4u) ? sc1 : ((ib < 6u) ? sc2 : sc3));
    u32 ssh = 6u * (ib & 1u) + ((l < 2u) ? 0u : 3u);
    d = dsup * (float)(2u * ((s16 >> ssh) & 7u) + 1u) * 0.125f;
    u32 qb = byte_at(w32, b + 32u + ib * 2u + (l >> 1));
    u32 hi = ((l & 1u) == 0u) ? ((qb << 8) & 0x700u) : ((qb << 4) & 0x700u);
    u32 gidx = byte_at(w32, b + ib * 4u + l) | hi;
    bool neg = (qb & (((l & 1u) == 0u) ? 0x08u : 0x80u)) != 0u;
    u32 c = neg ? 0x09090909u : 0x07070707u;
    q0 = sub_bytes(tbl[8592u + 2u * gidx], c);
    q1 = sub_bytes(tbl[8592u + 2u * gidx + 1u], c);
}

__device__ __forceinline__ void dqi8_iq2_xxs(
    const u32* __restrict__ w32, const u32* __restrict__ tbl,
    u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 t = (k0 & 255u) >> 3;
    u32 ib32 = t >> 2;
    u32 l = t & 3u;
    u32 b = rb + (k0 >> 8) * 66u;
    u32 a0 = bytes4(w32, b + 2u + ib32 * 8u);
    u32 a1 = bytes4(w32, b + 6u + ib32 * 8u);
    d = f16at(w32, b) * (0.5f + (float)(a1 >> 28)) * 0.25f;
    u32 gidx = (a0 >> (8u * l)) & 0xFFu;
    u32 sg = tbl[1792u + ((a1 >> (7u * l)) & 127u)];
    q0 = apply_signs4(tbl[1936u + 2u * gidx], sg);
    q1 = apply_signs4(tbl[1936u + 2u * gidx + 1u], sg >> 4);
}

__device__ __forceinline__ void dqi8_iq2_xs(
    const u32* __restrict__ w32, const u32* __restrict__ tbl,
    u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 t = (k0 & 255u) >> 3;
    u32 ib32 = t >> 2;
    u32 l = t & 3u;
    u32 b = rb + (k0 >> 8) * 74u;
    u32 sc = byte_at(w32, b + 66u + ib32);
    d = f16at(w32, b) * (0.5f + (float)((l < 2u) ? (sc & 0xFu) : (sc >> 4))) * 0.25f;
    u32 q = u16at(w32, b + 2u + 2u * t);
    u32 gidx = q & 511u;
    u32 sg = tbl[1792u + (q >> 9)];
    q0 = apply_signs4(tbl[2u * gidx], sg);
    q1 = apply_signs4(tbl[2u * gidx + 1u], sg >> 4);
}

__device__ __forceinline__ void dqi8_iq2_s(
    const u32* __restrict__ w32, const u32* __restrict__ tbl,
    u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 t = (k0 & 255u) >> 3;
    u32 ib32 = t >> 2;
    u32 l = t & 3u;
    u32 b = rb + (k0 >> 8) * 82u;
    u32 sc = byte_at(w32, b + 74u + ib32);
    d = f16at(w32, b) * (0.5f + (float)((l < 2u) ? (sc & 0xFu) : (sc >> 4))) * 0.25f;
    u32 qhb = byte_at(w32, b + 66u + ib32);
    u32 gidx = byte_at(w32, b + 2u + t) | ((qhb << (8u - 2u * l)) & 0x300u);
    u32 sg = byte_at(w32, b + 34u + t);
    q0 = apply_signs4(tbl[2448u + 2u * gidx], sg);
    q1 = apply_signs4(tbl[2448u + 2u * gidx + 1u], sg >> 4);
}

__device__ __forceinline__ void dqi8_iq3_xxs(
    const u32* __restrict__ w32, const u32* __restrict__ tbl,
    u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 t = (k0 & 255u) >> 3;
    u32 ib32 = t >> 2;
    u32 l = t & 3u;
    u32 b = rb + (k0 >> 8) * 98u;
    u32 aux32 = bytes4(w32, b + 66u + ib32 * 4u);
    d = f16at(w32, b) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    u32 sg = tbl[1792u + ((aux32 >> (7u * l)) & 127u)];
    u32 ww = u16at(w32, b + 2u + 2u * t);
    q0 = apply_signs4(tbl[1024u + (ww & 0xFFu)], sg);
    q1 = apply_signs4(tbl[1024u + (ww >> 8)], sg >> 4);
}

__device__ __forceinline__ void dqi8_iq3_s(
    const u32* __restrict__ w32, const u32* __restrict__ tbl,
    u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 t = (k0 & 255u) >> 3;
    u32 ib32 = t >> 2;
    u32 l = t & 3u;
    u32 b = rb + (k0 >> 8) * 110u;
    u32 sc = byte_at(w32, b + 106u + (ib32 >> 1));
    d = f16at(w32, b) * (float)(1u + 2u * (((ib32 & 1u) == 0u) ? (sc & 0xFu) : (sc >> 4)));
    u32 qh = byte_at(w32, b + 66u + ib32);
    u32 ww = u16at(w32, b + 2u + 2u * t);
    u32 i1 = (ww & 0xFFu) | ((qh << (8u - 2u * l)) & 256u);
    u32 i2 = (ww >> 8)   | ((qh << (7u - 2u * l)) & 256u);
    u32 sg = byte_at(w32, b + 74u + t);
    q0 = apply_signs4(tbl[1280u + i1], sg);
    q1 = apply_signs4(tbl[1280u + i2], sg >> 4);
}

__device__ __forceinline__ void dqi8_iq4_nl(
    const u32* __restrict__ w32, const u32* __restrict__ tbl,
    u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 b = rb + (k0 >> 5) * 18u;
    d = f16at(w32, b);
    u32 p = k0 & 31u;
    u32 so = b + 2u + (p & 15u);
    u32 nsh = (p >> 4) * 4u;
    u32 w0 = bytes4(w32, so);
    u32 w1 = bytes4(w32, so + 4u);
    q0 = pack4(
        (i32)tbl[1920u + ((w0 >> nsh) & 0xFu)],
        (i32)tbl[1920u + ((w0 >> (8u + nsh)) & 0xFu)],
        (i32)tbl[1920u + ((w0 >> (16u + nsh)) & 0xFu)],
        (i32)tbl[1920u + ((w0 >> (24u + nsh)) & 0xFu)]
    );
    q1 = pack4(
        (i32)tbl[1920u + ((w1 >> nsh) & 0xFu)],
        (i32)tbl[1920u + ((w1 >> (8u + nsh)) & 0xFu)],
        (i32)tbl[1920u + ((w1 >> (16u + nsh)) & 0xFu)],
        (i32)tbl[1920u + ((w1 >> (24u + nsh)) & 0xFu)]
    );
}

__device__ __forceinline__ void dqi8_iq4_xs(
    const u32* __restrict__ w32, const u32* __restrict__ tbl,
    u32 rb, u32 k0, u32& q0, u32& q1, float& d
) {
    u32 b = rb + (k0 >> 8) * 136u;
    u32 ib = (k0 >> 5) & 7u;
    u32 shb = u16at(w32, b + 2u);
    u32 sl = byte_at(w32, b + 4u + (ib >> 1));
    i32 ls = (i32)((sl >> (4u * (ib & 1u))) & 0xFu) | (i32)(((shb >> (2u * ib)) & 3u) << 4);
    d = f16at(w32, b) * (float)(ls - 32);
    u32 p = k0 & 31u;
    u32 so = (b + 8u + ib * 16u + (p & 15u)) >> 2;
    u32 nsh = (p >> 4) * 4u;
    u32 w0 = w32[so];
    u32 w1 = w32[so + 1u];
    q0 = pack4(
        (i32)tbl[1920u + ((w0 >> nsh) & 0xFu)],
        (i32)tbl[1920u + ((w0 >> (8u + nsh)) & 0xFu)],
        (i32)tbl[1920u + ((w0 >> (16u + nsh)) & 0xFu)],
        (i32)tbl[1920u + ((w0 >> (24u + nsh)) & 0xFu)]
    );
    q1 = pack4(
        (i32)tbl[1920u + ((w1 >> nsh) & 0xFu)],
        (i32)tbl[1920u + ((w1 >> (8u + nsh)) & 0xFu)],
        (i32)tbl[1920u + ((w1 >> (16u + nsh)) & 0xFu)],
        (i32)tbl[1920u + ((w1 >> (24u + nsh)) & 0xFu)]
    );
}

// ---------------------------------------------------------------------------
// 4. Tiled Split-K prefill matmul (64x128, BK=32)
// ---------------------------------------------------------------------------
#define TM 4u
#define BM 64u
#define TN 8u
#define BN 128u

#define DEFINE_IQ_QMM_S32(NAME, UNPACK) \
extern "C" __global__ __launch_bounds__(256, 2) \
void NAME( \
    const u32* __restrict__ w32, const u32* __restrict__ xq, const f32x4* __restrict__ xsc, \
    float* __restrict__ y, const u32* __restrict__ tbl, \
    u32 k, u32 m, u32 row_bytes, u32 batch, u32 kbchunk, u32 split_stride \
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
    UNPACK(w32, tbl, rb, (kb_beg << 5) + ld_part * 8u, pq0, pq1, pd); \
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
            UNPACK(w32, tbl, rb, ((kb + 1u) << 5) + ld_part * 8u, pq0, pq1, pd); \
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

#define DEFINE_IQ_QMM_S16(NAME, UNPACK) \
extern "C" __global__ __launch_bounds__(256, 2) \
void NAME( \
    const u32* __restrict__ w32, const u32* __restrict__ xq, const f32x4* __restrict__ xsc, \
    float* __restrict__ y, const u32* __restrict__ tbl, \
    u32 k, u32 m, u32 row_bytes, u32 batch, u32 kbchunk, u32 split_stride \
) { \
    __shared__ u32x4 ws[8u * (BM / 4u)]; \
    __shared__ u32x4 xs[8u * (BN / 4u)]; \
    __shared__ float wsd[2u * BM]; \
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
    UNPACK(w32, tbl, rb, (kb_beg << 5) + ld_part * 8u, pq0, pq1, pd); \
    for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb_beg) * 8u + x_part * 4u + c]; \
    pxs = xsc[xblk0 + kb_beg].x; \
    for (u32 kb = kb_beg; kb < kb_end; kb++) { \
        u32 kw = ld_part * 2u; \
        ((u32*)&ws[kw * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq0; \
        ((u32*)&ws[(kw + 1u) * (BM / 4u) + (ld_row >> 2)])[ld_row & 3u] = pq1; \
        if ((ld_part & 1u) == 0u) wsd[(ld_part >> 1) * BM + ld_row] = pd; \
        for (u32 c = 0; c < 4u; c++) \
            ((u32*)&xs[(x_part * 4u + c) * (BN / 4u) + (x_col >> 2)])[x_col & 3u] = pxv[c]; \
        if (x_part == 0u) xsd[x_col] = pxs; \
        lds_barrier(); \
        if (kb + 1u < kb_end) { \
            UNPACK(w32, tbl, rb, ((kb + 1u) << 5) + ld_part * 8u, pq0, pq1, pd); \
            for (u32 c = 0; c < 4u; c++) pxv[c] = xq[(xblk0 + kb + 1u) * 8u + x_part * 4u + c]; \
            pxs = xsc[xblk0 + kb + 1u].x; \
        } \
        const u32x4* wp = &ws[tx]; const u32x4* xp = &xs[ty * 2u]; \
        for (u32 h = 0; h < 2u; h++) { \
            i32 ia[TN][TM]; \
            for (u32 c = 0; c < 4u; c++) { \
                u32 kk = h * 4u + c; \
                u32x4 wv = wp[kk * (BM / 4u)]; \
                u32x4 xv0 = xp[kk * (BN / 4u)]; u32x4 xv1 = xp[kk * (BN / 4u) + 1u]; \
                u32 xv[8] = { xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w }; \
                for (u32 j = 0; j < TN; j++) { \
                    if (c == 0u) { \
                        ia[j][0] = sdot4_zero(wv.x, xv[j]); ia[j][1] = sdot4_zero(wv.y, xv[j]); \
                        ia[j][2] = sdot4_zero(wv.z, xv[j]); ia[j][3] = sdot4_zero(wv.w, xv[j]); \
                    } else { \
                        ia[j][0] = sdot4(wv.x, xv[j], ia[j][0]); ia[j][1] = sdot4(wv.y, xv[j], ia[j][1]); \
                        ia[j][2] = sdot4(wv.z, xv[j], ia[j][2]); ia[j][3] = sdot4(wv.w, xv[j], ia[j][3]); \
                    } \
                } \
            } \
            float wd[4]; for (u32 i = 0; i < 4u; i++) wd[i] = wsd[h * BM + tx * 4u + i]; \
            for (u32 j = 0; j < TN; j++) { \
                float dx = xsd[ty * TN + j]; \
                for (u32 i = 0; i < 4u; i++) acc[j][i] = __fmaf_rn(wd[i] * dx, (float)ia[j][i], acc[j][i]); \
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

DEFINE_IQ_QMM_S32(qmm_iq1_s,   dqi8_iq1_s)
DEFINE_IQ_QMM_S16(qmm_iq1_m,   dqi8_iq1_m)
DEFINE_IQ_QMM_S32(qmm_iq2_xxs, dqi8_iq2_xxs)
DEFINE_IQ_QMM_S16(qmm_iq2_xs,  dqi8_iq2_xs)
DEFINE_IQ_QMM_S16(qmm_iq2_s,   dqi8_iq2_s)
DEFINE_IQ_QMM_S32(qmm_iq3_xxs, dqi8_iq3_xxs)
DEFINE_IQ_QMM_S32(qmm_iq3_s,   dqi8_iq3_s)
DEFINE_IQ_QMM_S32(qmm_iq4_nl,  dqi8_iq4_nl)
DEFINE_IQ_QMM_S32(qmm_iq4_xs,  dqi8_iq4_xs)

extern "C" __global__ __launch_bounds__(256)
void iq_sum_splits_f32(
    const float* __restrict__ part,
    float* __restrict__ out,
    u32 n,
    u32 n_splits
) {
    u32 i = blockIdx.x * 256u + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (u32 z = 0u; z < n_splits; z++) s += part[(u64)z * n + i];
    out[i] = s;
}
"#;

pub struct HipIqKernels {
    _module: Arc<HipModule>,
    qmv1_iq1_s: HipFunction,
    qmv1_iq1_m: HipFunction,
    qmv1_iq2_xxs: HipFunction,
    qmv1_iq2_xs: HipFunction,
    qmv1_iq2_s: HipFunction,
    qmv1_iq3_xxs: HipFunction,
    qmv1_iq3_s: HipFunction,
    qmv1_iq4_nl: HipFunction,
    qmv1_iq4_xs: HipFunction,
    qmv_iq1_s: HipFunction,
    qmv_iq1_m: HipFunction,
    qmv_iq2_xxs: HipFunction,
    qmv_iq2_xs: HipFunction,
    qmv_iq2_s: HipFunction,
    qmv_iq3_xxs: HipFunction,
    qmv_iq3_s: HipFunction,
    qmv_iq4_nl: HipFunction,
    qmv_iq4_xs: HipFunction,
    qmm_iq1_s: HipFunction,
    qmm_iq1_m: HipFunction,
    qmm_iq2_xxs: HipFunction,
    qmm_iq2_xs: HipFunction,
    qmm_iq2_s: HipFunction,
    qmm_iq3_xxs: HipFunction,
    qmm_iq3_s: HipFunction,
    qmm_iq4_nl: HipFunction,
    qmm_iq4_xs: HipFunction,
    sum_splits_f32: HipFunction,
}

impl HipIqKernels {
    pub fn compile(ctx: &Arc<HipContext>) -> Result<Self, HipError> {
        let module = Arc::new(ctx.compile_module(IQ_KERNELS_SRC, "ocak_hip_iq_kernels.hip", &[])?);
        let get = |name: &str| module.function(name);
        Ok(Self {
            qmv1_iq1_s: get("qmv1_iq1_s")?,
            qmv1_iq1_m: get("qmv1_iq1_m")?,
            qmv1_iq2_xxs: get("qmv1_iq2_xxs")?,
            qmv1_iq2_xs: get("qmv1_iq2_xs")?,
            qmv1_iq2_s: get("qmv1_iq2_s")?,
            qmv1_iq3_xxs: get("qmv1_iq3_xxs")?,
            qmv1_iq3_s: get("qmv1_iq3_s")?,
            qmv1_iq4_nl: get("qmv1_iq4_nl")?,
            qmv1_iq4_xs: get("qmv1_iq4_xs")?,
            qmv_iq1_s: get("qmv_iq1_s")?,
            qmv_iq1_m: get("qmv_iq1_m")?,
            qmv_iq2_xxs: get("qmv_iq2_xxs")?,
            qmv_iq2_xs: get("qmv_iq2_xs")?,
            qmv_iq2_s: get("qmv_iq2_s")?,
            qmv_iq3_xxs: get("qmv_iq3_xxs")?,
            qmv_iq3_s: get("qmv_iq3_s")?,
            qmv_iq4_nl: get("qmv_iq4_nl")?,
            qmv_iq4_xs: get("qmv_iq4_xs")?,
            qmm_iq1_s: get("qmm_iq1_s")?,
            qmm_iq1_m: get("qmm_iq1_m")?,
            qmm_iq2_xxs: get("qmm_iq2_xxs")?,
            qmm_iq2_xs: get("qmm_iq2_xs")?,
            qmm_iq2_s: get("qmm_iq2_s")?,
            qmm_iq3_xxs: get("qmm_iq3_xxs")?,
            qmm_iq3_s: get("qmm_iq3_s")?,
            qmm_iq4_nl: get("qmm_iq4_nl")?,
            qmm_iq4_xs: get("qmm_iq4_xs")?,
            sum_splits_f32: get("iq_sum_splits_f32")?,
            _module: module,
        })
    }

    pub fn qmv(
        &self,
        ty: HipWType,
        stream: &HipStream,
        w: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        tbl: &HipBuffer,
        k: u32,
        m: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let rb = ty.row_bytes(k);
        if batch == 1 {
            let func = match ty {
                HipWType::IQ1S => &self.qmv1_iq1_s,
                HipWType::IQ1M => &self.qmv1_iq1_m,
                HipWType::IQ2XXS => &self.qmv1_iq2_xxs,
                HipWType::IQ2XS => &self.qmv1_iq2_xs,
                HipWType::IQ2S => &self.qmv1_iq2_s,
                HipWType::IQ3XXS => &self.qmv1_iq3_xxs,
                HipWType::IQ3S => &self.qmv1_iq3_s,
                HipWType::IQ4NL => &self.qmv1_iq4_nl,
                HipWType::IQ4XS => &self.qmv1_iq4_xs,
                _ => return Err(HipError::InvalidArg(format!("non-IQ type {ty:?} passed to HipIqKernels::qmv"))),
            };
            return self.launch_qmv1(func, stream, w, xq, xsc, y, tbl, k, m, rb);
        }
        let func = match ty {
            HipWType::IQ1S => &self.qmv_iq1_s,
            HipWType::IQ1M => &self.qmv_iq1_m,
            HipWType::IQ2XXS => &self.qmv_iq2_xxs,
            HipWType::IQ2XS => &self.qmv_iq2_xs,
            HipWType::IQ2S => &self.qmv_iq2_s,
            HipWType::IQ3XXS => &self.qmv_iq3_xxs,
            HipWType::IQ3S => &self.qmv_iq3_s,
            HipWType::IQ4NL => &self.qmv_iq4_nl,
            HipWType::IQ4XS => &self.qmv_iq4_xs,
            _ => return Err(HipError::InvalidArg(format!("non-IQ type {ty:?} passed to HipIqKernels::qmv"))),
        };
        self.launch_qmv(func, stream, w, xq, xsc, y, tbl, k, m, rb, batch)
    }

    pub fn qmm_with_split(
        &self,
        ty: HipWType,
        stream: &HipStream,
        w: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        tbl: &HipBuffer,
        split_buf: Option<&HipBuffer>,
        k: u32,
        m: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        if batch <= 8 {
            return self.qmv(ty, stream, w, xq, xsc, y, tbl, k, m, batch);
        }
        let func = match ty {
            HipWType::IQ1S => &self.qmm_iq1_s,
            HipWType::IQ1M => &self.qmm_iq1_m,
            HipWType::IQ2XXS => &self.qmm_iq2_xxs,
            HipWType::IQ2XS => &self.qmm_iq2_xs,
            HipWType::IQ2S => &self.qmm_iq2_s,
            HipWType::IQ3XXS => &self.qmm_iq3_xxs,
            HipWType::IQ3S => &self.qmm_iq3_s,
            HipWType::IQ4NL => &self.qmm_iq4_nl,
            HipWType::IQ4XS => &self.qmm_iq4_xs,
            _ => return Err(HipError::InvalidArg(format!("non-IQ type {ty:?} passed to HipIqKernels::qmm"))),
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
            self.launch_qmm_split(func, stream, w, xq, xsc, sb, tbl, k, m, ty.row_bytes(k), batch, n_splits)?;
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
            self.launch_qmm_split(func, stream, w, xq, xsc, y, tbl, k, m, ty.row_bytes(k), batch, 1)
        }
    }

    fn launch_qmv1(
        &self,
        func: &HipFunction,
        stream: &HipStream,
        w: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        tbl: &HipBuffer,
        k: u32,
        m: u32,
        row_bytes: u32,
    ) -> Result<(), HipError> {
        let mut wp = w.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut tp = tbl.as_device_ptr();
        let mut k = k;
        let mut m = m;
        let mut rb = row_bytes;
        let mut params: [*mut c_void; 8] = [
            &mut wp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
        ];
        unsafe { func.launch(stream, (m.div_ceil(16), 1, 1), (32, 16, 1), 0, &mut params) }
    }

    fn launch_qmv(
        &self,
        func: &HipFunction,
        stream: &HipStream,
        w: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        tbl: &HipBuffer,
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
        let mut tp = tbl.as_device_ptr();
        let mut k = k;
        let mut m = m;
        let mut rb = row_bytes;
        let mut b = batch;
        let mut params: [*mut c_void; 9] = [
            &mut wp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe { func.launch(stream, (m.div_ceil(16), 1, 1), (32, 16, 1), 0, &mut params) }
    }

    fn launch_qmm_split(
        &self,
        func: &HipFunction,
        stream: &HipStream,
        w: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        y: &HipBuffer,
        tbl: &HipBuffer,
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
        let mut tp = tbl.as_device_ptr();
        let mut k = k;
        let mut m = m;
        let mut rb = row_bytes;
        let mut b = batch;
        let mut kc = kbchunk;
        let mut ss = split_stride;
        let mut params: [*mut c_void; 11] = [
            &mut wp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
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
