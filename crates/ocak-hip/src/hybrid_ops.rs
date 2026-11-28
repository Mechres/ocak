//! Hybrid recurrent/gated HIP kernels for `LFM2`, `LFM3`, and `Qwen3.5`:
//! `shortconv_step`, `ssm_conv_silu`, `deltanet_step`, `gated_rms_norm`,
//! `split_qg`, `mul_sigmoid`, `mul_head_sigmoid`, and `rope_ext` (with `rope_freqs`).
use crate::ffi::HipError;
use crate::runtime::{HipBuffer, HipContext, HipFunction, HipModule, HipStream};
use std::ffi::c_void;
use std::sync::Arc;

pub const HYBRID_OPS_SRC: &str = r#"
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

struct __align__(16) f32x4 { float x, y, z, w; };
struct __align__(8)  f32x2 { float x, y; };

__device__ __forceinline__ u16 f32_to_f16(float x) {
    union { u16 u; _Float16 f; } v;
    v.f = (_Float16)x;
    return v.u;
}

__device__ __forceinline__ void lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__device__ __forceinline__ void warp_quant_q8_block(
    float v, u32 lane, u32 blk, u32* __restrict__ xq, f32x4* __restrict__ xsc
) {
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    float dx = amax * (1.0f / 127.0f);
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = min(127, max(-127, __float2int_rn(v * inv)));
    int q1 = __shfl_down(q, 1, 32);
    int q2 = __shfl_down(q, 2, 32);
    int q3 = __shfl_down(q, 3, 32);
    if ((lane & 3u) == 0u) {
        xq[blk * 8u + (lane >> 2)] = (u32)(q & 0xFF)
            | ((u32)(q1 & 0xFF) << 8)
            | ((u32)(q2 & 0xFF) << 16)
            | ((u32)(q3 & 0xFF) << 24);
    }
    int s0 = (lane < 16u) ? q : 0;
    int s1 = (lane < 16u) ? 0 : q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s0 += __shfl_xor(s0, off, 32);
        s1 += __shfl_xor(s1, off, 32);
    }
    if (lane == 0u) xsc[blk] = { dx, dx * (float)s0, dx * (float)s1, 0.0f };
}

// ---------------------------------------------------------------------------
// 1. LFM2 / LFM3 ShortConv: o = c * conv(b * x), state in registers across batch
// ---------------------------------------------------------------------------
#define MAXD 8u

extern "C" __global__ __launch_bounds__(256)
void shortconv_step_f32(
    const float* __restrict__ bcx,
    const float* __restrict__ cw,
    float* __restrict__ st,
    float* __restrict__ o,
    u32 n_embd,
    u32 l_cache,
    u32 batch
) {
    u32 e = blockIdx.x * 256u + threadIdx.x;
    if (e >= n_embd) return;
    u32 d_conv = l_cache - 1u;
    u32 sb = e * d_conv;
    u32 wb = e * l_cache;
    float h[MAXD];
    float w[MAXD + 1u];
    #pragma unroll
    for (u32 t = 0u; t < MAXD; t++) h[t] = (t < d_conv) ? st[sb + t] : 0.0f;
    #pragma unroll
    for (u32 t = 0u; t <= MAXD; t++) w[t] = (t <= d_conv) ? cw[wb + t] : 0.0f;

    for (u32 ti = 0u; ti < batch; ti++) {
        u64 bo = (u64)ti * 3u * n_embd;
        float bx = bcx[bo + e] * bcx[bo + 2u * n_embd + e];
        float c = bcx[bo + n_embd + e];
        float acc = bx * w[d_conv];
        #pragma unroll
        for (u32 t = 0u; t < MAXD; t++) if (t < d_conv) acc = __fmaf_rn(h[t], w[t], acc);
        #pragma unroll
        for (u32 t = 0u; t + 1u < MAXD; t++) if (t + 1u < d_conv) h[t] = h[t + 1u];
        #pragma unroll
        for (u32 t = 0u; t < MAXD; t++) if (t == d_conv - 1u) h[t] = bx;
        o[(u64)ti * n_embd + e] = c * acc;
    }
    #pragma unroll
    for (u32 t = 0u; t < MAXD; t++) if (t < d_conv) st[sb + t] = h[t];
}

// 1-token ShortConv + fused Q8 activation quantize (1 warp per 32-ch block, 8 blocks/WG).
extern "C" __global__ __launch_bounds__(256)
void shortconv_step_q8_f32(
    const float* __restrict__ bcx,
    const float* __restrict__ cw,
    float* __restrict__ st,
    float* __restrict__ o,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 n_embd,
    u32 l_cache
) {
    u32 lid = threadIdx.x;
    u32 warp = lid >> 5;
    u32 lane = lid & 31u;
    u32 blk = blockIdx.x * 8u + warp;
    if (blk >= (n_embd >> 5)) return;
    u32 e = blk * 32u + lane;
    u32 d_conv = l_cache - 1u;
    u32 sb = e * d_conv;
    u32 wb = e * l_cache;
    float bx = bcx[e] * bcx[2u * n_embd + e];
    float c = bcx[n_embd + e];
    float acc = bx * cw[wb + d_conv];
    for (u32 t = 0u; t < d_conv; t++) acc = __fmaf_rn(st[sb + t], cw[wb + t], acc);
    for (u32 t = 0u; t + 1u < d_conv; t++) st[sb + t] = st[sb + t + 1u];
    st[sb + d_conv - 1u] = bx;
    float v = c * acc;
    o[e] = v;
    warp_quant_q8_block(v, lane, blk, xq, xsc);
}

// ---------------------------------------------------------------------------
// 2. Qwen3.5 Causal depthwise conv + SiLU
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ssm_conv_silu_f32(
    const float* __restrict__ x,
    const float* __restrict__ cw,
    float* __restrict__ st,
    float* __restrict__ o,
    u32 channels,
    u32 k_conv,
    u32 batch
) {
    u32 ch = blockIdx.x * 256u + threadIdx.x;
    if (ch >= channels) return;
    u32 d_conv = k_conv - 1u;
    u32 sb = ch * d_conv;
    u32 wb = ch * k_conv;
    float h[MAXD];
    float w[MAXD + 1u];
    #pragma unroll
    for (u32 t = 0u; t < MAXD; t++) h[t] = (t < d_conv) ? st[sb + t] : 0.0f;
    #pragma unroll
    for (u32 t = 0u; t <= MAXD; t++) w[t] = (t <= d_conv) ? cw[wb + t] : 0.0f;

    for (u32 ti = 0u; ti < batch; ti++) {
        float xv = x[(u64)ti * channels + ch];
        float acc = xv * w[d_conv];
        #pragma unroll
        for (u32 t = 0u; t < MAXD; t++) if (t < d_conv) acc = __fmaf_rn(h[t], w[t], acc);
        #pragma unroll
        for (u32 t = 0u; t + 1u < MAXD; t++) if (t + 1u < d_conv) h[t] = h[t + 1u];
        #pragma unroll
        for (u32 t = 0u; t < MAXD; t++) if (t == d_conv - 1u) h[t] = xv;
        o[(u64)ti * channels + ch] = acc / (1.0f + __expf(-acc));
    }
    #pragma unroll
    for (u32 t = 0u; t < MAXD; t++) if (t < d_conv) st[sb + t] = h[t];
}

// ---------------------------------------------------------------------------
// 3. Qwen3.5 Gated DeltaNet recurrence (matches ocak-vk deltanet_step.comp)
//    8-lane cluster per state row, 32 rows/WG, TC=16 tokens staged in LDS.
// ---------------------------------------------------------------------------
#define DN_RPW  32u
#define DN_CL   8u
#define DN_MAXR 16u
#define DN_TC   16u

extern "C" __global__ __launch_bounds__(256)
void deltanet_step_f32(
    const float* __restrict__ qkv,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ st,
    float* __restrict__ o,
    u32 n_k_heads,
    u32 n_v_heads,
    u32 head_dim,
    float eps,
    u32 batch
) {
    __shared__ float sq[DN_TC * 128u];
    __shared__ float sk[DN_TC * 128u];
    __shared__ float sv[DN_TC * DN_RPW];
    __shared__ f32x4 sc[DN_TC];
    __shared__ f32x2 sgb[DN_TC];

    u32 tid = threadIdx.x;
    u32 lane = tid & 7u;
    u32 c = tid >> 3;
    u32 hd = head_dim;
    u32 row0 = blockIdx.x * DN_RPW;
    u32 h = row0 / hd;
    if (h >= n_v_heads) return;
    u32 j0 = row0 % hd;
    u32 j = j0 + c;
    u32 kh = h % n_k_heads;
    u32 k_total = n_k_heads * hd;
    u32 v_total = n_v_heads * hd;
    u32 conv_ch = 2u * k_total + v_total;

    u64 s_base = ((u64)h * hd + j) * hd;
    float s[DN_MAXR];
    #pragma unroll
    for (u32 r = 0u; r < DN_MAXR; r++) {
        u32 i = lane + DN_CL * r;
        s[r] = (i < hd) ? st[s_base + i] : 0.0f;
    }
    float dt_b = dt_bias[h];
    float sa_h = ssm_a[h];
    float inv_sqrt_hd = rsqrtf((float)hd);

    for (u32 t0 = 0u; t0 < batch; t0 += DN_TC) {
        u32 nt = min(DN_TC, batch - t0);
        for (u32 idx = tid; idx < nt * hd; idx += 256u) {
            u32 tt = idx / hd;
            u32 i = idx % hd;
            u64 qo = (u64)(t0 + tt) * conv_ch + (u64)kh * hd + i;
            sq[tt * 128u + i] = qkv[qo];
            sk[tt * 128u + i] = qkv[qo + k_total];
        }
        for (u32 idx = tid; idx < nt * DN_RPW; idx += 256u) {
            u32 tt = idx / DN_RPW;
            sv[idx] = qkv[(u64)(t0 + tt) * conv_ch + 2u * k_total + (u64)h * hd + j0 + (idx % DN_RPW)];
        }
        if (tid < nt) {
            u32 t = t0 + tid;
            float ab = alpha[(u64)t * n_v_heads + h] + dt_b;
            float sp = (ab > 20.0f) ? ab : logf(1.0f + expf(ab));
            sgb[tid] = { expf(sp * sa_h), 1.0f / (1.0f + expf(-beta[(u64)t * n_v_heads + h])) };
        }
        lds_barrier();

        for (u32 tt = c; tt < nt; tt += DN_RPW) {
            float qq = 0.0f, kk = 0.0f, kq = 0.0f;
            #pragma unroll
            for (u32 r = 0u; r < DN_MAXR; r++) {
                u32 i = lane + DN_CL * r;
                float qv = (i < hd) ? sq[tt * 128u + i] : 0.0f;
                float kv = (i < hd) ? sk[tt * 128u + i] : 0.0f;
                qq = __fmaf_rn(qv, qv, qq);
                kk = __fmaf_rn(kv, kv, kk);
                kq = __fmaf_rn(kv, qv, kq);
            }
            qq += __shfl_xor(qq, 4, 8); qq += __shfl_xor(qq, 2, 8); qq += __shfl_xor(qq, 1, 8);
            kk += __shfl_xor(kk, 4, 8); kk += __shfl_xor(kk, 2, 8); kk += __shfl_xor(kk, 1, 8);
            kq += __shfl_xor(kq, 4, 8); kq += __shfl_xor(kq, 2, 8); kq += __shfl_xor(kq, 1, 8);
            if (lane == 0u) {
                sc[tt] = { rsqrtf(qq + eps) * inv_sqrt_hd, rsqrtf(kk + eps), kq, 0.0f };
            }
        }
        lds_barrier();

        for (u32 tt = 0u; tt < nt; tt++) {
            float k[DN_MAXR];
            float sk0 = 0.0f, sq0 = 0.0f;
            #pragma unroll
            for (u32 r = 0u; r < DN_MAXR; r++) {
                u32 i = lane + DN_CL * r;
                float qv = (i < hd) ? sq[tt * 128u + i] : 0.0f;
                k[r] = (i < hd) ? sk[tt * 128u + i] : 0.0f;
                sk0 = __fmaf_rn(s[r], k[r], sk0);
                sq0 = __fmaf_rn(s[r], qv, sq0);
            }
            sk0 += __shfl_xor(sk0, 4, 8); sk0 += __shfl_xor(sk0, 2, 8); sk0 += __shfl_xor(sk0, 1, 8);
            sq0 += __shfl_xor(sq0, 4, 8); sq0 += __shfl_xor(sq0, 2, 8); sq0 += __shfl_xor(sq0, 1, 8);
            f32x4 scl = sc[tt];
            f32x2 gb = sgb[tt];
            float g = gb.x;
            float dj = (sv[tt * DN_RPW + c] - g * sk0 * scl.y) * gb.y;
            float kd = scl.y * dj;
            #pragma unroll
            for (u32 r = 0u; r < DN_MAXR; r++) s[r] = __fmaf_rn(k[r], kd, g * s[r]);
            if (lane == 0u) {
                o[(u64)(t0 + tt) * v_total + (u64)h * hd + j] = (g * sq0 + dj * scl.z * scl.y) * scl.x;
            }
        }
        lds_barrier();
    }
    #pragma unroll
    for (u32 r = 0u; r < DN_MAXR; r++) {
        u32 i = lane + DN_CL * r;
        if (i < hd) st[s_base + i] = s[r];
    }
}

// 3b. Dedicated 1-token / small-batch DeltaNet (32 lanes/row, 0 LDS, 0 barriers).
extern "C" __global__ __launch_bounds__(256)
void deltanet_step_tok_f32(
    const float* __restrict__ qkv,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ st,
    float* __restrict__ o,
    u32 n_k_heads,
    u32 n_v_heads,
    u32 head_dim,
    float eps,
    u32 batch
) {
    u32 lane = threadIdx.x & 31u;
    u32 row = blockIdx.x * 8u + (threadIdx.x >> 5);
    u32 hd = head_dim;
    u32 h = row / hd;
    u32 j = row % hd;
    if (h >= n_v_heads) return;
    u32 kh = h % n_k_heads;
    u32 k_total = n_k_heads * hd;
    u32 v_total = n_v_heads * hd;
    u32 conv_ch = 2u * k_total + v_total;

    u64 s_base = ((u64)h * hd + j) * hd;
    float s[4];
    #pragma unroll
    for (u32 r = 0u; r < 4u; r++) {
        u32 i = lane + 32u * r;
        s[r] = (i < hd) ? st[s_base + i] : 0.0f;
    }
    float dt_b = dt_bias[h];
    float sa_h = ssm_a[h];
    float inv_sqrt_hd = rsqrtf((float)hd);

    for (u32 t = 0u; t < batch; t++) {
        u64 qo = (u64)t * conv_ch + (u64)kh * hd;
        u64 ko = qo + k_total;
        float q[4], k[4];
        float qq = 0.0f, kk = 0.0f;
        #pragma unroll
        for (u32 r = 0u; r < 4u; r++) {
            u32 i = lane + 32u * r;
            q[r] = (i < hd) ? qkv[qo + i] : 0.0f;
            k[r] = (i < hd) ? qkv[ko + i] : 0.0f;
            qq = __fmaf_rn(q[r], q[r], qq);
            kk = __fmaf_rn(k[r], k[r], kk);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            qq += __shfl_xor(qq, off, 32);
            kk += __shfl_xor(kk, off, 32);
        }
        float qs = rsqrtf(qq + eps) * inv_sqrt_hd;
        float ks = rsqrtf(kk + eps);

        float ab = alpha[(u64)t * n_v_heads + h] + dt_b;
        float sp = (ab > 20.0f) ? ab : logf(1.0f + expf(ab));
        float g = expf(sp * sa_h);
        float b = 1.0f / (1.0f + expf(-beta[(u64)t * n_v_heads + h]));

        float sk = 0.0f;
        #pragma unroll
        for (u32 r = 0u; r < 4u; r++) {
            s[r] *= g;
            sk = __fmaf_rn(s[r], k[r], sk);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sk += __shfl_xor(sk, off, 32);
        sk *= ks;

        float vj = qkv[(u64)t * conv_ch + 2u * k_total + (u64)h * hd + j];
        float dj = (vj - sk) * b;
        float kd = ks * dj;
        float sq = 0.0f;
        #pragma unroll
        for (u32 r = 0u; r < 4u; r++) {
            s[r] = __fmaf_rn(k[r], kd, s[r]);
            sq = __fmaf_rn(s[r], q[r], sq);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off, 32);
        if (lane == 0u) o[(u64)t * v_total + (u64)h * hd + j] = sq * qs;
    }
    #pragma unroll
    for (u32 r = 0u; r < 4u; r++) {
        u32 i = lane + 32u * r;
        if (i < hd) st[s_base + i] = s[r];
    }
}

// ---------------------------------------------------------------------------
// 4. Gated RMSNorm, split_qg, mul_sigmoid, mul_head_sigmoid, rope_ext
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(128)
void gated_rms_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ z,
    float* __restrict__ o,
    u32 n_heads,
    u32 head_dim,
    float eps
) {
    __shared__ float smem[4];
    u32 h = blockIdx.x;
    u32 b = blockIdx.y;
    if (h >= n_heads) return;
    u32 tid = threadIdx.x;
    u64 base = ((u64)b * n_heads + h) * head_dim;
    float acc = 0.0f;
    for (u32 d = tid; d < head_dim; d += 128u) {
        float v = x[base + d];
        acc = __fmaf_rn(v, v, acc);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if ((tid & 31u) == 0u) smem[tid >> 5] = acc;
    lds_barrier();
    float tot = smem[0] + smem[1] + smem[2] + smem[3];
    float sc = rsqrtf(tot / (float)head_dim + eps);
    for (u32 d = tid; d < head_dim; d += 128u) {
        float zv = z[base + d];
        float silu = zv / (1.0f + __expf(-zv));
        o[base + d] = x[base + d] * sc * w[d] * silu;
    }
}

// 1-token gated RMSNorm + fused Q8 activation quantize (head_dim % 32 == 0).
extern "C" __global__ __launch_bounds__(128)
void gated_rms_norm_q8_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ z,
    float* __restrict__ o,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 n_heads,
    u32 head_dim,
    float eps
) {
    __shared__ float smem[4];
    u32 h = blockIdx.x;
    if (h >= n_heads) return;
    u32 tid = threadIdx.x;
    u32 warp = tid >> 5;
    u32 lane = tid & 31u;
    u64 base = (u64)h * head_dim;
    float acc = 0.0f;
    for (u32 d = tid; d < head_dim; d += 128u) {
        float v = x[base + d];
        acc = __fmaf_rn(v, v, acc);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0u) smem[warp] = acc;
    lds_barrier();
    float tot = smem[0] + smem[1] + smem[2] + smem[3];
    float sc = rsqrtf(tot / (float)head_dim + eps);
    u32 bph = head_dim >> 5;
    for (u32 b = warp; b < bph; b += 4u) {
        u32 d = b * 32u + lane;
        float zv = z[base + d];
        float silu = zv / (1.0f + __expf(-zv));
        float v = x[base + d] * sc * w[d] * silu;
        o[base + d] = v;
        warp_quant_q8_block(v, lane, h * bph + b, xq, xsc);
    }
}

// 1-token mul_sigmoid + fused Q8 quantize (1 warp per 32-element block, 8 blocks/WG).
extern "C" __global__ __launch_bounds__(256)
void mul_sigmoid_q8_f32(
    float* __restrict__ x,
    const float* __restrict__ g,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 n
) {
    u32 lid = threadIdx.x;
    u32 warp = lid >> 5;
    u32 lane = lid & 31u;
    u32 blk = blockIdx.x * 8u + warp;
    if (blk >= (n >> 5)) return;
    u32 i = blk * 32u + lane;
    float v = x[i] * (1.0f / (1.0f + __expf(-g[i])));
    x[i] = v;
    warp_quant_q8_block(v, lane, blk, xq, xsc);
}

// 1-token mul_head_sigmoid + fused Q8 quantize (head_dim % 32 == 0).
extern "C" __global__ __launch_bounds__(256)
void mul_head_sigmoid_q8_f32(
    float* __restrict__ x,
    const float* __restrict__ g,
    u32* __restrict__ xq,
    f32x4* __restrict__ xsc,
    u32 n_heads,
    u32 head_dim
) {
    u32 lid = threadIdx.x;
    u32 warp = lid >> 5;
    u32 lane = lid & 31u;
    u32 blk = blockIdx.x * 8u + warp;
    u32 total_blks = (n_heads * head_dim) >> 5;
    if (blk >= total_blks) return;
    u32 i = blk * 32u + lane;
    u32 h = i / head_dim;
    float v = x[i] * (1.0f / (1.0f + __expf(-g[h])));
    x[i] = v;
    warp_quant_q8_block(v, lane, blk, xq, xsc);
}

// LFM3 fused post_norm(r) + x += r_norm + next_norm(x) + Q8 quantize (1-token decode).
extern "C" __global__ __launch_bounds__(256)
void post_rms_add_rms_norm_q8_f32(
    float* __restrict__ x,
    const float* __restrict__ r,
    const float* __restrict__ w_post,
    const float* __restrict__ w_next,
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

    // 1) RMS of r
    float sr = 0.0f;
    for (u32 i = lid; i < n; i += 256u) {
        float v = r[i];
        sr = __fmaf_rn(v, v, sr);
    }
    for (int off = 16; off > 0; off >>= 1) sr += __shfl_xor(sr, off, 32);
    if (lane == 0u) smem[warp] = sr;
    lds_barrier();
    if (lid < 8u) {
        float v = smem[lid];
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor(v, off, 8);
        smem[lid] = v;
    }
    lds_barrier();
    float sc_post = rsqrtf(smem[0] / (float)n + eps);

    // 2) x += post_norm(r) and RMS of new x
    float sx = 0.0f;
    for (u32 i = lid; i < n; i += 256u) {
        float v = x[i] + r[i] * sc_post * w_post[i];
        x[i] = v;
        sx = __fmaf_rn(v, v, sx);
    }
    for (int off = 16; off > 0; off >>= 1) sx += __shfl_xor(sx, off, 32);
    if (lane == 0u) smem[warp] = sx;
    lds_barrier();
    if (lid < 8u) {
        float v = smem[lid];
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor(v, off, 8);
        smem[lid] = v;
    }
    lds_barrier();
    float sc_next = rsqrtf(smem[0] / (float)n + eps);

    // 3) y = x * sc_next * w_next + fused Q8 quantize
    u32 n_blocks = n >> 5;
    for (u32 blk = warp; blk < n_blocks; blk += 8u) {
        u32 idx = blk * 32u + lane;
        float v = x[idx] * sc_next * w_next[idx];
        y[idx] = v;
        warp_quant_q8_block(v, lane, blk, xq, xsc);
    }
}

extern "C" __global__ __launch_bounds__(256)
void split_qg_f32(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ g,
    u32 n_heads,
    u32 head_dim
) {
    u32 idx = blockIdx.x * 256u + threadIdx.x;
    u32 total = n_heads * head_dim;
    if (idx >= total) return;
    u32 b = blockIdx.y;
    u32 h = idx / head_dim;
    u32 d = idx % head_dim;
    q[(u64)b * total + idx] = qg[(u64)b * 2u * total + (u64)h * 2u * head_dim + d];
    g[(u64)b * total + idx] = qg[(u64)b * 2u * total + (u64)(h * 2u + 1u) * head_dim + d];
}

extern "C" __global__ __launch_bounds__(256)
void mul_sigmoid_f32(
    float* __restrict__ x,
    const float* __restrict__ g,
    u32 n
) {
    u32 i = blockIdx.x * 256u + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + __expf(-g[i]));
}

extern "C" __global__ __launch_bounds__(256)
void mul_head_sigmoid_f32(
    float* __restrict__ x,
    const float* __restrict__ g,
    u32 n_heads,
    u32 head_dim
) {
    u32 idx = blockIdx.x * 256u + threadIdx.x;
    u32 total = n_heads * head_dim;
    if (idx >= total) return;
    u32 b = blockIdx.y;
    u32 h = idx / head_dim;
    x[(u64)b * total + idx] *= 1.0f / (1.0f + __expf(-g[(u64)b * n_heads + h]));
}

// Extended RoPE supporting optional `rope_freqs` factors (for LFM3), batched.
extern "C" __global__ __launch_bounds__(64)
void rope_ext_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ factors,
    u32 n_heads,
    u32 n_kv_heads,
    u32 head_dim,
    u32 n_rot,
    u32 pos0,
    u32 mode,
    float freq_base,
    u32 use_factors
) {
    u32 wg = blockIdx.x;
    u32 tok = blockIdx.y;
    u32 pos = pos0 + tok;
    u32 half_rot = n_rot >> 1;
    for (u32 i = threadIdx.x; i < half_rot; i += 64u) {
        float inv_f = (use_factors != 0u) ? (1.0f / factors[i]) : 1.0f;
        float theta = (float)pos * powf(freq_base, -2.0f * (float)i / (float)n_rot) * inv_f;
        float c, s;
        sincosf(theta, &s, &c);
        u32 a = (mode == 0u) ? (2u * i) : i;
        u32 b = (mode == 0u) ? (2u * i + 1u) : (i + half_rot);
        if (wg < n_heads) {
            float* base = q + ((u64)tok * n_heads + wg) * head_dim;
            float x0 = base[a], x1 = base[b];
            base[a] = x0 * c - x1 * s;
            base[b] = x0 * s + x1 * c;
        } else {
            u32 h = wg - n_heads;
            if (h < n_kv_heads) {
                float* base = k + ((u64)tok * n_kv_heads + h) * head_dim;
                float x0 = base[a], x1 = base[b];
                base[a] = x0 * c - x1 * s;
                base[b] = x0 * s + x1 * c;
            }
        }
    }
}

// Device-meta fused RoPE(Q,K with optional rope_freqs) + paged FP16 KV write for HIP Graph decode.
extern "C" __global__ __launch_bounds__(128)
void rope_ext_kv_write_meta(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ factors,
    u32* __restrict__ kv_layer,
    const u32* __restrict__ page_table,
    const u32* __restrict__ step_meta,
    u32 n_heads,
    u32 n_kv_heads,
    u32 head_dim,
    u32 n_rot,
    float freq_base,
    u32 mode,
    u32 use_factors
) {
    u32 bid = blockIdx.x;
    u32 lid = threadIdx.x;
    u32 pos = step_meta[1];
    u32 half_rot = n_rot >> 1;
    u32 half_hd = head_dim >> 1;

    if (bid < n_heads) {
        if (lid < half_rot) {
            float inv_f = (use_factors != 0u) ? (1.0f / factors[lid]) : 1.0f;
            float theta = (float)pos * powf(freq_base, -2.0f * (float)lid / (float)n_rot) * inv_f;
            float c, s;
            sincosf(theta, &s, &c);
            float* base = q + (u64)bid * head_dim;
            u32 i0 = (mode == 0u) ? (2u * lid) : lid;
            u32 i1 = (mode == 0u) ? (2u * lid + 1u) : (lid + half_rot);
            float x0 = base[i0], x1 = base[i1];
            base[i0] = x0 * c - x1 * s;
            base[i1] = x0 * s + x1 * c;
        }
    } else {
        u32 kvh = bid - n_heads;
        if (kvh >= n_kv_heads) return;
        float* kbase = k + (u64)kvh * head_dim;
        if (lid < half_rot) {
            float inv_f = (use_factors != 0u) ? (1.0f / factors[lid]) : 1.0f;
            float theta = (float)pos * powf(freq_base, -2.0f * (float)lid / (float)n_rot) * inv_f;
            float c, s;
            sincosf(theta, &s, &c);
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
            kv_layer[k_off] = (u32)f32_to_f16(kbase[2u * lid]) | ((u32)f32_to_f16(kbase[2u * lid + 1u]) << 16);
            kv_layer[v_off] = (u32)f32_to_f16(vbase[2u * lid]) | ((u32)f32_to_f16(vbase[2u * lid + 1u]) << 16);
        }
    }
}
"#;

/// Compiled hybrid op bundle (`LFM2`, `LFM3`, `Qwen3.5`) for `HipContext`.
pub struct HipHybridOps {
    _module: Arc<HipModule>,
    shortconv_step_f32: HipFunction,
    shortconv_step_q8_f32: HipFunction,
    ssm_conv_silu_f32: HipFunction,
    deltanet_step_f32: HipFunction,
    deltanet_step_tok_f32: HipFunction,
    gated_rms_norm_f32: HipFunction,
    gated_rms_norm_q8_f32: HipFunction,
    split_qg_f32: HipFunction,
    mul_sigmoid_f32: HipFunction,
    mul_sigmoid_q8_f32: HipFunction,
    mul_head_sigmoid_f32: HipFunction,
    mul_head_sigmoid_q8_f32: HipFunction,
    post_rms_add_rms_norm_q8_f32: HipFunction,
    rope_ext_f32: HipFunction,
    rope_ext_kv_write_meta: HipFunction,
}

impl HipHybridOps {
    pub fn compile(ctx: &Arc<HipContext>) -> Result<Self, HipError> {
        let module = Arc::new(ctx.compile_module(HYBRID_OPS_SRC, "ocak_hip_hybrid_ops.hip", &[])?);
        Ok(Self {
            shortconv_step_f32: module.function("shortconv_step_f32")?,
            shortconv_step_q8_f32: module.function("shortconv_step_q8_f32")?,
            ssm_conv_silu_f32: module.function("ssm_conv_silu_f32")?,
            deltanet_step_f32: module.function("deltanet_step_f32")?,
            deltanet_step_tok_f32: module.function("deltanet_step_tok_f32")?,
            gated_rms_norm_f32: module.function("gated_rms_norm_f32")?,
            gated_rms_norm_q8_f32: module.function("gated_rms_norm_q8_f32")?,
            split_qg_f32: module.function("split_qg_f32")?,
            mul_sigmoid_f32: module.function("mul_sigmoid_f32")?,
            mul_sigmoid_q8_f32: module.function("mul_sigmoid_q8_f32")?,
            mul_head_sigmoid_f32: module.function("mul_head_sigmoid_f32")?,
            mul_head_sigmoid_q8_f32: module.function("mul_head_sigmoid_q8_f32")?,
            post_rms_add_rms_norm_q8_f32: module.function("post_rms_add_rms_norm_q8_f32")?,
            rope_ext_f32: module.function("rope_ext_f32")?,
            rope_ext_kv_write_meta: module.function("rope_ext_kv_write_meta")?,
            _module: module,
        })
    }

    pub fn shortconv_step_q8(
        &self,
        stream: &HipStream,
        bcx: &HipBuffer,
        cw: &HipBuffer,
        st: &HipBuffer,
        o: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n_embd: u32,
        l_cache: u32,
    ) -> Result<(), HipError> {
        let mut bp = bcx.as_device_ptr();
        let mut wp = cw.as_device_ptr();
        let mut sp = st.as_device_ptr();
        let mut op = o.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut scp = xsc.as_device_ptr();
        let mut ne = n_embd;
        let mut lc = l_cache;
        let mut params: [*mut c_void; 8] = [
            &mut bp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut scp as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut lc as *mut _ as *mut c_void,
        ];
        unsafe { self.shortconv_step_q8_f32.launch(stream, ((n_embd / 32).div_ceil(8), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn shortconv_step(
        &self,
        stream: &HipStream,
        bcx: &HipBuffer,
        cw: &HipBuffer,
        st: &HipBuffer,
        o: &HipBuffer,
        n_embd: u32,
        l_cache: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut bp = bcx.as_device_ptr();
        let mut wp = cw.as_device_ptr();
        let mut sp = st.as_device_ptr();
        let mut op = o.as_device_ptr();
        let mut ne = n_embd;
        let mut lc = l_cache;
        let mut b = batch;
        let mut params: [*mut c_void; 7] = [
            &mut bp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut lc as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe { self.shortconv_step_f32.launch(stream, (n_embd.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn ssm_conv_silu(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        cw: &HipBuffer,
        st: &HipBuffer,
        o: &HipBuffer,
        channels: u32,
        k_conv: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut wp = cw.as_device_ptr();
        let mut sp = st.as_device_ptr();
        let mut op = o.as_device_ptr();
        let mut ch = channels;
        let mut kc = k_conv;
        let mut b = batch;
        let mut params: [*mut c_void; 7] = [
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ch as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        unsafe { self.ssm_conv_silu_f32.launch(stream, (channels.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn deltanet_step(
        &self,
        stream: &HipStream,
        qkv: &HipBuffer,
        alpha: &HipBuffer,
        beta: &HipBuffer,
        dt_bias: &HipBuffer,
        ssm_a: &HipBuffer,
        st: &HipBuffer,
        o: &HipBuffer,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        eps: f32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut qp = qkv.as_device_ptr();
        let mut ap = alpha.as_device_ptr();
        let mut bp = beta.as_device_ptr();
        let mut dp = dt_bias.as_device_ptr();
        let mut sap = ssm_a.as_device_ptr();
        let mut stp = st.as_device_ptr();
        let mut op = o.as_device_ptr();
        let mut nkh = n_k_heads;
        let mut nvh = n_v_heads;
        let mut hd = head_dim;
        let mut eps = eps;
        let mut b = batch;
        let mut params: [*mut c_void; 12] = [
            &mut qp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut sap as *mut _ as *mut c_void,
            &mut stp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nkh as *mut _ as *mut c_void,
            &mut nvh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        if batch < 16 {
            let gx = (n_v_heads * head_dim).div_ceil(8);
            unsafe { self.deltanet_step_tok_f32.launch(stream, (gx, 1, 1), (256, 1, 1), 0, &mut params) }
        } else {
            let gx = (n_v_heads * head_dim).div_ceil(32);
            unsafe { self.deltanet_step_f32.launch(stream, (gx, 1, 1), (256, 1, 1), 0, &mut params) }
        }
    }

    pub fn gated_rms_norm_q8(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        w: &HipBuffer,
        z: &HipBuffer,
        o: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut wp = w.as_device_ptr();
        let mut zp = z.as_device_ptr();
        let mut op = o.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut eps = eps;
        let mut params: [*mut c_void; 9] = [
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut zp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe { self.gated_rms_norm_q8_f32.launch(stream, (n_heads, 1, 1), (128, 1, 1), 0, &mut params) }
    }

    pub fn mul_sigmoid_q8(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        g: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut gp = g.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut n = n;
        let mut params: [*mut c_void; 5] = [
            &mut xp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        unsafe { self.mul_sigmoid_q8_f32.launch(stream, ((n / 32).div_ceil(8), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn mul_head_sigmoid_q8(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        g: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n_heads: u32,
        head_dim: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut gp = g.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut params: [*mut c_void; 6] = [
            &mut xp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let total_blks = (n_heads * head_dim) / 32;
        unsafe { self.mul_head_sigmoid_q8_f32.launch(stream, (total_blks.div_ceil(8), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn post_rms_add_rms_norm_q8(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        r: &HipBuffer,
        w_post: &HipBuffer,
        w_next: &HipBuffer,
        y: &HipBuffer,
        xq: &HipBuffer,
        xsc: &HipBuffer,
        n: u32,
        eps: f32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut rp = r.as_device_ptr();
        let mut wpp = w_post.as_device_ptr();
        let mut wnp = w_next.as_device_ptr();
        let mut yp = y.as_device_ptr();
        let mut qp = xq.as_device_ptr();
        let mut sp = xsc.as_device_ptr();
        let mut n = n;
        let mut eps = eps;
        let mut params: [*mut c_void; 9] = [
            &mut xp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut wpp as *mut _ as *mut c_void,
            &mut wnp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe { self.post_rms_add_rms_norm_q8_f32.launch(stream, (1, 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn gated_rms_norm(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        w: &HipBuffer,
        z: &HipBuffer,
        o: &HipBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut wp = w.as_device_ptr();
        let mut zp = z.as_device_ptr();
        let mut op = o.as_device_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut eps = eps;
        let mut params: [*mut c_void; 7] = [
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut zp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe { self.gated_rms_norm_f32.launch(stream, (n_heads, batch, 1), (128, 1, 1), 0, &mut params) }
    }

    pub fn split_qg(
        &self,
        stream: &HipStream,
        qg: &HipBuffer,
        q: &HipBuffer,
        g: &HipBuffer,
        n_heads: u32,
        head_dim: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut qgp = qg.as_device_ptr();
        let mut qp = q.as_device_ptr();
        let mut gp = g.as_device_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut params: [*mut c_void; 5] = [
            &mut qgp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let total = n_heads * head_dim;
        unsafe { self.split_qg_f32.launch(stream, (total.div_ceil(256), batch, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn mul_sigmoid(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        g: &HipBuffer,
        n: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut gp = g.as_device_ptr();
        let mut n = n;
        let mut params: [*mut c_void; 3] = [
            &mut xp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        unsafe { self.mul_sigmoid_f32.launch(stream, (n.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn mul_head_sigmoid(
        &self,
        stream: &HipStream,
        x: &HipBuffer,
        g: &HipBuffer,
        n_heads: u32,
        head_dim: u32,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut xp = x.as_device_ptr();
        let mut gp = g.as_device_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut params: [*mut c_void; 4] = [
            &mut xp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let total = n_heads * head_dim;
        unsafe { self.mul_head_sigmoid_f32.launch(stream, (total.div_ceil(256), batch, 1), (256, 1, 1), 0, &mut params) }
    }

    pub fn rope_ext(
        &self,
        stream: &HipStream,
        q: &HipBuffer,
        k: &HipBuffer,
        factors: &HipBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        mode: u32,
        freq_base: f32,
        use_factors: bool,
        batch: u32,
    ) -> Result<(), HipError> {
        let mut qp = q.as_device_ptr();
        let mut kp = k.as_device_ptr();
        let mut fp = factors.as_device_ptr();
        let mut nh = n_heads;
        let mut nkv = n_kv_heads;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut p0 = pos0;
        let mut md = mode;
        let mut fb = freq_base;
        let mut uf = u32::from(use_factors);
        let mut params: [*mut c_void; 11] = [
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut fp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut p0 as *mut _ as *mut c_void,
            &mut md as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut uf as *mut _ as *mut c_void,
        ];
        unsafe { self.rope_ext_f32.launch(stream, (n_heads + n_kv_heads, batch, 1), (64, 1, 1), 0, &mut params) }
    }

    pub fn rope_ext_kv_write_meta(
        &self,
        stream: &HipStream,
        q: &HipBuffer,
        k: &HipBuffer,
        v: &HipBuffer,
        factors: &HipBuffer,
        kv: &HipBuffer,
        kv_byte_offset: usize,
        page_table: &HipBuffer,
        step_meta: &HipBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_rot: u32,
        freq_base: f32,
        mode: u32,
        use_factors: bool,
    ) -> Result<(), HipError> {
        let mut qp = q.as_device_ptr();
        let mut kp = k.as_device_ptr();
        let mut vp = v.as_device_ptr();
        let mut fp = factors.as_device_ptr();
        let mut kvp = kv.device_ptr_at(kv_byte_offset)?;
        let mut ptp = page_table.as_device_ptr();
        let mut mp = step_meta.as_device_ptr();
        let mut nh = n_heads;
        let mut nkv = n_kv_heads;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut md = mode;
        let mut uf = u32::from(use_factors);
        let mut params: [*mut c_void; 14] = [
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut fp as *mut _ as *mut c_void,
            &mut kvp as *mut _ as *mut c_void,
            &mut ptp as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut md as *mut _ as *mut c_void,
            &mut uf as *mut _ as *mut c_void,
        ];
        unsafe {
            self.rope_ext_kv_write_meta
                .launch(stream, (n_heads + n_kv_heads, 1, 1), (128, 1, 1), 0, &mut params)
        }
    }
}
