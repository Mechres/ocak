// Integer unpacking for the int8-dot matmul (qmm_q8.comp).
// dqi8(rb, k0, q0, q1, d, m): the 8 weights at columns k0..k0+8 (k0 % 8 == 0) of the row at
// byte offset rb are d * q[i] - m, with q[i] signed int8 packed little-endian into q0 (i < 4)
// and q1 (i >= 4). d and m are constant over the 16-column group containing k0.
// The including shader defines `w` (uint8_t[]) and `w32` (uint[], same buffer).
float f16at(uint o) { return unpackHalf2x16(uint(w[o]) | (uint(w[o + 1u]) << 8)).x; }
uint pack4(int a, int b, int c, int e) {
    return uint(a & 0xFF) | (uint(b & 0xFF) << 8) | (uint(c & 0xFF) << 16) | (uint(e & 0xFF) << 24);
}
uint bytes4(uint o) { return uint(w[o]) | (uint(w[o + 1u]) << 8) | (uint(w[o + 2u]) << 16) | (uint(w[o + 3u]) << 24); }

void dqi8(uint rb, uint k0, out uint q0, out uint q1, out float d, out float m) {
#if defined(Q4_K) || defined(Q5_K)
#if defined(Q4_K)
    uint b = rb + (k0 >> 8) * 144u;
#else
    uint b = rb + (k0 >> 8) * 176u;
#endif
    uint j = (k0 >> 5) & 7u;
    uint l = k0 & 31u;
    uint sb = b + 4u;
    uint sc, mn;
    if (j < 4u) {
        sc = uint(w[sb + j]) & 63u;
        mn = uint(w[sb + j + 4u]) & 63u;
    } else {
        sc = (uint(w[sb + j + 4u]) & 0xFu) | ((uint(w[sb + j - 4u]) >> 6) << 4);
        mn = (uint(w[sb + j + 4u]) >> 4) | ((uint(w[sb + j]) >> 6) << 4);
    }
    d = f16at(b) * float(sc);
    m = f16at(b + 2u) * float(mn);
    uint sh = (j & 1u) * 4u;
#if defined(Q4_K)
    uint qo = (b + 16u + (j >> 1) * 32u + l) >> 2;   // 4-byte aligned: 144-byte blocks
    q0 = (w32[qo] >> sh) & 0x0F0F0F0Fu;
    q1 = (w32[qo + 1u] >> sh) & 0x0F0F0F0Fu;
#else
    uint qo = (b + 48u + (j >> 1) * 32u + l) >> 2;   // 176-byte blocks
    uint ho = (b + 16u + l) >> 2;
    q0 = ((w32[qo] >> sh) & 0x0F0F0F0Fu) | (((w32[ho] >> j) & 0x01010101u) << 4);
    q1 = ((w32[qo + 1u] >> sh) & 0x0F0F0F0Fu) | (((w32[ho + 1u] >> j) & 0x01010101u) << 4);
#endif
#elif defined(Q6_K)
    uint b = rb + (k0 >> 8) * 210u;
    uint n = k0 & 255u;
    uint half_ = n >> 7;
    uint g = (n >> 5) & 3u;
    uint l = n & 31u;
    uint qlo = b + half_ * 64u + (g & 1u) * 32u + l;
    uint qho = b + 128u + half_ * 32u + l;
    uint sl = (g >> 1) * 4u;
    uint sh = g * 2u;
    d = f16at(b + 208u) * float(int8_t(w[b + 192u + half_ * 8u + (l >> 4) + 2u * g]));
    m = 0.0;
    // 210-byte blocks are only 2-byte aligned: assemble words from bytes.
    uint lo0 = bytes4(qlo), lo1 = bytes4(qlo + 4u);
    uint hi0 = bytes4(qho), hi1 = bytes4(qho + 4u);
    uint u0 = ((lo0 >> sl) & 0x0F0F0F0Fu) | (((hi0 >> sh) & 0x03030303u) << 4);
    uint u1 = ((lo1 >> sl) & 0x0F0F0F0Fu) | (((hi1 >> sh) & 0x03030303u) << 4);
    // q - 32 per byte without cross-byte borrows: bytes are 0..63, so (b + 0x80) - 0x20
    // stays in 0x60..0x9F, and flipping bit 7 back gives (b - 32) mod 256.
    q0 = ((u0 | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
    q1 = ((u1 | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
#elif defined(Q8_0)
    uint b = rb + (k0 >> 5) * 34u;
    uint o = b + 2u + (k0 & 31u);
    d = f16at(b);
    m = 0.0;
    q0 = bytes4(o);
    q1 = bytes4(o + 4u);
#else
#error "dequant_i8.glsl: unsupported weight format"
#endif
}
