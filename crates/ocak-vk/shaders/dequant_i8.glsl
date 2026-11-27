// Integer unpacking for the int8-dot matmul (qmm_q8.comp).
// dqi8(rb, k0, q0, q1, d, m): the 8 weights at columns k0..k0+8 (k0 % 8 == 0) of the row at
// byte offset rb are d * q[i] - m, with q[i] signed int8 packed little-endian into q0 (i < 4)
// and q1 (i >= 4). d and m are constant over the 16-column group containing k0.
// The including shader defines `w` (uint8_t[]), `w32` (uint[], same buffer) and `tbl` (IQ
// grid tables).
// fp16 at an even byte offset: one aligned 32-bit load.
float f16at(uint o) { return unpackHalf2x16(w32[o >> 2] >> ((o & 2u) << 3)).x; }
uint pack4(int a, int b, int c, int e) {
    return uint(a & 0xFF) | (uint(b & 0xFF) << 8) | (uint(c & 0xFF) << 16) | (uint(e & 0xFF) << 24);
}
// 4 bytes at any offset from two aligned 32-bit loads (funnel shift; sh = 0 adds nothing since
// (x << 1) << 31 drops every bit of x << 1).
uint bytes4(uint o) {
    uint a = o >> 2;
    uint sh = (o & 3u) << 3;
    return (w32[a] >> sh) | ((w32[a + 1u] << 1) << (31u - sh));
}
// Per-byte (b - c) for packed bytes with b in [0, 127], c <= 0x7F: no cross-byte borrows.
uint sub_bytes(uint x, uint c) { return ((x | 0x80808080u) - c) ^ 0x80808080u; }

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
    q0 = sub_bytes(u0, 0x20202020u);
    q1 = sub_bytes(u1, 0x20202020u);
#elif defined(Q8_0)
    uint b = rb + (k0 >> 5) * 34u;
    uint o = b + 2u + (k0 & 31u);
    d = f16at(b);
    m = 0.0;
    q0 = bytes4(o);
    q1 = bytes4(o + 4u);
#elif defined(Q2_K)
    uint b = rb + (k0 >> 8) * 84u;
    uint n = k0 & 255u;
    uint half_ = n >> 7;
    uint g = (n >> 5) & 3u;
    uint l = n & 31u;
    uint sc = uint(w[b + half_ * 8u + 2u * g + (l >> 4)]);
    // The 4-bit sub-scale is folded into the integer weights (q * sc <= 45 per byte, so the
    // packed multiply never carries), leaving the superblock d as the only weight scale; the
    // min (per 16) stays separate.
    d = f16at(b + 80u);
    m = f16at(b + 82u) * float(sc >> 4);
    uint qo = (b + 16u + half_ * 32u + l) >> 2;   // 84-byte blocks: 4-byte aligned
    q0 = ((w32[qo] >> (2u * g)) & 0x03030303u) * (sc & 0xFu);
    q1 = ((w32[qo + 1u] >> (2u * g)) & 0x03030303u) * (sc & 0xFu);
#elif defined(Q3_K)
    uint b = rb + (k0 >> 8) * 110u;
    uint n = k0 & 255u;
    uint half_ = n >> 7;
    uint g = (n >> 5) & 3u;
    uint l = n & 31u;
    uint is = half_ * 8u + 2u * g + (l >> 4);
    uint lo4 = (is < 8u) ? (uint(w[b + 96u + is]) & 0xFu) : (uint(w[b + 96u + is - 8u]) >> 4);
    uint hi2 = (uint(w[b + 104u + (is & 3u)]) >> (2u * (is >> 2))) & 3u;
    d = f16at(b + 108u) * float(int(lo4 | (hi2 << 4)) - 32);
    m = 0.0;
    uint qo = b + 32u + half_ * 32u + l;
    uint hs = half_ * 4u + g;
    // q = low 2 bits + 4 * hmask bit - 4, per byte.
    uint lo0 = (bytes4(qo) >> (2u * g)) & 0x03030303u, lo1 = (bytes4(qo + 4u) >> (2u * g)) & 0x03030303u;
    uint hm0 = (bytes4(b + l) >> hs) & 0x01010101u, hm1 = (bytes4(b + l + 4u) >> hs) & 0x01010101u;
    q0 = sub_bytes(lo0 | (hm0 << 2), 0x04040404u);
    q1 = sub_bytes(lo1 | (hm1 << 2), 0x04040404u);
#elif defined(IQ4_NL) || defined(IQ4_XS)
#if defined(IQ4_NL)
    uint b = rb + (k0 >> 5) * 18u;
    d = f16at(b);
    uint qo = b + 2u;
#else
    uint b = rb + (k0 >> 8) * 136u;
    uint ib = (k0 >> 5) & 7u;
    uint shb = uint(w[b + 2u]) | (uint(w[b + 3u]) << 8);
    uint sl = uint(w[b + 4u + ib / 2u]);
    int ls = int((sl >> (4u * (ib & 1u))) & 0xFu) | int(((shb >> (2u * ib)) & 3u) << 4);
    d = f16at(b) * float(ls - 32);
    uint qo = b + 8u + ib * 16u;
#endif
    m = 0.0;
    uint p = k0 & 31u;
    uint so = qo + (p & 15u);
    uint nsh = (p >> 4) * 4u;
    int v[8];
    for (uint i = 0u; i < 8u; i++) v[i] = int(tbl[1920u + ((uint(w[so + i]) >> nsh) & 0xFu)]);
    q0 = pack4(v[0], v[1], v[2], v[3]);
    q1 = pack4(v[4], v[5], v[6], v[7]);
#elif defined(IQ3_S) || defined(IQ3_XXS) || defined(IQ2_XS) || defined(IQ2_S) || defined(IQ2_XXS)
    // 256-weight superblock grid formats: 8 weights = 8 unsigned grid bytes with sign bits.
    uint t = (k0 & 255u) >> 3;
    uint ib32 = t >> 2;
    uint l = t & 3u;
    uint glo, ghi, sg;
#if defined(IQ3_S)
    uint b = rb + (k0 >> 8) * 110u;
    uint sc = uint(w[b + 106u + (ib32 >> 1)]);
    d = f16at(b) * float(1u + 2u * (((ib32 & 1u) == 0u) ? (sc & 0xFu) : (sc >> 4)));
    uint qh = uint(w[b + 66u + ib32]);
    uint i1 = uint(w[b + 2u + 2u * t]) | ((qh << (8u - 2u * l)) & 256u);
    uint i2 = uint(w[b + 2u + 2u * t + 1u]) | ((qh << (7u - 2u * l)) & 256u);
    glo = tbl[1280u + i1];
    ghi = tbl[1280u + i2];
    sg = uint(w[b + 74u + t]);
#elif defined(IQ3_XXS)
    uint b = rb + (k0 >> 8) * 98u;
    uint aux32 = bytes4(b + 66u + ib32 * 4u);
    d = f16at(b) * (0.5 + float(aux32 >> 28)) * 0.5;
    sg = tbl[1792u + ((aux32 >> (7u * l)) & 127u)];
    glo = tbl[1024u + uint(w[b + 2u + 2u * t])];
    ghi = tbl[1024u + uint(w[b + 2u + 2u * t + 1u])];
#elif defined(IQ2_XS)
    uint b = rb + (k0 >> 8) * 74u;
    uint sc = uint(w[b + 66u + ib32]);
    d = f16at(b) * (0.5 + float((l < 2u) ? (sc & 0xFu) : (sc >> 4))) * 0.25;
    uint qo = b + 2u + 2u * t;
    uint q = uint(w[qo]) | (uint(w[qo + 1u]) << 8);
    uint gidx = q & 511u;
    glo = tbl[2u * gidx];
    ghi = tbl[2u * gidx + 1u];
    sg = tbl[1792u + (q >> 9)];
#elif defined(IQ2_S)
    uint b = rb + (k0 >> 8) * 82u;
    uint sc = uint(w[b + 74u + ib32]);
    d = f16at(b) * (0.5 + float((l < 2u) ? (sc & 0xFu) : (sc >> 4))) * 0.25;
    uint qhb = uint(w[b + 66u + ib32]);
    uint gidx = uint(w[b + 2u + t]) | ((qhb << (8u - 2u * l)) & 0x300u);
    glo = tbl[2448u + 2u * gidx];
    ghi = tbl[2448u + 2u * gidx + 1u];
    sg = uint(w[b + 34u + t]);
#else
    uint b = rb + (k0 >> 8) * 66u;
    uint a0 = bytes4(b + 2u + ib32 * 8u);
    uint a1 = bytes4(b + 6u + ib32 * 8u);
    d = f16at(b) * (0.5 + float(a1 >> 28)) * 0.25;
    uint gidx = (a0 >> (8u * l)) & 0xFFu;
    glo = tbl[1936u + 2u * gidx];
    ghi = tbl[1936u + 2u * gidx + 1u];
    sg = tbl[1792u + ((a1 >> (7u * l)) & 127u)];
#endif
    m = 0.0;
    int v[8];
    for (uint j = 0u; j < 4u; j++) {
        int a = int((glo >> (8u * j)) & 0xFFu);
        int c = int((ghi >> (8u * j)) & 0xFFu);
        v[j] = ((sg & (1u << j)) != 0u) ? -a : a;
        v[j + 4u] = ((sg & (1u << (j + 4u))) != 0u) ? -c : c;
    }
    q0 = pack4(v[0], v[1], v[2], v[3]);
    q1 = pack4(v[4], v[5], v[6], v[7]);
#elif defined(IQ1_S) || defined(IQ1_M)
    // Weights are dl * (g + delta) with grid g in {-1, 0, 1} and delta = +-1/8: exact as
    // integers 8 g +- 1 with d = dl / 8.
    uint ib = (k0 >> 5) & 7u;
    uint l = (k0 >> 3) & 3u;
    float dl;
    bool neg;
    uint gidx;
#if defined(IQ1_S)
    uint b = rb + (k0 >> 8) * 50u;
    uint qhb = uint(w[b + 34u + 2u * ib]) | (uint(w[b + 35u + 2u * ib]) << 8);
    dl = f16at(b) * float(2u * ((qhb >> 12) & 7u) + 1u);
    neg = (qhb & 0x8000u) != 0u;
    gidx = uint(w[b + 2u + ib * 4u + l]) | (((qhb >> (3u * l)) & 7u) << 8);
#else
    uint b = rb + (k0 >> 8) * 56u;
    uint sc0 = uint(w[b + 48u]) | (uint(w[b + 49u]) << 8);
    uint sc1 = uint(w[b + 50u]) | (uint(w[b + 51u]) << 8);
    uint sc2 = uint(w[b + 52u]) | (uint(w[b + 53u]) << 8);
    uint sc3 = uint(w[b + 54u]) | (uint(w[b + 55u]) << 8);
    uint dbits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) | ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
    float dsup = unpackHalf2x16(dbits).x;
    uint s16 = (ib < 2u) ? sc0 : ((ib < 4u) ? sc1 : ((ib < 6u) ? sc2 : sc3));
    uint ssh = 6u * (ib & 1u) + ((l < 2u) ? 0u : 3u);
    dl = dsup * float(2u * ((s16 >> ssh) & 7u) + 1u);
    uint qb = uint(w[b + 32u + ib * 2u + (l >> 1)]);
    uint hi = (l & 1u) == 0u ? ((qb << 8) & 0x700u) : ((qb << 4) & 0x700u);
    gidx = uint(w[b + ib * 4u + l]) | hi;
    uint dbit = (l & 1u) == 0u ? 0x08u : 0x80u;
    neg = (qb & dbit) != 0u;
#endif
    d = dl * 0.125;
    m = 0.0;
    uint glo = tbl[4496u + 2u * gidx];
    uint ghi = tbl[4496u + 2u * gidx + 1u];
    int dd = neg ? -1 : 1;
    int v[8];
    for (uint j = 0u; j < 4u; j++) {
        v[j] = 8 * int(int8_t(int((glo >> (8u * j)) & 0xFFu))) + dd;
        v[j + 4u] = 8 * int(int8_t(int((ghi >> (8u * j)) & 0xFFu))) + dd;
    }
    q0 = pack4(v[0], v[1], v[2], v[3]);
    q1 = pack4(v[4], v[5], v[6], v[7]);
#elif defined(NVFP4)
    // 36-byte blocks of 64: 4 UE4M3 scales (one per 16) + E2M1 nibbles. E2M1 magnitudes are
    // {0, .5, 1, 1.5, 2, 3, 4, 6}: exact as integers {0, 1, 2, 3, 4, 6, 8, 12} with d = scale / 2.
    uint b = rb + (k0 >> 6) * 36u;
    uint sbk = (k0 >> 4) & 3u;
    uint sh = ((k0 >> 3) & 1u) * 4u;
    uint sv = uint(w[b + sbk]);
    d = 0.0;
    if (sv != 0u && sv != 0x7Fu) {
        int e = int((sv >> 3) & 0xFu);
        float mm = float(sv & 7u);
        d = ((e == 0) ? (mm * exp2(-9.0)) : ((1.0 + mm * 0.125) * exp2(float(e - 7)))) * 0.5;
    }
    m = 0.0;
    uint qo = b + 4u + sbk * 8u;
    int v[8];
    for (uint i = 0u; i < 8u; i++) {
        uint q = (uint(w[qo + i]) >> sh) & 0xFu;
        int mag = int((0xC8643210u >> (4u * (q & 7u))) & 0xFu);   // {0,1,2,3,4,6,8,12}
        v[i] = (q >= 8u) ? -mag : mag;
    }
    q0 = pack4(v[0], v[1], v[2], v[3]);
    q1 = pack4(v[4], v[5], v[6], v[7]);
#else
#error "dequant_i8.glsl: unsupported weight format"
#endif
}
