// Shared dequantization for the tiled / multi-token matmul kernels.
// The including shader defines buffers `w` (uint8_t[]) and `tbl` (uint[]) and one -D<TYPE>.
float f16at(uint o) { return unpackHalf2x16(uint(w[o]) | (uint(w[o + 1u]) << 8)).x; }
uint u16at(uint o) { return uint(w[o]) | (uint(w[o + 1u]) << 8); }
uint u32at(uint o) { return u16at(o) | (u16at(o + 2u) << 16); }
float s8(uint word, uint byte_idx) {
    int v = int((word >> (8u * byte_idx)) & 0xFFu);
    return float(v >= 128 ? v - 256 : v);
}

// Dequantize the 8 consecutive weights starting at column k0 (k0 % 8 == 0) of the row at rb.
void dq8(uint rb, uint k0, out float v[8]) {
#if defined(Q4_K)
    uint b = rb + (k0 >> 8) * 144u;
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
    float dl = f16at(b) * float(sc);
    float ml = f16at(b + 2u) * float(mn);
    uint qo = b + 16u + (j >> 1) * 32u + l;
    uint sh = (j & 1u) * 4u;
    for (uint i = 0u; i < 8u; i++) v[i] = dl * float((uint(w[qo + i]) >> sh) & 0xFu) - ml;
#elif defined(Q5_K)
    uint b = rb + (k0 >> 8) * 176u;
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
    float dl = f16at(b) * float(sc);
    float ml = f16at(b + 2u) * float(mn);
    uint qo = b + 48u + (j >> 1) * 32u + l;
    uint ho = b + 16u + l;
    uint sh = (j & 1u) * 4u;
    for (uint i = 0u; i < 8u; i++) {
        uint q = ((uint(w[qo + i]) >> sh) & 0xFu) | (((uint(w[ho + i]) >> j) & 1u) << 4);
        v[i] = dl * float(q) - ml;
    }
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
    float dl = f16at(b + 208u) * float(int8_t(w[b + 192u + half_ * 8u + (l >> 4) + 2u * g]));
    for (uint i = 0u; i < 8u; i++) {
        int q = int(((uint(w[qlo + i]) >> sl) & 0xFu) | (((uint(w[qho + i]) >> sh) & 3u) << 4));
        v[i] = dl * float(q - 32);
    }
#elif defined(Q2_K)
    uint b = rb + (k0 >> 8) * 84u;
    uint n = k0 & 255u;
    uint half_ = n >> 7;
    uint g = (n >> 5) & 3u;
    uint l = n & 31u;
    uint sc = uint(w[b + half_ * 8u + 2u * g + (l >> 4)]);
    float dl = f16at(b + 80u) * float(sc & 0xFu);
    float ml = f16at(b + 82u) * float(sc >> 4);
    uint qo = b + 16u + half_ * 32u + l;
    for (uint i = 0u; i < 8u; i++) v[i] = dl * float((uint(w[qo + i]) >> (2u * g)) & 3u) - ml;
#elif defined(Q3_K)
    uint b = rb + (k0 >> 8) * 110u;
    uint n = k0 & 255u;
    uint half_ = n >> 7;
    uint g = (n >> 5) & 3u;
    uint l = n & 31u;
    uint is = half_ * 8u + 2u * g + (l >> 4);
    uint lo4 = (is < 8u) ? (uint(w[b + 96u + is]) & 0xFu) : (uint(w[b + 96u + is - 8u]) >> 4);
    uint hi2 = (uint(w[b + 104u + (is & 3u)]) >> (2u * (is >> 2))) & 3u;
    float dl = f16at(b + 108u) * float(int(lo4 | (hi2 << 4)) - 32);
    uint qo = b + 32u + half_ * 32u + l;
    uint hs = half_ * 4u + g;
    for (uint i = 0u; i < 8u; i++) {
        int q = int((uint(w[qo + i]) >> (2u * g)) & 3u) - ((((uint(w[b + l + i]) >> hs) & 1u) != 0u) ? 0 : 4);
        v[i] = dl * float(q);
    }
#elif defined(Q8_0)
    uint b = rb + (k0 >> 5) * 34u;
    uint l = k0 & 31u;
    float d = f16at(b);
    for (uint i = 0u; i < 8u; i++) v[i] = d * float(int8_t(w[b + 2u + l + i]));
#elif defined(Q4_0)
    uint b = rb + (k0 >> 5) * 18u;
    uint l = k0 & 31u;
    float d = f16at(b);
    uint qo = b + 2u + (l & 15u);
    uint sh = (l >> 4) * 4u;
    for (uint i = 0u; i < 8u; i++) v[i] = d * (float((uint(w[qo + i]) >> sh) & 0xFu) - 8.0);
#elif defined(F16)
    uint o = rb + k0 * 2u;
    for (uint i = 0u; i < 8u; i++) v[i] = f16at(o + 2u * i);
#elif defined(BF16)
    uint o = rb + k0 * 2u;
    for (uint i = 0u; i < 8u; i++) v[i] = uintBitsToFloat(u16at(o + 2u * i) << 16);
#elif defined(NVFP4)
    // 36-byte blocks of 64: 4 UE4M3 scales (one per 16) + 32 bytes of E2M1 nibbles; in each
    // 16-weight sub-block, byte j holds weight j (low nibble) and weight j + 8 (high nibble).
    uint b = rb + (k0 >> 6) * 36u;
    uint sbk = (k0 >> 4) & 3u;
    uint sh = ((k0 >> 3) & 1u) * 4u;
    uint sv = uint(w[b + sbk]);
    float d = 0.0;
    if (sv != 0u && sv != 0x7Fu) {
        int e = int((sv >> 3) & 0xFu);
        float mm = float(sv & 7u);
        d = ((e == 0) ? (mm * exp2(-9.0)) : ((1.0 + mm * 0.125) * exp2(float(e - 7)))) * 0.5;
    }
    uint qo = b + 4u + sbk * 8u;
    for (uint i = 0u; i < 8u; i++) {
        uint q = (uint(w[qo + i]) >> sh) & 0xFu;
        float mag = float((0xC8643210u >> (4u * (q & 7u))) & 0xFu);   // {0,1,2,3,4,6,8,12}
        v[i] = d * ((q >= 8u) ? -mag : mag);
    }
#elif defined(F32)
    uint o = rb + k0 * 4u;
    for (uint i = 0u; i < 8u; i++) {
        uint p = o + 4u * i;
        v[i] = uintBitsToFloat(uint(w[p]) | (uint(w[p + 1u]) << 8) | (uint(w[p + 2u]) << 16) | (uint(w[p + 3u]) << 24));
    }
#elif defined(IQ4_NL) || defined(IQ4_XS)
#if defined(IQ4_NL)
    uint b = rb + (k0 >> 5) * 18u;
    float dl = f16at(b);
    uint qo = b + 2u;
#else
    uint b = rb + (k0 >> 8) * 136u;
    uint ib = (k0 >> 5) & 7u;
    uint sh = u16at(b + 2u);
    uint sl = uint(w[b + 4u + ib / 2u]);
    int ls = int((sl >> (4u * (ib & 1u))) & 0xFu) | int(((sh >> (2u * ib)) & 3u) << 4);
    float dl = f16at(b) * float(ls - 32);
    uint qo = b + 8u + ib * 16u;
#endif
    uint p = k0 & 31u;
    uint so = qo + (p & 15u);
    uint nsh = (p >> 4) * 4u;
    for (uint i = 0u; i < 8u; i++) v[i] = dl * float(int(tbl[1920u + ((uint(w[so + i]) >> nsh) & 0xFu)]));
#elif defined(IQ1_S) || defined(IQ1_M)
    uint ib = (k0 >> 5) & 7u;
    uint l = (k0 >> 3) & 3u;
    float dl, delta;
    uint gidx;
#if defined(IQ1_S)
    uint b = rb + (k0 >> 8) * 50u;
    uint qhb = u16at(b + 34u + 2u * ib);
    dl = f16at(b) * float(2u * ((qhb >> 12) & 7u) + 1u);
    delta = ((qhb & 0x8000u) != 0u) ? -0.125 : 0.125;
    gidx = uint(w[b + 2u + ib * 4u + l]) | (((qhb >> (3u * l)) & 7u) << 8);
#else
    uint b = rb + (k0 >> 8) * 56u;
    uint sc0 = u16at(b + 48u), sc1 = u16at(b + 50u), sc2 = u16at(b + 52u), sc3 = u16at(b + 54u);
    uint dbits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) | ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
    float d = unpackHalf2x16(dbits).x;
    uint s16 = (ib < 2u) ? sc0 : ((ib < 4u) ? sc1 : ((ib < 6u) ? sc2 : sc3));
    uint ssh = 6u * (ib & 1u) + ((l < 2u) ? 0u : 3u);
    dl = d * float(2u * ((s16 >> ssh) & 7u) + 1u);
    uint qb = uint(w[b + 32u + ib * 2u + (l >> 1)]);
    uint hi = (l & 1u) == 0u ? ((qb << 8) & 0x700u) : ((qb << 4) & 0x700u);
    gidx = uint(w[b + ib * 4u + l]) | hi;
    uint dbit = (l & 1u) == 0u ? 0x08u : 0x80u;
    delta = ((qb & dbit) != 0u) ? -0.125 : 0.125;
#endif
    uint glo = tbl[4496u + 2u * gidx];
    uint ghi = tbl[4496u + 2u * gidx + 1u];
    for (uint j = 0u; j < 4u; j++) {
        v[j] = dl * (s8(glo, j) + delta);
        v[j + 4u] = dl * (s8(ghi, j) + delta);
    }
#else
    // 256-weight superblock IQ formats: thread-of-32 index t covers weights t*8 .. t*8+7.
    uint t = (k0 & 255u) >> 3;
    uint ib32 = t >> 2;
    uint l = t & 3u;
    uint glo, ghi, sg;
    float dl;
#if defined(IQ3_S)
    uint b = rb + (k0 >> 8) * 110u;
    float d = f16at(b);
    uint sc = uint(w[b + 106u + (ib32 >> 1)]);
    dl = d * float(1u + 2u * (((ib32 & 1u) == 0u) ? (sc & 0xFu) : (sc >> 4)));
    uint qh = uint(w[b + 66u + ib32]);
    uint i1 = uint(w[b + 2u + 2u * t]) | ((qh << (8u - 2u * l)) & 256u);
    uint i2 = uint(w[b + 2u + 2u * t + 1u]) | ((qh << (7u - 2u * l)) & 256u);
    glo = tbl[1280u + i1];
    ghi = tbl[1280u + i2];
    sg = uint(w[b + 74u + t]);
#elif defined(IQ3_XXS)
    uint b = rb + (k0 >> 8) * 98u;
    float d = f16at(b);
    uint so = b + 66u + ib32 * 4u;
    uint aux32 = uint(w[so]) | (uint(w[so + 1u]) << 8) | (uint(w[so + 2u]) << 16) | (uint(w[so + 3u]) << 24);
    dl = d * (0.5 + float(aux32 >> 28)) * 0.5;
    sg = tbl[1792u + ((aux32 >> (7u * l)) & 127u)];
    glo = tbl[1024u + uint(w[b + 2u + 2u * t])];
    ghi = tbl[1024u + uint(w[b + 2u + 2u * t + 1u])];
#elif defined(IQ2_XS)
    uint b = rb + (k0 >> 8) * 74u;
    float d = f16at(b);
    uint sc = uint(w[b + 66u + ib32]);
    dl = d * (0.5 + float((l < 2u) ? (sc & 0xFu) : (sc >> 4))) * 0.25;
    uint qo = b + 2u + 2u * t;
    uint q = uint(w[qo]) | (uint(w[qo + 1u]) << 8);
    uint gidx = q & 511u;
    glo = tbl[2u * gidx];
    ghi = tbl[2u * gidx + 1u];
    sg = tbl[1792u + (q >> 9)];
#elif defined(IQ2_S)
    uint b = rb + (k0 >> 8) * 82u;
    float d = f16at(b);
    uint sc = uint(w[b + 74u + ib32]);
    dl = d * (0.5 + float((l < 2u) ? (sc & 0xFu) : (sc >> 4))) * 0.25;
    uint qhb = uint(w[b + 66u + ib32]);
    uint gidx = uint(w[b + 2u + t]) | ((qhb << (8u - 2u * l)) & 0x300u);
    glo = tbl[2448u + 2u * gidx];
    ghi = tbl[2448u + 2u * gidx + 1u];
    sg = uint(w[b + 34u + t]);
#elif defined(IQ2_XXS)
    uint b = rb + (k0 >> 8) * 66u;
    uint q0 = b + 2u + ib32 * 8u;
    uint a0 = u32at(q0);
    uint a1 = u32at(q0 + 4u);
    dl = f16at(b) * (0.5 + float(a1 >> 28)) * 0.25;
    uint gidx = (a0 >> (8u * l)) & 0xFFu;
    glo = tbl[1936u + 2u * gidx];
    ghi = tbl[1936u + 2u * gidx + 1u];
    sg = tbl[1792u + ((a1 >> (7u * l)) & 127u)];
#else
#error "qmm_tiled: no weight type defined"
#endif
    for (uint j = 0u; j < 4u; j++) {
        v[j] = dl * float((glo >> (8u * j)) & 0xFFu) * (((sg & (1u << j)) != 0u) ? -1.0 : 1.0);
        v[j + 4u] = dl * float((ghi >> (8u * j)) & 0xFFu) * (((sg & (1u << (j + 4u))) != 0u) ? -1.0 : 1.0);
    }
#endif
}

