// Fixed-rate zfp for 2-D float32 arrays on Intel GPUs (SYCL).
//
// A port of the 2-D float path of zfp's CUDA backend (LLNL zfp, develop
// 70fecac, src/cuda_zfp: encode.cuh, encode2.cuh, decode.cuh, decode2.cuh,
// shared.h, constants.h; BSD-3-Clause, Copyright (c) 2014-2023 Lawrence
// Livermore National Security, LLC). Upstream zfp has no SYCL backend, and
// clio-core's zfp-sycl codec expects one at /opt/zfp-sycl, which Aurora does
// not have.
//
// The codestream is the one libzfp writes in fixed-rate mode with 64-bit
// stream words and ZFP_ROUND_NEVER (both defaults): one GPU work-item per 4x4
// block, blocks in x-fastest order, each exactly maxbits = 16 * rate bits.
// zfp_bench.cpp checks it bit for bit against libzfp.
//
// Restriction: rate must be a multiple of 4, so every block is a whole number
// of 64-bit words and a work-item owns its words outright (the CUDA encoder
// handles the general case with atomicAdd on shared words).
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace zfp_sycl {

using Word = uint64_t;
constexpr int kBlock = 16;               // values per 2-D block
constexpr int kEbits = 9;                // float exponent bits + 1
constexpr int kEbias = 127;
constexpr int kIntPrec = 32;
constexpr int kMaxWords = 32;            // up to rate 128
constexpr uint32_t kNbMask = 0xaaaaaaaau;

// order coefficients (i, j) by i + j, then i^2 + j^2 (constants.h perm_2)
inline constexpr unsigned char kPerm2[16] = {0, 1, 4, 5, 2, 8, 6, 9, 3, 12, 10, 7, 13, 11, 14, 15};

inline int words_per_block(int rate) { return rate * kBlock / 64; }
inline size_t stream_words(int nx, int ny, int rate) {
  return size_t((nx + 3) / 4) * ((ny + 3) / 4) * words_per_block(rate);
}

// frexp exponent of a positive finite float, clamped to 1 - ebias for
// subnormals, as zfp's exponent(); -ebias for 0.
inline int exponent(float x) {
  if (!(x > 0)) return -kEbias;
  int biased = int((sycl::bit_cast<uint32_t>(x) >> 23) & 0xff);
  return sycl::max(biased - 126, 1 - kEbias);
}

inline void fwd_lift(int32_t *p, int s) {
  int32_t x = p[0], y = p[s], z = p[2 * s], w = p[3 * s];
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;
  p[0] = x; p[s] = y; p[2 * s] = z; p[3 * s] = w;
}

inline void inv_lift(int32_t *p, int s) {
  int32_t x = p[0], y = p[s], z = p[2 * s], w = p[3 * s];
  y += w >> 1; w -= y >> 1;
  y += w; w -= y - w;
  z += x; x -= z - x;
  y += z; z -= y - z;
  w += x; x -= w - x;
  p[0] = x; p[s] = y; p[2 * s] = z; p[3 * s] = w;
}

// encode2.cuh pad_block: fill the n valid values of a row/column out to 4
inline void pad_block(float *p, int n, int s) {
  switch (n) {
    case 0: p[0] = 0; [[fallthrough]];
    case 1: p[s] = p[0]; [[fallthrough]];
    case 2: p[2 * s] = p[s]; [[fallthrough]];
    case 3: p[3 * s] = p[0]; [[fallthrough]];
    default: break;
  }
}

struct Writer {
  Word *w; int pos = 0;
  void bits(uint64_t v, int n) {              // n <= 64, writes the low n bits
    if (n == 0) return;
    if (n < 64) v &= (uint64_t(1) << n) - 1;
    int i = pos >> 6, o = pos & 63;
    w[i] |= v << o;
    if (o && o + n > 64) w[i + 1] |= v >> (64 - o);
    pos += n;
  }
  bool bit(bool b) { if (b) w[pos >> 6] |= uint64_t(1) << (pos & 63); ++pos; return b; }
};

struct Reader {
  const Word *w; int pos = 0;
  uint64_t bits(int n) {                      // n <= 64
    if (n == 0) return 0;
    int i = pos >> 6, o = pos & 63;
    uint64_t v = w[i] >> o;
    if (o && o + n > 64) v |= w[i + 1] << (64 - o);
    if (n < 64) v &= (uint64_t(1) << n) - 1;
    pos += n;
    return v;
  }
  bool bit() { bool b = (w[pos >> 6] >> (pos & 63)) & 1; ++pos; return b; }
};

// One 4x4 block: values -> maxbits bits at out (zeroed by the caller).
inline void encode_block(const float *f, int maxbits, Word *out) {
  Writer wr{out};
  float m = 0;
  for (int i = 0; i < kBlock; i++) m = sycl::fmax(m, sycl::fabs(f[i]));
  int emax = exponent(m);
  // precision(emax, 32, -1074) is 32 for every float exponent
  unsigned e = unsigned(emax + kEbias);
  if (!e) return;                             // all zero: a 0 bit, then padding
  wr.bits(2 * e + 1, kEbits);
  float s = sycl::ldexp(1.0f, kIntPrec - 2 - emax);
  int32_t ib[kBlock];
  for (int i = 0; i < kBlock; i++) ib[i] = int32_t(s * f[i]);
  for (int y = 0; y < 4; y++) fwd_lift(ib + 4 * y, 1);
  for (int x = 0; x < 4; x++) fwd_lift(ib + x, 4);
  uint32_t ub[kBlock];
  for (int i = 0; i < kBlock; i++) ub[i] = (uint32_t(ib[kPerm2[i]]) + kNbMask) ^ kNbMask;
  unsigned bits = maxbits - kEbits;
  for (unsigned k = kIntPrec, n = 0; bits && k-- > 0;) {
    uint64_t x = 0;
    for (int i = 0; i < kBlock; i++) x += uint64_t((ub[i] >> k) & 1u) << i;
    unsigned mm = sycl::min(n, bits);
    bits -= mm;
    wr.bits(x, mm); x >>= mm;
    for (; n < kBlock && bits && (bits--, wr.bit(x != 0)); x >>= 1, n++)
      for (; n < kBlock - 1 && bits && (bits--, !wr.bit(x & 1u)); x >>= 1, n++)
        ;
  }
}

// One 4x4 block: maxbits bits at in -> values.
inline void decode_block(const Word *in, int maxbits, float *f) {
  Reader rd{in};
  for (int i = 0; i < kBlock; i++) f[i] = 0;
  if (!rd.bit()) return;
  int emax = int(rd.bits(kEbits - 1)) - kEbias;
  unsigned bits = maxbits - kEbits;
  uint32_t ub[kBlock] = {};
  for (unsigned k = kIntPrec, n = 0; bits && k-- > 0;) {
    unsigned mm = sycl::min(n, bits);
    bits -= mm;
    uint64_t x = rd.bits(mm);
    for (; bits && n < kBlock; n++) {
      bits--;
      if (rd.bit()) {
        for (; bits && n < kBlock - 1; n++) { bits--; if (rd.bit()) break; }
        x += uint64_t(1) << n;
      } else {
        break;
      }
    }
    for (int i = 0; i < kBlock; i++, x >>= 1) ub[i] += uint32_t(x & 1u) << k;
  }
  int32_t ib[kBlock];
  for (int i = 0; i < kBlock; i++) ib[kPerm2[i]] = int32_t((ub[i] ^ kNbMask) - kNbMask);
  for (int x = 0; x < 4; x++) inv_lift(ib + x, 4);
  for (int y = 0; y < 4; y++) inv_lift(ib + 4 * y, 1);
  float s = sycl::ldexp(1.0f, emax - (kIntPrec - 2));
  for (int i = 0; i < kBlock; i++) f[i] = s * float(ib[i]);
}

// data: ny rows of nx floats (C order, x fastest), on the device.
// stream: stream_words(nx, ny, rate) words, on the device.
inline sycl::event encode(sycl::queue &q, const float *data, int nx, int ny, int rate, Word *stream) {
  const int bx = (nx + 3) / 4, by = (ny + 3) / 4, W = words_per_block(rate), maxbits = 16 * rate;
  const size_t nb = size_t(bx) * by;
  return q.parallel_for(sycl::range<1>((nb + 255) / 256 * 256), [=](sycl::id<1> id) {
    size_t b = id[0];
    if (b >= nb) return;
    int x0 = int(b % bx) * 4, y0 = int(b / bx) * 4;
    int cx = sycl::min(4, nx - x0), cy = sycl::min(4, ny - y0);
    float f[kBlock];
    const float *p = data + size_t(y0) * nx + x0;
    if (cx == 4 && cy == 4) {
      for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) f[4 * y + x] = p[size_t(y) * nx + x];
    } else {                                  // encode2.cuh gather_partial2
      for (int y = 0; y < 4; y++)
        if (y < cy) {
          for (int x = 0; x < 4; x++) if (x < cx) f[4 * y + x] = p[size_t(y) * nx + x];
          pad_block(f + 4 * y, cx, 1);
        }
      for (int x = 0; x < 4; x++) pad_block(f + x, cy, 4);
    }
    Word buf[kMaxWords];
    for (int i = 0; i < W; i++) buf[i] = 0;
    encode_block(f, maxbits, buf);
    Word *o = stream + b * W;
    for (int i = 0; i < W; i++) o[i] = buf[i];
  });
}

inline sycl::event decode(sycl::queue &q, const Word *stream, int nx, int ny, int rate, float *data) {
  const int bx = (nx + 3) / 4, by = (ny + 3) / 4, W = words_per_block(rate), maxbits = 16 * rate;
  const size_t nb = size_t(bx) * by;
  return q.parallel_for(sycl::range<1>((nb + 255) / 256 * 256), [=](sycl::id<1> id) {
    size_t b = id[0];
    if (b >= nb) return;
    float f[kBlock];
    decode_block(stream + b * W, maxbits, f);
    int x0 = int(b % bx) * 4, y0 = int(b / bx) * 4;
    int cx = sycl::min(4, nx - x0), cy = sycl::min(4, ny - y0);
    float *p = data + size_t(y0) * nx + x0;
    for (int y = 0; y < cy; y++) for (int x = 0; x < cx; x++) p[size_t(y) * nx + x] = f[4 * y + x];
  });
}

}  // namespace zfp_sycl
