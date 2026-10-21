// CPU (libzfp) vs PVC (zfp_sycl.hpp) fixed-rate zfp on a simulated 2-D float32 field.
//
//   zfp_bench <nx> <ny> <rate> <field> [reps] [threads]   -> one JSON line
//   field: smooth (geolocation-like: a gradient with gentle curvature)
//          radiance (smooth scene + structure at 4-40 px + 0.5% noise)
//
// Compression:   libzfp serial | libzfp OpenMP (threads) | PVC
// Decompression: libzfp serial | libzfp on `threads` std::threads, each
//                decoding its own strip of block rows from the same stream
//                (libzfp has no parallel decompression; in fixed-rate mode a
//                strip starts at a known word) | PVC
// The SYCL device is the default one (a PVC tile on a compute node;
// ONEAPI_DEVICE_SELECTOR=opencl:cpu runs the same kernels on the CPU).
// PVC times are given as kernel only and end to end (host -> device copy,
// kernel, device -> host copy) from pinned host memory.
//
// Run without OMP_PROC_BIND: libgomp then pins the main thread to one core,
// the strip decoders inherit that mask, and they run one at a time.
//
// Every stream and every decoded array must match libzfp serial bit for bit;
// "match" in the output reports it. Times are the median of `reps`, after
// one untimed warm-up of each path.
#include <sycl/sycl.hpp>
#include <zfp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include "zfp_sycl.hpp"

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
template <class F> static double timed(int reps, F f) {
  f();
  std::vector<double> t;
  for (int r = 0; r < reps; r++) { auto a = clk::now(); f(); t.push_back(secs(a, clk::now())); }
  return median(t);
}

static void simulate(float *a, int nx, int ny, const std::string &kind) {
  uint64_t s = 88172645463325252ull;
  auto rnd = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return double(s >> 11) / double(1ull << 53) - 0.5; };
  for (int y = 0; y < ny; y++)
    for (int x = 0; x < nx; x++) {
      double u = double(x) / nx, v = double(y) / ny, val;
      if (kind == "smooth") val = -30 + 25 * v + 3 * u * u - 1.5 * u * v;
      else val = 90 + 40 * std::sin(6.3 * u) * std::cos(4.1 * v) + 12 * std::sin(x / 4.0) * std::sin(y / 6.5)
                 + 6 * std::cos(x / 37.0 + y / 23.0) + 0.45 * rnd();
      a[size_t(y) * nx + x] = float(val);
    }
}

static zfp_stream *open_stream(int rate, void *buf, size_t bytes) {
  zfp_stream *z = zfp_stream_open(nullptr);
  zfp_stream_set_rate(z, rate, zfp_type_float, 2, zfp_false);
  bitstream *b = stream_open(buf, bytes);
  zfp_stream_set_bit_stream(z, b);
  zfp_stream_rewind(z);
  return z;
}
static void close_stream(zfp_stream *z) { stream_close(zfp_stream_bit_stream(z)); zfp_stream_close(z); }

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: zfp_bench nx ny rate smooth|radiance [reps] [threads]\n"); return 2; }
  const int nx = atoi(argv[1]), ny = atoi(argv[2]), rate = atoi(argv[3]);
  const std::string kind = argv[4];
  const int reps = argc > 5 ? atoi(argv[5]) : 5;
  const int T = argc > 6 ? atoi(argv[6]) : int(std::thread::hardware_concurrency());
  if (rate % 4 || rate <= 0 || rate > 128) { fprintf(stderr, "rate must be a multiple of 4, <= 128\n"); return 2; }

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};
  const size_t n = size_t(nx) * ny, words = zfp_sycl::stream_words(nx, ny, rate), sbytes = words * 8;
  const int bx = (nx + 3) / 4, by = (ny + 3) / 4, W = zfp_sycl::words_per_block(rate);

  float *src = sycl::malloc_host<float>(n, q), *out_cpu = sycl::malloc_host<float>(n, q),
        *out_par = sycl::malloc_host<float>(n, q), *out_gpu = sycl::malloc_host<float>(n, q);
  uint64_t *s_cpu = sycl::malloc_host<uint64_t>(words + 8, q), *s_omp = sycl::malloc_host<uint64_t>(words + 8, q),
           *s_gpu = sycl::malloc_host<uint64_t>(words + 8, q);
  float *d_data = sycl::malloc_device<float>(n, q);
  uint64_t *d_stream = sycl::malloc_device<uint64_t>(words, q);
  simulate(src, nx, ny, kind);
  zfp_field *field = zfp_field_2d(src, zfp_type_float, nx, ny);

  // ---- compression
  size_t zbytes = 0;
  double c_cpu = timed(reps, [&] {
    memset(s_cpu, 0, sbytes); zfp_stream *z = open_stream(rate, s_cpu, sbytes + 64);
    zbytes = zfp_compress(z, field); close_stream(z);
  });
  double c_omp = timed(reps, [&] {
    memset(s_omp, 0, sbytes); zfp_stream *z = open_stream(rate, s_omp, sbytes + 64);
    zfp_stream_set_execution(z, zfp_exec_omp); zfp_stream_set_omp_threads(z, T);
    zfp_compress(z, field); close_stream(z);
  });
  double c_gpu = timed(reps, [&] {
    q.memcpy(d_data, src, n * 4).wait();
    zfp_sycl::encode(q, d_data, nx, ny, rate, d_stream).wait();
    q.memcpy(s_gpu, d_stream, sbytes).wait();
  });
  std::vector<double> kt;
  for (int r = 0; r < reps; r++) {
    auto e = zfp_sycl::encode(q, d_data, nx, ny, rate, d_stream); e.wait();
    kt.push_back((e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                  e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9);
  }
  double c_gpu_k = median(kt);
  bool cs_omp = memcmp(s_cpu, s_omp, sbytes) == 0, cs_gpu = memcmp(s_cpu, s_gpu, sbytes) == 0;

  // ---- decompression (all from the libzfp serial stream)
  zfp_field *fout = zfp_field_2d(out_cpu, zfp_type_float, nx, ny);
  double d_cpu = timed(reps, [&] {
    zfp_stream *z = open_stream(rate, s_cpu, sbytes + 64); zfp_decompress(z, fout); close_stream(z);
  });
  double d_par = timed(reps, [&] {
    std::vector<std::thread> th;
    for (int t = 0; t < T; t++)
      th.emplace_back([&, t] {
        int r0 = int(int64_t(by) * t / T), r1 = int(int64_t(by) * (t + 1) / T);
        if (r0 == r1) return;
        int rows = std::min(4 * r1, ny) - 4 * r0;
        size_t w0 = size_t(r0) * bx * W;
        zfp_field *f = zfp_field_2d(out_par + size_t(4 * r0) * nx, zfp_type_float, nx, rows);
        zfp_stream *z = open_stream(rate, s_cpu + w0, sbytes - w0 * 8 + 64);
        zfp_decompress(z, f); close_stream(z); zfp_field_free(f);
      });
    for (auto &x : th) x.join();
  });
  q.memcpy(d_stream, s_cpu, sbytes).wait();
  double d_gpu = timed(reps, [&] {
    q.memcpy(d_stream, s_cpu, sbytes).wait();
    zfp_sycl::decode(q, d_stream, nx, ny, rate, d_data).wait();
    q.memcpy(out_gpu, d_data, n * 4).wait();
  });
  kt.clear();
  for (int r = 0; r < reps; r++) {
    auto e = zfp_sycl::decode(q, d_stream, nx, ny, rate, d_data); e.wait();
    kt.push_back((e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                  e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9);
  }
  double d_gpu_k = median(kt);
  double h2d = timed(reps, [&] { q.memcpy(d_data, src, n * 4).wait(); });
  bool ds_par = memcmp(out_cpu, out_par, n * 4) == 0, ds_gpu = memcmp(out_cpu, out_gpu, n * 4) == 0;

  double emax = 0, esq = 0;
  for (size_t i = 0; i < n; i++) { double e = std::fabs(double(out_cpu[i]) - src[i]); emax = std::max(emax, e); esq += e * e; }
  const double mb = n * 4 / 1e6;
  auto gbs = [&](double t) { return mb / 1e3 / t; };
  printf("{\"nx\":%d,\"ny\":%d,\"field\":\"%s\",\"rate\":%d,\"MB\":%.1f,\"zfp_MB\":%.2f,\"threads\":%d,\"reps\":%d,"
         "\"device\":\"%s\",\"max_err\":%.4g,\"rms_err\":%.4g,"
         "\"compress_s\":{\"cpu_1\":%.5f,\"cpu_omp\":%.5f,\"gpu_kernel\":%.5f,\"gpu_e2e\":%.5f},"
         "\"decompress_s\":{\"cpu_1\":%.5f,\"cpu_strips\":%.5f,\"gpu_kernel\":%.5f,\"gpu_e2e\":%.5f},"
         "\"h2d_raw_s\":%.5f,"
         "\"compress_GBs\":{\"cpu_1\":%.2f,\"cpu_omp\":%.2f,\"gpu_kernel\":%.2f,\"gpu_e2e\":%.2f},"
         "\"decompress_GBs\":{\"cpu_1\":%.2f,\"cpu_strips\":%.2f,\"gpu_kernel\":%.2f,\"gpu_e2e\":%.2f},"
         "\"match\":{\"stream_omp\":%s,\"stream_gpu\":%s,\"values_strips\":%s,\"values_gpu\":%s,\"zfp_bytes\":%s}}\n",
         nx, ny, kind.c_str(), rate, mb, sbytes / 1e6, T, reps,
         q.get_device().get_info<sycl::info::device::name>().c_str(), emax, std::sqrt(esq / n),
         c_cpu, c_omp, c_gpu_k, c_gpu, d_cpu, d_par, d_gpu_k, d_gpu, h2d,
         gbs(c_cpu), gbs(c_omp), gbs(c_gpu_k), gbs(c_gpu), gbs(d_cpu), gbs(d_par), gbs(d_gpu_k), gbs(d_gpu),
         cs_omp ? "true" : "false", cs_gpu ? "true" : "false", ds_par ? "true" : "false", ds_gpu ? "true" : "false",
         zbytes == sbytes ? "true" : "false");
  return (cs_omp && cs_gpu && ds_par && ds_gpu) ? 0 : 1;
}
