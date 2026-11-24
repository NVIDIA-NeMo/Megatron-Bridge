// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace {
constexpr int THREADS = 256;
constexpr int VEC = 8;

struct Multimax {
  float r[4], t[4];

  __device__ __forceinline__ float operator()(float x) const {
    const float a = fmaxf(r[0] - x, 0.f);
    const float b = fmaxf(x - r[1], 0.f);
    const float c = fmaxf(r[2] - x, 0.f);
    const float d = fmaxf(x - r[3], 0.f);
    float y = __fadd_rn(x, __fmul_rn(t[0], a));
    y = __fadd_rn(y, __fmul_rn(t[1], b));
    y = __fadd_rn(y, __fmul_rn(t[2], __fmul_rn(c, c)));
    y = __fadd_rn(y, __fmul_rn(t[3], __fmul_rn(d, d)));
    return __bfloat162float(__float2bfloat16_rn(y));
  }
};

__device__ __forceinline__ void load_eight(const __nv_bfloat16* ptr, uint32_t (&out)[4]) {
  const uint4 value = *reinterpret_cast<const uint4*>(ptr);
  out[0] = value.x;
  out[1] = value.y;
  out[2] = value.z;
  out[3] = value.w;
}

__device__ __forceinline__ float warp_max(float m) {
#pragma unroll
  for (int i = 16; i > 0; i >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, i));
  return m;
}
__device__ __forceinline__ float warp_sum(float s) {
#pragma unroll
  for (int i = 16; i > 0; i >>= 1) s += __shfl_xor_sync(0xffffffff, s, i);
  return s;
}
__device__ __forceinline__ int warp_min(int x) {
#pragma unroll
  for (int i = 16; i > 0; i >>= 1) x = min(x, __shfl_xor_sync(0xffffffff, x, i));
  return x;
}
struct Result {
  float m, s;
  int id;
};
struct Shared {
  float m[32], s[32];
  int id[32];
  Result row;
};

template <bool PRED, bool ANCHORED>
__device__ __forceinline__ Result reduce_row(const __nv_bfloat16* row, int vocab,
                                             const Multimax& transform, Shared& sm) {
  float lane_m[VEC], lane_s[VEC];
  int lane_id[VEC];
#pragma unroll
  for (int j = 0; j < VEC; j++) {
    lane_m[j] = -INFINITY;
    lane_s[j] = 0.f;
    lane_id[j] = INT_MAX;
  }
  // Vocab is divisible by eight. A whole vector is in or out of bounds.
  for (int offset = threadIdx.x * VEC; offset < vocab; offset += THREADS * VEC) {
    uint32_t words[VEC / 2];
    load_eight(row + offset, words);
#pragma unroll
    for (int j = 0; j < VEC; j++) {
      uint16_t bits = uint16_t(words[j / 2] >> (16 * (j % 2)));
      float y = transform(__bfloat162float(__ushort_as_bfloat16(bits)));
      float old = lane_m[j];
      if constexpr (ANCHORED) {
        lane_s[j] += __expf(y);  // discarded on outlying rows, which are recomputed stably
      } else {
        float adjust = __expf(-fabsf(y - old));
        lane_s[j] = (y > old) ? fmaf(lane_s[j], adjust, 1.f) : (lane_s[j] + adjust);
      }
      if constexpr (PRED)
        if (y > old) lane_id[j] = offset + j;
      lane_m[j] = fmaxf(old, y);
    }
  }
  float m = -INFINITY, s = 0.f;
  int id = INT_MAX;
#pragma unroll
  for (int j = 0; j < VEC; j++) m = fmaxf(m, lane_m[j]);
  float wm = warp_max(m);
#pragma unroll
  for (int j = 0; j < VEC; j++) {
    if constexpr (ANCHORED)
      s += lane_s[j];
    else
      s += lane_s[j] * __expf(lane_m[j] - wm);
    if constexpr (PRED)
      if (lane_m[j] == wm) id = min(id, lane_id[j]);
  }
  float ws = warp_sum(s);
  if constexpr (PRED) id = warp_min(id);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (!lane) {
    sm.m[warp] = wm;
    sm.s[warp] = ws;
    if constexpr (PRED) sm.id[warp] = id;
  }
  __syncthreads();
  if (warp == 0) {
    wm = lane < THREADS / 32 ? sm.m[lane] : -INFINITY;
    float mm = warp_max(wm);
    ws = lane < THREADS / 32 ? sm.s[lane] : 0.f;
    if constexpr (!ANCHORED) ws *= __expf(wm - mm);
    ws = warp_sum(ws);
    if constexpr (PRED) id = warp_min((lane < THREADS / 32 && wm == mm) ? sm.id[lane] : INT_MAX);
    if (lane == 0) {
      sm.row.m = mm;
      sm.row.s = ANCHORED ? ws * __expf(-mm) : ws;
      sm.row.id = id;
    }
  }
  __syncthreads();
  return sm.row;
}

template <bool PRED>
__global__ void head(const __nv_bfloat16* raw, const float* rt, const int64_t* labels, float* row_max,
                     float* row_sum, float* loss, int64_t* predictions, int vocab) {
  int row = blockIdx.x;
  const __nv_bfloat16* x = raw + int64_t(row) * vocab;
  Multimax transform;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    transform.r[i] = rt[i];
    transform.t[i] = rt[i + 4];
  }
  __shared__ Shared sm;
  Result r = reduce_row<PRED, true>(x, vocab, transform, sm);
  if (!(r.m >= -60.f && r.m <= 60.f)) {
    // The first pass's zero-anchor exponential sum may underflow/overflow here.
    // Reload the row with exact-tie, max-relative online normalization.
    r = reduce_row<PRED, false>(x, vocab, transform, sm);
  }
  if (threadIdx.x == 0) {
    const int64_t label = labels[row];
    bool valid = label >= 0 && label < vocab;
    float shifted = valid ? transform(__bfloat162float(x[label])) - r.m : 0.f;
    row_max[row] = r.m;
    row_sum[row] = r.s;
    loss[row] = __logf(r.s) - shifted;
    if constexpr (PRED) predictions[row] = r.id;
  }
}

}  // namespace

// Fingerprint and target are supplied by the matching Bridge prebuild command.
extern "C" const char* ernie55_multimax_ce_source() { return ERNIE55_CUDA_HEAD_SOURCE; }
extern "C" const char* ernie55_multimax_ce_arch() { return ERNIE55_CUDA_HEAD_ARCH; }

extern "C" int ernie55_multimax_ce_forward(const __nv_bfloat16* raw, const float* rt,
                                           const int64_t* labels, float* row_max, float* row_sum,
                                           float* loss, int64_t* predictions, int rows, int vocab,
                                           cudaStream_t stream) {
  if (rows <= 0 || vocab < THREADS * VEC || vocab % VEC != 0 ||
      (reinterpret_cast<uintptr_t>(raw) & 15U) != 0) {
    return int(cudaErrorInvalidValue);
  }
  if (predictions) {
    head<true>
        <<<rows, THREADS, 0, stream>>>(raw, rt, labels, row_max, row_sum, loss, predictions, vocab);
  } else {
    head<false><<<rows, THREADS, 0, stream>>>(raw, rt, labels, row_max, row_sum, loss, nullptr, vocab);
  }
  return int(cudaGetLastError());
}
