// Copyright © 2023-2024 Apple Inc.
// Derived MIT source from MLX quantized.h. See LICENSE and ATTRIBUTION.md.
// Experimental: no parity or speed claim before real GPU acceptance gates.
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// BEGIN PINNED HELPERS
template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U slot_load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U slot_load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U slot_qdot(
    const thread uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const thread uint16_t* ws = (const thread uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U slot_qdot_safe(
    const thread uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const thread uint16_t* ws = (const thread uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}
// END PINNED HELPERS

// Address-space adaptation above is mechanical. Slot scheduling below is new.
template <typename T, int GROUP_SIZE, int BITS, int SLOTS, int K, int N, int PACKS, int ROWS, bool SHARED, bool SAFE = false>
inline void slot_qmv_block(
    const device uint8_t* ws,
    const device T* scales,
    const device T* biases,
    const device T* x,
    thread float (&result)[SLOTS][ROWS],
    int out_row,
    int simd_lid,
    int k,
    int remaining = 0) {
  constexpr int values_per_thread = get_pack_factor<BITS, 32>() * PACKS;
  constexpr int weight_bytes = values_per_thread * BITS / 8;
  constexpr int row_bytes = K * get_bytes_per_pack<BITS, 32>() /
      get_pack_factor<BITS, 32>();
  constexpr int row_groups = K / GROUP_SIZE;
  constexpr int scale_step_per_thread = GROUP_SIZE / values_per_thread;
  thread float x_thread[SLOTS][values_per_thread];
  thread float sums[SLOTS];
  for (int slot = 0; slot < SLOTS; slot++) {
    const device T* xl = x + slot * K + k + simd_lid * values_per_thread;
    if constexpr (SAFE) {
      sums[slot] = slot_load_vector_safe<T, float, values_per_thread, BITS>(
          xl, x_thread[slot], remaining);
    } else {
      sums[slot] = slot_load_vector<T, float, values_per_thread, BITS>(
          xl, x_thread[slot]);
    }
  }
  for (int row = 0; row < ROWS; row++) {
    const device uint8_t* wl = ws + (out_row + row) * row_bytes +
        k * BITS / 8 + simd_lid * weight_bytes;
    const int group_index = (out_row + row) * row_groups + k / GROUP_SIZE +
        simd_lid / scale_step_per_thread;
    float s = scales[group_index];
    float b = biases[group_index];
    if constexpr (SHARED && !SAFE && BITS == 4 && (weight_bytes % 4) == 0) {
      // 031 shared dequant: each 4-weight chunk is decoded once and applied to every slot, using the
      // pinned 4-bit qdot expression and per-slot accumulation order unchanged.
      const device uint32_t* wl32 = (const device uint32_t*)wl;
      float acc[SLOTS];
      for (int slot = 0; slot < SLOTS; slot++) acc[slot] = 0;
      for (int word = 0; word < weight_bytes / 4; word++) {
        const uint32_t v = wl32[word];
        for (int hw = 0; hw < 2; hw++) {
          const uint16_t h = uint16_t(hw ? (v >> 16) : (v & 0xffffu));
          const int i = 2 * word + hw;
          const float f0 = float(h & 0x000f), f1 = float(h & 0x00f0),
                      f2 = float(h & 0x0f00), f3 = float(h & 0xf000);
          for (int slot = 0; slot < SLOTS; slot++) {
            acc[slot] += (x_thread[slot][4 * i] * f0 + x_thread[slot][4 * i + 1] * f1 +
                          x_thread[slot][4 * i + 2] * f2 + x_thread[slot][4 * i + 3] * f3);
          }
        }
      }
      for (int slot = 0; slot < SLOTS; slot++) {
        result[slot][row] += s * acc[slot] + sums[slot] * b;
      }
    } else {
      thread uint32_t weight_storage[(weight_bytes + 3) / 4];
      thread uint8_t* staged = (thread uint8_t*)weight_storage;
      const int loaded_bytes = SAFE ? remaining * BITS / 8 : weight_bytes;
      for (int byte = 0; byte < loaded_bytes; byte++) {
        staged[byte] = wl[byte];
      }
      for (int slot = 0; slot < SLOTS; slot++) {
        if constexpr (SAFE) {
          result[slot][row] += slot_qdot_safe<float, values_per_thread, BITS>(
              staged, x_thread[slot], s, b, sums[slot], remaining);
        } else {
          result[slot][row] += slot_qdot<float, values_per_thread, BITS>(
              staged, x_thread[slot], s, b, sums[slot]);
        }
      }
    }
  }
}
// END SLOT BLOCK

// 031 MODE 2 (slot-outer): packed weights for ROWS rows held in registers (shared read), x for one slot
// at a time (low register pressure). Pinned per-slot 4-bit qdot expression and per-output order unchanged.
template <typename T, int GROUP_SIZE, int BITS, int SLOTS, int K, int N, int PACKS, int ROWS>
inline void slot_qmv_block_outer(
    const device uint8_t* ws,
    const device T* scales,
    const device T* biases,
    const device T* x,
    thread float (&result)[SLOTS][ROWS],
    int out_row,
    int simd_lid,
    int k) {
  constexpr int values_per_thread = get_pack_factor<BITS, 32>() * PACKS;
  constexpr int weight_bytes = values_per_thread * BITS / 8;
  constexpr int words = weight_bytes / 4;
  constexpr int row_bytes = K * get_bytes_per_pack<BITS, 32>() / get_pack_factor<BITS, 32>();
  constexpr int row_groups = K / GROUP_SIZE;
  constexpr int scale_step_per_thread = GROUP_SIZE / values_per_thread;
  static_assert(BITS == 4 && weight_bytes % 4 == 0, "outer mode is 4-bit fast only");
  uint32_t wreg[ROWS][words];
  float sc[ROWS];
  float bi[ROWS];
  for (int row = 0; row < ROWS; row++) {
    const device uint32_t* wl32 = (const device uint32_t*)(ws + (out_row + row) * row_bytes +
        k * BITS / 8 + simd_lid * weight_bytes);
    for (int w = 0; w < words; w++) wreg[row][w] = wl32[w];
    const int group_index = (out_row + row) * row_groups + k / GROUP_SIZE +
        simd_lid / scale_step_per_thread;
    sc[row] = scales[group_index];
    bi[row] = biases[group_index];
  }
  for (int slot = 0; slot < SLOTS; slot++) {
    thread float xt[values_per_thread];
    const float sum = slot_load_vector<T, float, values_per_thread, BITS>(
        x + slot * K + k + simd_lid * values_per_thread, xt);
    for (int row = 0; row < ROWS; row++) {
      float accum = 0;
      for (int i = 0; i < values_per_thread / 4; i++) {
        const uint32_t v = wreg[row][i / 2];
        const uint16_t h = uint16_t((i & 1) ? (v >> 16) : (v & 0xffffu));
        accum += (xt[4 * i] * (h & 0x000f) + xt[4 * i + 1] * (h & 0x00f0) +
                  xt[4 * i + 2] * (h & 0x0f00) + xt[4 * i + 3] * (h & 0xf000));
      }
      result[slot][row] += sc[row] * accum + sum * bi[row];
    }
  }
}


template <typename T, int GROUP_SIZE, int BITS, int SLOTS, int K, int N, bool FAST, int ROWS, int SIMDS, int MODE>
inline void slot_qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    uint3 group,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BITS == 3 || BITS == 4, "only affine 3/4-bit admitted");
  static_assert(GROUP_SIZE == 32 || GROUP_SIZE == 64 || GROUP_SIZE == 128,
                "unsupported affine group size");
  static_assert(K > 0 && K % GROUP_SIZE == 0, "incomplete quantization group");
  static_assert(FAST == (K % 512 == 0), "must match stock single-token dispatch");
  static_assert(N >= 8 && N % 8 == 0, "no ragged output rows");
  static_assert(SLOTS >= 1 && SLOTS <= 9, "unsupported slot count");

  constexpr int packs_per_thread = FAST ? 2 : 1;
  constexpr int block_size = get_pack_factor<BITS, 32>() * packs_per_thread * 32;
  static_assert(N % (ROWS * SIMDS) == 0, "rows must tile N");
  const int out_row = int(group.y) * (ROWS * SIMDS) + int(simd_gid) * ROWS;
  const device uint8_t* ws = (const device uint8_t*)w;
  thread float result[SLOTS][ROWS] = {0};
  if constexpr (FAST && MODE == 2 && BITS == 4) {
    for (int k = 0; k < K; k += block_size) {
      slot_qmv_block_outer<T, GROUP_SIZE, BITS, SLOTS, K, N, packs_per_thread, ROWS>(
          ws, scales, biases, x, result, out_row, int(simd_lid), k);
    }
  } else if constexpr (FAST) {
    for (int k = 0; k < K; k += block_size) {
      slot_qmv_block<T, GROUP_SIZE, BITS, SLOTS, K, N, packs_per_thread, ROWS, (MODE == 1)>(
          ws, scales, biases, x, result, out_row, int(simd_lid), k);
    }
  } else {
    // Stock QMV deliberately uses safe helpers for the last step even if full.
    int k = 0;
    for (; k < K - block_size; k += block_size) {
      slot_qmv_block<T, GROUP_SIZE, BITS, SLOTS, K, N, packs_per_thread, ROWS, (MODE == 1)>(
          ws, scales, biases, x, result, out_row, int(simd_lid), k);
    }
    constexpr int values_per_thread = get_pack_factor<BITS, 32>() * packs_per_thread;
    const int remaining = clamp(K - k - int(simd_lid) * values_per_thread,
                                0, values_per_thread);
    if (remaining > 0) {
      slot_qmv_block<T, GROUP_SIZE, BITS, SLOTS, K, N, packs_per_thread, ROWS, (MODE == 1), true>(
          ws, scales, biases, x, result, out_row, int(simd_lid), k, remaining);
    }
  }
  for (int slot = 0; slot < SLOTS; slot++) {
    for (int row = 0; row < ROWS; row++) {
      result[slot][row] = simd_sum(result[slot][row]);
      if (simd_lid == 0) {
        y[slot * N + out_row + row] = static_cast<T>(result[slot][row]);
      }
    }
  }
}
