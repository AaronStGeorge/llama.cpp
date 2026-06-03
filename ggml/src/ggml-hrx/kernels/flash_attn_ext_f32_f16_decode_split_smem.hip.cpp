#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

// P058 fa_dense_01: LDS-SHARED-K/V split-K decode flash attention (lever B2). Byte-identical math /
// reductions / scratch layout / ABI to hrx_flash_attn_ext_f32_f16_decode_split (the proven float4 8-way
// split). The ONLY change: the 4 row_groups (4 wavefronts) of one workgroup are the 4 GQA query heads of
// ONE kv_head and scan the SAME K/V; instead of each wavefront re-loading K/V from global f16, all 128
// threads cooperatively stage each BC=32 K-tile and V-tile into __shared__ ONCE per tile, then the 4
// row_groups read K/V from LDS. This cuts global K/V traffic ~4x at long KV (where the per-element K/V
// loads dominate the split kernel), keeping the same 8-way split / 64-wg grid {H_KV*SPLITS, N, S} (the
// occupancy the prior dead-ends proved is required) and the same _combine. Online softmax stays f32.
struct hrx_flash_attn_ext_f32_f16_decode_constants {
    long long D;
    long long KV;
    long long N;
    long long H;
    long long H_KV;
    long long S;
    long long q_nb1;
    long long q_nb2;
    long long q_nb3;
    long long k_nb1;
    long long k_nb2;
    long long k_nb3;
    long long v_nb1;
    long long v_nb2;
    long long v_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    long long mask_nb0;
    long long mask_nb1;
    long long mask_nb3;
    float scale;
    int has_mask;
    float max_bias;
    float m0;
    float m1;
    float logit_softcap;
    int n_head_log2;
    int has_sinks;
};

static __device__ __forceinline__ float hrx_fa_split_load_f16(const __half * base, long long byte_offset) {
    return __half2float(*reinterpret_cast<const __half *>(reinterpret_cast<const char *>(base) + byte_offset));
}

static __device__ __forceinline__ float4 hrx_fa_split_load_f16x4(const __half * h) {
    return make_float4(__half2float(h[0]), __half2float(h[1]), __half2float(h[2]), __half2float(h[3]));
}

static __device__ __forceinline__ float4 hrx_fa_split_load_f32x4(const char * ptr) {
    return *reinterpret_cast<const float4 *>(ptr);
}

static __device__ __forceinline__ float hrx_fa_split_dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static __device__ __forceinline__ float4 hrx_fa_split_f4_zero() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

static __device__ __forceinline__ float4 hrx_fa_split_f4_madd(float4 acc, float scale, float4 v) {
    acc.x += scale * v.x;
    acc.y += scale * v.y;
    acc.z += scale * v.z;
    acc.w += scale * v.w;
    return acc;
}

// Reduce across the d_tid dimension (D_SPLIT=8 lanes -> masks 4,2,1). Unchanged from the global-load split.
static __device__ __forceinline__ float hrx_fa_split_sum_dsplit(float v) {
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        v += __shfl_xor(v, mask, 32);
    }
    return v;
}

// Reduce across the col_tid dimension (col_tid in [0,4) -> masks 8,16). Unchanged from the global-load split.
static __device__ __forceinline__ float hrx_fa_split_sum_cols(float v) {
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 16, 32);
    return v;
}

static __device__ __forceinline__ float hrx_fa_split_max_cols(float v) {
    v = fmaxf(v, __shfl_xor(v, 8, 32));
    v = fmaxf(v, __shfl_xor(v, 16, 32));
    return v;
}

static __device__ __forceinline__ float4 hrx_fa_split_sum_cols4(float4 v) {
    v.x = hrx_fa_split_sum_cols(v.x);
    v.y = hrx_fa_split_sum_cols(v.y);
    v.z = hrx_fa_split_sum_cols(v.z);
    v.w = hrx_fa_split_sum_cols(v.w);
    return v;
}

static __device__ __forceinline__ float4 hrx_fa_split_scale4(float4 v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

static __device__ __forceinline__ float hrx_fa_split_alibi_slope(
        const hrx_flash_attn_ext_f32_f16_decode_constants c,
        long long head) {
    if (c.max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = head < c.n_head_log2 ? c.m0 : c.m1;
    const int exp_h = head < c.n_head_log2 ? static_cast<int>(head + 1) :
        static_cast<int>(2 * (head - c.n_head_log2) + 1);
    return powf(base, exp_h);
}

extern "C" __global__ __launch_bounds__(128) void hrx_flash_attn_ext_f32_f16_decode_split_smem(
        const float * q,
        const __half * k,
        const __half * v,
        const __half * mask,
        const float * sinks,
        float * dst,
        float * scratch,
        hrx_flash_attn_ext_f32_f16_decode_constants c) {
    (void) sinks;   // sinks + normalization are the combine kernel's job
    (void) dst;     // present only to keep the binding order identical to the single-kernel ABI
    constexpr int GQA = 4;
    constexpr int SPLITS = 8;
    constexpr int D = 128;
    constexpr int D_SPLIT = 8;
    constexpr int BC = 32;
    constexpr int COLS_PER_THREAD = 8;
    constexpr int VEC_PER_THREAD = D / (4 * D_SPLIT);   // 128 / 32 = 4

    // One BC=32 x D=128 f16 tile each for K and V, staged ONCE per tile and shared by the 4 row_groups.
    // 32*128 = 4096 halves = 8 KiB each, 16 KiB total (gfx1100 has 64 KiB LDS/wg -> >= 4 wg/CU by LDS).
    __shared__ __half smem_k[BC * D];
    __shared__ __half smem_v[BC * D];

    const long long x = __builtin_amdgcn_workgroup_id_x();
    const long long split = x % SPLITS;
    const long long kv_head = x / SPLITS;
    const long long token = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 31;
    const unsigned int row_group = tid >> 5;                 // one query head per row_group (GQA-4 -> 4 heads)
    const unsigned int d_tid = lane & (D_SPLIT - 1);
    const unsigned int col_tid = lane >> 3;

    if (kv_head >= c.H_KV || token >= c.N || seq >= c.S || c.D != D || c.H != c.H_KV * GQA) {
        return;
    }

    const long long split_chunk = (((c.KV + SPLITS - 1) / SPLITS) + (BC - 1)) & ~(static_cast<long long>(BC) - 1);
    const long long split_begin = split * split_chunk;
    const long long split_end = split_begin + split_chunk < c.KV ? split_begin + split_chunk : c.KV;

    const long long head = kv_head * GQA + row_group;
    const bool valid = row_group < GQA && head < c.H;

    const char * k_head = reinterpret_cast<const char *>(k) + kv_head * c.k_nb2 + seq * c.k_nb3;
    const char * v_head = reinterpret_cast<const char *>(v) + kv_head * c.v_nb2 + seq * c.v_nb3;
    const char * q_head = reinterpret_cast<const char *>(q) + token * c.q_nb1 + head * c.q_nb2 + seq * c.q_nb3;
    const char * mask_row = reinterpret_cast<const char *>(mask) + token * c.mask_nb1 + seq * c.mask_nb3;
    const float slope = valid ? hrx_fa_split_alibi_slope(c, head) : 1.0f;

    float l = 0.0f;
    float m = -FLT_MAX * 0.5f;
    float4 out[VEC_PER_THREAD];
#pragma unroll
    for (int d = 0; d < VEC_PER_THREAD; ++d) {
        out[d] = hrx_fa_split_f4_zero();
    }

    // Cooperative-staging geometry: the 32x128 f16 K (and V) tile is 8 KiB = 512 x 16-byte chunks. 128
    // threads each copy 4 chunks. chunk g -> column (g >> 4) in [0,32), within-column float4 (g & 15) in
    // [0,16) (16 float4 = 128 halves per column). Address from k_head + col*k_nb1 honours padded row
    // strides (k_nb1 may exceed 256). The f16 bits are copied verbatim (no convert) via a 16-byte load.
    constexpr int CHUNKS = (BC * D * (int) sizeof(__half)) / 16;   // 8192 / 16 = 512
    constexpr int CHUNKS_PER_THREAD = CHUNKS / 128;                // 512 / 128 = 4

    for (long long jb = split_begin; jb < split_end; jb += BC) {
        // Stage this tile's K and V into LDS, cooperatively, once for all 4 row_groups.
#pragma unroll
        for (int t = 0; t < CHUNKS_PER_THREAD; ++t) {
            const int g = static_cast<int>(tid) + t * 128;
            const int col = g >> 4;                     // 0..31
            const int f4  = g & 15;                     // 0..15 (float4 within the 128-half column)
            const long long kv_col = jb + col;
            const int half_off = f4 * 8;                // 0..127, first of 8 halves this chunk copies
            float4 * dst_k = reinterpret_cast<float4 *>(smem_k + col * D + half_off);
            float4 * dst_v = reinterpret_cast<float4 *>(smem_v + col * D + half_off);
            if (kv_col < c.KV) {
                const char * k_src = k_head + kv_col * c.k_nb1 + half_off * static_cast<int>(sizeof(__half));
                const char * v_src = v_head + kv_col * c.v_nb1 + half_off * static_cast<int>(sizeof(__half));
                *dst_k = *reinterpret_cast<const float4 *>(k_src);
                *dst_v = *reinterpret_cast<const float4 *>(v_src);
            } else {
                *dst_k = hrx_fa_split_f4_zero();
                *dst_v = hrx_fa_split_f4_zero();
            }
        }
        __builtin_amdgcn_s_barrier();

        float scores[COLS_PER_THREAD];
#pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const int tile_col = ci * 4 + static_cast<int>(col_tid);
            const long long kv_col = jb + tile_col;
            const bool valid_col = kv_col < split_end;
            float mask_value = 0.0f;
            if (c.has_mask && valid_col) {
                mask_value = hrx_fa_split_load_f16(reinterpret_cast<const __half *>(mask_row), kv_col * c.mask_nb0);
            }

            float s = 0.0f;
            if (valid_col && (!c.has_mask || mask_value > -60000.0f)) {
                const __half * k_row = smem_k + tile_col * D;
#pragma unroll
                for (int d = 0; d < VEC_PER_THREAD; ++d) {
                    const int vec_index = d * D_SPLIT + static_cast<int>(d_tid);
                    const float4 kv = hrx_fa_split_load_f16x4(k_row + vec_index * 4);
                    if (valid) {
                        const int byte_offset_f32 = vec_index * 4 * static_cast<int>(sizeof(float));
                        const float4 qv = hrx_fa_split_scale4(
                            hrx_fa_split_load_f32x4(q_head + byte_offset_f32), c.scale);
                        s += hrx_fa_split_dot4(qv, kv);
                    }
                }
                s = hrx_fa_split_sum_dsplit(s);
                if (c.logit_softcap != 0.0f) {
                    s = c.logit_softcap * tanhf(s);
                }
                if (c.has_mask) {
                    s += slope * mask_value;
                }
            } else {
                s = -FLT_MAX * 0.5f;
            }
            scores[ci] = valid ? s : -FLT_MAX * 0.5f;
        }

        float row_max = -FLT_MAX * 0.5f;
#pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            row_max = fmaxf(row_max, scores[ci]);
        }
        row_max = hrx_fa_split_max_cols(row_max);

        const float old_m = m;
        m = fmaxf(m, row_max);
        const float old_scale = expf(old_m - m);
        l *= old_scale;
#pragma unroll
        for (int d = 0; d < VEC_PER_THREAD; ++d) {
            out[d] = hrx_fa_split_scale4(out[d], old_scale);
        }

#pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const int tile_col = ci * 4 + static_cast<int>(col_tid);
            const long long kv_col = jb + tile_col;
            if (kv_col >= split_end) {
                continue;
            }
            const float p = expf(scores[ci] - m);
            l += p;
            const __half * v_row = smem_v + tile_col * D;
#pragma unroll
            for (int d = 0; d < VEC_PER_THREAD; ++d) {
                const int vec_index = d * D_SPLIT + static_cast<int>(d_tid);
                const float4 vv = hrx_fa_split_load_f16x4(v_row + vec_index * 4);
                out[d] = hrx_fa_split_f4_madd(out[d], p, vv);
            }
        }
        // WAR guard: the next tile's staging must not overwrite LDS that lagging lanes still read here.
        __builtin_amdgcn_s_barrier();
    }

    l = hrx_fa_split_sum_cols(l);
#pragma unroll
    for (int d = 0; d < VEC_PER_THREAD; ++d) {
        out[d] = hrx_fa_split_sum_cols4(out[d]);
    }

    if (col_tid == 0) {
        const size_t partial_count = static_cast<size_t>(c.S) * static_cast<size_t>(c.N) *
            static_cast<size_t>(c.H) * SPLITS;
        float * scratch_o = scratch;
        float * scratch_l = scratch_o + partial_count * D;
        float * scratch_m = scratch_l + partial_count;
        const size_t base = (((static_cast<size_t>(seq) * c.N + static_cast<size_t>(token)) *
            c.H + static_cast<size_t>(head)) * SPLITS + static_cast<size_t>(split));
        if (d_tid == 0 && valid) {
            scratch_l[base] = l;
            scratch_m[base] = m;
        }
#pragma unroll
        for (int d = 0; d < VEC_PER_THREAD; ++d) {
            const int vec_index = d * D_SPLIT + static_cast<int>(d_tid);
            if (valid) {
                *reinterpret_cast<float4 *>(scratch_o + base * D + vec_index * 4) = out[d];
            }
        }
    }
}
