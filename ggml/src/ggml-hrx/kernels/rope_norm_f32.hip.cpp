#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// P058: NORMAL-mode (adjacent-pair) rope with llama3 freq_factors, single position, no YaRN (ext_factor == 0).
// Llama-3.x rope is GGML_ROPE_TYPE_NORMAL (mode 0): each adjacent pair (2k, 2k+1) is rotated by theta for
// dimension k, with theta scaled by the per-dimension freq_factors[k] (rope_freqs.weight). Math mirrors ggml's
// rope_norm (ggml/src/ggml-cuda/rope.cu); the sibling hrx_rope_f32 kernel does IMROPE (sectioned, split-half).
//
// ABI mirrors hrx_rope_f32 (rope_f32.hip.cpp): the bindings are the leading pointer args and the constants
// struct is the trailing by-value kernarg. Bindings (4): src0, pos, freq_factors, dst. Reuses the 120-byte
// hrx_rope_f32_constants layout (the section fields are unused here).
struct hrx_rope_f32_constants {
    long long ne00;
    long long ne01;
    long long ne02;
    long long nrows;
    long long src_s1;
    long long src_s2;
    long long src_s3;
    long long dst_s1;
    long long dst_s2;
    long long dst_s3;
    int n_dims;
    int mode;
    int section0;
    int section1;
    int section2;
    int section3;
    float freq_base;
    float freq_scale;
    float attn_factor;
    float _pad;
};

extern "C" __global__ void hrx_rope_norm_f32(
        const float * src, const int * pos, const float * freq_factors, float * dst,
        hrx_rope_f32_constants c) {
    const long long pair = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long pairs_per_row = c.ne00 / 2;
    const long long total_pairs = c.nrows * pairs_per_row;
    if (pair >= total_pairs) {
        return;
    }

    const long long row = pair / pairs_per_row;
    const long long pair_in_row = pair - row * pairs_per_row;
    const int i0 = static_cast<int>(2 * pair_in_row);
    const long long i3 = row / (c.ne01 * c.ne02);
    const long long i2 = (row - i3 * c.ne01 * c.ne02) / c.ne01;
    const long long i1 = row - i3 * c.ne01 * c.ne02 - i2 * c.ne01;
    const long long src_base = i1 * c.src_s1 + i2 * c.src_s2 + i3 * c.src_s3;
    const long long dst_base = i1 * c.dst_s1 + i2 * c.dst_s2 + i3 * c.dst_s3;

    // Dimensions at/after n_dims are passed through unchanged (adjacent layout). For Llama n_dims == ne00, so
    // this branch is not taken; kept for partial-rope generality.
    if (i0 >= c.n_dims) {
        dst[dst_base + i0 + 0] = src[src_base + i0 + 0];
        dst[dst_base + i0 + 1] = src[src_base + i0 + 1];
        return;
    }

    const float theta_scale  = powf(c.freq_base, -2.0f / static_cast<float>(c.n_dims));
    const float freq_factor  = freq_factors ? freq_factors[i0 / 2] : 1.0f;
    const float theta_base   = static_cast<float>(pos[i2]) * powf(theta_scale, static_cast<float>(i0) / 2.0f);
    const float theta        = c.freq_scale * (theta_base / freq_factor);
    const float cos_theta    = cosf(theta) * c.attn_factor;
    const float sin_theta    = sinf(theta) * c.attn_factor;

    // NORMAL mode: rotate the adjacent pair (i0, i0+1).
    const float x0 = src[src_base + i0 + 0];
    const float x1 = src[src_base + i0 + 1];
    dst[dst_base + i0 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[dst_base + i0 + 1] = x0 * sin_theta + x1 * cos_theta;
}
