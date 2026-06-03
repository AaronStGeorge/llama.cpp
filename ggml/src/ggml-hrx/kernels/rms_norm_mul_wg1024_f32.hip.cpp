// A 1024-thread workgroup variant of the fused RMS_NORM + MUL kernel, selected
// for wide rows (ncols >= 1024) at decode (nrows == 1); opt out with
// GGML_HRX_DISABLE_RMS_NORM_MUL_WG1024. The 512-thread hrx_rms_norm_mul_f32
// launches ONE 512-thread
// workgroup per row; for a single decode row of 4096 cols that one workgroup is
// latency-bound on the strided gather (8 loads/thread). The HIP rms_norm_f32
// launches 1024 threads for ncols >= 1024 (norm.cu: block_dims(1024)); doubling
// the threads doubles the in-flight memory operations in the single workgroup,
// hiding more load latency. This is a pure width change: the per-thread math,
// the constants ABI (hrx_rms_norm_mul_constants, 144 bytes) and the f32
// arithmetic are identical to the 512-thread kernel, so the result is
// bit-for-bit the same as hrx_rms_norm_mul_f32. Builtins-only (freestanding,
// device-only); no hip_runtime.h symbols beyond the struct/types.
#include <hip/hip_runtime.h>

struct hrx_rms_norm_mul_constants {
    long long ncols;
    long long nrows;
    long long ne1;
    long long ne2;
    long long src_nb1;
    long long src_nb2;
    long long src_nb3;
    long long weight_ne0;
    long long weight_ne1;
    long long weight_ne2;
    long long weight_ne3;
    long long weight_nb1;
    long long weight_nb2;
    long long weight_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    float eps;
    int _pad;
};

template <int WG_SIZE>
static __device__ float hrx_rms_norm_mul_wg1024_reduce(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 31;
    const unsigned int wave = tid >> 5;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __builtin_amdgcn_s_barrier();

    sum = lane < waves ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
        if (lane == 0) {
            shared[0] = sum;
        }
    }
    __builtin_amdgcn_s_barrier();
    return shared[0];
}

template <int WG_SIZE>
static __device__ void hrx_rms_norm_mul_wg1024_impl(
        const float * src, const float * weight, float * dst,
        hrx_rms_norm_mul_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float sumsh[(WG_SIZE + 31) / 32];

    const long long i3 = row / (c.ne1 * c.ne2);
    const long long i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    const long long i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;
    const long long wi1 = c.weight_ne1 == 1 ? 0 : i1;
    const long long wi2 = c.weight_ne2 == 1 ? 0 : i2;
    const long long wi3 = c.weight_ne3 == 1 ? 0 : i3;

    const char * src_row = reinterpret_cast<const char *>(src) +
        i1 * c.src_nb1 + i2 * c.src_nb2 + i3 * c.src_nb3;
    const char * weight_row = reinterpret_cast<const char *>(weight) +
        wi1 * c.weight_nb1 + wi2 * c.weight_nb2 + wi3 * c.weight_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) +
        i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;

    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += WG_SIZE) {
        const float value = *reinterpret_cast<const float *>(src_row + col * sizeof(float));
        sum += value * value;
    }

    const float scale = 1.0f / __builtin_sqrtf(hrx_rms_norm_mul_wg1024_reduce<WG_SIZE>(sum, sumsh) / (float) c.ncols + c.eps);
    for (long long col = tid; col < c.ncols; col += WG_SIZE) {
        const long long wcol = c.weight_ne0 == 1 ? 0 : col;
        const float src_value = *reinterpret_cast<const float *>(src_row + col * sizeof(float));
        const float weight_value = *reinterpret_cast<const float *>(weight_row + wcol * sizeof(float));
        *reinterpret_cast<float *>(dst_row + col * sizeof(float)) = src_value * scale * weight_value;
    }
}

extern "C" __global__ void hrx_rms_norm_mul_wg1024_f32(
        const float * src, const float * weight, float * dst,
        hrx_rms_norm_mul_constants c) {
    hrx_rms_norm_mul_wg1024_impl<1024>(src, weight, dst, c);
}
