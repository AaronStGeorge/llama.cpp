#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

// Default-on (opt out via GGML_HRX_DISABLE_SET_ROWS_FASTDIV): a drop-in faster
// variant of hrx_set_rows_f32_f16. Identical element-per-thread mapping and f32->f16 store,
// but the linear-index decomposition is done with host-precomputed multiply-shift
// fast-division (the technique CUDA/HIP k_set_rows uses, common.cuh
// init_fastdiv_values / fast_div_modulo) instead of six 64-bit integer % and /
// per thread. gfx11 has no hardware 64-bit idiv, so the original kernel pays a
// long emulated-division sequence per thread; the multiply-shift form is a single
// 32x32->64 high-multiply plus a shift. At decode the set_rows that write the KV
// cache are tiny (nc = n_embd_v_gqa = 1024, nr = ne02 = ne03 = 1), so the kernel
// is latency/compute-bound on exactly that division, not bandwidth-bound.

// A packed <mp, shift> pair == init_fastdiv_values()'s <x=mp, y=L>; the divisor
// itself is already carried in the constants below, so it is not repacked here.
struct hrx_fastdiv {
    unsigned int mp;
    unsigned int shift;
};

struct hrx_set_rows_f32_f16_fastdiv_constants {
    long long nc;
    long long nr;
    long long ne02;
    long long ne03;
    long long ne1;
    long long ne11;
    long long ne12;
    long long src0_nb1;
    long long src0_nb2;
    long long src0_nb3;
    long long idx_nb0;
    long long idx_nb1;
    long long idx_nb2;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    // fast-division magics for the decomposition divisors (host-precomputed)
    hrx_fastdiv fd_nc;
    hrx_fastdiv fd_nr;
    hrx_fastdiv fd_ne02;
    hrx_fastdiv fd_ne11;
    hrx_fastdiv fd_ne12;
};

// high 32 bits of n*mp, without __umulhi (freestanding device-only build): one
// 32x32->64 multiply, then take the top word. Lowers to v_mul_hi_u32 on gfx11.
__device__ static inline unsigned int hrx_umulhi(unsigned int n, unsigned int mp) {
    return (unsigned int) (((unsigned long long) n * (unsigned long long) mp) >> 32);
}

// n / d using <mp, shift>; mirrors common.cuh fastdiv (the .z divisor is unused).
__device__ static inline unsigned int hrx_fastdiv_div(unsigned int n, hrx_fastdiv fd) {
    const unsigned int hi = hrx_umulhi(n, fd.mp);
    return (hi + n) >> fd.shift;
}

// n % d, given the divisor d (carried separately in the constants).
__device__ static inline unsigned int hrx_fastdiv_mod(unsigned int n, hrx_fastdiv fd, unsigned int d) {
    return n - hrx_fastdiv_div(n, fd) * d;
}

extern "C" __global__ void hrx_set_rows_f32_f16_fastdiv(
        const float * src0, const long long * idxs, __half * dst,
        hrx_set_rows_f32_f16_fastdiv_constants c) {
    const long long linear = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long total = c.nc * c.nr * c.ne02 * c.ne03;
    if (linear >= total) {
        return;
    }

    // Decompose `linear` over (nc, nr, ne02, ne03) with multiply-shift division.
    // `total` (and thus `linear`) fits in uint32 at the model's set_rows shapes;
    // the host gates this kernel on total < 2^31 (supports_*_fastdiv).
    unsigned int rem = static_cast<unsigned int>(linear);

    const unsigned int q0 = hrx_fastdiv_div(rem, c.fd_nc);
    const long long i0    = static_cast<long long>(rem) - static_cast<long long>(q0) * c.nc;
    rem = q0;

    const unsigned int q1 = hrx_fastdiv_div(rem, c.fd_nr);
    const long long i     = static_cast<long long>(rem) - static_cast<long long>(q1) * c.nr;
    rem = q1;

    const unsigned int q2 = hrx_fastdiv_div(rem, c.fd_ne02);
    const long long i2    = static_cast<long long>(rem) - static_cast<long long>(q2) * c.ne02;
    const long long i3    = static_cast<long long>(q2);

    const long long i12 = static_cast<long long>(
        hrx_fastdiv_mod(static_cast<unsigned int>(i3), c.fd_ne12, static_cast<unsigned int>(c.ne12)));
    const long long i11 = static_cast<long long>(
        hrx_fastdiv_mod(static_cast<unsigned int>(i2), c.fd_ne11, static_cast<unsigned int>(c.ne11)));

    const long long row = *reinterpret_cast<const long long *>(
        reinterpret_cast<const char *>(idxs) + i * c.idx_nb0 + i11 * c.idx_nb1 + i12 * c.idx_nb2);
    if (row < 0 || row >= c.ne1) {
        return;
    }

    const float value = *reinterpret_cast<const float *>(
        reinterpret_cast<const char *>(src0) + i0 * sizeof(float) + i * c.src0_nb1 + i2 * c.src0_nb2 + i3 * c.src0_nb3);
    *reinterpret_cast<__half *>(
        reinterpret_cast<char *>(dst) + i0 * sizeof(__half) + row * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3) =
        __float2half(value);
}
