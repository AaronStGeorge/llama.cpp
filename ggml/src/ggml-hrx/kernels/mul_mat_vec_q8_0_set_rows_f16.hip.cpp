// P058 experiment 02 (default-on; opt out GGML_HRX_DISABLE_MUL_MAT_Q8_0_SET_ROWS_FUSION):
// fuse the q8_0 V-projection matmul with the SET_ROWS that writes its result into the
// (non-transposed, FA) V cache. Llama-3.x decode dispatches these as two
// separate kernels: hrx_mul_mat_vec_q8_0_f32 writes an f32 [n_embd_v_gqa,1]
// Vcur, then hrx_set_rows_f32_f16 reads that f32 back and writes it as f16 into
// one cache_v row. The existing MUL_MAT+SET_ROWS fusion only covers BF16
// weights (mul_mat_vec_bf16_set_rows), so the q8_0 path round-trips through
// VRAM and pays an extra dispatch. This kernel does the q8_0 dequant-dot
// exactly like hrx_mul_mat_vec_q8_0_f32 (same f32 math) but stores the result
// straight to the cache as f16, removing the standalone set_rows dispatch and
// the f32 intermediate round-trip.
//
// Geometry (decode, single token): one workgroup per output feature `row`
// (0..rows-1, rows == n_embd_v_gqa); 256 threads reduce the k-length dot; a
// single destination row index (set_rows->src[1], one idx for the whole token)
// is loaded once and the feature is written at
//   cache_v + dst_row*dst_nb1 + row*sizeof(__half).
// Builtins for ids/shuffle/barrier; hip_fp16/hip_runtime device intrinsics only
// (the same set the shipping bf16 set_rows / q8_0 kernels use; compiles
// freestanding device-only).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q8_0 {
    unsigned short d;
    int8_t qs[32];
};

struct hrx_mul_mat_vec_q8_0_set_rows_constants {
    long long k;            // src0->ne[0]  (== n_embd, contraction length)
    long long rows;         // src0->ne[1]  (== n_embd_v_gqa, output features)
    long long set_rows_ne1; // set_rows->ne[1] (cache rows; bounds-check dst_row)
    long long idx_nb0;      // set_rows->src[1]->nb[0] (byte stride of idx[0])
    long long dst_nb1;      // set_rows->nb[1] (byte stride between cache rows)
};

static __device__ __forceinline__ long long hrx_load_i64(const long long * base, long long byte_offset) {
    return *reinterpret_cast<const long long *>(reinterpret_cast<const char *>(base) + byte_offset);
}

static __device__ __forceinline__ float hrx_reduce_256(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < (256 / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_set_rows_f16(
        const hrx_block_q8_0 * src0, const float * src1, const long long * idxs, __half * dst,
        hrx_mul_mat_vec_q8_0_set_rows_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = c.k / 32;
    const hrx_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    float sum = 0.0f;

    // Same per-thread q8_0 dequant-dot tiling as hrx_mul_mat_vec_q8_0_f32:
    // 32 block-slots x 8 lanes, each lane owns 4 contiguous quants of a block.
    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 32) {
        const hrx_block_q8_0 * block = row_blocks + block_idx;
        const float d = __half2float(__ushort_as_half(block->d));
        const long long src_base = block_idx * 32 + in_block_base;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float value = d * static_cast<float>(block->qs[in_block_base + j]);
            sum += value * src1[src_base + j];
        }
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        const long long dst_row = hrx_load_i64(idxs, 0);
        if (dst_row >= 0 && dst_row < c.set_rows_ne1) {
            *reinterpret_cast<__half *>(
                reinterpret_cast<char *>(dst) + dst_row * c.dst_nb1 + row * static_cast<long long>(sizeof(__half))) =
                    __float2half(sum);
        }
    }
}
