#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

// Fused q8_0 gate/up matmul + SwiGLU epilogue (GLU split SwiGLU).
//
// DESIGN ANGLE #1: dual-wave occupancy. The fused kernel runs at a reduced
// workgroup size (128 threads == 4 wave32 waves on gfx1100) compared to the
// pure q8_0 scalar mul_mat_vec kernel (256 threads). Every thread accumulates
// BOTH the gate and the up dot products over the q8_0 blocks of its assigned
// output row, the two partials are reduced independently across the workgroup,
// and the SwiGLU combine (up * silu(gate)) is applied once at the end. This
// keeps register pressure low, raises occupancy / latency hiding, and folds the
// element-wise GLU into the matmul tail so no separate GLU launch is needed.
//
// All compute is performed in f32 after dequantising q8_0, so the result is
// bit-for-bit identical to the unfused {MUL_MAT(q8_0), MUL_MAT(q8_0), SwiGLU}
// path (no extra rounding is introduced by the fusion).

struct hrx_block_q8_0_swiglu {
    unsigned short d;
    int8_t qs[32];
};

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_reduce2_q8_0_swiglu(
        float & gate_sum,
        float & up_sum,
        float * gate_shared,
        float * up_shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down(gate_sum, offset);
        up_sum += __shfl_down(up_sum, offset);
    }
    if (WG_SIZE <= warpSize) {
        return;
    }
    if (lane == 0) {
        gate_shared[wave] = gate_sum;
        up_shared[wave] = up_sum;
    }
    __syncthreads();

    gate_sum = lane < waves ? gate_shared[lane] : 0.0f;
    up_sum = lane < waves ? up_shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            gate_sum += __shfl_down(gate_sum, offset);
            up_sum += __shfl_down(up_sum, offset);
        }
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_vec_q8_0_swiglu_f32_impl(
        const hrx_block_q8_0_swiglu * gate,
        const hrx_block_q8_0_swiglu * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float gate_sumsh[(WG_SIZE + 31) / 32];
    __shared__ float up_sumsh[(WG_SIZE + 31) / 32];

    const long long blocks_per_row = k / 32;
    const hrx_block_q8_0_swiglu * gate_blocks = gate + row * blocks_per_row;
    const hrx_block_q8_0_swiglu * up_blocks = up + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    // 8 lanes cooperate on one 32-element block (4 values each), mirroring the
    // pure q8_0 scalar kernel's intra-block layout.
    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;
    constexpr int block_slots = WG_SIZE / 8;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_slots) {
        const hrx_block_q8_0_swiglu * gate_block = gate_blocks + block_idx;
        const hrx_block_q8_0_swiglu * up_block = up_blocks + block_idx;
        const float gate_d = __half2float(__ushort_as_half(gate_block->d));
        const float up_d = __half2float(__ushort_as_half(up_block->d));
        const long long src_base = block_idx * 32 + in_block_base;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float b = src1_col[src_base + j];
            gate_sum += gate_d * static_cast<float>(gate_block->qs[in_block_base + j]) * b;
            up_sum += up_d * static_cast<float>(up_block->qs[in_block_base + j]) * b;
        }
    }

    hrx_reduce2_q8_0_swiglu<WG_SIZE>(gate_sum, up_sum, gate_sumsh, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        dst[col * rows + row] = up_sum * silu_gate;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_swiglu_f32(
        const hrx_block_q8_0_swiglu * gate,
        const hrx_block_q8_0_swiglu * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    hrx_mul_mat_vec_q8_0_swiglu_f32_impl<128>(gate, up, src1, dst, k, rows, cols);
}
