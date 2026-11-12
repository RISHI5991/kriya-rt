// File: kernels/metal/q4_gemm_and_moe.metal
//! Native Metal 4 Compute Kernels for KriyaRT:
//! 1. In-kernel Q4_G64 dequantization for M=1 Decode GEMV (memory-bandwidth-bound)
//! 2. Threadgroup-tiled Q4_G64 Prefill GEMM (compute-bound)
//! 3. 100% GPU-Resident MoE Routing (Top-K -> Histogram -> Prefix Scan -> Scatter)
//! 4. PowerInfer-style Sparse-Neuron-Aware GEMV for hot GPU neuron subsets

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_atomic>

using namespace metal;

// ============================================================================
// 1. Data Structures & In-Register Dequantization Helpers
// ============================================================================

/// Matches `BlockQ4Group64` in `crates/kriya-core/src/tensor.rs` (36 bytes packed).
struct BlockQ4Group64 {
    ushort scale_bf16;
    ushort bias_bf16;
    uchar  qs[32];
};

/// Constant kernel dispatch arguments for Linear / MLP projections.
struct GemmParams {
    uint M;              // Number of tokens in batch/chunk (M=1 for decode, M>1 for prefill)
    uint N;              // Output dimension (rows in column-major / row-major weight matrix)
    uint K;              // Input hidden dimension (must be a multiple of 64)
    uint num_k_blocks;   // K / 64
};

/// Constant kernel dispatch arguments for fused QKV linear projections.
struct QkvGemmParams {
    uint M;              // Number of tokens in batch/chunk (M=1 for decode, M>1 for prefill)
    uint K;              // Input hidden dimension (must be a multiple of 64)
    uint num_k_blocks;   // K / 64
    uint q_dim;          // Q projection dimension (num_q_heads * head_dim)
    uint k_dim;          // K projection dimension (num_kv_heads * head_dim)
    uint v_dim;          // V projection dimension (num_kv_heads * head_dim)
    uint has_bias;       // 1 if QKV bias added in-kernel, 0 otherwise
    uint _pad;           // Explicit 32-byte struct alignment
};

/// Constant kernel dispatch arguments for fused Gate+Up MLP linear projections.
struct GateUpGemmParams {
    uint M;                 // Number of tokens in batch/chunk (M=1 for decode, M>1 for prefill)
    uint K;                 // Input hidden dimension (must be a multiple of 64)
    uint num_k_blocks;      // K / 64
    uint intermediate_size; // Intermediate MLP projection dimension
};

/// Constant kernel dispatch arguments for fused Gate+Up+SwiGLU MLP projection.
struct GateUpSwigluGemmParams {
    uint M;                  // Number of tokens in batch/chunk
    uint K;                  // Input hidden dimension (multiple of 64)
    uint num_k_blocks;       // K / 64
    uint intermediate_size;  // Intermediate MLP projection dimension
    uint activation_variant; // 0 = SwiGLU, 1 = GeGLU, 2 = ReLU
    uint _pad[3];            // Explicit 32-byte struct alignment
};

/// Constant kernel dispatch arguments for GPU-resident MoE routing.
struct MoeRouteParams {
    uint num_tokens;         // Number of tokens being routed
    uint num_total_experts;  // Total experts in layer (e.g., 256)
    uint num_active_experts; // Active top-k routed experts per token (e.g., 8)
    uint normalize_weights;  // 1 if top-k probabilities are renormalized
};

/// Converts raw bfloat16 bits (ushort) to 32-bit IEEE 754 float in registers.
inline float bf16_to_f32(ushort bits) {
    uint u32_bits = uint(bits) << 16;
    return as_type<float>(u32_bits);
}

/// Converts 32-bit IEEE 754 float to raw bfloat16 bits (round-to-nearest-even).
inline ushort f32_to_bf16(float val) {
    uint bits = as_type<uint>(val);
    uint lsb = (bits >> 16) & 1u;
    uint rounding_bias = 0x7FFFu + lsb;
    return ushort((bits + rounding_bias) >> 16);
}

// ============================================================================
// 2. Decode Path: Memory-Bandwidth-Bound M=1 GEMV Kernel (coalesced vector loads)
// ============================================================================

/// In-register distributive 64-element block dot product:
/// sum(w_i * x_i) = scale * sum(q_i * x_i) + bias * sum(x_i)
/// Computes with 3x fewer floating-point operations and zero data dependency stalls.
inline float q4_g64_block_dot_distributive(
    device const BlockQ4Group64& blk,
    device const ushort* x_ptr
) {
    const float scale = bf16_to_f32(blk.scale_bf16);
    const float bias  = bf16_to_f32(blk.bias_bf16);

    device const packed_uint4* qs_vec = reinterpret_cast<device const packed_uint4*>(blk.qs);
    const uint4 vec_lo = uint4(qs_vec[0]);
    const uint4 vec_hi = uint4(qs_vec[1]);

    float sum_qx0 = 0.0f;
    float sum_qx1 = 0.0f;
    float sum_x0  = 0.0f;
    float sum_x1  = 0.0f;

    // Unroll first 16 bytes (32 elements) from `vec_lo` with 128-bit vectorized activation fetches
    #pragma unroll
    for (uint w_i = 0u; w_i < 4u; ++w_i) {
        uint word = vec_lo[w_i];
        uint x_base = w_i * 8u;
        uint4 x_vec = *reinterpret_cast<device const uint4*>(x_ptr + x_base);
        #pragma unroll
        for (uint b = 0u; b < 4u; ++b) {
            uint byte_val = (word >> (b * 8u)) & 0xFFu;
            float q0 = float(byte_val & 0x0Fu);
            float q1 = float(byte_val >> 4u);
            uint x_pair = (b == 0u) ? x_vec.x : ((b == 1u) ? x_vec.y : ((b == 2u) ? x_vec.z : x_vec.w));
            float x0 = as_type<float>((x_pair & 0xFFFFu) << 16u);
            float x1 = as_type<float>(x_pair & 0xFFFF0000u);
            sum_qx0 = fma(q0, x0, sum_qx0);
            sum_qx1 = fma(q1, x1, sum_qx1);
            sum_x0 += x0;
            sum_x1 += x1;
        }
    }

    // Unroll second 16 bytes (32 elements) from `vec_hi` with 128-bit vectorized activation fetches
    #pragma unroll
    for (uint w_i = 0u; w_i < 4u; ++w_i) {
        uint word = vec_hi[w_i];
        uint x_base = 32u + w_i * 8u;
        uint4 x_vec = *reinterpret_cast<device const uint4*>(x_ptr + x_base);
        #pragma unroll
        for (uint b = 0u; b < 4u; ++b) {
            uint byte_val = (word >> (b * 8u)) & 0xFFu;
            float q0 = float(byte_val & 0x0Fu);
            float q1 = float(byte_val >> 4u);
            uint x_pair = (b == 0u) ? x_vec.x : ((b == 1u) ? x_vec.y : ((b == 2u) ? x_vec.z : x_vec.w));
            float x0 = as_type<float>((x_pair & 0xFFFFu) << 16u);
            float x1 = as_type<float>(x_pair & 0xFFFF0000u);
            sum_qx0 = fma(q0, x0, sum_qx0);
            sum_qx1 = fma(q1, x1, sum_qx1);
            sum_x0 += x0;
            sum_x1 += x1;
        }
    }

    return fma(sum_qx0 + sum_qx1, scale, (sum_x0 + sum_x1) * bias);
}

/// 128-Bit Vector-Coalesced M=1 Decode GEMV Kernel (32 lanes per row).
kernel void q4_g64_decode_gemv(
    device const BlockQ4Group64* weights   [[buffer(0)]],
    device const ushort*         input_x   [[buffer(1)]], // bf16 [1, K]
    device ushort*               output_y  [[buffer(2)]], // bf16 [1, N]
    constant GemmParams&         params    [[buffer(3)]],
    uint3                        tgp_pos   [[threadgroup_position_in_grid]],
    uint                         simd_id   [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane [[thread_index_in_simdgroup]]
) {
    const uint row_idx = tgp_pos.x * 4u + simd_id;
    if (row_idx >= params.N) {
        return;
    }

    const uint num_blocks = params.num_k_blocks;
    device const BlockQ4Group64* row_blocks = weights + (row_idx * num_blocks);

    float acc = 0.0f;
    for (uint blk_idx = simd_lane; blk_idx < num_blocks; blk_idx += 32u) {
        acc += q4_g64_block_dot_distributive(row_blocks[blk_idx], input_x + (blk_idx * 64u));
    }

    float row_result = simd_sum(acc);
    if (simd_lane == 0u) {
        output_y[row_idx] = f32_to_bf16(row_result);
    }
}

/// Multi-Row Decode GEMV Kernel for small/medium dimensions (num_k_blocks <= 16).
/// Each 32-lane SIMD group computes 2 rows simultaneously (16 threads per row).
/// Grid: N.div_ceil(8) threadgroups, each threadgroup has 128 threads (4 simdgroups * 2 rows = 8 rows).
kernel void q4_g64_decode_gemv_sub16(
    device const BlockQ4Group64* weights   [[buffer(0)]],
    device const ushort*         input_x   [[buffer(1)]], // bf16 [1, K]
    device ushort*               output_y  [[buffer(2)]], // bf16 [1, N]
    constant GemmParams&         params    [[buffer(3)]],
    uint3                        tgp_pos   [[threadgroup_position_in_grid]],
    uint                         simd_id   [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane [[thread_index_in_simdgroup]]
) {
    const uint sub_group = simd_lane / 16u; // 0 or 1
    const uint sub_lane  = simd_lane % 16u; // 0..15
    const uint row_idx   = tgp_pos.x * 8u + simd_id * 2u + sub_group;
    if (row_idx >= params.N) {
        return;
    }

    const uint num_blocks = params.num_k_blocks;
    device const BlockQ4Group64* row_blocks = weights + (row_idx * num_blocks);

    float acc = 0.0f;
    for (uint blk_idx = sub_lane; blk_idx < num_blocks; blk_idx += 16u) {
        acc += q4_g64_block_dot_distributive(row_blocks[blk_idx], input_x + (blk_idx * 64u));
    }

    // 16-lane reduction within half-simdgroup
    acc += simd_shuffle_down(acc, 8u);
    acc += simd_shuffle_down(acc, 4u);
    acc += simd_shuffle_down(acc, 2u);
    acc += simd_shuffle_down(acc, 1u);

    if (sub_lane == 0u) {
        output_y[row_idx] = f32_to_bf16(acc);
    }
}

/// Fused Decode GEMV + Residual Accumulate Kernel (M = 1).
/// Projects input_x and adds directly into residual_inout in registers, eliminating
/// the separate residual_add kernel and avoiding extra unified memory roundtrips.
kernel void q4_g64_decode_gemv_accumulate(
    device const BlockQ4Group64* weights        [[buffer(0)]],
    device const ushort*         input_x        [[buffer(1)]], // bf16 [1, K]
    device ushort*               residual_inout [[buffer(2)]], // bf16 [1, N]
    constant GemmParams&         params         [[buffer(3)]],
    uint3                        tgp_pos        [[threadgroup_position_in_grid]],
    uint                         simd_id        [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane      [[thread_index_in_simdgroup]]
) {
    const uint row_idx = tgp_pos.x * 4u + simd_id;
    if (row_idx >= params.N) {
        return;
    }

    const uint num_blocks = params.num_k_blocks;
    device const BlockQ4Group64* row_blocks = weights + (row_idx * num_blocks);

    float acc = 0.0f;
    for (uint blk_idx = simd_lane; blk_idx < num_blocks; blk_idx += 32u) {
        acc += q4_g64_block_dot_distributive(row_blocks[blk_idx], input_x + (blk_idx * 64u));
    }

    float row_result = simd_sum(acc);
    if (simd_lane == 0u) {
        float r = bf16_to_f32(residual_inout[row_idx]);
        residual_inout[row_idx] = f32_to_bf16(row_result + r);
    }
}

/// 16-Lane Sub-Warp Fused Decode GEMV + Residual Accumulate Kernel (M = 1, K <= 1024).
/// Each SIMDgroup of 32 lanes computes TWO rows concurrently (16 lanes per row),
/// doubling occupancy and hardware throughput for hidden sizes <= 1024.
kernel void q4_g64_decode_gemv_accumulate_sub16(
    device const BlockQ4Group64* weights        [[buffer(0)]],
    device const ushort*         input_x        [[buffer(1)]], // bf16 [1, K]
    device ushort*               residual_inout [[buffer(2)]], // bf16 [1, N]
    constant GemmParams&         params         [[buffer(3)]],
    uint3                        tgp_pos        [[threadgroup_position_in_grid]],
    uint                         simd_id        [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane      [[thread_index_in_simdgroup]]
) {
    const uint sub_id   = simd_lane >> 4u;
    const uint sub_lane = simd_lane & 0x0Fu;
    const uint row_idx  = tgp_pos.x * 8u + simd_id * 2u + sub_id;
    if (row_idx >= params.N) {
        return;
    }

    const uint num_blocks = params.num_k_blocks;
    device const BlockQ4Group64* row_blocks = weights + (row_idx * num_blocks);

    float acc = 0.0f;
    for (uint blk_idx = sub_lane; blk_idx < num_blocks; blk_idx += 16u) {
        acc += q4_g64_block_dot_distributive(row_blocks[blk_idx], input_x + (blk_idx * 64u));
    }

    // 16-lane reduction within half-simdgroup
    acc += simd_shuffle_down(acc, 8u);
    acc += simd_shuffle_down(acc, 4u);
    acc += simd_shuffle_down(acc, 2u);
    acc += simd_shuffle_down(acc, 1u);

    if (sub_lane == 0u) {
        float r = bf16_to_f32(residual_inout[row_idx]);
        residual_inout[row_idx] = f32_to_bf16(acc + r);
    }
}

/// Fused QKV Decode GEMV Kernel (M=1, 32 lanes per row).
/// Projects input_x into Q, K, and V buffers in a SINGLE compute dispatch,
/// directly incorporating optional learned QKV bias in registers before unified memory store.
kernel void q4_g64_decode_qkv_gemv(
    device const BlockQ4Group64* q_weights [[buffer(0)]],
    device const BlockQ4Group64* k_weights [[buffer(1)]],
    device const BlockQ4Group64* v_weights [[buffer(2)]],
    device const ushort*         input_x   [[buffer(3)]], // bf16 [1, K]
    device ushort*               q_out     [[buffer(4)]], // bf16 [1, q_dim]
    device ushort*               k_out     [[buffer(5)]], // bf16 [1, k_dim]
    device ushort*               v_out     [[buffer(6)]], // bf16 [1, v_dim]
    device const ushort*         q_bias    [[buffer(7)]], // bf16 [q_dim]
    device const ushort*         k_bias    [[buffer(8)]], // bf16 [k_dim]
    device const ushort*         v_bias    [[buffer(9)]], // bf16 [v_dim]
    constant QkvGemmParams&      params    [[buffer(10)]],
    uint3                        tgp_pos   [[threadgroup_position_in_grid]],
    uint                         simd_id   [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane [[thread_index_in_simdgroup]]
) {
    const uint total_rows = params.q_dim + params.k_dim + params.v_dim;
    const uint row_idx = tgp_pos.x * 4u + simd_id;
    if (row_idx >= total_rows) {
        return;
    }

    device const BlockQ4Group64* row_blocks;
    device ushort* out_ptr;
    uint out_idx;
    device const ushort* bias_ptr = nullptr;

    if (row_idx < params.q_dim) {
        row_blocks = q_weights + (row_idx * params.num_k_blocks);
        out_ptr = q_out;
        out_idx = row_idx;
        if (params.has_bias != 0u) {
            bias_ptr = q_bias + row_idx;
        }
    } else if (row_idx < params.q_dim + params.k_dim) {
        uint k_idx = row_idx - params.q_dim;
        row_blocks = k_weights + (k_idx * params.num_k_blocks);
        out_ptr = k_out;
        out_idx = k_idx;
        if (params.has_bias != 0u) {
            bias_ptr = k_bias + k_idx;
        }
    } else {
        uint v_idx = row_idx - (params.q_dim + params.k_dim);
        row_blocks = v_weights + (v_idx * params.num_k_blocks);
        out_ptr = v_out;
        out_idx = v_idx;
        if (params.has_bias != 0u) {
            bias_ptr = v_bias + v_idx;
        }
    }

    const uint num_blocks = params.num_k_blocks;
    float acc = 0.0f;
    for (uint blk_idx = simd_lane; blk_idx < num_blocks; blk_idx += 32u) {
        acc += q4_g64_block_dot_distributive(row_blocks[blk_idx], input_x + (blk_idx * 64u));
    }

    float row_result = simd_sum(acc);
    if (simd_lane == 0u) {
        if (bias_ptr != nullptr) {
            row_result += bf16_to_f32(*bias_ptr);
        }
        out_ptr[out_idx] = f32_to_bf16(row_result);
    }
}

/// Multi-Row Fused QKV Decode GEMV Kernel (num_k_blocks <= 16).
/// Each SIMD group computes 2 rows simultaneously (16 threads per row).
/// Grid: total_rows.div_ceil(8) threadgroups of 128 threads.
kernel void q4_g64_decode_qkv_gemv_sub16(
    device const BlockQ4Group64* q_weights [[buffer(0)]],
    device const BlockQ4Group64* k_weights [[buffer(1)]],
    device const BlockQ4Group64* v_weights [[buffer(2)]],
    device const ushort*         input_x   [[buffer(3)]], // bf16 [1, K]
    device ushort*               q_out     [[buffer(4)]], // bf16 [1, q_dim]
    device ushort*               k_out     [[buffer(5)]], // bf16 [1, k_dim]
    device ushort*               v_out     [[buffer(6)]], // bf16 [1, v_dim]
    device const ushort*         q_bias    [[buffer(7)]], // bf16 [q_dim]
    device const ushort*         k_bias    [[buffer(8)]], // bf16 [k_dim]
    device const ushort*         v_bias    [[buffer(9)]], // bf16 [v_dim]
    constant QkvGemmParams&      params    [[buffer(10)]],
    uint3                        tgp_pos   [[threadgroup_position_in_grid]],
    uint                         simd_id   [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane [[thread_index_in_simdgroup]]
) {
    const uint total_rows = params.q_dim + params.k_dim + params.v_dim;
    const uint sub_group = simd_lane / 16u;
    const uint sub_lane  = simd_lane % 16u;
    const uint row_idx   = tgp_pos.x * 8u + simd_id * 2u + sub_group;
    if (row_idx >= total_rows) {
        return;
    }

    device const BlockQ4Group64* row_blocks;
    device ushort* out_ptr;
    uint out_idx;
    device const ushort* bias_ptr = nullptr;

    if (row_idx < params.q_dim) {
        row_blocks = q_weights + (row_idx * params.num_k_blocks);
        out_ptr = q_out;
        out_idx = row_idx;
        if (params.has_bias != 0u) {
            bias_ptr = q_bias + row_idx;
        }
    } else if (row_idx < params.q_dim + params.k_dim) {
        uint k_idx = row_idx - params.q_dim;
        row_blocks = k_weights + (k_idx * params.num_k_blocks);
        out_ptr = k_out;
        out_idx = k_idx;
        if (params.has_bias != 0u) {
            bias_ptr = k_bias + k_idx;
        }
    } else {
        uint v_idx = row_idx - (params.q_dim + params.k_dim);
        row_blocks = v_weights + (v_idx * params.num_k_blocks);
        out_ptr = v_out;
        out_idx = v_idx;
        if (params.has_bias != 0u) {
            bias_ptr = v_bias + v_idx;
        }
    }

    const uint num_blocks = params.num_k_blocks;
    float acc = 0.0f;
    for (uint blk_idx = sub_lane; blk_idx < num_blocks; blk_idx += 16u) {
        acc += q4_g64_block_dot_distributive(row_blocks[blk_idx], input_x + (blk_idx * 64u));
    }

    acc += simd_shuffle_down(acc, 8u);
    acc += simd_shuffle_down(acc, 4u);
    acc += simd_shuffle_down(acc, 2u);
    acc += simd_shuffle_down(acc, 1u);

    if (sub_lane == 0u) {
        if (bias_ptr != nullptr) {
            acc += bf16_to_f32(*bias_ptr);
        }
        out_ptr[out_idx] = f32_to_bf16(acc);
    }
}

/// Fused Gate+Up Decode GEMV Kernel (M = 1, 32 lanes per row).
kernel void q4_g64_decode_gate_up_gemv(
    device const BlockQ4Group64* gate_weights [[buffer(0)]],
    device const BlockQ4Group64* up_weights   [[buffer(1)]],
    device const ushort*         input_x      [[buffer(2)]], // bf16 [1, K]
    device ushort*               gate_out     [[buffer(3)]], // bf16 [1, intermediate_size]
    device ushort*               up_out       [[buffer(4)]], // bf16 [1, intermediate_size]
    constant GateUpGemmParams&   params       [[buffer(5)]],
    uint3                        tgp_pos      [[threadgroup_position_in_grid]],
    uint                         simd_id      [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane    [[thread_index_in_simdgroup]]
) {
    const uint total_rows = params.intermediate_size * 2u;
    const uint row_idx = tgp_pos.x * 4u + simd_id;
    if (row_idx >= total_rows) {
        return;
    }

    device const BlockQ4Group64* row_blocks;
    device ushort* out_ptr;
    uint out_idx;

    if (row_idx < params.intermediate_size) {
        row_blocks = gate_weights + (row_idx * params.num_k_blocks);
        out_ptr = gate_out;
        out_idx = row_idx;
    } else {
        uint u_idx = row_idx - params.intermediate_size;
        row_blocks = up_weights + (u_idx * params.num_k_blocks);
        out_ptr = up_out;
        out_idx = u_idx;
    }

    const uint num_blocks = params.num_k_blocks;
    float acc = 0.0f;
    for (uint blk_idx = simd_lane; blk_idx < num_blocks; blk_idx += 32u) {
        acc += q4_g64_block_dot_distributive(row_blocks[blk_idx], input_x + (blk_idx * 64u));
    }

    float row_result = simd_sum(acc);
    if (simd_lane == 0u) {
        out_ptr[out_idx] = f32_to_bf16(row_result);
    }
}


// ============================================================================
// 3. Prefill Path: 2D Block-Tiled GEMM (Threadgroup SRAM + FMA)
// ============================================================================

/// Dequantizes 4-bit weight tiles directly inside threadgroup SRAM and computes
/// Y[M, N] = X[M, K] * W^T[N, K] for bounded prefill chunks (e.g., M <= 2048).
///
/// Uses 2D block tiling: each threadgroup (16x16 threads) computes a 16x16 output tile
/// by cooperatively loading activation and weight tiles into SRAM, then accumulating
/// via scalar FMA. Each thread loads 8 elements of X and W tiles (2048/256 = 8).
///
/// Grid: tgp_id.x = N tile index, tgp_id.y = M tile index
kernel void q4_g64_prefill_gemm(
    device const BlockQ4Group64* weights   [[buffer(0)]],
    device const ushort*         input_x   [[buffer(1)]], // bf16 [M, K]
    device ushort*               output_y  [[buffer(2)]], // bf16 [M, N]
    constant GemmParams&         params    [[buffer(3)]],
    uint2                        tgp_id    [[threadgroup_position_in_grid]],
    uint2                        tid       [[thread_position_in_threadgroup]]
) {
    // Tile dimensions: 16 tokens (M) x 16 output channels (N) x 64 inner dim (K=1 block)
    constexpr uint TGP_M  = 16u;
    constexpr uint TGP_N  = 16u;
    constexpr uint TILE_K = 64u;

    threadgroup float x_sram[TGP_M][TILE_K];   // [16, 64] activation tile
    threadgroup float w_sram[TILE_K][TGP_N];   // [64, 16] dequantized weight tile

    const uint global_m_base = tgp_id.y * TGP_M;
    const uint global_n_base = tgp_id.x * TGP_N;
    const uint global_m = global_m_base + tid.y;
    const uint global_n = global_n_base + tid.x;

    // Linear thread index within threadgroup (0..255)
    const uint linear_tid = tid.y * TGP_N + tid.x;

    float acc = 0.0f;

    for (uint blk_idx = 0; blk_idx < params.num_k_blocks; ++blk_idx) {
        const uint k_base = blk_idx * TILE_K;

        // Cooperatively load X tile [16, 64] into SRAM (bf16 -> f32)
        // 1024 floats / 256 threads = 4 elements/thread via row-major mapping
        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint elem_idx = linear_tid * 4 + step;
            if (elem_idx < TGP_M * TILE_K) {
                uint tm = elem_idx / TILE_K;
                uint tk = elem_idx % TILE_K;
                uint gm = global_m_base + tm;
                uint gk = k_base + tk;
                x_sram[tm][tk] = (gm < params.M && gk < params.K)
                    ? bf16_to_f32(input_x[gm * params.K + gk])
                    : 0.0f;
            }
        }

        // Cooperatively dequantize W tile [64, 16] from Q4_G64 into SRAM
        // 1024 float values / 256 threads = 4 values/thread via row-major mapping
        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint elem_idx = linear_tid * 4 + step;
            if (elem_idx < TILE_K * TGP_N) {
                uint k = elem_idx / TGP_N;
                uint n = elem_idx % TGP_N;
                uint gn = global_n_base + n;
                if (gn < params.N) {
                    device const BlockQ4Group64& blk = weights[gn * params.num_k_blocks + blk_idx];
                    float scale = bf16_to_f32(blk.scale_bf16);
                    float bias  = bf16_to_f32(blk.bias_bf16);
                    uchar packed = blk.qs[k / 2];
                    float w = (k % 2 == 0)
                        ? fma(float(packed & 0x0Fu), scale, bias)
                        : fma(float(packed >> 4u),   scale, bias);
                    w_sram[k][n] = w;
                } else {
                    w_sram[k][n] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute inner product across the dequantized tile
        #pragma unroll
        for (uint k = 0; k < TILE_K; ++k) {
            acc = fma(x_sram[tid.y][k], w_sram[k][tid.x], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_m < params.M && global_n < params.N) {
        output_y[global_m * params.N + global_n] = f32_to_bf16(acc);
    }
}

/// Fused QKV Prefill GEMM Kernel (M >= 1).
/// Projects batch/chunk of M tokens into Q, K, and V buffers in a SINGLE compute dispatch,
/// directly incorporating optional learned QKV bias in registers before unified memory store.
/// 2D Block-tiled with SRAM staging: 16 tokens (M) x 16 channels (N) x 64 inner (K).
kernel void q4_g64_prefill_qkv_gemm(
    device const BlockQ4Group64* q_weights [[buffer(0)]],
    device const BlockQ4Group64* k_weights [[buffer(1)]],
    device const BlockQ4Group64* v_weights [[buffer(2)]],
    device const ushort*         input_x   [[buffer(3)]], // bf16 [M, K]
    device ushort*               q_out     [[buffer(4)]], // bf16 [M, q_dim]
    device ushort*               k_out     [[buffer(5)]], // bf16 [M, k_dim]
    device ushort*               v_out     [[buffer(6)]], // bf16 [M, v_dim]
    device const ushort*         q_bias    [[buffer(7)]], // bf16 [q_dim]
    device const ushort*         k_bias    [[buffer(8)]], // bf16 [k_dim]
    device const ushort*         v_bias    [[buffer(9)]], // bf16 [v_dim]
    constant QkvGemmParams&      params    [[buffer(10)]],
    uint2                        tgp_id    [[threadgroup_position_in_grid]],
    uint2                        tid       [[thread_position_in_threadgroup]]
) {
    constexpr uint TGP_M  = 16u;
    constexpr uint TGP_N  = 16u;
    constexpr uint TILE_K = 64u;

    threadgroup float x_sram[TGP_M][TILE_K];   // [16, 64] activation tile
    threadgroup float w_sram[TILE_K][TGP_N];   // [64, 16] dequantized weight tile

    const uint total_n = params.q_dim + params.k_dim + params.v_dim;
    const uint global_m_base = tgp_id.y * TGP_M;
    const uint global_n_base = tgp_id.x * TGP_N;
    const uint global_m = global_m_base + tid.y;
    const uint global_n = global_n_base + tid.x;

    const uint linear_tid = tid.y * TGP_N + tid.x;

    float acc = 0.0f;

    for (uint blk_idx = 0; blk_idx < params.num_k_blocks; ++blk_idx) {
        const uint k_base = blk_idx * TILE_K;

        // Cooperatively load X tile [16, 64] into SRAM (bf16 -> f32)
        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint elem_idx = linear_tid * 4 + step;
            if (elem_idx < TGP_M * TILE_K) {
                uint tm = elem_idx / TILE_K;
                uint tk = elem_idx % TILE_K;
                uint gm = global_m_base + tm;
                uint gk = k_base + tk;
                x_sram[tm][tk] = (gm < params.M && gk < params.K)
                    ? bf16_to_f32(input_x[gm * params.K + gk])
                    : 0.0f;
            }
        }

        // Cooperatively dequantize W tile [64, 16] from Q/K/V weights into SRAM
        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint elem_idx = linear_tid * 4 + step;
            if (elem_idx < TILE_K * TGP_N) {
                uint k = elem_idx / TGP_N;
                uint n = elem_idx % TGP_N;
                uint gn = global_n_base + n;
                if (gn < params.q_dim) {
                    device const BlockQ4Group64& blk = q_weights[gn * params.num_k_blocks + blk_idx];
                    float scale = bf16_to_f32(blk.scale_bf16);
                    float bias  = bf16_to_f32(blk.bias_bf16);
                    uchar packed = blk.qs[k / 2];
                    float w = (k % 2 == 0)
                        ? fma(float(packed & 0x0Fu), scale, bias)
                        : fma(float(packed >> 4u),   scale, bias);
                    w_sram[k][n] = w;
                } else if (gn < params.q_dim + params.k_dim) {
                    uint k_n = gn - params.q_dim;
                    device const BlockQ4Group64& blk = k_weights[k_n * params.num_k_blocks + blk_idx];
                    float scale = bf16_to_f32(blk.scale_bf16);
                    float bias  = bf16_to_f32(blk.bias_bf16);
                    uchar packed = blk.qs[k / 2];
                    float w = (k % 2 == 0)
                        ? fma(float(packed & 0x0Fu), scale, bias)
                        : fma(float(packed >> 4u),   scale, bias);
                    w_sram[k][n] = w;
                } else if (gn < total_n) {
                    uint v_n = gn - (params.q_dim + params.k_dim);
                    device const BlockQ4Group64& blk = v_weights[v_n * params.num_k_blocks + blk_idx];
                    float scale = bf16_to_f32(blk.scale_bf16);
                    float bias  = bf16_to_f32(blk.bias_bf16);
                    uchar packed = blk.qs[k / 2];
                    float w = (k % 2 == 0)
                        ? fma(float(packed & 0x0Fu), scale, bias)
                        : fma(float(packed >> 4u),   scale, bias);
                    w_sram[k][n] = w;
                } else {
                    w_sram[k][n] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k = 0; k < TILE_K; ++k) {
            acc = fma(x_sram[tid.y][k], w_sram[k][tid.x], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_m < params.M) {
        if (global_n < params.q_dim) {
            float val = acc;
            if (params.has_bias != 0u) {
                val += bf16_to_f32(q_bias[global_n]);
            }
            q_out[global_m * params.q_dim + global_n] = f32_to_bf16(val);
        } else if (global_n < params.q_dim + params.k_dim) {
            uint k_n = global_n - params.q_dim;
            float val = acc;
            if (params.has_bias != 0u) {
                val += bf16_to_f32(k_bias[k_n]);
            }
            k_out[global_m * params.k_dim + k_n] = f32_to_bf16(val);
        } else if (global_n < total_n) {
            uint v_n = global_n - (params.q_dim + params.k_dim);
            float val = acc;
            if (params.has_bias != 0u) {
                val += bf16_to_f32(v_bias[v_n]);
            }
            v_out[global_m * params.v_dim + v_n] = f32_to_bf16(val);
        }
    }
}

/// Fused Gate+Up Prefill GEMM Kernel (M >= 1).
/// Projects batch/chunk of M tokens into Gate and Up buffers in a SINGLE compute dispatch.
kernel void q4_g64_prefill_gate_up_gemm(
    device const BlockQ4Group64* gate_weights [[buffer(0)]],
    device const BlockQ4Group64* up_weights   [[buffer(1)]],
    device const ushort*         input_x      [[buffer(2)]], // bf16 [M, K]
    device ushort*               gate_out     [[buffer(3)]], // bf16 [M, intermediate_size]
    device ushort*               up_out       [[buffer(4)]], // bf16 [M, intermediate_size]
    constant GateUpGemmParams&   params       [[buffer(5)]],
    uint2                        tgp_id       [[threadgroup_position_in_grid]],
    uint2                        tid          [[thread_position_in_threadgroup]]
) {
    constexpr uint TGP_M  = 16u;
    constexpr uint TGP_N  = 16u;
    constexpr uint TILE_K = 64u;

    threadgroup float x_sram[TGP_M][TILE_K];
    threadgroup float w_sram[TILE_K][TGP_N];

    const uint total_n = params.intermediate_size * 2u;
    const uint global_m_base = tgp_id.y * TGP_M;
    const uint global_n_base = tgp_id.x * TGP_N;
    const uint global_m = global_m_base + tid.y;
    const uint global_n = global_n_base + tid.x;

    const uint linear_tid = tid.y * TGP_N + tid.x;

    float acc = 0.0f;

    for (uint blk_idx = 0; blk_idx < params.num_k_blocks; ++blk_idx) {
        const uint k_base = blk_idx * TILE_K;

        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint elem_idx = linear_tid * 4 + step;
            if (elem_idx < TGP_M * TILE_K) {
                uint tm = elem_idx / TILE_K;
                uint tk = elem_idx % TILE_K;
                uint gm = global_m_base + tm;
                uint gk = k_base + tk;
                x_sram[tm][tk] = (gm < params.M && gk < params.K)
                    ? bf16_to_f32(input_x[gm * params.K + gk])
                    : 0.0f;
            }
        }

        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint elem_idx = linear_tid * 4 + step;
            if (elem_idx < TILE_K * TGP_N) {
                uint k = elem_idx / TGP_N;
                uint n = elem_idx % TGP_N;
                uint gn = global_n_base + n;
                if (gn < params.intermediate_size) {
                    device const BlockQ4Group64& blk = gate_weights[gn * params.num_k_blocks + blk_idx];
                    float scale = bf16_to_f32(blk.scale_bf16);
                    float bias  = bf16_to_f32(blk.bias_bf16);
                    uchar packed = blk.qs[k / 2];
                    float w = (k % 2 == 0)
                        ? fma(float(packed & 0x0Fu), scale, bias)
                        : fma(float(packed >> 4u),   scale, bias);
                    w_sram[k][n] = w;
                } else if (gn < total_n) {
                    uint u_n = gn - params.intermediate_size;
                    device const BlockQ4Group64& blk = up_weights[u_n * params.num_k_blocks + blk_idx];
                    float scale = bf16_to_f32(blk.scale_bf16);
                    float bias  = bf16_to_f32(blk.bias_bf16);
                    uchar packed = blk.qs[k / 2];
                    float w = (k % 2 == 0)
                        ? fma(float(packed & 0x0Fu), scale, bias)
                        : fma(float(packed >> 4u),   scale, bias);
                    w_sram[k][n] = w;
                } else {
                    w_sram[k][n] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k = 0; k < TILE_K; ++k) {
            acc = fma(x_sram[tid.y][k], w_sram[k][tid.x], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_m < params.M) {
        if (global_n < params.intermediate_size) {
            gate_out[global_m * params.intermediate_size + global_n] = f32_to_bf16(acc);
        } else if (global_n < total_n) {
            uint u_n = global_n - params.intermediate_size;
            up_out[global_m * params.intermediate_size + u_n] = f32_to_bf16(acc);
        }
    }
}

// ============================================================================
// 4. 100% GPU-Resident MoE Routing (Zero CPU Synchronization)
// ============================================================================

/// Stage 1 of GPU-resident MoE routing:
/// Computes Softmax + Top-K expert selection per token and atomically increments
/// `expert_counts` in GPU memory so the CPU never inspects routing decisions.
kernel void moe_router_topk_and_histogram(
    device const float*       router_logits      [[buffer(0)]], // [num_tokens, num_total_experts]
    device uint*              selected_experts   [[buffer(1)]], // [num_tokens, num_active_experts]
    device float*             selected_weights   [[buffer(2)]], // [num_tokens, num_active_experts]
    device atomic_uint*       expert_counts      [[buffer(3)]], // [num_total_experts]
    device uint*              token_expert_slot  [[buffer(4)]], // [num_tokens, num_active_experts] intra-expert offset
    constant MoeRouteParams&  params             [[buffer(5)]],
    uint                      token_idx          [[thread_position_in_grid]]
) {
    if (token_idx >= params.num_tokens) {
        return;
    }

    device const float* logits = router_logits + (token_idx * params.num_total_experts);
    const uint E = params.num_total_experts;
    const uint K = min(params.num_active_experts, 16u);

    // Numerically stable softmax denominator over all experts
    float max_logit = -INFINITY;
    for (uint e = 0; e < E; ++e) {
        max_logit = max(max_logit, logits[e]);
    }

    float sum_exp = 0.0f;
    for (uint e = 0; e < E; ++e) {
        sum_exp += exp(logits[e] - max_logit);
    }

    uint  top_idx[16];
    float top_val[16];
    for (uint k = 0; k < K; ++k) {
        top_idx[k] = 0u;
        top_val[k] = -INFINITY;
    }

    // Single-pass register insertion sort for Top-K experts
    for (uint e = 0; e < E; ++e) {
        float prob = exp(logits[e] - max_logit) / sum_exp;
        if (prob > top_val[K - 1u]) {
            top_val[K - 1u] = prob;
            top_idx[K - 1u] = e;
            for (int j = int(K) - 2; j >= 0; --j) {
                if (top_val[j + 1] > top_val[j]) {
                    float tmp_v = top_val[j];
                    top_val[j] = top_val[j + 1];
                    top_val[j + 1] = tmp_v;

                    uint tmp_i = top_idx[j];
                    top_idx[j] = top_idx[j + 1];
                    top_idx[j + 1] = tmp_i;
                } else {
                    break;
                }
            }
        }
    }

    float weight_norm = 0.0f;
    if (params.normalize_weights != 0u) {
        for (uint k = 0; k < K; ++k) {
            weight_norm += top_val[k];
        }
        weight_norm = max(weight_norm, 1e-8f);
    } else {
        weight_norm = 1.0f;
    }

    // Write out Top-K indices/weights and atomically reserve slot inside each expert's bucket
    const uint out_base = token_idx * K;
    for (uint k = 0; k < K; ++k) {
        uint exp_id = top_idx[k];
        selected_experts[out_base + k] = exp_id;
        selected_weights[out_base + k] = top_val[k] / weight_norm;

        uint slot = atomic_fetch_add_explicit(&expert_counts[exp_id], 1u, memory_order_relaxed);
        token_expert_slot[out_base + k] = slot;
    }
}

/// Stage 2 of GPU-resident MoE routing:
/// Computes exclusive prefix scan over `expert_counts` to produce `expert_offsets`
/// and writes the contiguous `scatter_token_indices` array for Grouped Expert GEMM.
kernel void moe_router_prefix_scan_and_scatter(
    device const uint*        expert_counts        [[buffer(0)]], // [num_total_experts]
    device uint*              expert_offsets       [[buffer(1)]], // [num_total_experts + 1]
    device const uint*        selected_experts     [[buffer(2)]], // [num_tokens * num_active_experts]
    device const uint*        token_expert_slot    [[buffer(3)]], // [num_tokens * num_active_experts]
    device uint*              scatter_token_map    [[buffer(4)]], // [num_tokens * num_active_experts]
    constant MoeRouteParams&  params               [[buffer(5)]],
    uint                      tid                  [[thread_position_in_grid]]
) {
    // Thread 0 performs fast sequential prefix scan over 256 experts in L1 cache
    if (tid == 0u) {
        uint running_offset = 0u;
        for (uint e = 0; e < params.num_total_experts; ++e) {
            expert_offsets[e] = running_offset;
            running_offset += expert_counts[e];
        }
        expert_offsets[params.num_total_experts] = running_offset;
    }

    threadgroup_barrier(mem_flags::mem_device);

    const uint total_assignments = params.num_tokens * params.num_active_experts;
    if (tid < total_assignments) {
        uint exp_id = selected_experts[tid];
        uint intra_slot = token_expert_slot[tid];
        uint dst_idx = expert_offsets[exp_id] + intra_slot;
        scatter_token_map[dst_idx] = tid;
    }
}


// ============================================================================
// 6. GPU-Resident Routed Expert Gather-Scatter GEMV (Closes Gap 4)
// ============================================================================

struct RoutedExpertGemvParams {
    uint num_tokens;
    uint num_active_experts; // K (e.g., 8)
    uint out_dim;            // N (intermediate_size or hidden_size)
    uint in_dim;             // K_dim (hidden_size or intermediate_size)
    uint num_k_blocks;       // in_dim / 64
    uint expert_stride_blks; // Blocks per expert matrix: out_dim * num_k_blocks
};

/// Executes gather-scatter GEMV across dynamically routed top-K experts per token
/// using `selected_experts[tok, k]` directly on the GPU with zero CPU readback.
kernel void q4_g64_routed_moe_gemv(
    device const BlockQ4Group64* expert_weights    [[buffer(0)]], // [num_experts, out_dim, num_k_blocks]
    device const ushort*         input_x           [[buffer(1)]], // [num_tokens, K_slots_or_1, in_dim]
    device const uint*           selected_experts  [[buffer(2)]], // [num_tokens, num_active_experts]
    device ushort*               expert_outputs    [[buffer(3)]], // [num_tokens, num_active_experts, out_dim]
    constant RoutedExpertGemvParams& params        [[buffer(4)]],
    constant uint&               input_is_per_slot [[buffer(5)]], // 0 if input is [tok, in_dim], 1 if [tok, k, in_dim]
    uint3                        tgp_pos           [[threadgroup_position_in_grid]],
    uint                         simd_id           [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane         [[thread_index_in_simdgroup]]
) {
    const uint row_idx  = tgp_pos.x * 4u + simd_id; // Output row in [0..out_dim)
    const uint slot_idx = tgp_pos.y;                // Active expert slot in [0..K)
    const uint tok_idx  = tgp_pos.z;                // Token index in [0..num_tokens)

    if (row_idx >= params.out_dim || slot_idx >= params.num_active_experts || tok_idx >= params.num_tokens) {
        return;
    }

    const uint assign_idx = tok_idx * params.num_active_experts + slot_idx;
    const uint expert_id  = selected_experts[assign_idx];

    // Offset into the selected expert's weight matrix.
    // expert_id is in [0, num_total_experts). The expert_weights buffer must be
    // allocated with space for all experts (see MetalExecutor::allocate_for_descriptor).
    device const BlockQ4Group64* row_blocks =
        expert_weights + (expert_id * params.expert_stride_blks) + (row_idx * params.num_k_blocks);

    const uint x_offset = (input_is_per_slot != 0u)
        ? (assign_idx * params.in_dim)
        : (tok_idx * params.in_dim);
    device const ushort* x_ptr_base = input_x + x_offset;

    float acc = 0.0f;
    for (uint blk_idx = simd_lane; blk_idx < params.num_k_blocks; blk_idx += 32u) {
        BlockQ4Group64 blk = row_blocks[blk_idx];
        const float scale = bf16_to_f32(blk.scale_bf16);
        const float bias  = bf16_to_f32(blk.bias_bf16);
        device const ushort* x_ptr = x_ptr_base + (blk_idx * 64u);

        float block_sum = 0.0f;
        #pragma unroll
        for (uint byte_i = 0; byte_i < 32u; ++byte_i) {
            uchar packed = blk.qs[byte_i];
            float w0 = fma(float(packed & 0x0Fu), scale, bias);
            float w1 = fma(float(packed >> 4u),   scale, bias);
            block_sum = fma(w0, bf16_to_f32(x_ptr[byte_i * 2u]),     block_sum);
            block_sum = fma(w1, bf16_to_f32(x_ptr[byte_i * 2u + 1u]), block_sum);
        }
        acc += block_sum;
    }

    float row_result = simd_sum(acc);
    if (simd_lane == 0u) {
        expert_outputs[assign_idx * params.out_dim + row_idx] = f32_to_bf16(row_result);
    }
}

// ============================================================================
// 7. Adaptive Multi-Tile Dispatch: TGP_M = 64 Long Context Prefill Kernels (M >= 64)
// ============================================================================
// 1. 64x16 output tile per threadgroup (4x weight reuse in registers across 4 accumulators)
// 2. 14 KB threadgroup SRAM: 2 concurrent threadgroups per Apple GPU core for 100% latency hiding
// 3. 128-bit vectorized activation loading (uint4 pairs)
// 4. Cuts DRAM weight bandwidth by 75% vs TGP_M = 16

kernel void q4_g64_batched_simdgroup_gemm_m64(
    device const BlockQ4Group64* weights   [[buffer(0)]],
    device const ushort*         input_x   [[buffer(1)]], // bf16 [M, K]
    device ushort*               output_y  [[buffer(2)]], // bf16 [M, N]
    constant GemmParams&         params    [[buffer(3)]],
    uint2                        tgp_id    [[threadgroup_position_in_grid]],
    uint                         simd_id   [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint TGP_M  = 64u;
    constexpr uint TGP_N  = 16u;
    constexpr uint TILE_K = 64u;
    constexpr uint STRIDE_K = 68u; // Bank padding: 68 * 2 = 136 bytes eliminates 32-way L1 bank conflict

    threadgroup half  x_tile[TGP_M][STRIDE_K];  // [64, 68] = 8.5 KB
    threadgroup half  w_tile[TILE_K][TGP_N];  // [64, 16] = 2 KB
    threadgroup float out_tile[TGP_M][TGP_N]; // [64, 16] = 4 KB

    const uint global_m_base = tgp_id.y * TGP_M;
    const uint global_n_base = tgp_id.x * TGP_N;
    const uint linear_tid = simd_id * 32u + simd_lane;

    // 4 SIMDgroups map to 2x2 grid of 32x8 output tiles
    const uint m_sub = (simd_id / 2u) * 32u;
    const uint n_sub = (simd_id % 2u) * 8u;

    simdgroup_matrix<float, 8, 8> acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint blk_idx = 0; blk_idx < params.num_k_blocks; ++blk_idx) {
        const uint k_base = blk_idx * TILE_K;

        // Cooperatively load X tile [64, 64] using all 128 threads (32 elements per thread)
        const uint tm_base = linear_tid >> 1u;        // 0..63
        const uint tk_base = (linear_tid & 1u) * 32u; // 0, 32
        const uint gm = global_m_base + tm_base;
        const uint gk_base = k_base + tk_base;

        if (gm < params.M && gk_base + 31u < params.K) {
            #pragma unroll
            for (uint half_idx = 0u; half_idx < 2u; ++half_idx) {
                uint k_offset = tk_base + half_idx * 16u;
                uint4 x_raw0 = *reinterpret_cast<device const uint4*>(input_x + gm * params.K + k_base + k_offset);
                uint4 x_raw1 = *reinterpret_cast<device const uint4*>(input_x + gm * params.K + k_base + k_offset + 8u);
                #pragma unroll
                for (uint w = 0u; w < 4u; ++w) {
                    uint pair0 = (w == 0u) ? x_raw0.x : ((w == 1u) ? x_raw0.y : ((w == 2u) ? x_raw0.z : x_raw0.w));
                    uint pair1 = (w == 0u) ? x_raw1.x : ((w == 1u) ? x_raw1.y : ((w == 2u) ? x_raw1.z : x_raw1.w));
                    float f0 = as_type<float>((pair0 & 0xFFFFu) << 16u);
                    float f1 = as_type<float>(pair0 & 0xFFFF0000u);
                    float f2 = as_type<float>((pair1 & 0xFFFFu) << 16u);
                    float f3 = as_type<float>(pair1 & 0xFFFF0000u);
                    x_tile[tm_base][k_offset + w * 2u]          = half(f0);
                    x_tile[tm_base][k_offset + w * 2u + 1u]     = half(f1);
                    x_tile[tm_base][k_offset + 8u + w * 2u]     = half(f2);
                    x_tile[tm_base][k_offset + 8u + w * 2u + 1u] = half(f3);
                }
            }
        } else {
            #pragma unroll
            for (uint i = 0u; i < 32u; ++i) {
                uint gk = gk_base + i;
                x_tile[tm_base][tk_base + i] = (gm < params.M && gk < params.K)
                    ? half(bf16_to_f32(input_x[gm * params.K + gk]))
                    : 0.0h;
            }
        }

        // Cooperatively load and dequantize W tile [64, 16] using all 128 threads (8 elements per thread)
        const uint col_n = linear_tid >> 3u;      // 0..15
        const uint sub_k = linear_tid & 7u;       // 0..7
        const uint gn    = global_n_base + col_n;

        if (gn < params.N) {
            device const BlockQ4Group64& blk = weights[gn * params.num_k_blocks + blk_idx];
            uint sb_raw = *reinterpret_cast<device const uint*>(&blk);
            float scale = as_type<float>((sb_raw & 0xFFFFu) << 16u);
            float bias  = as_type<float>(sb_raw & 0xFFFF0000u);

            uint byte_start = sub_k * 4u;
            uint q_words = *reinterpret_cast<device const uint*>(blk.qs + byte_start);

            #pragma unroll
            for (uint b = 0u; b < 4u; ++b) {
                uchar packed = uchar((q_words >> (b * 8u)) & 0xFFu);
                float w0 = fma(float(packed & 0x0Fu), scale, bias);
                float w1 = fma(float(packed >> 4u),   scale, bias);
                uint k0 = (byte_start + b) * 2u;
                w_tile[k0][col_n]      = half(w0);
                w_tile[k0 + 1u][col_n] = half(w1);
            }
        } else {
            uint k_start = sub_k * 8u;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                w_tile[k_start + i][col_n] = 0.0h;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Hardware matrix multiply-accumulate with 4x weight reuse across acc0, acc1, acc2, acc3
        #pragma unroll
        for (uint k_step = 0; k_step < 8; ++k_step) {
            simdgroup_matrix<half, 8, 8> b_mat;
            simdgroup_load(b_mat, &w_tile[k_step * 8u][n_sub], TGP_N);

            simdgroup_matrix<half, 8, 8> a_mat0;
            simdgroup_load(a_mat0, &x_tile[m_sub][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc0, a_mat0, b_mat, acc0);

            simdgroup_matrix<half, 8, 8> a_mat1;
            simdgroup_load(a_mat1, &x_tile[m_sub + 8u][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc1, a_mat1, b_mat, acc1);

            simdgroup_matrix<half, 8, 8> a_mat2;
            simdgroup_load(a_mat2, &x_tile[m_sub + 16u][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc2, a_mat2, b_mat, acc2);

            simdgroup_matrix<half, 8, 8> a_mat3;
            simdgroup_load(a_mat3, &x_tile[m_sub + 24u][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc3, a_mat3, b_mat, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc0, &out_tile[m_sub][n_sub], TGP_N);
    simdgroup_store(acc1, &out_tile[m_sub + 8u][n_sub], TGP_N);
    simdgroup_store(acc2, &out_tile[m_sub + 16u][n_sub], TGP_N);
    simdgroup_store(acc3, &out_tile[m_sub + 24u][n_sub], TGP_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint step = 0; step < 8; ++step) {
        uint elem = linear_tid * 8u + step;
        if (elem < TGP_M * TGP_N) {
            uint tm = elem / TGP_N;
            uint tn = elem % TGP_N;
            uint gm = global_m_base + tm;
            uint gn = global_n_base + tn;
            if (gm < params.M && gn < params.N) {
                output_y[gm * params.N + gn] = f32_to_bf16(out_tile[tm][tn]);
            }
        }
    }
}

kernel void q4_g64_batched_simdgroup_qkv_gemm_m64(
    device const BlockQ4Group64* q_weights [[buffer(0)]],
    device const BlockQ4Group64* k_weights [[buffer(1)]],
    device const BlockQ4Group64* v_weights [[buffer(2)]],
    device const ushort*         input_x   [[buffer(3)]], // bf16 [M, K]
    device ushort*               q_out     [[buffer(4)]], // bf16 [M, q_dim]
    device ushort*               k_out     [[buffer(5)]], // bf16 [M, k_dim]
    device ushort*               v_out     [[buffer(6)]], // bf16 [M, v_dim]
    device const ushort*         q_bias    [[buffer(7)]], // bf16 [q_dim]
    device const ushort*         k_bias    [[buffer(8)]], // bf16 [k_dim]
    device const ushort*         v_bias    [[buffer(9)]], // bf16 [v_dim]
    constant QkvGemmParams&      params    [[buffer(10)]],
    uint2                        tgp_id    [[threadgroup_position_in_grid]],
    uint                         simd_id   [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint TGP_M  = 64u;
    constexpr uint TGP_N  = 16u;
    constexpr uint TILE_K = 64u;
    constexpr uint STRIDE_K = 68u; // Bank padding: 68 * 2 = 136 bytes eliminates 32-way L1 bank conflict

    threadgroup half  x_tile[TGP_M][STRIDE_K];
    threadgroup half  w_tile[TILE_K][TGP_N];
    threadgroup float out_tile[TGP_M][TGP_N];

    const uint total_n = params.q_dim + params.k_dim + params.v_dim;
    const uint global_m_base = tgp_id.y * TGP_M;
    const uint global_n_base = tgp_id.x * TGP_N;
    const uint linear_tid = simd_id * 32u + simd_lane;

    const uint m_sub = (simd_id / 2u) * 32u;
    const uint n_sub = (simd_id % 2u) * 8u;

    simdgroup_matrix<float, 8, 8> acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint blk_idx = 0; blk_idx < params.num_k_blocks; ++blk_idx) {
        const uint k_base = blk_idx * TILE_K;

        const uint tm_base = linear_tid >> 1u;
        const uint tk_base = (linear_tid & 1u) * 32u;
        const uint gm = global_m_base + tm_base;
        const uint gk_base = k_base + tk_base;

        if (gm < params.M && gk_base + 31u < params.K) {
            #pragma unroll
            for (uint half_idx = 0u; half_idx < 2u; ++half_idx) {
                uint k_offset = tk_base + half_idx * 16u;
                uint4 x_raw0 = *reinterpret_cast<device const uint4*>(input_x + gm * params.K + k_base + k_offset);
                uint4 x_raw1 = *reinterpret_cast<device const uint4*>(input_x + gm * params.K + k_base + k_offset + 8u);
                #pragma unroll
                for (uint w = 0u; w < 4u; ++w) {
                    uint pair0 = (w == 0u) ? x_raw0.x : ((w == 1u) ? x_raw0.y : ((w == 2u) ? x_raw0.z : x_raw0.w));
                    uint pair1 = (w == 0u) ? x_raw1.x : ((w == 1u) ? x_raw1.y : ((w == 2u) ? x_raw1.z : x_raw1.w));
                    float f0 = as_type<float>((pair0 & 0xFFFFu) << 16u);
                    float f1 = as_type<float>(pair0 & 0xFFFF0000u);
                    float f2 = as_type<float>((pair1 & 0xFFFFu) << 16u);
                    float f3 = as_type<float>(pair1 & 0xFFFF0000u);
                    x_tile[tm_base][k_offset + w * 2u]          = half(f0);
                    x_tile[tm_base][k_offset + w * 2u + 1u]     = half(f1);
                    x_tile[tm_base][k_offset + 8u + w * 2u]     = half(f2);
                    x_tile[tm_base][k_offset + 8u + w * 2u + 1u] = half(f3);
                }
            }
        } else {
            #pragma unroll
            for (uint i = 0u; i < 32u; ++i) {
                uint gk = gk_base + i;
                x_tile[tm_base][tk_base + i] = (gm < params.M && gk < params.K)
                    ? half(bf16_to_f32(input_x[gm * params.K + gk]))
                    : 0.0h;
            }
        }

        const uint col_n = linear_tid >> 3u;
        const uint sub_k = linear_tid & 7u;
        const uint gn    = global_n_base + col_n;

        if (gn < total_n) {
            device const BlockQ4Group64* target_weights;
            uint local_col;

            if (gn < params.q_dim) {
                target_weights = q_weights;
                local_col = gn;
            } else if (gn < params.q_dim + params.k_dim) {
                target_weights = k_weights;
                local_col = gn - params.q_dim;
            } else {
                target_weights = v_weights;
                local_col = gn - params.q_dim - params.k_dim;
            }

            device const BlockQ4Group64& blk = target_weights[local_col * params.num_k_blocks + blk_idx];
            uint sb_raw = *reinterpret_cast<device const uint*>(&blk);
            float scale = as_type<float>((sb_raw & 0xFFFFu) << 16u);
            float bias  = as_type<float>(sb_raw & 0xFFFF0000u);

            uint byte_start = sub_k * 4u;
            uint q_words = *reinterpret_cast<device const uint*>(blk.qs + byte_start);

            #pragma unroll
            for (uint b = 0u; b < 4u; ++b) {
                uchar packed = uchar((q_words >> (b * 8u)) & 0xFFu);
                float w0 = fma(float(packed & 0x0Fu), scale, bias);
                float w1 = fma(float(packed >> 4u),   scale, bias);
                uint k0 = (byte_start + b) * 2u;
                w_tile[k0][col_n]      = half(w0);
                w_tile[k0 + 1u][col_n] = half(w1);
            }
        } else {
            uint k_start = sub_k * 8u;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                w_tile[k_start + i][col_n] = 0.0h;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k_step = 0; k_step < 8; ++k_step) {
            simdgroup_matrix<half, 8, 8> b_mat;
            simdgroup_load(b_mat, &w_tile[k_step * 8u][n_sub], TGP_N);

            simdgroup_matrix<half, 8, 8> a_mat0;
            simdgroup_load(a_mat0, &x_tile[m_sub][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc0, a_mat0, b_mat, acc0);

            simdgroup_matrix<half, 8, 8> a_mat1;
            simdgroup_load(a_mat1, &x_tile[m_sub + 8u][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc1, a_mat1, b_mat, acc1);

            simdgroup_matrix<half, 8, 8> a_mat2;
            simdgroup_load(a_mat2, &x_tile[m_sub + 16u][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc2, a_mat2, b_mat, acc2);

            simdgroup_matrix<half, 8, 8> a_mat3;
            simdgroup_load(a_mat3, &x_tile[m_sub + 24u][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc3, a_mat3, b_mat, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc0, &out_tile[m_sub][n_sub], TGP_N);
    simdgroup_store(acc1, &out_tile[m_sub + 8u][n_sub], TGP_N);
    simdgroup_store(acc2, &out_tile[m_sub + 16u][n_sub], TGP_N);
    simdgroup_store(acc3, &out_tile[m_sub + 24u][n_sub], TGP_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint step = 0; step < 8; ++step) {
        uint elem = linear_tid * 8u + step;
        if (elem < TGP_M * TGP_N) {
            uint tm = elem / TGP_N;
            uint tn = elem % TGP_N;
            uint gm = global_m_base + tm;
            uint gn = global_n_base + tn;
            if (gm < params.M) {
                if (gn < params.q_dim) {
                    float bias = (params.has_bias != 0u && q_bias != nullptr) ? bf16_to_f32(q_bias[gn]) : 0.0f;
                    q_out[gm * params.q_dim + gn] = f32_to_bf16(out_tile[tm][tn] + bias);
                } else if (gn < params.q_dim + params.k_dim) {
                    uint k_idx = gn - params.q_dim;
                    float bias = (params.has_bias != 0u && k_bias != nullptr) ? bf16_to_f32(k_bias[k_idx]) : 0.0f;
                    k_out[gm * params.k_dim + k_idx] = f32_to_bf16(out_tile[tm][tn] + bias);
                } else if (gn < total_n) {
                    uint v_idx = gn - params.q_dim - params.k_dim;
                    float bias = (params.has_bias != 0u && v_bias != nullptr) ? bf16_to_f32(v_bias[v_idx]) : 0.0f;
                    v_out[gm * params.v_dim + v_idx] = f32_to_bf16(out_tile[tm][tn] + bias);
                }
            }
        }
    }
}




kernel void q4_g64_batched_simdgroup_gate_up_swiglu_gemm_m64(
    device const BlockQ4Group64*     gate_weights [[buffer(0)]],
    device const BlockQ4Group64*     up_weights   [[buffer(1)]],
    device const ushort*             input_x      [[buffer(2)]], // bf16 [M, K]
    device ushort*                   mlp_out      [[buffer(3)]], // bf16 [M, intermediate_size]
    constant GateUpSwigluGemmParams& params       [[buffer(4)]],
    uint2                            tgp_id       [[threadgroup_position_in_grid]],
    uint                             simd_id      [[simdgroup_index_in_threadgroup]],
    uint                             simd_lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint TGP_M  = 64u;
    constexpr uint TGP_N  = 16u;
    constexpr uint TILE_K = 64u;
    constexpr uint STRIDE_K = 68u; // Bank padding: 68 * 2 = 136 bytes eliminates 32-way L1 bank conflict

    threadgroup half  x_tile[TGP_M][STRIDE_K];       // [64, 68] = 8.5 KB
    threadgroup half  w_gate_tile[TILE_K][TGP_N];  // [64, 16] = 2 KB
    threadgroup half  w_up_tile[TILE_K][TGP_N];    // [64, 16] = 2 KB
    threadgroup float gate_out_tile[TGP_M][TGP_N]; // [64, 16] = 4 KB
    threadgroup float up_out_tile[TGP_M][TGP_N];   // [64, 16] = 4 KB

    const uint global_m_base = tgp_id.y * TGP_M;
    const uint global_n_base = tgp_id.x * TGP_N;
    const uint linear_tid = simd_id * 32u + simd_lane;

    const uint m_sub = (simd_id / 2u) * 32u;
    const uint n_sub = (simd_id % 2u) * 8u;

    simdgroup_matrix<float, 8, 8> acc_gate0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc_gate1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc_gate2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc_gate3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    simdgroup_matrix<float, 8, 8> acc_up0   = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc_up1   = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc_up2   = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc_up3   = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint blk_idx = 0; blk_idx < params.num_k_blocks; ++blk_idx) {
        const uint k_base = blk_idx * TILE_K;

        const uint tm_base = linear_tid >> 1u;
        const uint tk_base = (linear_tid & 1u) * 32u;
        const uint gm = global_m_base + tm_base;
        const uint gk_base = k_base + tk_base;

        if (gm < params.M && gk_base + 31u < params.K) {
            #pragma unroll
            for (uint half_idx = 0u; half_idx < 2u; ++half_idx) {
                uint k_offset = tk_base + half_idx * 16u;
                uint4 x_raw0 = *reinterpret_cast<device const uint4*>(input_x + gm * params.K + k_base + k_offset);
                uint4 x_raw1 = *reinterpret_cast<device const uint4*>(input_x + gm * params.K + k_base + k_offset + 8u);
                #pragma unroll
                for (uint w = 0u; w < 4u; ++w) {
                    uint pair0 = (w == 0u) ? x_raw0.x : ((w == 1u) ? x_raw0.y : ((w == 2u) ? x_raw0.z : x_raw0.w));
                    uint pair1 = (w == 0u) ? x_raw1.x : ((w == 1u) ? x_raw1.y : ((w == 2u) ? x_raw1.z : x_raw1.w));
                    float f0 = as_type<float>((pair0 & 0xFFFFu) << 16u);
                    float f1 = as_type<float>(pair0 & 0xFFFF0000u);
                    float f2 = as_type<float>((pair1 & 0xFFFFu) << 16u);
                    float f3 = as_type<float>(pair1 & 0xFFFF0000u);
                    x_tile[tm_base][k_offset + w * 2u]          = half(f0);
                    x_tile[tm_base][k_offset + w * 2u + 1u]     = half(f1);
                    x_tile[tm_base][k_offset + 8u + w * 2u]     = half(f2);
                    x_tile[tm_base][k_offset + 8u + w * 2u + 1u] = half(f3);
                }
            }
        } else {
            #pragma unroll
            for (uint i = 0u; i < 32u; ++i) {
                uint gk = gk_base + i;
                x_tile[tm_base][tk_base + i] = (gm < params.M && gk < params.K)
                    ? half(bf16_to_f32(input_x[gm * params.K + gk]))
                    : 0.0h;
            }
        }

        const uint col_n = linear_tid >> 3u;
        const uint sub_k = linear_tid & 7u;
        const uint gn    = global_n_base + col_n;

        if (gn < params.intermediate_size) {
            device const BlockQ4Group64& gate_blk = gate_weights[gn * params.num_k_blocks + blk_idx];
            device const BlockQ4Group64& up_blk   = up_weights[gn * params.num_k_blocks + blk_idx];

            uint g_sb = *reinterpret_cast<device const uint*>(&gate_blk);
            uint u_sb = *reinterpret_cast<device const uint*>(&up_blk);
            float g_scale = as_type<float>((g_sb & 0xFFFFu) << 16u);
            float g_bias  = as_type<float>(g_sb & 0xFFFF0000u);
            float u_scale = as_type<float>((u_sb & 0xFFFFu) << 16u);
            float u_bias  = as_type<float>(u_sb & 0xFFFF0000u);

            uint byte_start = sub_k * 4u;
            uint g_q = *reinterpret_cast<device const uint*>(gate_blk.qs + byte_start);
            uint u_q = *reinterpret_cast<device const uint*>(up_blk.qs + byte_start);

            #pragma unroll
            for (uint b = 0u; b < 4u; ++b) {
                uchar g_packed = uchar((g_q >> (b * 8u)) & 0xFFu);
                uchar u_packed = uchar((u_q >> (b * 8u)) & 0xFFu);

                float gw0 = fma(float(g_packed & 0x0Fu), g_scale, g_bias);
                float gw1 = fma(float(g_packed >> 4u),   g_scale, g_bias);
                float uw0 = fma(float(u_packed & 0x0Fu), u_scale, u_bias);
                float uw1 = fma(float(u_packed >> 4u),   u_scale, u_bias);

                uint k0 = (byte_start + b) * 2u;
                w_gate_tile[k0][col_n]      = half(gw0);
                w_gate_tile[k0 + 1u][col_n] = half(gw1);
                w_up_tile[k0][col_n]        = half(uw0);
                w_up_tile[k0 + 1u][col_n]   = half(uw1);
            }
        } else {
            uint k_start = sub_k * 8u;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                w_gate_tile[k_start + i][col_n] = 0.0h;
                w_up_tile[k_start + i][col_n]   = 0.0h;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k_step = 0; k_step < 8; ++k_step) {
            simdgroup_matrix<half, 8, 8> b_gate;
            simdgroup_matrix<half, 8, 8> b_up;
            simdgroup_load(b_gate, &w_gate_tile[k_step * 8u][n_sub], TGP_N);
            simdgroup_load(b_up,   &w_up_tile[k_step * 8u][n_sub],   TGP_N);

            simdgroup_matrix<half, 8, 8> a_mat0;
            simdgroup_load(a_mat0, &x_tile[m_sub][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc_gate0, a_mat0, b_gate, acc_gate0);
            simdgroup_multiply_accumulate(acc_up0,   a_mat0, b_up,   acc_up0);

            simdgroup_matrix<half, 8, 8> a_mat1;
            simdgroup_load(a_mat1, &x_tile[m_sub + 8u][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc_gate1, a_mat1, b_gate, acc_gate1);
            simdgroup_multiply_accumulate(acc_up1,   a_mat1, b_up,   acc_up1);

            simdgroup_matrix<half, 8, 8> a_mat2;
            simdgroup_load(a_mat2, &x_tile[m_sub + 16u][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc_gate2, a_mat2, b_gate, acc_gate2);
            simdgroup_multiply_accumulate(acc_up2,   a_mat2, b_up,   acc_up2);

            simdgroup_matrix<half, 8, 8> a_mat3;
            simdgroup_load(a_mat3, &x_tile[m_sub + 24u][k_step * 8u], STRIDE_K);
            simdgroup_multiply_accumulate(acc_gate3, a_mat3, b_gate, acc_gate3);
            simdgroup_multiply_accumulate(acc_up3,   a_mat3, b_up,   acc_up3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc_gate0, &gate_out_tile[m_sub][n_sub], TGP_N);
    simdgroup_store(acc_gate1, &gate_out_tile[m_sub + 8u][n_sub], TGP_N);
    simdgroup_store(acc_gate2, &gate_out_tile[m_sub + 16u][n_sub], TGP_N);
    simdgroup_store(acc_gate3, &gate_out_tile[m_sub + 24u][n_sub], TGP_N);

    simdgroup_store(acc_up0,   &up_out_tile[m_sub][n_sub],   TGP_N);
    simdgroup_store(acc_up1,   &up_out_tile[m_sub + 8u][n_sub],   TGP_N);
    simdgroup_store(acc_up2,   &up_out_tile[m_sub + 16u][n_sub],   TGP_N);
    simdgroup_store(acc_up3,   &up_out_tile[m_sub + 24u][n_sub],   TGP_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint step = 0; step < 8; ++step) {
        uint elem = linear_tid * 8u + step;
        if (elem < TGP_M * TGP_N) {
            uint tm = elem / TGP_N;
            uint tn = elem % TGP_N;
            uint gm = global_m_base + tm;
            uint gn = global_n_base + tn;
            if (gm < params.M && gn < params.intermediate_size) {
                float g = gate_out_tile[tm][tn];
                float u = up_out_tile[tm][tn];
                float act_g = 0.0f;
                if (params.activation_variant == 0u) {
                    act_g = g / (1.0f + exp(-g));
                } else if (params.activation_variant == 1u) {
                    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
                    float cubic = fma(0.044715f * g, g * g, g);
                    act_g = 0.5f * g * (1.0f + tanh(SQRT_2_OVER_PI * cubic));
                } else {
                    act_g = max(g, 0.0f);
                }
                mlp_out[gm * params.intermediate_size + gn] = f32_to_bf16(act_g * u);
            }
        }
    }
}
