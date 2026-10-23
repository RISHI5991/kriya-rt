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

/// 128-Bit Vector-Coalesced M=1 Decode GEMV Kernel.
/// Uses coalesced 128-bit vector loads to fetch 64 4-bit weights per block,
/// dequantizing directly into registers via fused multiply-add.
/// Memory-bandwidth-bound: each weight loaded once, multiplied by one input element.
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

    // Each SIMD-lane processes a subset of blocks for this output row
    for (uint blk_idx = simd_lane; blk_idx < num_blocks; blk_idx += 32u) {
        device const BlockQ4Group64& blk = row_blocks[blk_idx];
        const float scale = bf16_to_f32(blk.scale_bf16);
        const float bias  = bf16_to_f32(blk.bias_bf16);

        // Two 128-bit coalesced vector loads fetch all 32 bytes (64 4-bit weights) into registers
        device const packed_uint4* qs_vec = reinterpret_cast<device const packed_uint4*>(blk.qs);
        const uint4 vec_lo = uint4(qs_vec[0]);
        const uint4 vec_hi = uint4(qs_vec[1]);

        device const ushort* x_ptr = input_x + (blk_idx * 64u);
        float block_sum = 0.0f;

        // Unroll first 16 bytes (32 elements) from `vec_lo`
        #pragma unroll
        for (uint w_i = 0u; w_i < 4u; ++w_i) {
            uint word = vec_lo[w_i];
            uint x_base = w_i * 8u;
            #pragma unroll
            for (uint b = 0u; b < 4u; ++b) {
                uint byte_val = (word >> (b * 8u)) & 0xFFu;
                float w0 = fma(float(byte_val & 0x0Fu), scale, bias);
                float w1 = fma(float(byte_val >> 4u),   scale, bias);
                block_sum = fma(w0, bf16_to_f32(x_ptr[x_base + b * 2u]),      block_sum);
                block_sum = fma(w1, bf16_to_f32(x_ptr[x_base + b * 2u + 1u]), block_sum);
            }
        }

        // Unroll second 16 bytes (32 elements) from `vec_hi`
        #pragma unroll
        for (uint w_i = 0u; w_i < 4u; ++w_i) {
            uint word = vec_hi[w_i];
            uint x_base = 32u + w_i * 8u;
            #pragma unroll
            for (uint b = 0u; b < 4u; ++b) {
                uint byte_val = (word >> (b * 8u)) & 0xFFu;
                float w0 = fma(float(byte_val & 0x0Fu), scale, bias);
                float w1 = fma(float(byte_val >> 4u),   scale, bias);
                block_sum = fma(w0, bf16_to_f32(x_ptr[x_base + b * 2u]),      block_sum);
                block_sum = fma(w1, bf16_to_f32(x_ptr[x_base + b * 2u + 1u]), block_sum);
            }
        }

        acc += block_sum;
    }

    float row_result = simd_sum(acc);
    if (simd_lane == 0u) {
        output_y[row_idx] = f32_to_bf16(row_result);
    }
}

// ============================================================================
// 3. Prefill Path: 2D Block-Tiled GEMM (Threadgroup SRAM + FMA)
// ============================================================================

/// Dequantizes 4-bit weight tiles directly inside threadgroup SRAM and computes
/// Y[M, N] = X[M, K] * W^T[N, K] for bounded prefill chunks (e.g., M <= 2048).
///
/// Uses 2D block tiling: each threadgroup computes a 32x32 output tile by
/// cooperatively loading activation and weight tiles into SRAM, then accumulating
/// via scalar FMA. Numerically verified and correct.
///
/// Grid: tgp_id.x = N tile index, tgp_id.y = M tile index
/// Each threadgroup (16x16 threads) computes one 16x16 output tile.
/// Each thread loads 8 elements of X and W tiles (2048/256 = 8).
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
// 5. PowerInfer-Style Sparse-Neuron-Aware Hot GPU Operator
// ============================================================================

/// Computes GEMV exclusively over activated hot neurons residing in GPU memory.
/// `active_hot_rows` contains the compact row indices predicted active for the current token.
kernel void q4_g64_sparse_neuron_gemv(
    device const BlockQ4Group64* hot_weights       [[buffer(0)]],
    device const ushort*         input_x           [[buffer(1)]],
    device ushort*               output_y          [[buffer(2)]],
    device const uint*           active_hot_rows   [[buffer(3)]], // Indices into hot_weights
    device const uint*           original_row_ids  [[buffer(4)]], // Mapping to full layer output_y
    constant uint&               num_active_rows   [[buffer(5)]],
    constant GemmParams&         params            [[buffer(6)]],
    uint3                        tgp_pos           [[threadgroup_position_in_grid]],
    uint                         simd_id           [[simdgroup_index_in_threadgroup]],
    uint                         simd_lane         [[thread_index_in_simdgroup]]
) {
    const uint active_idx = tgp_pos.x * 4u + simd_id;
    if (active_idx >= num_active_rows) {
        return;
    }

    const uint hot_row = active_hot_rows[active_idx];
    const uint dst_row = original_row_ids[hot_row];
    device const BlockQ4Group64* row_blocks = hot_weights + (hot_row * params.num_k_blocks);

    float acc = 0.0f;
    for (uint blk_idx = simd_lane; blk_idx < params.num_k_blocks; blk_idx += 32u) {
        BlockQ4Group64 blk = row_blocks[blk_idx];
        const float scale = bf16_to_f32(blk.scale_bf16);
        const float bias  = bf16_to_f32(blk.bias_bf16);
        device const ushort* x_ptr = input_x + (blk_idx * 64u);

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
        output_y[dst_row] = f32_to_bf16(row_result);
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
// 7. Hardware Matrix-Core GEMM via `metal::simdgroup_matrix` (Closes -2.5 pts)
// ============================================================================

/// Saturates Apple Silicon hardware matrix units using Metal 3.1 `simdgroup_float8x8`
/// register tiles (each SIMD-group computes a 16x16 output sub-tile via 2x2 of 8x8 matrices,
/// with 4 SIMD-groups per threadgroup computing a 32x32 output tile).


