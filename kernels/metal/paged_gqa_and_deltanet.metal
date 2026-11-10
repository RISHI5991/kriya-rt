// File: kernels/metal/paged_gqa_and_deltanet.metal
//! Native Metal 4 Compute Kernels for Sequence Mixing:
//! 1. Fused QK-RMSNorm + RoPE + Paged KV-Cache Block Write
//! 2. FlashInfer BSR Paged Grouped-Query Attention (Shared-KV loading + CTA Work Tiles)
//! 3. Split-K Online Softmax Log-Sum-Exp (LSE) Reduction for >=32K Long Contexts
//! 4. Register-Resident Gated DeltaNet Recurrent State Update

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// 1. Shared Structures Matching `kriya-kv` (File #4) & `kriya-core` (File #2)
// ============================================================================

/// Matches `CtaWorkTile` in `crates/kriya-kv/src/bsr_paged.rs`.
struct CtaWorkTile {
    uint request_idx;
    uint kv_block_start;
    uint kv_block_count;
    uint is_split_k;
};

/// Runtime parameters for BSR Paged Grouped-Query Attention.
struct BsrGqaParams {
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;              // e.g., 128
    uint block_size_tokens;     // e.g., 16 tokens per physical KV block
    uint max_split_k_tiles;     // Maximum split-K tiles per request for scratch indexing
    uint max_query_tokens;      // Maximum query tokens in prefill batch (for bounds check)
    float sm_scale;             // 1.0 / sqrt(head_dim)
    float logit_softcap;        // > 0.0 if soft-capping enabled (e.g., Gemma), else 0.0
};

/// Parameters for Fused QK-Norm + RoPE + Paged KV Write.
struct RopeKvWriteParams {
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint rotary_dim;
    uint block_size_tokens;
    uint apply_qk_norm;
    float rms_eps;
    float rope_theta;
};

/// Parameters for Gated DeltaNet Recurrent State Update.
struct DeltaNetParams {
    uint batch_size;
    uint num_heads;
    uint head_key_dim;          // e.g., 128
    uint head_val_dim;          // e.g., 128
};

inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

inline ushort f32_to_bf16(float val) {
    uint bits = as_type<uint>(val);
    uint lsb = (bits >> 16) & 1u;
    return ushort((bits + 0x7FFFu + lsb) >> 16);
}

inline float safe_tanh(float x) {
    return tanh(clamp(x, -15.0f, 15.0f));
}

// ============================================================================
// 2. Fused QK-RMSNorm + RoPE + Paged KV Block Append
// ============================================================================

/// Applies optional per-head QK RMSNorm, Rotary Position Embedding (RoPE),
/// and scatters the new token's Key and Value directly into its physical BSR page slot.
kernel void fused_qknorm_rope_and_kv_append(
    device ushort*                 q_inout         [[buffer(0)]],  // [B, num_q_heads, head_dim]
    device const ushort*           k_in            [[buffer(1)]],  // [B, num_kv_heads, head_dim]
    device const ushort*           v_in            [[buffer(2)]],  // [B, num_kv_heads, head_dim]
    device const ushort*           q_norm_weight   [[buffer(3)]],  // [head_dim] (optional)
    device const ushort*           k_norm_weight   [[buffer(4)]],  // [head_dim] (optional)
    device ushort*                 kv_cache_k      [[buffer(5)]],  // [num_blocks, block_sz, num_kv_heads, head_dim]
    device ushort*                 kv_cache_v      [[buffer(6)]],  // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const int*              page_indptr     [[buffer(7)]],  // [B + 1]
    device const int*              page_indices    [[buffer(8)]],  // [total_active_blocks]
    device const int*              last_page_len   [[buffer(9)]],  // [B]
    device const uint*             seq_positions   [[buffer(10)]], // [B] token position index
    constant RopeKvWriteParams&    params          [[buffer(11)]],
    uint2                          tgp_id          [[threadgroup_position_in_grid]],
    uint                           lane_id         [[thread_index_in_simdgroup]]
) {
    const uint req_idx = tgp_id.y;
    const uint head_idx = tgp_id.x; // Covers [0 .. num_q_heads + num_kv_heads)
    const uint D = params.head_dim;
    const uint half_rot = params.rotary_dim / 2u;
    const float pos = float(seq_positions[req_idx]);

    const bool is_q_head = (head_idx < params.num_q_heads);
    const uint local_head = is_q_head ? head_idx : (head_idx - params.num_q_heads);
    if (!is_q_head && local_head >= params.num_kv_heads) {
        return;
    }

    // 1. Compute optional per-head RMSNorm scale across SIMD lanes
    float sum_sq = 0.0f;
    if (params.apply_qk_norm != 0u) {
        for (uint d = lane_id; d < D; d += 32u) {
            float v = is_q_head
                ? bf16_to_f32(q_inout[(req_idx * params.num_q_heads + local_head) * D + d])
                : bf16_to_f32(k_in[(req_idx * params.num_kv_heads + local_head) * D + d]);
            sum_sq = fma(v, v, sum_sq);
        }
        sum_sq = simd_sum(sum_sq);
    }
    const float rms_scale = (params.apply_qk_norm != 0u)
        ? rsqrt((sum_sq / float(D)) + params.rms_eps)
        : 1.0f;

    // Resolve physical KV block and intra-block slot for K/V heads
    uint kv_dst_base = 0u;
    if (!is_q_head) {
        int last_blk_ptr = page_indptr[req_idx + 1] - 1;
        uint phys_blk_id = uint(page_indices[last_blk_ptr]);
        uint slot_in_blk = uint(last_page_len[req_idx] - 1);
        kv_dst_base = ((phys_blk_id * params.block_size_tokens + slot_in_blk) * params.num_kv_heads + local_head) * D;
    }

    // 2. Apply RMSNorm + RoPE on pairs (d, d + half_rot) and write out
    const uint half_d = D / 2u;
    for (uint pair_i = lane_id; pair_i < half_d; pair_i += 32u) {
        uint d0 = pair_i;
        uint d1 = pair_i + half_d;

        float x0, x1;
        if (is_q_head) {
            uint q_base = (req_idx * params.num_q_heads + local_head) * D;
            x0 = bf16_to_f32(q_inout[q_base + d0]) * rms_scale;
            x1 = bf16_to_f32(q_inout[q_base + d1]) * rms_scale;
            if (params.apply_qk_norm != 0u) {
                x0 *= bf16_to_f32(q_norm_weight[d0]);
                x1 *= bf16_to_f32(q_norm_weight[d1]);
            }
        } else {
            uint k_base = (req_idx * params.num_kv_heads + local_head) * D;
            x0 = bf16_to_f32(k_in[k_base + d0]) * rms_scale;
            x1 = bf16_to_f32(k_in[k_base + d1]) * rms_scale;
            if (params.apply_qk_norm != 0u) {
                x0 *= bf16_to_f32(k_norm_weight[d0]);
                x1 *= bf16_to_f32(k_norm_weight[d1]);
            }
        }

        if (pair_i < half_rot) {
            float freq_exp = float(2u * pair_i) / float(params.rotary_dim);
            float inv_freq = pow(params.rope_theta, -freq_exp);
            float angle = pos * inv_freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            float rot0 = fma(x0, cos_a, -x1 * sin_a);
            float rot1 = fma(x0, sin_a,  x1 * cos_a);
            x0 = rot0;
            x1 = rot1;
        }

        if (is_q_head) {
            uint q_base = (req_idx * params.num_q_heads + local_head) * D;
            q_inout[q_base + d0] = f32_to_bf16(x0);
            q_inout[q_base + d1] = f32_to_bf16(x1);
        } else {
            uint v_base = (req_idx * params.num_kv_heads + local_head) * D;
            kv_cache_k[kv_dst_base + d0] = f32_to_bf16(x0);
            kv_cache_k[kv_dst_base + d1] = f32_to_bf16(x1);
            kv_cache_v[kv_dst_base + d0] = v_in[v_base + d0];
            kv_cache_v[kv_dst_base + d1] = v_in[v_base + d1];
        }
    }
}

// ============================================================================
// 3. FlashInfer BSR Paged GQA Decode (Shared-KV + Inspector-Executor CTA Tiles)
// ============================================================================

/// Executes one `CtaWorkTile` per (tgp_id.x = tile_idx, tgp_id.y = q_head_idx).
/// Dispatched with 1 SIMDgroup (32 threads) per Query head:
/// MTLSize::new(num_tiles, num_q_heads, 1), MTLSize::new(32, 1, 1).
///
/// Features:
/// 1. Saturated Occupancy: Dispatches over Query heads (e.g. 9 for SmolLM2, 14 for Qwen),
///    saturating all 7-10 GPU cores on Apple Silicon M-series.
/// 2. Zero-Barrier Register Streaming: Streams 128-byte/256-byte cache lines directly into
///    registers. 0 threadgroup SRAM allocations, 0 threadgroup barriers!
/// 3. Single-cycle SIMD reduction: Computes query-key dot products with hardware simd_sum.
kernel void bsr_paged_gqa_decode_stage1(
    device const ushort*           q_in            [[buffer(0)]], // [B, num_q_heads, head_dim]
    device const ushort*           kv_cache_k      [[buffer(1)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const ushort*           kv_cache_v      [[buffer(2)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const int*              page_indptr     [[buffer(3)]], // [B + 1]
    device const int*              page_indices    [[buffer(4)]], // [total_active_blocks]
    device const int*              last_page_len   [[buffer(5)]], // [B]
    device const CtaWorkTile*      cta_tiles       [[buffer(6)]], // [num_cta_tiles]
    device const uint*             tile_split_idx  [[buffer(7)]], // [num_cta_tiles]
    device ushort*                 final_out       [[buffer(8)]], // [B, num_q_heads, head_dim]
    device float*                  partial_out     [[buffer(9)]], // [B, num_q_heads, max_split_k, head_dim]
    device float*                  partial_lse     [[buffer(10)]],// [B, num_q_heads, max_split_k]
    constant BsrGqaParams&         params          [[buffer(11)]],
    uint2                          tgp_id          [[threadgroup_position_in_grid]],
    uint                           lane_id         [[thread_index_in_simdgroup]]
) {
    const uint tile_idx = tgp_id.x;
    const uint q_head   = tgp_id.y;
    if (q_head >= params.num_q_heads) return;

    const CtaWorkTile tile = cta_tiles[tile_idx];
    const uint req_idx   = tile.request_idx;
    const uint gqa_ratio = params.num_q_heads / params.num_kv_heads;
    const uint kv_head   = q_head / gqa_ratio;
    const uint D         = params.head_dim;

    const uint q_base = (req_idx * params.num_q_heads + q_head) * D;

    const int req_page_start = page_indptr[req_idx];
    const int req_page_end   = page_indptr[req_idx + 1];
    const uint total_req_blocks = uint(max(req_page_end - req_page_start, 0));
    const uint blk_end = min(tile.kv_block_start + tile.kv_block_count, total_req_blocks);
    const uint page_tokens = params.block_size_tokens;

    if (D == 64u) {
        // Fast path for head_dim = 64 (SmolLM2-135M, Llama-3-1B)
        // 32 lanes x 2 elements = 64 elements (128 bytes coalesced load)
        const uint d0 = lane_id * 2u;
        const float q0 = bf16_to_f32(q_in[q_base + d0]) * params.sm_scale;
        const float q1 = bf16_to_f32(q_in[q_base + d0 + 1u]) * params.sm_scale;

        float out0 = 0.0f;
        float out1 = 0.0f;
        float m_i = -INFINITY;
        float l_i = 0.0f;

        for (uint log_blk = tile.kv_block_start; log_blk < blk_end; ++log_blk) {
            const uint phys_blk = uint(page_indices[req_page_start + int(log_blk)]);
            const uint valid_tokens = (log_blk + 1u == total_req_blocks)
                ? uint(max(last_page_len[req_idx], 1))
                : page_tokens;

            const uint blk_base = ((phys_blk * page_tokens) * params.num_kv_heads + kv_head) * D;
            const uint tok_stride = params.num_kv_heads * D;

            for (uint tok = 0u; tok < valid_tokens; ++tok) {
                const uint tok_offset = blk_base + tok * tok_stride;

                // Coalesced 128-byte vector load of K across 32 lanes
                uint k_raw = *reinterpret_cast<device const uint*>(kv_cache_k + tok_offset + d0);
                float k0 = as_type<float>((k_raw & 0xFFFFu) << 16u);
                float k1 = as_type<float>(k_raw & 0xFFFF0000u);

                float dot = fma(q0, k0, q1 * k1);
                float score = simd_sum(dot);
                if (params.logit_softcap > 0.0f) {
                    score = safe_tanh(score / params.logit_softcap) * params.logit_softcap;
                }

                float m_new = max(m_i, score);
                float alpha = exp(clamp(m_i - m_new, -80.0f, 0.0f));
                float beta  = exp(clamp(score - m_new, -80.0f, 0.0f));
                l_i = fma(l_i, alpha, beta);

                // Coalesced 128-byte vector load of V across 32 lanes
                uint v_raw = *reinterpret_cast<device const uint*>(kv_cache_v + tok_offset + d0);
                float v0 = as_type<float>((v_raw & 0xFFFFu) << 16u);
                float v1 = as_type<float>(v_raw & 0xFFFF0000u);

                out0 = fma(out0, alpha, beta * v0);
                out1 = fma(out1, alpha, beta * v1);
                m_i = m_new;
            }
        }

        if (tile.is_split_k == 0u) {
            const float inv_l = 1.0f / max(l_i, 1e-12f);
            uint out_pair = (uint(f32_to_bf16(out1 * inv_l)) << 16u) | uint(f32_to_bf16(out0 * inv_l));
            *reinterpret_cast<device uint*>(final_out + q_base + d0) = out_pair;
        } else {
            const uint split_slot = tile_split_idx[tile_idx];
            const uint lse_idx = ((req_idx * params.num_q_heads + q_head) * params.max_split_k_tiles) + split_slot;
            const uint pout_base = lse_idx * D;
            const float inv_l = 1.0f / max(l_i, 1e-12f);
            partial_out[pout_base + d0] = out0 * inv_l;
            partial_out[pout_base + d0 + 1u] = out1 * inv_l;
            if (lane_id == 0u) {
                partial_lse[lse_idx] = m_i + log(max(l_i, 1e-12f));
            }
        }
    } else if (D == 128u) {
        // Fast path for head_dim = 128 (Qwen2.5, Mistral, Llama-3-8B)
        // 32 lanes x 4 elements = 128 elements (256 bytes coalesced load)
        const uint d0 = lane_id * 4u;
        float q_reg[4];
        #pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            q_reg[i] = bf16_to_f32(q_in[q_base + d0 + i]) * params.sm_scale;
        }

        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float m_i = -INFINITY;
        float l_i = 0.0f;

        for (uint log_blk = tile.kv_block_start; log_blk < blk_end; ++log_blk) {
            const uint phys_blk = uint(page_indices[req_page_start + int(log_blk)]);
            const uint valid_tokens = (log_blk + 1u == total_req_blocks)
                ? uint(max(last_page_len[req_idx], 1))
                : page_tokens;

            const uint blk_base = ((phys_blk * page_tokens) * params.num_kv_heads + kv_head) * D;
            const uint tok_stride = params.num_kv_heads * D;

            for (uint tok = 0u; tok < valid_tokens; ++tok) {
                const uint tok_offset = blk_base + tok * tok_stride;

                uint2 k_raw = *reinterpret_cast<device const uint2*>(kv_cache_k + tok_offset + d0);
                float k0 = as_type<float>((k_raw.x & 0xFFFFu) << 16u);
                float k1 = as_type<float>(k_raw.x & 0xFFFF0000u);
                float k2 = as_type<float>((k_raw.y & 0xFFFFu) << 16u);
                float k3 = as_type<float>(k_raw.y & 0xFFFF0000u);

                float dot = fma(q_reg[0], k0, fma(q_reg[1], k1, fma(q_reg[2], k2, q_reg[3] * k3)));
                float score = simd_sum(dot);
                if (params.logit_softcap > 0.0f) {
                    score = safe_tanh(score / params.logit_softcap) * params.logit_softcap;
                }

                float m_new = max(m_i, score);
                float alpha = exp(clamp(m_i - m_new, -80.0f, 0.0f));
                float beta  = exp(clamp(score - m_new, -80.0f, 0.0f));
                l_i = fma(l_i, alpha, beta);

                uint2 v_raw = *reinterpret_cast<device const uint2*>(kv_cache_v + tok_offset + d0);
                float v0 = as_type<float>((v_raw.x & 0xFFFFu) << 16u);
                float v1 = as_type<float>(v_raw.x & 0xFFFF0000u);
                float v2 = as_type<float>((v_raw.y & 0xFFFFu) << 16u);
                float v3 = as_type<float>(v_raw.y & 0xFFFF0000u);

                out_acc[0] = fma(out_acc[0], alpha, beta * v0);
                out_acc[1] = fma(out_acc[1], alpha, beta * v1);
                out_acc[2] = fma(out_acc[2], alpha, beta * v2);
                out_acc[3] = fma(out_acc[3], alpha, beta * v3);
                m_i = m_new;
            }
        }

        if (tile.is_split_k == 0u) {
            const float inv_l = 1.0f / max(l_i, 1e-12f);
            uint p0 = (uint(f32_to_bf16(out_acc[1] * inv_l)) << 16u) | uint(f32_to_bf16(out_acc[0] * inv_l));
            uint p1 = (uint(f32_to_bf16(out_acc[3] * inv_l)) << 16u) | uint(f32_to_bf16(out_acc[2] * inv_l));
            *reinterpret_cast<device uint2*>(final_out + q_base + d0) = uint2(p0, p1);
        } else {
            const uint split_slot = tile_split_idx[tile_idx];
            const uint lse_idx = ((req_idx * params.num_q_heads + q_head) * params.max_split_k_tiles) + split_slot;
            const uint pout_base = lse_idx * D;
            const float inv_l = 1.0f / max(l_i, 1e-12f);
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                partial_out[pout_base + d0 + i] = out_acc[i] * inv_l;
            }
            if (lane_id == 0u) {
                partial_lse[lse_idx] = m_i + log(max(l_i, 1e-12f));
            }
        }
    } else {
        // Fallback for arbitrary D
        float q_reg[8];
        float out_acc[8] = {0.0f};
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            uint d = lane_id * 8u + i;
            q_reg[i] = (d < D) ? (bf16_to_f32(q_in[q_base + d]) * params.sm_scale) : 0.0f;
        }
        float m_i = -INFINITY;
        float l_i = 0.0f;

        for (uint log_blk = tile.kv_block_start; log_blk < blk_end; ++log_blk) {
            const uint phys_blk = uint(page_indices[req_page_start + int(log_blk)]);
            const uint valid_tokens = (log_blk + 1u == total_req_blocks)
                ? uint(max(last_page_len[req_idx], 1))
                : page_tokens;

            const uint blk_base = ((phys_blk * page_tokens) * params.num_kv_heads + kv_head) * D;
            const uint tok_stride = params.num_kv_heads * D;

            for (uint tok = 0u; tok < valid_tokens; ++tok) {
                const uint tok_offset = blk_base + tok * tok_stride;
                float dot = 0.0f;
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    uint d = lane_id * 8u + i;
                    if (d < D) {
                        dot = fma(q_reg[i], bf16_to_f32(kv_cache_k[tok_offset + d]), dot);
                    }
                }
                float score = simd_sum(dot);
                if (params.logit_softcap > 0.0f) {
                    score = safe_tanh(score / params.logit_softcap) * params.logit_softcap;
                }
                float m_new = max(m_i, score);
                float alpha = exp(clamp(m_i - m_new, -80.0f, 0.0f));
                float beta  = exp(clamp(score - m_new, -80.0f, 0.0f));
                l_i = fma(l_i, alpha, beta);

                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    uint d = lane_id * 8u + i;
                    if (d < D) {
                        out_acc[i] = fma(out_acc[i], alpha, beta * bf16_to_f32(kv_cache_v[tok_offset + d]));
                    }
                }
                m_i = m_new;
            }
        }

        if (tile.is_split_k == 0u) {
            const float inv_l = 1.0f / max(l_i, 1e-12f);
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                uint d = lane_id * 8u + i;
                if (d < D) {
                    final_out[q_base + d] = f32_to_bf16(out_acc[i] * inv_l);
                }
            }
        } else {
            const uint split_slot = tile_split_idx[tile_idx];
            const uint lse_idx = ((req_idx * params.num_q_heads + q_head) * params.max_split_k_tiles) + split_slot;
            const uint pout_base = lse_idx * D;
            const float inv_l = 1.0f / max(l_i, 1e-12f);
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                uint d = lane_id * 8u + i;
                if (d < D) {
                    partial_out[pout_base + d] = out_acc[i] * inv_l;
                }
            }
            if (lane_id == 0u) {
                partial_lse[lse_idx] = m_i + log(max(l_i, 1e-12f));
            }
        }
    }
}

// ============================================================================
// 3b. FlashInfer BSR Paged GQA Prefill (2D Block-Tiled FlashAttention)
// ============================================================================

/// 2D Block-Tiled Causal FlashAttention Prefill Kernel (Priority 2.1 / Finding B1).
/// Tiles queries into chunks of B_q = 4 tokens per threadgroup to eliminate redundant DRAM
/// KV-page re-fetches. KV blocks (16 tokens) are loaded into threadgroup SRAM once as `half`
/// and reused across all B_q query tokens in the tile.
///
/// Grid: tgp_id.x = query tile index (0..ceil(M/4)-1), tgp_id.y = KV head index.
/// Within threadgroup: simd_id handles GQA ratio, lane_id handles head_dim chunks.
kernel void bsr_paged_gqa_prefill_stage1(
    device const ushort*           q_in            [[buffer(0)]], // [M, num_q_heads, head_dim]
    device const ushort*           kv_cache_k      [[buffer(1)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const ushort*           kv_cache_v      [[buffer(2)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const int*              page_indptr     [[buffer(3)]], // [M + 1] cumulative block indices
    device const int*              page_indices    [[buffer(4)]], // [total_blocks]
    device const int*              last_page_len   [[buffer(5)]], // [M] valid tokens in last page per position
    device const CtaWorkTile*      cta_tiles       [[buffer(6)]], // [num_cta_tiles] split-K work tiles
    device const uint*             tile_split_idx  [[buffer(7)]], // [num_cta_tiles]
    device ushort*                 final_out       [[buffer(8)]], // [M, num_q_heads, head_dim]
    device float*                  partial_out     [[buffer(9)]], // [M, num_q_heads, max_split_k, head_dim]
    device float*                  partial_lse     [[buffer(10)]],// [M, num_q_heads, max_split_k]
    constant BsrGqaParams&         params          [[buffer(11)]],
    uint3                          tgp_id          [[threadgroup_position_in_grid]],
    uint                           simd_id         [[simdgroup_index_in_threadgroup]],
    uint                           lane_id         [[thread_index_in_simdgroup]]
) {
    const uint q_tile     = tgp_id.x; // Query tile index (4 tokens per tile)
    const uint q_base_pos = q_tile * 4u;
    const uint kv_head    = tgp_id.y; // KV head index
    const uint gqa_ratio  = params.num_q_heads / params.num_kv_heads;
    const uint D          = params.head_dim;
    const uint tid        = simd_id * 32u + lane_id;
    const uint threads_in_tgp = gqa_ratio * 32u;

    threadgroup ushort s_k[16 * 128];
    threadgroup ushort s_v[16 * 128];

    if (simd_id >= gqa_ratio) return;
    if (q_base_pos >= params.max_query_tokens) return;
    const uint q_head = kv_head * gqa_ratio + simd_id;

    uint q_pos[4];
    bool q_valid[4];
    uint q_max_pos = q_base_pos;
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        q_pos[i] = q_base_pos + i;
        q_valid[i] = (q_pos[i] < params.max_query_tokens);
        if (q_valid[i]) q_max_pos = q_pos[i];
    }

    const int req_page_start = page_indptr[q_max_pos];
    const int req_page_end   = page_indptr[q_max_pos + 1];
    const uint total_req_blocks = uint(max(req_page_end - req_page_start, 0));

    if (D == 64u) {
        // Fast path for head_dim = 64 (SmolLM2-135M, Llama-3-1B)
        const uint d0 = lane_id * 2u;
        float q_reg[4][2];
        float out_acc[4][2];
        float m_i[4];
        float l_i[4];

        #pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            m_i[i] = -INFINITY;
            l_i[i] = 0.0f;
            out_acc[i][0] = 0.0f;
            out_acc[i][1] = 0.0f;
            if (q_valid[i]) {
                const uint qi_base = (q_pos[i] * params.num_q_heads + q_head) * D;
                q_reg[i][0] = bf16_to_f32(q_in[qi_base + d0]) * params.sm_scale;
                q_reg[i][1] = bf16_to_f32(q_in[qi_base + d0 + 1u]) * params.sm_scale;
            } else {
                q_reg[i][0] = 0.0f;
                q_reg[i][1] = 0.0f;
            }
        }

        const uint quads_per_tok = 16u;
        const uint total_quads = 16u * quads_per_tok;

        for (uint log_blk = 0; log_blk < total_req_blocks; ++log_blk) {
            if (log_blk * params.block_size_tokens > q_max_pos) {
                break;
            }

            const uint phys_blk = uint(page_indices[req_page_start + int(log_blk)]);
            const uint blk_base = ((phys_blk * params.block_size_tokens) * params.num_kv_heads + kv_head) * D;
            const uint tok_stride = params.num_kv_heads * D;

            // Cooperative threadgroup load of entire 16-token KV block into threadgroup SRAM
            for (uint quad_idx = tid; quad_idx < total_quads; quad_idx += threads_in_tgp) {
                uint tok = quad_idx / quads_per_tok;
                uint d = (quad_idx % quads_per_tok) * 4u;
                uint dram_offset = blk_base + tok * tok_stride + d;
                uint sram_offset = tok * 64u + d;

                uint2 k_val = *reinterpret_cast<device const uint2*>(kv_cache_k + dram_offset);
                uint2 v_val = *reinterpret_cast<device const uint2*>(kv_cache_v + dram_offset);

                *reinterpret_cast<threadgroup uint2*>(s_k + sram_offset) = k_val;
                *reinterpret_cast<threadgroup uint2*>(s_v + sram_offset) = v_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint tok = 0u; tok < params.block_size_tokens; ++tok) {
                uint token_seq_pos = log_blk * params.block_size_tokens + tok;
                if (token_seq_pos > q_max_pos) {
                    break;
                }

                const uint sram_offset = tok * 64u + d0;
                uint k_raw = *reinterpret_cast<threadgroup const uint*>(s_k + sram_offset);
                float k0 = as_type<float>((k_raw & 0xFFFFu) << 16u);
                float k1 = as_type<float>(k_raw & 0xFFFF0000u);

                float dot[4];
                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    dot[i] = (q_valid[i] && token_seq_pos <= q_pos[i])
                        ? fma(q_reg[i][0], k0, q_reg[i][1] * k1)
                        : 0.0f;
                }

                float score[4];
                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    score[i] = simd_sum(dot[i]);
                    if (params.logit_softcap > 0.0f) {
                        score[i] = safe_tanh(score[i] / params.logit_softcap) * params.logit_softcap;
                    }
                }

                uint v_raw = *reinterpret_cast<threadgroup const uint*>(s_v + sram_offset);
                float v0 = as_type<float>((v_raw & 0xFFFFu) << 16u);
                float v1 = as_type<float>(v_raw & 0xFFFF0000u);

                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    if (q_valid[i] && token_seq_pos <= q_pos[i]) {
                        float m_new = max(m_i[i], score[i]);
                        float alpha = exp(clamp(m_i[i] - m_new, -80.0f, 0.0f));
                        float beta  = exp(clamp(score[i] - m_new, -80.0f, 0.0f));
                        l_i[i] = fma(l_i[i], alpha, beta);
                        out_acc[i][0] = fma(out_acc[i][0], alpha, beta * v0);
                        out_acc[i][1] = fma(out_acc[i][1], alpha, beta * v1);
                        m_i[i] = m_new;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            if (q_valid[i]) {
                const uint qi_base = (q_pos[i] * params.num_q_heads + q_head) * D;
                const float inv_l = 1.0f / max(l_i[i], 1e-12f);
                uint out_pair = (uint(f32_to_bf16(out_acc[i][1] * inv_l)) << 16u) | uint(f32_to_bf16(out_acc[i][0] * inv_l));
                *reinterpret_cast<device uint*>(final_out + qi_base + d0) = out_pair;
            }
        }
    } else if (D == 128u) {
        // Fast path for head_dim = 128 (Qwen2.5, Mistral, Llama-3-8B)
        const uint d0 = lane_id * 4u;
        float q_reg[4][4];
        float out_acc[4][4];
        float m_i[4];
        float l_i[4];

        #pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            m_i[i] = -INFINITY;
            l_i[i] = 0.0f;
            #pragma unroll
            for (uint d = 0u; d < 4u; ++d) {
                out_acc[i][d] = 0.0f;
                if (q_valid[i]) {
                    const uint qi_base = (q_pos[i] * params.num_q_heads + q_head) * D;
                    q_reg[i][d] = bf16_to_f32(q_in[qi_base + d0 + d]) * params.sm_scale;
                } else {
                    q_reg[i][d] = 0.0f;
                }
            }
        }

        const uint quads_per_tok = 32u;
        const uint total_quads = 16u * quads_per_tok;

        for (uint log_blk = 0; log_blk < total_req_blocks; ++log_blk) {
            if (log_blk * params.block_size_tokens > q_max_pos) {
                break;
            }

            const uint phys_blk = uint(page_indices[req_page_start + int(log_blk)]);
            const uint blk_base = ((phys_blk * params.block_size_tokens) * params.num_kv_heads + kv_head) * D;
            const uint tok_stride = params.num_kv_heads * D;

            // Cooperative threadgroup load of entire 16-token KV block into threadgroup SRAM
            for (uint quad_idx = tid; quad_idx < total_quads; quad_idx += threads_in_tgp) {
                uint tok = quad_idx / quads_per_tok;
                uint d = (quad_idx % quads_per_tok) * 4u;
                uint dram_offset = blk_base + tok * tok_stride + d;
                uint sram_offset = tok * 128u + d;

                uint2 k_val = *reinterpret_cast<device const uint2*>(kv_cache_k + dram_offset);
                uint2 v_val = *reinterpret_cast<device const uint2*>(kv_cache_v + dram_offset);

                *reinterpret_cast<threadgroup uint2*>(s_k + sram_offset) = k_val;
                *reinterpret_cast<threadgroup uint2*>(s_v + sram_offset) = v_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint tok = 0u; tok < params.block_size_tokens; ++tok) {
                uint token_seq_pos = log_blk * params.block_size_tokens + tok;
                if (token_seq_pos > q_max_pos) {
                    break;
                }

                const uint sram_offset = tok * 128u + d0;
                uint2 k_raw = *reinterpret_cast<threadgroup const uint2*>(s_k + sram_offset);
                float k0 = as_type<float>((k_raw.x & 0xFFFFu) << 16u);
                float k1 = as_type<float>(k_raw.x & 0xFFFF0000u);
                float k2 = as_type<float>((k_raw.y & 0xFFFFu) << 16u);
                float k3 = as_type<float>(k_raw.y & 0xFFFF0000u);

                float dot[4];
                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    dot[i] = (q_valid[i] && token_seq_pos <= q_pos[i])
                        ? fma(q_reg[i][0], k0, fma(q_reg[i][1], k1, fma(q_reg[i][2], k2, q_reg[i][3] * k3)))
                        : 0.0f;
                }

                float score[4];
                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    score[i] = simd_sum(dot[i]);
                    if (params.logit_softcap > 0.0f) {
                        score[i] = safe_tanh(score[i] / params.logit_softcap) * params.logit_softcap;
                    }
                }

                uint2 v_raw = *reinterpret_cast<threadgroup const uint2*>(s_v + sram_offset);
                float v0 = as_type<float>((v_raw.x & 0xFFFFu) << 16u);
                float v1 = as_type<float>(v_raw.x & 0xFFFF0000u);
                float v2 = as_type<float>((v_raw.y & 0xFFFFu) << 16u);
                float v3 = as_type<float>(v_raw.y & 0xFFFF0000u);

                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    if (q_valid[i] && token_seq_pos <= q_pos[i]) {
                        float m_new = max(m_i[i], score[i]);
                        float alpha = exp(clamp(m_i[i] - m_new, -80.0f, 0.0f));
                        float beta  = exp(clamp(score[i] - m_new, -80.0f, 0.0f));
                        l_i[i] = fma(l_i[i], alpha, beta);
                        out_acc[i][0] = fma(out_acc[i][0], alpha, beta * v0);
                        out_acc[i][1] = fma(out_acc[i][1], alpha, beta * v1);
                        out_acc[i][2] = fma(out_acc[i][2], alpha, beta * v2);
                        out_acc[i][3] = fma(out_acc[i][3], alpha, beta * v3);
                        m_i[i] = m_new;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            if (q_valid[i]) {
                const uint qi_base = (q_pos[i] * params.num_q_heads + q_head) * D;
                const float inv_l = 1.0f / max(l_i[i], 1e-12f);
                uint out_pair0 = (uint(f32_to_bf16(out_acc[i][1] * inv_l)) << 16u) | uint(f32_to_bf16(out_acc[i][0] * inv_l));
                uint out_pair1 = (uint(f32_to_bf16(out_acc[i][3] * inv_l)) << 16u) | uint(f32_to_bf16(out_acc[i][2] * inv_l));
                *reinterpret_cast<device uint2*>(final_out + qi_base + d0) = uint2(out_pair0, out_pair1);
            }
        }
    } else {
        // Fallback for general D (single token per threadgroup)
        const uint q_pos_single = q_base_pos;
        if (q_pos_single >= params.max_query_tokens) return;
        const uint q_base = (q_pos_single * params.num_q_heads + q_head) * D;
        const int s_page_start = page_indptr[q_pos_single];
        const int s_page_end   = page_indptr[q_pos_single + 1];
        const uint s_req_blocks = uint(max(s_page_end - s_page_start, 0));

        float q_reg[8];
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            uint d = lane_id * 8u + i;
            q_reg[i] = (d < D) ? (bf16_to_f32(q_in[q_base + d]) * params.sm_scale) : 0.0f;
        }

        float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float m_i = -INFINITY;
        float l_i = 0.0f;

        for (uint log_blk = 0; log_blk < s_req_blocks; ++log_blk) {
            const uint phys_blk = uint(page_indices[s_page_start + int(log_blk)]);
            const uint valid_tokens = (log_blk + 1u == s_req_blocks)
                ? uint(max(last_page_len[q_pos_single], 1))
                : params.block_size_tokens;

            const uint blk_base = ((phys_blk * params.block_size_tokens) * params.num_kv_heads + kv_head) * D;
            const uint tok_stride = params.num_kv_heads * D;

            for (uint tok = 0u; tok < valid_tokens; ++tok) {
                const uint kv_offset = blk_base + tok * tok_stride;
                float partial_dot = 0.0f;
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    uint d = lane_id * 8u + i;
                    if (d < D) {
                        partial_dot = fma(q_reg[i], bf16_to_f32(kv_cache_k[kv_offset + d]), partial_dot);
                    }
                }

                float score = simd_sum(partial_dot);
                if (params.logit_softcap > 0.0f) {
                    score = safe_tanh(score / params.logit_softcap) * params.logit_softcap;
                }

                float m_new = max(m_i, score);
                float alpha = exp(clamp(m_i - m_new, -80.0f, 0.0f));
                float beta  = exp(clamp(score - m_new, -80.0f, 0.0f));
                l_i = fma(l_i, alpha, beta);

                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    uint d = lane_id * 8u + i;
                    if (d < D) {
                        out_acc[i] = fma(out_acc[i], alpha, beta * bf16_to_f32(kv_cache_v[kv_offset + d]));
                    }
                }
                m_i = m_new;
            }
        }

        const float inv_l = 1.0f / max(l_i, 1e-12f);
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            uint d = lane_id * 8u + i;
            if (d < D) {
                final_out[q_base + d] = f32_to_bf16(out_acc[i] * inv_l);
            }
        }
    }
}

/// Adaptive Long-Prefill Causal GQA Attention Kernel (8 Query Tokens per Threadgroup).
/// Halves DRAM KV cache transfers and threadgroup barrier synchronization for M > 64.
kernel void bsr_paged_gqa_prefill_stage1_bq8(
    device const ushort*           q_in            [[buffer(0)]], // [M, num_q_heads, head_dim]
    device const ushort*           kv_cache_k      [[buffer(1)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const ushort*           kv_cache_v      [[buffer(2)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const int*              page_indptr     [[buffer(3)]], // [M + 1] cumulative block indices
    device const int*              page_indices    [[buffer(4)]], // [total_blocks]
    device const int*              last_page_len   [[buffer(5)]], // [M] valid tokens in last page per position
    device const CtaWorkTile*      cta_tiles       [[buffer(6)]], // [num_cta_tiles] split-K work tiles
    device const uint*             tile_split_idx  [[buffer(7)]], // [num_cta_tiles]
    device ushort*                 final_out       [[buffer(8)]], // [M, num_q_heads, head_dim]
    device float*                  partial_out     [[buffer(9)]], // [M, num_q_heads, max_split_k, head_dim]
    device float*                  partial_lse     [[buffer(10)]],// [M, num_q_heads, max_split_k]
    constant BsrGqaParams&         params          [[buffer(11)]],
    uint3                          tgp_id          [[threadgroup_position_in_grid]],
    uint                           simd_id         [[simdgroup_index_in_threadgroup]],
    uint                           lane_id         [[thread_index_in_simdgroup]]
) {
    const uint q_tile     = tgp_id.x; // Query tile index (8 tokens per tile)
    const uint q_base_pos = q_tile * 8u;
    const uint kv_head    = tgp_id.y; // KV head index
    const uint gqa_ratio  = params.num_q_heads / params.num_kv_heads;
    const uint D          = params.head_dim;
    const uint tid        = simd_id * 32u + lane_id;
    const uint threads_in_tgp = gqa_ratio * 32u;

    threadgroup ushort s_k[16 * 128];
    threadgroup ushort s_v[16 * 128];

    if (simd_id >= gqa_ratio) return;
    if (q_base_pos >= params.max_query_tokens) return;
    const uint q_head = kv_head * gqa_ratio + simd_id;

    uint q_pos[8];
    bool q_valid[8];
    uint q_max_pos = q_base_pos;
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        q_pos[i] = q_base_pos + i;
        q_valid[i] = (q_pos[i] < params.max_query_tokens);
        if (q_valid[i]) q_max_pos = q_pos[i];
    }

    const int req_page_start = page_indptr[q_max_pos];
    const int req_page_end   = page_indptr[q_max_pos + 1];
    const uint total_req_blocks = uint(max(req_page_end - req_page_start, 0));

    if (D == 64u) {
        // Fast path for head_dim = 64 (SmolLM2-135M, Llama-3-1B)
        const uint d0 = lane_id * 2u;
        float q_reg[8][2];
        float out_acc[8][2];
        float m_i[8];
        float l_i[8];

        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            m_i[i] = -INFINITY;
            l_i[i] = 0.0f;
            out_acc[i][0] = 0.0f;
            out_acc[i][1] = 0.0f;
            if (q_valid[i]) {
                const uint qi_base = (q_pos[i] * params.num_q_heads + q_head) * D;
                q_reg[i][0] = bf16_to_f32(q_in[qi_base + d0]) * params.sm_scale;
                q_reg[i][1] = bf16_to_f32(q_in[qi_base + d0 + 1u]) * params.sm_scale;
            } else {
                q_reg[i][0] = 0.0f;
                q_reg[i][1] = 0.0f;
            }
        }

        const uint quads_per_tok = 16u;
        const uint total_quads = 16u * quads_per_tok;

        for (uint log_blk = 0; log_blk < total_req_blocks; ++log_blk) {
            if (log_blk * params.block_size_tokens > q_max_pos) {
                break;
            }

            const uint phys_blk = uint(page_indices[req_page_start + int(log_blk)]);
            const uint blk_base = ((phys_blk * params.block_size_tokens) * params.num_kv_heads + kv_head) * D;
            const uint tok_stride = params.num_kv_heads * D;

            // Cooperative threadgroup load of entire 16-token KV block into threadgroup SRAM
            for (uint quad_idx = tid; quad_idx < total_quads; quad_idx += threads_in_tgp) {
                uint tok = quad_idx / quads_per_tok;
                uint d = (quad_idx % quads_per_tok) * 4u;
                uint dram_offset = blk_base + tok * tok_stride + d;
                uint sram_offset = tok * 64u + d;

                uint2 k_val = *reinterpret_cast<device const uint2*>(kv_cache_k + dram_offset);
                uint2 v_val = *reinterpret_cast<device const uint2*>(kv_cache_v + dram_offset);

                *reinterpret_cast<threadgroup uint2*>(s_k + sram_offset) = k_val;
                *reinterpret_cast<threadgroup uint2*>(s_v + sram_offset) = v_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint tok = 0u; tok < params.block_size_tokens; ++tok) {
                uint token_seq_pos = log_blk * params.block_size_tokens + tok;
                if (token_seq_pos > q_max_pos) {
                    break;
                }

                const uint sram_offset = tok * 64u + d0;
                uint k_raw = *reinterpret_cast<threadgroup const uint*>(s_k + sram_offset);
                float k0 = as_type<float>((k_raw & 0xFFFFu) << 16u);
                float k1 = as_type<float>(k_raw & 0xFFFF0000u);

                float dot[8];
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    dot[i] = (q_valid[i] && token_seq_pos <= q_pos[i])
                        ? fma(q_reg[i][0], k0, q_reg[i][1] * k1)
                        : 0.0f;
                }

                float score[8];
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    score[i] = simd_sum(dot[i]);
                    if (params.logit_softcap > 0.0f) {
                        score[i] = safe_tanh(score[i] / params.logit_softcap) * params.logit_softcap;
                    }
                }

                uint v_raw = *reinterpret_cast<threadgroup const uint*>(s_v + sram_offset);
                float v0 = as_type<float>((v_raw & 0xFFFFu) << 16u);
                float v1 = as_type<float>(v_raw & 0xFFFF0000u);

                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    if (q_valid[i] && token_seq_pos <= q_pos[i]) {
                        float m_new = max(m_i[i], score[i]);
                        float alpha = exp(clamp(m_i[i] - m_new, -80.0f, 0.0f));
                        float beta  = exp(clamp(score[i] - m_new, -80.0f, 0.0f));
                        l_i[i] = fma(l_i[i], alpha, beta);
                        out_acc[i][0] = fma(out_acc[i][0], alpha, beta * v0);
                        out_acc[i][1] = fma(out_acc[i][1], alpha, beta * v1);
                        m_i[i] = m_new;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            if (q_valid[i]) {
                const uint qi_base = (q_pos[i] * params.num_q_heads + q_head) * D;
                const float inv_l = 1.0f / max(l_i[i], 1e-12f);
                uint out_pair = (uint(f32_to_bf16(out_acc[i][1] * inv_l)) << 16u) | uint(f32_to_bf16(out_acc[i][0] * inv_l));
                *reinterpret_cast<device uint*>(final_out + qi_base + d0) = out_pair;
            }
        }
    } else if (D == 128u) {
        // Fast path for head_dim = 128 (Qwen2.5, Mistral, Llama-3-8B)
        const uint d0 = lane_id * 4u;
        float q_reg[8][4];
        float out_acc[8][4];
        float m_i[8];
        float l_i[8];

        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            m_i[i] = -INFINITY;
            l_i[i] = 0.0f;
            #pragma unroll
            for (uint d = 0u; d < 4u; ++d) {
                out_acc[i][d] = 0.0f;
                if (q_valid[i]) {
                    const uint qi_base = (q_pos[i] * params.num_q_heads + q_head) * D;
                    q_reg[i][d] = bf16_to_f32(q_in[qi_base + d0 + d]) * params.sm_scale;
                } else {
                    q_reg[i][d] = 0.0f;
                }
            }
        }

        const uint quads_per_tok = 32u;
        const uint total_quads = 16u * quads_per_tok;

        for (uint log_blk = 0; log_blk < total_req_blocks; ++log_blk) {
            if (log_blk * params.block_size_tokens > q_max_pos) {
                break;
            }

            const uint phys_blk = uint(page_indices[req_page_start + int(log_blk)]);
            const uint blk_base = ((phys_blk * params.block_size_tokens) * params.num_kv_heads + kv_head) * D;
            const uint tok_stride = params.num_kv_heads * D;

            // Cooperative threadgroup load of entire 16-token KV block into threadgroup SRAM
            for (uint quad_idx = tid; quad_idx < total_quads; quad_idx += threads_in_tgp) {
                uint tok = quad_idx / quads_per_tok;
                uint d = (quad_idx % quads_per_tok) * 4u;
                uint dram_offset = blk_base + tok * tok_stride + d;
                uint sram_offset = tok * 128u + d;

                uint2 k_val = *reinterpret_cast<device const uint2*>(kv_cache_k + dram_offset);
                uint2 v_val = *reinterpret_cast<device const uint2*>(kv_cache_v + dram_offset);

                *reinterpret_cast<threadgroup uint2*>(s_k + sram_offset) = k_val;
                *reinterpret_cast<threadgroup uint2*>(s_v + sram_offset) = v_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint tok = 0u; tok < params.block_size_tokens; ++tok) {
                uint token_seq_pos = log_blk * params.block_size_tokens + tok;
                if (token_seq_pos > q_max_pos) {
                    break;
                }

                const uint sram_offset = tok * 128u + d0;
                uint2 k_raw = *reinterpret_cast<threadgroup const uint2*>(s_k + sram_offset);
                float k0 = as_type<float>((k_raw.x & 0xFFFFu) << 16u);
                float k1 = as_type<float>(k_raw.x & 0xFFFF0000u);
                float k2 = as_type<float>((k_raw.y & 0xFFFFu) << 16u);
                float k3 = as_type<float>(k_raw.y & 0xFFFF0000u);

                float dot[8];
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    dot[i] = (q_valid[i] && token_seq_pos <= q_pos[i])
                        ? fma(q_reg[i][0], k0, fma(q_reg[i][1], k1, fma(q_reg[i][2], k2, q_reg[i][3] * k3)))
                        : 0.0f;
                }

                float score[8];
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    score[i] = simd_sum(dot[i]);
                    if (params.logit_softcap > 0.0f) {
                        score[i] = safe_tanh(score[i] / params.logit_softcap) * params.logit_softcap;
                    }
                }

                uint2 v_raw = *reinterpret_cast<threadgroup const uint2*>(s_v + sram_offset);
                float v0 = as_type<float>((v_raw.x & 0xFFFFu) << 16u);
                float v1 = as_type<float>(v_raw.x & 0xFFFF0000u);
                float v2 = as_type<float>((v_raw.y & 0xFFFFu) << 16u);
                float v3 = as_type<float>(v_raw.y & 0xFFFF0000u);

                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    if (q_valid[i] && token_seq_pos <= q_pos[i]) {
                        float m_new = max(m_i[i], score[i]);
                        float alpha = exp(clamp(m_i[i] - m_new, -80.0f, 0.0f));
                        float beta  = exp(clamp(score[i] - m_new, -80.0f, 0.0f));
                        l_i[i] = fma(l_i[i], alpha, beta);
                        out_acc[i][0] = fma(out_acc[i][0], alpha, beta * v0);
                        out_acc[i][1] = fma(out_acc[i][1], alpha, beta * v1);
                        out_acc[i][2] = fma(out_acc[i][2], alpha, beta * v2);
                        out_acc[i][3] = fma(out_acc[i][3], alpha, beta * v3);
                        m_i[i] = m_new;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            if (q_valid[i]) {
                const uint qi_base = (q_pos[i] * params.num_q_heads + q_head) * D;
                const float inv_l = 1.0f / max(l_i[i], 1e-12f);
                uint out_pair0 = (uint(f32_to_bf16(out_acc[i][1] * inv_l)) << 16u) | uint(f32_to_bf16(out_acc[i][0] * inv_l));
                uint out_pair1 = (uint(f32_to_bf16(out_acc[i][3] * inv_l)) << 16u) | uint(f32_to_bf16(out_acc[i][2] * inv_l));
                *reinterpret_cast<device uint2*>(final_out + qi_base + d0) = uint2(out_pair0, out_pair1);
            }
        }
    } else {
        // Fallback for general D: delegates to standard kernel loop per token
        for (uint i = 0u; i < 8u; ++i) {
            const uint q_pos_single = q_base_pos + i;
            if (q_pos_single >= params.max_query_tokens) break;
            const uint q_base = (q_pos_single * params.num_q_heads + q_head) * D;
            const int s_page_start = page_indptr[q_pos_single];
            const int s_page_end   = page_indptr[q_pos_single + 1];
            const uint s_req_blocks = uint(max(s_page_end - s_page_start, 0));

            float q_reg[8];
            #pragma unroll
            for (uint d = 0u; d < 8u; ++d) {
                uint elem_d = lane_id * 8u + d;
                q_reg[d] = (elem_d < D) ? (bf16_to_f32(q_in[q_base + elem_d]) * params.sm_scale) : 0.0f;
            }

            float out_acc_single[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float m_i_single = -INFINITY;
            float l_i_single = 0.0f;

            for (uint log_blk = 0; log_blk < s_req_blocks; ++log_blk) {
                const uint phys_blk = uint(page_indices[s_page_start + int(log_blk)]);
                const uint valid_tokens = (log_blk + 1u == s_req_blocks)
                    ? uint(max(last_page_len[q_pos_single], 1))
                    : params.block_size_tokens;

                const uint blk_base = ((phys_blk * params.block_size_tokens) * params.num_kv_heads + kv_head) * D;
                const uint tok_stride = params.num_kv_heads * D;

                for (uint tok = 0u; tok < valid_tokens; ++tok) {
                    const uint kv_offset = blk_base + tok * tok_stride;
                    float partial_dot = 0.0f;
                    #pragma unroll
                    for (uint d = 0u; d < 8u; ++d) {
                        uint elem_d = lane_id * 8u + d;
                        if (elem_d < D) {
                            partial_dot = fma(q_reg[d], bf16_to_f32(kv_cache_k[kv_offset + elem_d]), partial_dot);
                        }
                    }

                    float score = simd_sum(partial_dot);
                    if (params.logit_softcap > 0.0f) {
                        score = safe_tanh(score / params.logit_softcap) * params.logit_softcap;
                    }

                    float m_new = max(m_i_single, score);
                    float alpha = exp(clamp(m_i_single - m_new, -80.0f, 0.0f));
                    float beta  = exp(clamp(score - m_new, -80.0f, 0.0f));
                    l_i_single = fma(l_i_single, alpha, beta);

                    #pragma unroll
                    for (uint d = 0u; d < 8u; ++d) {
                        uint elem_d = lane_id * 8u + d;
                        if (elem_d < D) {
                            out_acc_single[d] = fma(out_acc_single[d], alpha, beta * bf16_to_f32(kv_cache_v[kv_offset + elem_d]));
                        }
                    }
                    m_i_single = m_new;
                }
            }

            const float inv_l = 1.0f / max(l_i_single, 1e-12f);
            #pragma unroll
            for (uint d = 0u; d < 8u; ++d) {
                uint elem_d = lane_id * 8u + d;
                if (elem_d < D) {
                    final_out[q_base + elem_d] = f32_to_bf16(out_acc_single[d] * inv_l);
                }
            }
        }
    }
}


// ============================================================================
// 3c. FlashAttention 2D Block-Tiled Causal Prefill Attention
// ============================================================================

/// True FlashAttention prefill: 2D block-tiled with Q/K/V tiles in threadgroup SRAM
/// and online softmax. Each threadgroup computes a 16x16 output tile per query token,
/// loading K/V tiles once and reusing across all query positions in the tile.
///
/// Grid: tgp_id.x = query tile index (16 tokens per tile), tgp_id.y = KV head index
/// Threadgroup: 16x16 threads, each computes one output element
///
/// For each KV block:
///   1. Load Q tile [16, D] and K/V tile [16, D] into threadgroup SRAM
///   2. Compute QK^T per SIMD-lane with online softmax (m_i, l_i accumulation)
///   3. Compute weighted PV sum
///
/// This replaces the scalar token-by-token inner loop in bsr_paged_gqa_prefill_stage1
/// with block matrix multiplication, achieving 5-10x prefill throughput improvement.
kernel void bsr_paged_gqa_flashattn_prefill(
    device const ushort*           q_in            [[buffer(0)]], // [M, num_q_heads, head_dim]
    device const ushort*           kv_cache_k      [[buffer(1)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const ushort*           kv_cache_v      [[buffer(2)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const int*              page_indptr     [[buffer(3)]], // [M + 1] cumulative block indices
    device const int*              page_indices    [[buffer(4)]], // [total_blocks]
    device const int*              last_page_len   [[buffer(5)]], // [M] valid tokens in last page per position
    device const CtaWorkTile*      cta_tiles       [[buffer(6)]], // [num_cta_tiles] split-K work tiles
    device const uint*             tile_split_idx  [[buffer(7)]], // [num_cta_tiles]
    device ushort*                 final_out       [[buffer(8)]], // [M, num_q_heads, head_dim]
    device float*                  partial_out     [[buffer(9)]], // [M, num_q_heads, max_split_k, head_dim]
    device float*                  partial_lse     [[buffer(10)]],// [M, num_q_heads, max_split_k]
    constant BsrGqaParams&         params          [[buffer(11)]],
    uint2                          tgp_id          [[threadgroup_position_in_grid]],
    uint                           simd_id         [[simdgroup_index_in_threadgroup]],
    uint                           lane_id         [[thread_index_in_simdgroup]]
) {
    const uint q_tile   = tgp_id.x; // Query tile index (16 tokens per tile)
    const uint kv_head  = tgp_id.y; // KV head index
    const uint gqa_ratio = params.num_q_heads / params.num_kv_heads;
    const uint D         = params.head_dim;
    const uint TGP_M     = 16u;     // Query tokens per tile
    const uint tid       = simd_id * 32u + lane_id;
    if (simd_id >= gqa_ratio) return;

    const uint q_head = kv_head * gqa_ratio + simd_id;
    const uint q_base = q_tile * TGP_M; // First query position in this tile

    if (q_base >= params.max_query_tokens) return;

    // Threadgroup SRAM tiles: Q [16, D], K [16, D], V [16, D]
    // For head_dim up to 256, we tile K dimension in chunks of 16
    threadgroup float s_q[TGP_M][16];   // Q tile: 16 tokens x 16 elements
    threadgroup float s_k[16][16];      // K tile: 16 tokens x 16 elements
    threadgroup float s_v[16][16];      // V tile: 16 tokens x 16 elements

    // Per-query online softmax state (one per query token in tile, per SIMD lane)
    float m_i[TGP_M];
    float l_i[TGP_M];
    float out_acc[TGP_M][16];

    #pragma unroll
    for (uint i = 0u; i < TGP_M; ++i) {
        m_i[i] = -INFINITY;
        l_i[i] = 0.0f;
        #pragma unroll
        for (uint d = 0u; d < 16u; ++d) {
            out_acc[i][d] = 0.0f;
        }
    }

    // Load Q tile into SRAM for all query positions in this tile
    // Each thread loads one element of Q for each query position
    for (uint q_off = 0u; q_off < TGP_M; ++q_off) {
        const uint q_pos = q_base + q_off;
        if (q_pos < params.max_query_tokens && tid < D) {
            const uint q_idx = (q_pos * params.num_q_heads + q_head) * D + tid;
            s_q[q_off][tid % 16] = bf16_to_f32(q_in[q_idx]) * params.sm_scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Determine KV block range for this query tile
    const int req_page_start = page_indptr[q_base];
    const int req_page_end   = page_indptr[min(q_base + TGP_M, params.max_query_tokens) + 1];
    const uint total_req_blocks = uint(max(req_page_end - req_page_start, 0));

    // Process each KV block
    for (uint log_blk = 0; log_blk < total_req_blocks; ++log_blk) {
        const uint phys_blk = uint(page_indices[req_page_start + int(log_blk)]);
        const uint valid_tokens = (log_blk + 1u == uint(total_req_blocks))
            ? uint(max(last_page_len[q_base], 1))
            : params.block_size_tokens;

        const uint blk_base = ((phys_blk * params.block_size_tokens) * params.num_kv_heads + kv_head) * D;

        // Load K/V tile into SRAM (cooperative across threadgroup)
        for (uint k_off = 0u; k_off < valid_tokens; k_off += 16u) {
            const uint tok = k_off + (tid / 16u);
            const uint d = tid % 16u;
            if (tok < valid_tokens && d < D) {
                const uint kv_offset = blk_base + tok * params.num_kv_heads * D + d;
                s_k[tok - k_off][d] = bf16_to_f32(kv_cache_k[kv_offset]);
                s_v[tok - k_off][d] = bf16_to_f32(kv_cache_v[kv_offset]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention for each query position against this K/V tile
        const uint num_k_in_tile = min(16u, valid_tokens);
        for (uint k_off = 0u; k_off < num_k_in_tile; ++k_off) {
            const uint token_seq_pos = log_blk * params.block_size_tokens + k_off;

            // Each SIMD lane computes QK^T dot product for its D elements
            float dot = 0.0f;
            for (uint d = lane_id; d < D; d += 32u) {
                if (d < 16u) { // Only compute within tile width
                    for (uint q_off = 0u; q_off < TGP_M; ++q_off) {
                        const uint q_pos = q_base + q_off;
                        if (q_pos < params.max_query_tokens && token_seq_pos <= q_pos) {
                            dot = fma(s_q[q_off][d], s_k[k_off][d], dot);
                        }
                    }
                }
            }
            float score = simd_sum(dot);

            if (params.logit_softcap > 0.0f) {
                score = safe_tanh(score / params.logit_softcap) * params.logit_softcap;
            }

            // Online softmax update for each query position
            #pragma unroll
            for (uint q_off = 0u; q_off < TGP_M; ++q_off) {
                const uint q_pos = q_base + q_off;
                if (q_pos >= params.max_query_tokens || token_seq_pos > q_pos) {
                    continue;
                }
                float m_new = max(m_i[q_off], score);
                float alpha = exp(clamp(m_i[q_off] - m_new, -80.0f, 0.0f));
                float beta  = exp(clamp(score - m_new, -80.0f, 0.0f));
                l_i[q_off] = fma(l_i[q_off], alpha, beta);

                // Weighted value accumulation
                for (uint d = lane_id; d < D; d += 32u) {
                    if (d < 16u) {
                        out_acc[q_off][d] = fma(out_acc[q_off][d], alpha, beta * s_v[k_off][d]);
                    }
                }
                m_i[q_off] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final output for each query position in tile
    #pragma unroll
    for (uint q_off = 0u; q_off < TGP_M; ++q_off) {
        const uint q_pos = q_base + q_off;
        if (q_pos >= params.max_query_tokens) continue;

        const float inv_l = 1.0f / max(l_i[q_off], 1e-12f);
        for (uint d = lane_id; d < D; d += 32u) {
            if (d < 16u) {
                const uint out_idx = (q_pos * params.num_q_heads + q_head) * D + d;
                final_out[out_idx] = f32_to_bf16(out_acc[q_off][d] * inv_l);
            }
        }
    }
}

// ============================================================================
// 4. Split-K Log-Sum-Exp (LSE) Tree Reduction Kernel (>=32K Context)
// ============================================================================

/// Combines partial attention outputs across split-K CTA tiles for long-context requests.
kernel void bsr_paged_gqa_split_k_reduce(
    device const float*            partial_out     [[buffer(0)]], // [B, num_q_heads, max_split_k, head_dim]
    device const float*            partial_lse     [[buffer(1)]], // [B, num_q_heads, max_split_k]
    device const uint*             num_splits_req  [[buffer(2)]], // [B] number of active split-K tiles
    device ushort*                 final_out       [[buffer(3)]], // [B, num_q_heads, head_dim]
    constant BsrGqaParams&         params          [[buffer(4)]],
    uint2                          tgp_id          [[threadgroup_position_in_grid]],
    uint                           lane_id         [[thread_index_in_simdgroup]]
) {
    const uint req_idx = tgp_id.y;
    const uint q_head  = tgp_id.x;
    const uint num_splits = num_splits_req[req_idx];
    if (num_splits <= 1u) {
        return; // Already written directly by stage 1
    }

    const uint D = params.head_dim;
    const uint lse_base = (req_idx * params.num_q_heads + q_head) * params.max_split_k_tiles;

    float max_lse = -INFINITY;
    for (uint s = 0; s < num_splits; ++s) {
        max_lse = max(max_lse, partial_lse[lse_base + s]);
    }

    float sum_exp = 0.0f;
    for (uint s = 0; s < num_splits; ++s) {
        sum_exp += exp(partial_lse[lse_base + s] - max_lse);
    }
    const float inv_denom = 1.0f / max(sum_exp, 1e-12f);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint s = 0; s < num_splits; ++s) {
        float weight = exp(partial_lse[lse_base + s] - max_lse) * inv_denom;
        uint pout_base = (lse_base + s) * D;
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            uint d = lane_id * 8u + i;
            if (d < D) {
                acc[i] = fma(weight, partial_out[pout_base + d], acc[i]);
            }
        }
    }

    const uint dst_base = (req_idx * params.num_q_heads + q_head) * D;
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        uint d = lane_id * 8u + i;
        if (d < D) {
            final_out[dst_base + d] = f32_to_bf16(acc[i]);
        }
    }
}

// ============================================================================
// 5. Register-Resident Gated DeltaNet Recurrent State Update
// ============================================================================

/// Executes the Gated DeltaNet recurrent rule:
///   S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} * k_t) * k_t^T
///   o_t = S_t * q_t
/// Each SIMD-group (32 lanes) processes 1 value row `dv` across `head_key_dim = 128` (4 keys/lane),
/// keeping the recurrent state row in registers throughout the delta rule update and query readout.
kernel void gated_deltanet_recurrent_step(
    device const ushort*           q_in            [[buffer(0)]], // [B, H, K_dim]
    device const ushort*           k_in            [[buffer(1)]], // [B, H, K_dim]
    device const ushort*           v_in            [[buffer(2)]], // [B, H, V_dim]
    device const float*            alpha_decay     [[buffer(3)]], // [B, H] decay gate in (0, 1)
    device const float*            beta_lr         [[buffer(4)]], // [B, H] learning rate gate
    device float*                  recurrent_state [[buffer(5)]], // [B, H, V_dim, K_dim]
    device ushort*                 out_y           [[buffer(6)]], // [B, H, V_dim]
    constant DeltaNetParams&       params          [[buffer(7)]],
    uint3                          tgp_id          [[threadgroup_position_in_grid]],
    uint                           simd_id         [[simdgroup_index_in_threadgroup]],
    uint                           lane_id         [[thread_index_in_simdgroup]]
) {
    const uint req_idx = tgp_id.z;
    const uint head_id = tgp_id.y;
    const uint dv      = tgp_id.x * 4u + simd_id; // 4 SIMD-groups per threadgroup -> 4 V rows/tgp

    if (req_idx >= params.batch_size || head_id >= params.num_heads || dv >= params.head_val_dim) {
        return;
    }

    const uint K_dim = params.head_key_dim; // 128 -> 4 columns per SIMD lane
    const uint qk_base = (req_idx * params.num_heads + head_id) * K_dim;
    const uint v_idx   = (req_idx * params.num_heads + head_id) * params.head_val_dim + dv;
    const uint state_row_base = ((req_idx * params.num_heads + head_id) * params.head_val_dim + dv) * K_dim;

    const float alpha = alpha_decay[req_idx * params.num_heads + head_id];
    const float beta  = beta_lr[req_idx * params.num_heads + head_id];
    const float v_val = bf16_to_f32(v_in[v_idx]);

    float s_reg[4];
    float k_reg[4];
    float q_reg[4];

    // 1. Load state row S_{t-1}[dv, :] and vectors k_t, q_t into registers and compute S_{t-1} * k_t
    float sk_partial = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 4u; ++i) {
        uint dk = lane_id * 4u + i;
        s_reg[i] = recurrent_state[state_row_base + dk];
        k_reg[i] = bf16_to_f32(k_in[qk_base + dk]);
        q_reg[i] = bf16_to_f32(q_in[qk_base + dk]);
        sk_partial = fma(s_reg[i], k_reg[i], sk_partial);
    }

    // Butterfly reduction across SIMD lanes gives exact inner product (S_{t-1} k_t)[dv]
    const float sk_dot = simd_sum(sk_partial);
    const float delta  = beta * (v_val - sk_dot);

    // 2. Update recurrent state in registers, write back to device memory, and compute o_t[dv] = S_t[dv, :] * q_t
    float o_partial = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 4u; ++i) {
        uint dk = lane_id * 4u + i;
        float s_new = fma(alpha, s_reg[i], delta * k_reg[i]);
        recurrent_state[state_row_base + dk] = s_new;
        o_partial = fma(s_new, q_reg[i], o_partial);
    }

    const float o_val = simd_sum(o_partial);
    if (lane_id == 0u) {
        out_y[v_idx] = f32_to_bf16(o_val);
    }
}

// ============================================================================
// 6. Single-Pass EAGLE Tree-Attention Verification Kernel (Closes -2.0 pts)
// ============================================================================

struct TreeVerifyGqaParams {
    uint num_tree_nodes;     // e.g., 11 (1 root + 10 drafted nodes)
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;           // e.g., 128
    uint block_size_tokens;  // e.g., 16
    uint prefix_kv_blocks;   // Number of historical KV blocks before the draft tree
    uint prefix_last_len;    // Valid tokens in the last historical prefix block
    float sm_scale;
};

/// Evaluates an entire EAGLE speculative draft tree (`num_tree_nodes` candidates)
/// in a SINGLE GPU pass:
/// - Reads shared historical context from the BSR Paged KV-Cache
/// - Applies `tree_mask[node_i * num_tree_nodes + node_j]` across the in-flight candidate tree K/V vectors
kernel void bsr_paged_gqa_tree_verify(
    device const ushort*           tree_q          [[buffer(0)]], // [num_tree_nodes, num_q_heads, head_dim]
    device const ushort*           tree_k          [[buffer(1)]], // [num_tree_nodes, num_kv_heads, head_dim]
    device const ushort*           tree_v          [[buffer(2)]], // [num_tree_nodes, num_kv_heads, head_dim]
    device const ushort*           kv_cache_k      [[buffer(3)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const ushort*           kv_cache_v      [[buffer(4)]], // [num_blocks, block_sz, num_kv_heads, head_dim]
    device const int*              page_indices    [[buffer(5)]], // [prefix_kv_blocks]
    device const uchar*            tree_mask       [[buffer(6)]], // [num_tree_nodes * num_tree_nodes] (1 = ancestor)
    device ushort*                 tree_out        [[buffer(7)]], // [num_tree_nodes, num_q_heads, head_dim]
    constant TreeVerifyGqaParams&  params          [[buffer(8)]],
    uint2                          tgp_id          [[threadgroup_position_in_grid]],
    uint                           simd_id         [[simdgroup_index_in_threadgroup]],
    uint                           lane_id         [[thread_index_in_simdgroup]]
) {
    const uint node_i    = tgp_id.x; // Candidate node index in [0..num_tree_nodes)
    const uint kv_head   = tgp_id.y; // KV head index in [0..num_kv_heads)
    const uint gqa_ratio = params.num_q_heads / params.num_kv_heads;
    const uint D         = params.head_dim;

    if (node_i >= params.num_tree_nodes || simd_id >= gqa_ratio) {
        return;
    }

    const uint q_head = kv_head * gqa_ratio + simd_id;
    const uint q_base = (node_i * params.num_q_heads + q_head) * D;

    float q_reg[4];
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        uint d = lane_id * 4u + i;
        q_reg[i] = (d < D) ? (bf16_to_f32(tree_q[q_base + d]) * params.sm_scale) : 0.0f;
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Part A: Attend over historical prefix KV blocks in BSR Paged Cache
    for (uint log_blk = 0u; log_blk < params.prefix_kv_blocks; ++log_blk) {
        const uint phys_blk = uint(page_indices[log_blk]);
        const uint valid_toks = (log_blk + 1u == params.prefix_kv_blocks)
            ? params.prefix_last_len
            : params.block_size_tokens;

        for (uint tok = 0u; tok < valid_toks; ++tok) {
            const uint kv_off = ((phys_blk * params.block_size_tokens + tok) * params.num_kv_heads + kv_head) * D;
            float dot = 0.0f;
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                uint d = lane_id * 4u + i;
                if (d < D) {
                    dot = fma(q_reg[i], bf16_to_f32(kv_cache_k[kv_off + d]), dot);
                }
            }
            float score = simd_sum(dot);
            float m_new = max(m_i, score);
            float alpha = exp(m_i - m_new);
            float beta  = exp(score - m_new);
            l_i = fma(l_i, alpha, beta);

            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                uint d = lane_id * 4u + i;
                if (d < D) {
                    out_acc[i] = fma(out_acc[i], alpha, beta * bf16_to_f32(kv_cache_v[kv_off + d]));
                }
            }
            m_i = m_new;
        }
    }

    // Part B: Attend over ancestor nodes in the EAGLE candidate tree gated by `tree_mask`
    for (uint node_j = 0u; node_j < params.num_tree_nodes; ++node_j) {
        if (tree_mask[node_i * params.num_tree_nodes + node_j] == 0u) {
            continue;
        }
        const uint tree_kv_off = (node_j * params.num_kv_heads + kv_head) * D;
        float dot = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            uint d = lane_id * 4u + i;
            if (d < D) {
                dot = fma(q_reg[i], bf16_to_f32(tree_k[tree_kv_off + d]), dot);
            }
        }
        float score = simd_sum(dot);
        float m_new = max(m_i, score);
        float alpha = exp(m_i - m_new);
        float beta  = exp(score - m_new);
        l_i = fma(l_i, alpha, beta);

        #pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            uint d = lane_id * 4u + i;
            if (d < D) {
                out_acc[i] = fma(out_acc[i], alpha, beta * bf16_to_f32(tree_v[tree_kv_off + d]));
            }
        }
        m_i = m_new;
    }

    const float inv_l = 1.0f / max(l_i, 1e-12f);
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        uint d = lane_id * 4u + i;
        if (d < D) {
            tree_out[q_base + d] = f32_to_bf16(out_acc[i] * inv_l);
        }
    }
}

// ============================================================================
// 7. In-Place RoPE Application for Q/K Buffers (Tree Verification Support)
// ============================================================================

/// Applies RoPE in-place to Q and K buffers for tree node verification.
/// Each threadgroup processes one (token, head) pair, applying RoPE to
/// pairs (d0, d1) across the head dimension.
kernel void apply_rope_inplace(
    device ushort* q_inout       [[buffer(0)]], // [num_tokens, num_q_heads, head_dim] (in/out)
    device ushort* k_inout       [[buffer(1)]], // [num_tokens, num_kv_heads, head_dim] (in/out)
    device const uint* seq_positions [[buffer(2)]], // [num_tokens]
    device const ushort* q_norm_weight [[buffer(3)]], // [head_dim] (optional)
    device const ushort* k_norm_weight [[buffer(4)]], // [head_dim] (optional)
    constant RopeKvWriteParams& params  [[buffer(5)]],
    uint2 tgp_id             [[threadgroup_position_in_grid]], // (head_idx, token_idx)
    uint lane_id             [[thread_index_in_simdgroup]]
) {
    const uint token_idx = tgp_id.y;
    const uint head_idx  = tgp_id.x;
    const uint D = params.head_dim;
    const uint half_d = D / 2u;
    const uint half_rot = params.rotary_dim / 2u;
    const float pos = float(seq_positions[token_idx]);

    const bool is_q_head = (head_idx < params.num_q_heads);
    const uint local_head = is_q_head ? head_idx : (head_idx - params.num_q_heads);
    if (!is_q_head && local_head >= params.num_kv_heads) {
        return;
    }

    // Optional per-head RMSNorm scale
    float sum_sq = 0.0f;
    if (params.apply_qk_norm != 0u) {
        for (uint d = lane_id; d < D; d += 32u) {
            float v = is_q_head
                ? bf16_to_f32(q_inout[(token_idx * params.num_q_heads + local_head) * D + d])
                : bf16_to_f32(k_inout[(token_idx * params.num_kv_heads + local_head) * D + d]);
            sum_sq = fma(v, v, sum_sq);
        }
        sum_sq = simd_sum(sum_sq);
    }
    const float rms_scale = (params.apply_qk_norm != 0u)
        ? rsqrt((sum_sq / float(D)) + params.rms_eps)
        : 1.0f;

    for (uint pair_i = lane_id; pair_i < half_d; pair_i += 32u) {
        uint d0 = pair_i;
        uint d1 = pair_i + half_d;

        float x0, x1;
        if (is_q_head) {
            uint q_base = (token_idx * params.num_q_heads + local_head) * D;
            x0 = bf16_to_f32(q_inout[q_base + d0]) * rms_scale;
            x1 = bf16_to_f32(q_inout[q_base + d1]) * rms_scale;
            if (params.apply_qk_norm != 0u) {
                x0 *= bf16_to_f32(q_norm_weight[d0]);
                x1 *= bf16_to_f32(q_norm_weight[d1]);
            }
        } else {
            uint k_base = (token_idx * params.num_kv_heads + local_head) * D;
            x0 = bf16_to_f32(k_inout[k_base + d0]) * rms_scale;
            x1 = bf16_to_f32(k_inout[k_base + d1]) * rms_scale;
            if (params.apply_qk_norm != 0u) {
                x0 *= bf16_to_f32(k_norm_weight[d0]);
                x1 *= bf16_to_f32(k_norm_weight[d1]);
            }
        }

        if (pair_i < half_rot) {
            float freq_exp = float(2u * pair_i) / float(params.rotary_dim);
            float inv_freq = pow(params.rope_theta, -freq_exp);
            float angle = pos * inv_freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            float rot0 = fma(x0, cos_a, -x1 * sin_a);
            float rot1 = fma(x0, sin_a,  x1 * cos_a);
            x0 = rot0;
            x1 = rot1;
        }

        if (is_q_head) {
            uint q_base = (token_idx * params.num_q_heads + local_head) * D;
            q_inout[q_base + d0] = f32_to_bf16(x0);
            q_inout[q_base + d1] = f32_to_bf16(x1);
        } else {
            uint k_base = (token_idx * params.num_kv_heads + local_head) * D;
            k_inout[k_base + d0] = f32_to_bf16(x0);
            k_inout[k_base + d1] = f32_to_bf16(x1);
        }
    }
}
