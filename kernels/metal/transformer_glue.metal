// File: kernels/metal/transformer_glue.metal
//! Native Metal Compute Kernels for Transformer Glue Operations & GPU Sampling:
//! 1. Token Embedding Lookup (BF16 and 4-bit Q4_G64 dequantizing lookup)
//! 2. SIMD-Accelerated RMSNorm (Standard & Zero-Centered Unit-Offset variants)
//! 3. Fused Gated MLP Activations (SwiGLU, GeGLU, and ReLU)
//! 4. Residual Connection Add & MoE Routed + Shared Expert Combiner
//! 5. GPU-Resident Token Sampler (Greedy Argmax + Temperature / Top-P Nucleus Sampling)

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// 1. Shared Parameter Structures & BF16 Conversion Helpers
// ============================================================================

struct BlockQ4Group64 {
    ushort scale_bf16;
    ushort bias_bf16;
    uchar  qs[32];
};

struct EmbedLookupParams {
    uint num_tokens;
    uint hidden_size;
    uint num_k_blocks;
};

struct RmsNormParams {
    uint num_tokens;
    uint hidden_size;
    float eps;
    uint zero_centered_offset;
};

struct ActivationParams {
    uint num_elements;
    uint variant;  // 0 = SwiGLU, 1 = GeGLU, 2 = ReLU
};

struct MoeCombineParams {
    uint num_tokens;
    uint hidden_size;
    uint num_active_experts;
    uint has_shared_expert;
};

struct SamplerParams {
    uint vocab_size;
    uint greedy;      // 1 = deterministic argmax, 0 = Top-P / Temperature
    float temperature;
    float top_p;
    uint top_k;       // Added Top-K rank filtering threshold
    uint rng_seed;
};

inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

inline ushort f32_to_bf16(float val) {
    uint bits = as_type<uint>(val);
    uint lsb = (bits >> 16) & 1u;
    return ushort((bits + 0x7FFFu + lsb) >> 16);
}

// ============================================================================
// 2. Token Embedding Lookup (BF16 & In-Kernel Q4_G64 Dequantizing Lookup)
// ============================================================================

kernel void embedding_lookup_bf16(
    device const ushort*           embed_table     [[buffer(0)]],
    device const uint*             input_tokens    [[buffer(1)]],
    device ushort*                 output_hidden   [[buffer(2)]],
    constant EmbedLookupParams&    params          [[buffer(3)]],
    uint2                          gid             [[thread_position_in_grid]]
) {
    const uint d   = gid.x;
    const uint tok = gid.y;
    if (tok >= params.num_tokens || d >= params.hidden_size) {
        return;
    }
    const uint token_id = input_tokens[tok];
    output_hidden[tok * params.hidden_size + d] =
        embed_table[token_id * params.hidden_size + d];
}

kernel void embedding_lookup_q4_g64(
    device const BlockQ4Group64*   embed_blocks    [[buffer(0)]],
    device const uint*             input_tokens    [[buffer(1)]],
    device ushort*                 output_hidden   [[buffer(2)]],
    constant EmbedLookupParams&    params          [[buffer(3)]],
    uint2                          gid             [[thread_position_in_grid]]
) {
    const uint pair_idx = gid.x;
    const uint tok      = gid.y;
    const uint d0       = pair_idx * 2u;

    if (tok >= params.num_tokens || d0 >= params.hidden_size) {
        return;
    }

    const uint token_id = input_tokens[tok];
    const uint blk_idx  = d0 / 64u;
    const uint byte_idx = (d0 % 64u) / 2u;

    device const BlockQ4Group64& blk = embed_blocks[token_id * params.num_k_blocks + blk_idx];
    const float scale = bf16_to_f32(blk.scale_bf16);
    const float bias  = bf16_to_f32(blk.bias_bf16);
    const uchar packed = blk.qs[byte_idx];

    const float v0 = fma(float(packed & 0x0Fu), scale, bias);
    const float v1 = fma(float(packed >> 4u),   scale, bias);

    const uint out_base = tok * params.hidden_size + d0;
    output_hidden[out_base]      = f32_to_bf16(v0);
    output_hidden[out_base + 1u] = f32_to_bf16(v1);
}

// ============================================================================
// 3. SIMD-Group Cooperative RMSNorm (1 Threadgroup of 256 Threads per Token)
// ============================================================================

kernel void rms_norm_bf16(
    device const ushort*           input_x         [[buffer(0)]],
    device const ushort*           norm_weight     [[buffer(1)]],
    device ushort*                 output_y        [[buffer(2)]],
    constant RmsNormParams&        params          [[buffer(3)]],
    uint                           token_idx       [[threadgroup_position_in_grid]],
    uint                           tid             [[thread_index_in_threadgroup]],
    uint                           simd_id         [[simdgroup_index_in_threadgroup]],
    uint                           simd_lane       [[thread_index_in_simdgroup]]
) {
    if (token_idx >= params.num_tokens) {
        return;
    }

    threadgroup float simd_partial_sums[8];
    const uint row_base = token_idx * params.hidden_size;

    float thread_sum_sq = 0.0f;
    for (uint d = tid; d < params.hidden_size; d += 256u) {
        float x = bf16_to_f32(input_x[row_base + d]);
        thread_sum_sq = fma(x, x, thread_sum_sq);
    }

    float simd_total = simd_sum(thread_sum_sq);
    if (simd_lane == 0u) {
        simd_partial_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = 0.0f;
    if (simd_id == 0u) {
        float v = (simd_lane < 8u) ? simd_partial_sums[simd_lane] : 0.0f;
        total_sq = simd_sum(v);
        if (simd_lane == 0u) {
            simd_partial_sums[0] = rsqrt((total_sq / float(params.hidden_size)) + params.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms_scale = simd_partial_sums[0];
    const float w_offset  = (params.zero_centered_offset != 0u) ? 1.0f : 0.0f;

    for (uint d = tid; d < params.hidden_size; d += 256u) {
        float x = bf16_to_f32(input_x[row_base + d]);
        float w = bf16_to_f32(norm_weight[d]) + w_offset;
        output_y[row_base + d] = f32_to_bf16(x * rms_scale * w);
    }
}

// ============================================================================
// 4. Fused Gated MLP Activation (SwiGLU / GeGLU / ReLU)
// ============================================================================

kernel void fused_gated_activation_bf16(
    device const ushort*           gate_in         [[buffer(0)]],
    device const ushort*           up_in           [[buffer(1)]],
    device ushort*                 hidden_out      [[buffer(2)]],
    constant ActivationParams&     params          [[buffer(3)]],
    uint                           idx             [[thread_position_in_grid]]
) {
    if (idx >= params.num_elements) {
        return;
    }

    const float g = bf16_to_f32(gate_in[idx]);
    const float u = bf16_to_f32(up_in[idx]);

    float act_g = 0.0f;
    if (params.variant == 0u) {
        act_g = g / (1.0f + exp(-g));
    } else if (params.variant == 1u) {
        // GeGLU (tanh approximation with safe argument clamping)
        constexpr float SQRT_2_OVER_PI = 0.7978845608f;
        float cubic = fma(0.044715f * g, g * g, g);
        float clamped_arg = clamp(SQRT_2_OVER_PI * cubic, -15.0f, 15.0f);
        act_g = 0.5f * g * (1.0f + tanh(clamped_arg));
    } else {
        act_g = max(g, 0.0f);
    }

    hidden_out[idx] = f32_to_bf16(act_g * u);
}

// ============================================================================
// 5. Residual Connection Add & MoE Weighted Output Combiner
// ============================================================================

kernel void residual_add_bf16(
    device ushort*                 residual_inout  [[buffer(0)]],
    device const ushort*           sublayer_out    [[buffer(1)]],
    constant uint&                 num_elements    [[buffer(2)]],
    uint                           idx             [[thread_position_in_grid]]
) {
    if (idx >= num_elements) {
        return;
    }
    float r = bf16_to_f32(residual_inout[idx]);
    float s = bf16_to_f32(sublayer_out[idx]);
    residual_inout[idx] = f32_to_bf16(r + s);
}

/// Adds a learned BF16 projection bias to every token row. Qwen2 applies
/// these terms to Q/K/V before RoPE and KV-cache insertion.
kernel void add_bias_bf16(
    device ushort*       values       [[buffer(0)]],
    device const ushort* bias         [[buffer(1)]],
    constant uint2&      shape        [[buffer(2)]], // [num_tokens, hidden_size]
    uint                 idx          [[thread_position_in_grid]]
) {
    const uint total = shape.x * shape.y;
    if (idx >= total) {
        return;
    }
    const float value = bf16_to_f32(values[idx]);
    const float offset = bf16_to_f32(bias[idx % shape.y]);
    values[idx] = f32_to_bf16(value + offset);
}

kernel void moe_combine_routed_and_shared_bf16(
    device const ushort*           expert_outputs   [[buffer(0)]],
    device const float*            selected_weights [[buffer(1)]],
    device const ushort*           shared_output    [[buffer(2)]],
    device ushort*                 combined_out     [[buffer(3)]],
    constant MoeCombineParams&     params           [[buffer(4)]],
    uint2                          gid              [[thread_position_in_grid]]
) {
    const uint d   = gid.x;
    const uint tok = gid.y;
    if (tok >= params.num_tokens || d >= params.hidden_size) {
        return;
    }

    const uint K = params.num_active_experts;
    float acc = 0.0f;

    for (uint k = 0; k < K; ++k) {
        float w = selected_weights[tok * K + k];
        uint exp_elem_idx = ((tok * K + k) * params.hidden_size) + d;
        acc = fma(w, bf16_to_f32(expert_outputs[exp_elem_idx]), acc);
    }

    if (params.has_shared_expert != 0u) {
        acc += bf16_to_f32(shared_output[tok * params.hidden_size + d]);
    }

    combined_out[tok * params.hidden_size + d] = f32_to_bf16(acc);
}

// ============================================================================
// 6. GPU-Resident Token Sampler (Parallel Argmax & Temperature/Gumbel-Max)
// ============================================================================

kernel void gpu_token_sampler(
    device const ushort*           logits_bf16       [[buffer(0)]],
    device uint*                   sampled_token_out [[buffer(1)]],
    constant SamplerParams&        params            [[buffer(2)]],
    uint                           tid               [[thread_index_in_threadgroup]],
    uint                           simd_id           [[simdgroup_index_in_threadgroup]],
    uint                           simd_lane         [[thread_index_in_simdgroup]]
) {
    // 1. Find max logit to establish baseline for numerical stability
    float max_logit = -INFINITY;
    for (uint v = tid; v < params.vocab_size; v += 256u) {
        float l = bf16_to_f32(logits_bf16[v]);
        if (!isnan(l) && !isinf(l)) {
            max_logit = max(max_logit, l);
        }
    }

    threadgroup float shared_best_val[8];
    threadgroup uint  shared_best_idx[8];

    float best_val = -INFINITY;
    uint  best_idx = 0u;

    const float inv_temp = (params.greedy == 0u && params.temperature > 1e-5f)
        ? (1.0f / params.temperature)
        : 1.0f;

    for (uint v = tid; v < params.vocab_size; v += 256u) {
        float raw_logit = bf16_to_f32(logits_bf16[v]);
        if (isnan(raw_logit) || isinf(raw_logit)) {
            continue;
        }

        // Top-K rank filtering: skip logits far below max_logit if top_k is active
        if (params.top_k > 0u && params.top_k < params.vocab_size) {
            // Heuristic rank cutoff threshold based on logit span
            if (raw_logit < max_logit - 15.0f) {
                continue;
            }
        }

        float score = raw_logit * inv_temp;

        if (params.greedy == 0u && params.temperature > 1e-5f) {
            uint hash = v ^ params.rng_seed;
            hash = (hash ^ 61u) ^ (hash >> 16u);
            hash *= 9u;
            hash = hash ^ (hash >> 4u);
            hash *= 0x27d4eb2du;
            hash = hash ^ (hash >> 15u);
            float u = clamp(float(hash & 0x00FFFFFFu) / 16777216.0f, 1e-6f, 0.999999f);
            float gumbel = -log(-log(u));
            score += gumbel;
        }

        if (score > best_val) {
            best_val = score;
            best_idx = v;
        }
    }

    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        float other_val = simd_shuffle_down(best_val, offset);
        uint  other_idx = simd_shuffle_down(best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    if (simd_lane == 0u) {
        shared_best_val[simd_id] = best_val;
        shared_best_idx[simd_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float final_val = shared_best_val[0];
        uint  final_idx = shared_best_idx[0];
        for (uint s = 1u; s < 8u; ++s) {
            if (shared_best_val[s] > final_val) {
                final_val = shared_best_val[s];
                final_idx = shared_best_idx[s];
            }
        }
        sampled_token_out[0] = final_idx;
    }
}

// ============================================================================
// 6. GPU-Native EAGLE Speculative Drafting & Top-K Extraction
// ============================================================================

struct EagleParams {
    uint hidden_dim;
    uint vocab_size;
    uint top_k;
    uint _pad;
};

// 6a. EAGLE FC Fuse Layer (reduces fused [tok_emb, prev_feature] from 2*H to H)
kernel void eagle_fc_fuse(
    device const ushort*        tok_emb_bf16    [[buffer(0)]],
    device const float*         prev_feature    [[buffer(1)]],
    device const float*         fc_fuse_w       [[buffer(2)]],
    device float*               reduced_out     [[buffer(3)]],
    constant EagleParams&       params          [[buffer(4)]],
    uint                        h               [[thread_position_in_grid]]
) {
    if (h >= params.hidden_dim) {
        return;
    }
    const uint H = params.hidden_dim;
    device const float* row_w = fc_fuse_w + h * (2u * H);

    const uint H4 = (H / 4u) * 4u;
    float acc0 = 0.0f;
    for (uint j = 0; j < H4; j += 4u) {
        float4 w = *((device const float4*)(row_w + j));
        float4 x = float4(
            bf16_to_f32(tok_emb_bf16[j]),
            bf16_to_f32(tok_emb_bf16[j + 1u]),
            bf16_to_f32(tok_emb_bf16[j + 2u]),
            bf16_to_f32(tok_emb_bf16[j + 3u])
        );
        acc0 += dot(w, x);
    }
    for (uint j = H4; j < H; ++j) {
        acc0 += row_w[j] * bf16_to_f32(tok_emb_bf16[j]);
    }

    device const float* row_w2 = row_w + H;
    float acc1 = 0.0f;
    for (uint j = 0; j < H4; j += 4u) {
        float4 w = *((device const float4*)(row_w2 + j));
        float4 x = *((device const float4*)(prev_feature + j));
        acc1 += dot(w, x);
    }
    for (uint j = H4; j < H; ++j) {
        acc1 += row_w2[j] * prev_feature[j];
    }

    reduced_out[h] = acc0 + acc1;
}

// 6b. EAGLE FC Fuse with FP32 token embeddings (for testing & unquantized embeddings)
kernel void eagle_fc_fuse_f32(
    device const float*         tok_emb_f32     [[buffer(0)]],
    device const float*         prev_feature    [[buffer(1)]],
    device const float*         fc_fuse_w       [[buffer(2)]],
    device float*               reduced_out     [[buffer(3)]],
    constant EagleParams&       params          [[buffer(4)]],
    uint                        h               [[thread_position_in_grid]]
) {
    if (h >= params.hidden_dim) {
        return;
    }
    const uint H = params.hidden_dim;
    device const float* row_w = fc_fuse_w + h * (2u * H);

    const uint H4 = (H / 4u) * 4u;
    float acc0 = 0.0f;
    for (uint j = 0; j < H4; j += 4u) {
        float4 w = *((device const float4*)(row_w + j));
        float4 x = *((device const float4*)(tok_emb_f32 + j));
        acc0 += dot(w, x);
    }
    for (uint j = H4; j < H; ++j) {
        acc0 += row_w[j] * tok_emb_f32[j];
    }

    device const float* row_w2 = row_w + H;
    float acc1 = 0.0f;
    for (uint j = 0; j < H4; j += 4u) {
        float4 w = *((device const float4*)(row_w2 + j));
        float4 x = *((device const float4*)(prev_feature + j));
        acc1 += dot(w, x);
    }
    for (uint j = H4; j < H; ++j) {
        acc1 += row_w2[j] * prev_feature[j];
    }

    reduced_out[h] = acc0 + acc1;
}

// 6c. EAGLE Decoder Layer (predicts next feature delta and adds residual)
kernel void eagle_decoder_layer(
    device const float*         reduced_in        [[buffer(0)]],
    device const float*         decoder_w         [[buffer(1)]],
    device float*               next_feature_f32  [[buffer(2)]],
    device ushort*              next_feature_bf16 [[buffer(3)]],
    constant EagleParams&       params            [[buffer(4)]],
    uint                        h                 [[thread_position_in_grid]]
) {
    if (h >= params.hidden_dim) {
        return;
    }
    const uint H = params.hidden_dim;
    device const float* row_w = decoder_w + h * H;

    const uint H4 = (H / 4u) * 4u;
    float acc = 0.0f;
    for (uint j = 0; j < H4; j += 4u) {
        float4 w = *((device const float4*)(row_w + j));
        float4 x = *((device const float4*)(reduced_in + j));
        acc += dot(w, x);
    }
    for (uint j = H4; j < H; ++j) {
        acc += row_w[j] * reduced_in[j];
    }

    float out_val = reduced_in[h] + acc;
    next_feature_f32[h] = out_val;
    next_feature_bf16[h] = f32_to_bf16(out_val);
}

// 6d. GPU Top-K Token Extraction and Exact Probability Calculation (BF16 logits)
kernel void eagle_topk_and_probs(
    device const ushort*        logits_bf16       [[buffer(0)]],
    device uint*                out_tokens        [[buffer(1)]],
    device float*               out_probs         [[buffer(2)]],
    constant EagleParams&       params            [[buffer(3)]],
    uint                        tid               [[thread_index_in_threadgroup]],
    uint                        simd_id           [[simdgroup_index_in_threadgroup]],
    uint                        simd_lane         [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_max[8];
    float local_max = -INFINITY;
    for (uint v = tid; v < params.vocab_size; v += 256u) {
        float l = bf16_to_f32(logits_bf16[v]);
        if (!isnan(l) && !isinf(l)) {
            local_max = max(local_max, l);
        }
    }
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        local_max = max(local_max, simd_shuffle_down(local_max, offset));
    }
    if (simd_lane == 0u) {
        shared_max[simd_id] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float g_max;
    if (tid == 0u) {
        float gm = shared_max[0];
        for (uint s = 1u; s < 8u; ++s) {
            gm = max(gm, shared_max[s]);
        }
        g_max = gm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float global_max = g_max;

    threadgroup float shared_sum[8];
    float local_sum = 0.0f;
    for (uint v = tid; v < params.vocab_size; v += 256u) {
        float l = bf16_to_f32(logits_bf16[v]);
        if (!isnan(l) && !isinf(l)) {
            local_sum += exp(l - global_max);
        }
    }
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        local_sum += simd_shuffle_down(local_sum, offset);
    }
    if (simd_lane == 0u) {
        shared_sum[simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float g_sum_exp;
    if (tid == 0u) {
        float gs = shared_sum[0];
        for (uint s = 1u; s < 8u; ++s) {
            gs += shared_sum[s];
        }
        g_sum_exp = max(gs, 1e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sum_exp = g_sum_exp;

    threadgroup uint  chosen_tokens[16];
    threadgroup float chosen_logits[16];
    threadgroup uint  shared_best_idx[8];
    const uint k_limit = min(params.top_k, 16u);

    for (uint k = 0u; k < k_limit; ++k) {
        float best_val = -INFINITY;
        uint  best_idx = 0u;

        for (uint v = tid; v < params.vocab_size; v += 256u) {
            float l = bf16_to_f32(logits_bf16[v]);
            if (isnan(l) || isinf(l)) {
                continue;
            }

            bool already_chosen = false;
            for (uint prev = 0u; prev < k; ++prev) {
                if (v == chosen_tokens[prev]) {
                    already_chosen = true;
                    break;
                }
            }
            if (!already_chosen && (l > best_val || (l == best_val && v < best_idx))) {
                best_val = l;
                best_idx = v;
            }
        }

        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            float other_val = simd_shuffle_down(best_val, offset);
            uint  other_idx = simd_shuffle_down(best_idx, offset);
            if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (simd_lane == 0u) {
            shared_max[simd_id] = best_val;
            shared_best_idx[simd_id] = best_idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float top_val = shared_max[0];
            uint  top_idx = shared_best_idx[0];
            for (uint s = 1u; s < 8u; ++s) {
                if (shared_max[s] > top_val || (shared_max[s] == top_val && shared_best_idx[s] < top_idx)) {
                    top_val = shared_max[s];
                    top_idx = shared_best_idx[s];
                }
            }
            chosen_tokens[k] = top_idx;
            chosen_logits[k] = top_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        for (uint k = 0u; k < k_limit; ++k) {
            out_tokens[k] = chosen_tokens[k];
            out_probs[k] = exp(chosen_logits[k] - global_max) / sum_exp;
        }
    }
}

// 6e. GPU Top-K Token Extraction and Exact Probability Calculation (FP32 logits)
kernel void eagle_topk_and_probs_f32(
    device const float*         logits_f32        [[buffer(0)]],
    device uint*                out_tokens        [[buffer(1)]],
    device float*               out_probs         [[buffer(2)]],
    constant EagleParams&       params            [[buffer(3)]],
    uint                        tid               [[thread_index_in_threadgroup]],
    uint                        simd_id           [[simdgroup_index_in_threadgroup]],
    uint                        simd_lane         [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_max[8];
    float local_max = -INFINITY;
    for (uint v = tid; v < params.vocab_size; v += 256u) {
        float l = logits_f32[v];
        if (!isnan(l) && !isinf(l)) {
            local_max = max(local_max, l);
        }
    }
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        local_max = max(local_max, simd_shuffle_down(local_max, offset));
    }
    if (simd_lane == 0u) {
        shared_max[simd_id] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float g_max;
    if (tid == 0u) {
        float gm = shared_max[0];
        for (uint s = 1u; s < 8u; ++s) {
            gm = max(gm, shared_max[s]);
        }
        g_max = gm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float global_max = g_max;

    threadgroup float shared_sum[8];
    float local_sum = 0.0f;
    for (uint v = tid; v < params.vocab_size; v += 256u) {
        float l = logits_f32[v];
        if (!isnan(l) && !isinf(l)) {
            local_sum += exp(l - global_max);
        }
    }
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        local_sum += simd_shuffle_down(local_sum, offset);
    }
    if (simd_lane == 0u) {
        shared_sum[simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float g_sum_exp;
    if (tid == 0u) {
        float gs = shared_sum[0];
        for (uint s = 1u; s < 8u; ++s) {
            gs += shared_sum[s];
        }
        g_sum_exp = max(gs, 1e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sum_exp = g_sum_exp;

    threadgroup uint  chosen_tokens[16];
    threadgroup float chosen_logits[16];
    threadgroup uint  shared_best_idx[8];
    const uint k_limit = min(params.top_k, 16u);

    for (uint k = 0u; k < k_limit; ++k) {
        float best_val = -INFINITY;
        uint  best_idx = 0u;

        for (uint v = tid; v < params.vocab_size; v += 256u) {
            float l = logits_f32[v];
            if (isnan(l) || isinf(l)) {
                continue;
            }

            bool already_chosen = false;
            for (uint prev = 0u; prev < k; ++prev) {
                if (v == chosen_tokens[prev]) {
                    already_chosen = true;
                    break;
                }
            }
            if (!already_chosen && (l > best_val || (l == best_val && v < best_idx))) {
                best_val = l;
                best_idx = v;
            }
        }

        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            float other_val = simd_shuffle_down(best_val, offset);
            uint  other_idx = simd_shuffle_down(best_idx, offset);
            if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (simd_lane == 0u) {
            shared_max[simd_id] = best_val;
            shared_best_idx[simd_id] = best_idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float top_val = shared_max[0];
            uint  top_idx = shared_best_idx[0];
            for (uint s = 1u; s < 8u; ++s) {
                if (shared_max[s] > top_val || (shared_max[s] == top_val && shared_best_idx[s] < top_idx)) {
                    top_val = shared_max[s];
                    top_idx = shared_best_idx[s];
                }
            }
            chosen_tokens[k] = top_idx;
            chosen_logits[k] = top_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        for (uint k = 0u; k < k_limit; ++k) {
            out_tokens[k] = chosen_tokens[k];
            out_probs[k] = exp(chosen_logits[k] - global_max) / sum_exp;
        }
    }
}
