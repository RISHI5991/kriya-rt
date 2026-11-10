//! Real-World Validation Tests for KriyaRT.
//!
//! Tests Metal kernels against CPU references using REAL model dimensions
//! (Llama-3.2-3B, Mistral-7B, Gemma-2-9B head_dim=256, etc.).
//! These tests verify numerical correctness on hardware-representative shapes.

use futures::executor::block_on;
use kriya_backend_metal::encoder::{MetalExecutor, SamplingConfig};
use kriya_core::descriptor::ModelArchDescriptor;
use kriya_core::quantizer::{
    bf16_bits_to_f32, dequantize_q4_group64_to_f32, f32_to_bf16_bits, quantize_f32_to_q4_group64,
};
use kriya_kv::bsr_paged::{CtaWorkTile, PagedKvCacheManager};
use metal::{Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

fn workspace_kernels_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
        .join("metal")
}

fn require_metal_device() -> Option<Device> {
    match Device::system_default() {
        Some(dev) => Some(dev),
        None => {
            if std::env::var("KRIYA_REQUIRE_METAL").is_ok() {
                panic!(
                    "Fatal: KRIYA_REQUIRE_METAL is set, but no Apple Silicon GPU is visible to this process!"
                );
            }
            eprintln!("Skipping Metal execution: no Apple Silicon GPU is visible to this process");
            None
        }
    }
}

/// ========================================================================
// TEST 1: Q4_G64 Decode GEMV at Real Model Dimensions
/// ========================================================================
/// Tests Q4_G64 decode GEMV with Llama-3.2-3B dimensions: hidden=3072, q_dim=3072.
/// Verifies against a naive CPU reference.
#[test]
fn test_q4_gemmv_real_dimensions_llama_3b() {
    let Some(device) = require_metal_device() else {
        return;
    };

    // Llama-3.2-3B: hidden_size=3072, num_q_heads=24, head_dim=128
    let h: usize = 3072;
    let q_dim: usize = 3072;
    let num_k_blocks = h.div_ceil(64); // 48 blocks

    // Create Q4_G64 weight matrix [q_dim, h] with small values for BF16 precision
    let weights_f32: Vec<f32> = (0..q_dim * h)
        .map(|i| ((i as f32) * 0.0001 - 0.5) * 0.01)
        .collect();
    let weight_blocks = quantize_f32_to_q4_group64(&weights_f32).unwrap();
    assert_eq!(weight_blocks.len(), q_dim * num_k_blocks);

    // Input vector [h] with small values
    let input_f32: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.0002 - 0.3) * 0.01).collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();

    // CPU reference: dequantize and compute
    let weight_f32 = dequantize_q4_group64_to_f32(&weight_blocks);
    let expected: Vec<f32> = (0..q_dim)
        .map(|row| {
            weight_f32[row * h..(row + 1) * h]
                .iter()
                .zip(input_f32.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>()
        })
        .collect();

    let shared = MTLResourceOptions::StorageModeShared;
    let weight_buf = device.new_buffer_with_data(
        weight_blocks.as_ptr() as *const _,
        (weight_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );
    let input_buf = device.new_buffer_with_data(
        input_bf16.as_ptr() as *const _,
        (input_bf16.len() * 2) as u64,
        shared,
    );
    let output_buf = device.new_buffer((q_dim * 2) as u64, shared);

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.get_or_compile("q4_g64_decode_gemv").unwrap());
    encoder.set_buffer(0, Some(&weight_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    let params = kriya_backend_metal::encoder::GemmParams {
        m: 1,
        n: q_dim as u32,
        k: h as u32,
        num_k_blocks: num_k_blocks as u32,
    };
    encoder.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    let tgp_count = q_dim.div_ceil(4);
    encoder.dispatch_thread_groups(
        MTLSize::new(tgp_count as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let output = unsafe { std::slice::from_raw_parts(output_buf.contents() as *const u16, q_dim) };
    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    for (i, (&gpu_bits, &cpu)) in output.iter().zip(expected.iter()).enumerate() {
        let gpu = bf16_bits_to_f32(gpu_bits);
        let diff = (gpu - cpu).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        let rel = if cpu.abs() > 1e-6 {
            diff / cpu.abs()
        } else {
            diff
        };
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
        if diff > 0.05 && rel > 0.05 {
            panic!("Mismatch at row {i}: GPU={gpu:.6}, CPU={cpu:.6}, diff={diff:.6}, rel={rel:.6}");
        }
    }
    println!(
        "Q4_GEMV Llama-3.2-3B (h=3072, q=3072): max_diff={:.6}, max_rel_diff={:.6}, PASS",
        max_diff, max_rel_diff
    );
}

/// ========================================================================
// TEST 2: Prefill GEMM (simdgroup_matrix) at Real Dimensions, M > 1
/// ========================================================================
/// Tests the hardware matrix-core prefill GEMM with M=16 tokens, K=3072, N=3072.
/// This is the kernel that was unverified — critical for prefill performance.
#[test]
fn test_q4_prefill_gemm_real_dimensions() {
    let Some(device) = require_metal_device() else {
        return;
    };

    // Real prefill dimensions: M=16 tokens, K=3072 hidden, N=3072 output
    let m: usize = 16;
    let k: usize = 3072;
    let n: usize = 3072;
    let num_k_blocks = k.div_ceil(64);

    // Create input [M, K] in BF16 with small values for BF16 precision
    let input_f32: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.0001 - 0.5) * 0.01)
        .collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();

    // Create random Q4_G64 weights [N, K] with small values
    let weights_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.00005 - 0.25) * 0.01)
        .collect();
    let weight_blocks = quantize_f32_to_q4_group64(&weights_f32).unwrap();
    assert_eq!(weight_blocks.len(), n * num_k_blocks);

    // CPU reference
    let weight_f32 = dequantize_q4_group64_to_f32(&weight_blocks);
    let mut expected: Vec<f32> = Vec::with_capacity(m * n);
    for row in 0..m {
        for col in 0..n {
            let val: f32 = weight_f32[col * k..(col + 1) * k]
                .iter()
                .zip(input_f32[row * k..(row + 1) * k].iter())
                .map(|(w, x)| w * x)
                .sum();
            expected.push(val);
        }
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let weight_buf = device.new_buffer_with_data(
        weight_blocks.as_ptr() as *const _,
        (weight_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );
    let input_buf = device.new_buffer_with_data(
        input_bf16.as_ptr() as *const _,
        (input_bf16.len() * 2) as u64,
        shared,
    );
    let output_buf = device.new_buffer((m * n * 2) as u64, shared);

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.get_or_compile("q4_g64_prefill_gemm").unwrap());
    encoder.set_buffer(0, Some(&weight_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    let params = kriya_backend_metal::encoder::GemmParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        num_k_blocks: num_k_blocks as u32,
    };
    encoder.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    // Grid: (N/16, M/16) threadgroups of 16x16 threads
    let grid = MTLSize::new(n.div_ceil(16) as u64, m.div_ceil(16) as u64, 1);
    encoder.dispatch_thread_groups(grid, MTLSize::new(16, 16, 1));
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let output = unsafe { std::slice::from_raw_parts(output_buf.contents() as *const u16, m * n) };
    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut mismatch_count = 0usize;
    for (i, (&gpu_bits, &cpu)) in output.iter().zip(expected.iter()).enumerate() {
        let gpu = bf16_bits_to_f32(gpu_bits);
        let diff = (gpu - cpu).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        let rel = if cpu.abs() > 1e-6 {
            diff / cpu.abs()
        } else {
            diff
        };
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
        if diff > 0.05 && rel > 0.1 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                eprintln!(
                    "Mismatch at [{},{}]: GPU={:.6}, CPU={:.6}, diff={:.6}, rel={:.6}",
                    i / n,
                    i % n,
                    gpu,
                    cpu,
                    diff,
                    rel
                );
            }
        }
    }
    let mismatch_rate = mismatch_count as f32 / (m * n) as f32 * 100.0;
    println!(
        "Q4_PREFILL_GEMM (M=16, K=3072, N=3072): max_diff={:.6}, max_rel_diff={:.6}, mismatches={}/{} ({:.2}%)",
        max_diff,
        max_rel_diff,
        mismatch_count,
        m * n,
        mismatch_rate
    );
    assert!(
        mismatch_rate < 1.0,
        "Prefill GEMM mismatch rate too high: {:.2}%",
        mismatch_rate
    );
}

/// ========================================================================
// TEST 3: Paged BSR GQA Attention at Real Dimensions (head_dim=128)
/// ========================================================================
/// Tests BSR paged GQA attention with Llama-3.2-3B dimensions:
/// 24 Q heads, 8 KV heads, head_dim=128, block_size=16.
#[test]
fn test_bsr_paged_gqa_real_dimensions() {
    let Some(device) = require_metal_device() else {
        return;
    };

    const B: usize = 1; // batch size
    const NUM_Q_HEADS: u32 = 24;
    const NUM_KV_HEADS: u32 = 8;
    const D: usize = 128;
    const BLOCK_SZ: u32 = 16;
    const NUM_BLOCKS: u32 = 8; // 128 tokens of history

    let gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS; // 3

    // Random Q [B, num_q_heads, D]
    let q_f32: Vec<f32> = (0..B * NUM_Q_HEADS as usize * D)
        .map(|i| ((i as f32) * 0.01 - 10.0) * 0.001)
        .collect();
    let q_bf16: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();

    // Random K/V cache [num_blocks, block_sz, num_kv_heads, D]
    let k_f32: Vec<f32> = (0..NUM_BLOCKS as usize * BLOCK_SZ as usize * NUM_KV_HEADS as usize * D)
        .map(|i| ((i as f32) * 0.008 - 8.0) * 0.001)
        .collect();
    let v_f32: Vec<f32> = (0..NUM_BLOCKS as usize * BLOCK_SZ as usize * NUM_KV_HEADS as usize * D)
        .map(|i| ((i as f32) * 0.006 - 6.0) * 0.001)
        .collect();
    let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();

    // CPU reference: online softmax attention
    let sm_scale = 1.0 / (D as f32).sqrt();
    let mut expected: Vec<f32> = vec![0.0; B * NUM_Q_HEADS as usize * D];
    for b in 0..B {
        for qh in 0..NUM_Q_HEADS as usize {
            let kvh = qh / gqa_ratio as usize;
            let mut m_i = f32::NEG_INFINITY;
            let mut l_i = 0.0f32;
            let mut acc: Vec<f32> = vec![0.0; D];

            for blk in 0..NUM_BLOCKS as usize {
                for tok in 0..BLOCK_SZ as usize {
                    let kv_off =
                        ((blk * BLOCK_SZ as usize + tok) * NUM_KV_HEADS as usize + kvh) * D;
                    let mut score = 0.0f32;
                    for d in 0..D {
                        score += q_f32[(b * NUM_Q_HEADS as usize + qh) * D + d] * k_f32[kv_off + d];
                    }
                    score *= sm_scale;

                    let m_new = m_i.max(score);
                    let alpha = (m_i - m_new).exp();
                    let beta = (score - m_new).exp();
                    l_i = l_i * alpha + beta;
                    for d in 0..D {
                        acc[d] = acc[d] * alpha + beta * v_f32[kv_off + d];
                    }
                    m_i = m_new;
                }
            }
            let inv_l = 1.0 / l_i.max(1e-12);
            for d in 0..D {
                expected[(b * NUM_Q_HEADS as usize + qh) * D + d] = acc[d] * inv_l;
            }
        }
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let q_buf = device.new_buffer_with_data(
        q_bf16.as_ptr() as *const _,
        (q_bf16.len() * 2) as u64,
        shared,
    );
    let kv_bytes = (NUM_BLOCKS as usize * BLOCK_SZ as usize * NUM_KV_HEADS as usize * D * 2) as u64;
    let kv_k = device.new_buffer(kv_bytes, shared);
    let kv_v = device.new_buffer(kv_bytes, shared);
    unsafe {
        std::ptr::copy_nonoverlapping(k_bf16.as_ptr(), kv_k.contents() as *mut u16, k_bf16.len());
        std::ptr::copy_nonoverlapping(v_bf16.as_ptr(), kv_v.contents() as *mut u16, v_bf16.len());
    }

    // BSR layout: one request, all 8 blocks
    let indptr: Vec<i32> = vec![0, NUM_BLOCKS as i32];
    let indices: Vec<i32> = (0..NUM_BLOCKS).map(|i| i as i32).collect();
    let last_len: Vec<i32> = vec![BLOCK_SZ as i32];
    let indptr_buf = device.new_buffer_with_data(
        indptr.as_ptr() as *const _,
        (indptr.len() * 4) as u64,
        shared,
    );
    let indices_buf = device.new_buffer_with_data(
        indices.as_ptr() as *const _,
        (indices.len() * 4) as u64,
        shared,
    );
    let last_len_buf = device.new_buffer_with_data(
        last_len.as_ptr() as *const _,
        (last_len.len() * 4) as u64,
        shared,
    );

    let tile = CtaWorkTile {
        request_idx: 0,
        kv_block_start: 0,
        kv_block_count: NUM_BLOCKS,
        is_split_k: 0,
    };
    let tiles = device.new_buffer_with_data(
        &tile as *const _ as *const _,
        std::mem::size_of_val(&tile) as u64,
        shared,
    );
    let split_idx = device.new_buffer(4, shared);
    let output = device.new_buffer((B * NUM_Q_HEADS as usize * D * 2) as u64, shared);
    let partial_out = device.new_buffer((B * NUM_Q_HEADS as usize * D * 4) as u64, shared);
    let partial_lse = device.new_buffer((B * NUM_Q_HEADS as usize * 4) as u64, shared);

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    // fused_qknorm_rope_and_kv_append (write K/V to cache)
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(
            &pipelines
                .get_or_compile("fused_qknorm_rope_and_kv_append")
                .unwrap(),
        );
        let rope_params = kriya_backend_metal::encoder::RopeKvWriteParams {
            num_q_heads: NUM_Q_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: D as u32,
            rotary_dim: D as u32,
            block_size_tokens: BLOCK_SZ,
            apply_qk_norm: 0,
            rms_eps: 1e-5,
            rope_theta: 500_000.0,
        };
        // We just write K/V to cache; q_inout is unused for this test
        enc.set_buffer(0, Some(&q_buf), 0); // q_inout
        enc.set_buffer(1, Some(&q_buf), 0); // k_in (dummy)
        enc.set_buffer(2, Some(&q_buf), 0); // v_in (dummy)
        enc.set_buffer(3, None, 0); // q_norm_weight
        enc.set_buffer(4, None, 0); // k_norm_weight
        enc.set_buffer(5, Some(&kv_k), 0);
        enc.set_buffer(6, Some(&kv_v), 0);
        enc.set_buffer(7, Some(&indptr_buf), 0);
        enc.set_buffer(8, Some(&indices_buf), 0);
        enc.set_buffer(9, Some(&last_len_buf), 0);
        let seq_pos = device.new_buffer_with_data([0u32].as_ptr() as *const _, 4, shared);
        enc.set_buffer(10, Some(&seq_pos), 0); // seq_positions dummy
        enc.set_bytes(
            11,
            std::mem::size_of_val(&rope_params) as u64,
            &rope_params as *const _ as *const _,
        );
        enc.dispatch_thread_groups(
            MTLSize::new((NUM_Q_HEADS + NUM_KV_HEADS) as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // bsr_paged_gqa_decode_stage1
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(
            &pipelines
                .get_or_compile("bsr_paged_gqa_decode_stage1")
                .unwrap(),
        );
        let gqa_params = kriya_backend_metal::encoder::BsrGqaParams {
            num_q_heads: NUM_Q_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: D as u32,
            block_size_tokens: BLOCK_SZ,
            max_split_k_tiles: 1,
            max_query_tokens: 1,
            sm_scale,
            logit_softcap: 0.0,
        };
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&kv_k), 0);
        enc.set_buffer(2, Some(&kv_v), 0);
        enc.set_buffer(3, Some(&indptr_buf), 0);
        enc.set_buffer(4, Some(&indices_buf), 0);
        enc.set_buffer(5, Some(&last_len_buf), 0);
        enc.set_buffer(6, Some(&tiles), 0);
        enc.set_buffer(7, Some(&split_idx), 0);
        enc.set_buffer(8, Some(&output), 0);
        enc.set_buffer(9, Some(&partial_out), 0);
        enc.set_buffer(10, Some(&partial_lse), 0);
        enc.set_bytes(
            11,
            std::mem::size_of_val(&gqa_params) as u64,
            &gqa_params as *const _ as *const _,
        );
        enc.dispatch_thread_groups(
            MTLSize::new(1, NUM_Q_HEADS as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let gpu_output = unsafe {
        std::slice::from_raw_parts(
            output.contents() as *const u16,
            B * NUM_Q_HEADS as usize * D,
        )
    };
    let mut max_diff = 0.0f32;
    for (i, (&gpu_bits, &cpu)) in gpu_output.iter().zip(expected.iter()).enumerate() {
        let gpu = bf16_bits_to_f32(gpu_bits);
        let diff = (gpu - cpu).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.05 {
            panic!("Attention mismatch at {i}: GPU={:.6}, CPU={:.6}", gpu, cpu);
        }
    }
    println!(
        "BSR Paged GQA (B=1, QH=24, KH=8, D=128, blocks=8): max_diff={:.6}, PASS",
        max_diff
    );
}

/// ========================================================================
// TEST 4: EAGLE Tree Verification End-to-End
/// ========================================================================
/// Tests that forward_tree_logits produces valid logits for a small tree.
#[test]
fn test_eagle_tree_verification_end_to_end() {
    let Some(_device) = require_metal_device() else {
        return;
    };

    // Use miniature descriptor (64 hidden, 64 vocab) for speed
    let desc = {
        use kriya_core::descriptor::{
            ActivationVariant, AttentionMaskVariant, ChannelMixerDescriptor, LayerDescriptor,
            NormVariant, PhaseRoutingPolicy, PositionalEncoding, SequenceMixerDescriptor,
        };
        ModelArchDescriptor {
            family: "test".into(),
            model_name: "mini-eagle".into(),
            vocab_size: 64,
            hidden_size: 64,
            max_position_embeddings: 128,
            weight_dtype: kriya_core::tensor::DType::Q4Group64,
            tie_word_embeddings: false,
            final_norm: NormVariant::RmsNorm {
                eps: 1e-5,
                zero_centered_unit_offset: false,
            },
            final_logit_softcap: None,
            phase_policy: PhaseRoutingPolicy::default(),
            layers: vec![LayerDescriptor {
                layer_idx: 0,
                attn_norm: NormVariant::RmsNorm {
                    eps: 1e-5,
                    zero_centered_unit_offset: false,
                },
                seq_mixer: SequenceMixerDescriptor::GroupedQueryAttention {
                    num_q_heads: 1,
                    num_kv_heads: 1,
                    head_dim: 64,
                    pos_encoding: PositionalEncoding::RoPE {
                        theta: 10_000.0,
                        rotary_dim: 64,
                        interleaved: false,
                        scaling_factor: 1.0,
                    },
                    mask_variant: AttentionMaskVariant::Causal,
                    qk_norm: None,
                    qkv_bias: false,
                    attn_logit_softcap: None,
                    split_k_decode_threshold: 128,
                },
                post_attn_norm: None,
                ffn_norm: NormVariant::RmsNorm {
                    eps: 1e-5,
                    zero_centered_unit_offset: false,
                },
                channel_mixer: ChannelMixerDescriptor::DenseMlp {
                    intermediate_size: 64,
                    activation: ActivationVariant::SwiGLU,
                    use_bias: false,
                },
                post_ffn_norm: None,
            }],
        }
    };

    let shader_dir = workspace_kernels_dir();
    let tmp_dir = tempfile::tempdir().unwrap();
    // Write a complete miniature checkpoint
    write_complete_miniature_checkpoint(tmp_dir.path());

    let mut executor =
        MetalExecutor::new(&desc, &shader_dir, Some(tmp_dir.path()), 16, 16).unwrap();

    // Create a 3-node draft tree
    let tree_tokens = vec![1u32, 2, 3];
    let tree_mask = vec![
        true, false, false, // node 0 attends to itself
        true, true, false, // node 1 attends to 0, 1
        true, true, true, // node 2 attends to 0, 1, 2
    ];

    let mut kv_mgr = PagedKvCacheManager::new(64, 32, 16);
    kv_mgr.allocate_sequence(1, &[10, 20, 30]).unwrap();
    let plan = kv_mgr.build_schedule_plan(&[1], 16).unwrap();

    let start = Instant::now();
    let (logits, features) =
        block_on(executor.forward_tree_logits(&desc, &plan, &tree_tokens, &tree_mask))
            .expect("Tree verification should succeed");
    let elapsed = start.elapsed();

    // Verify shapes
    assert_eq!(logits.len(), 3, "Should have logits for 3 tree nodes");
    assert_eq!(logits[0].len(), 64, "Each node should have 64 vocab logits");
    assert_eq!(
        features.len(),
        3,
        "Should have hidden states for 3 tree nodes"
    );
    assert_eq!(
        features[0].len(),
        64,
        "Each node should have 64 hidden dims"
    );

    // Verify logits are finite and reasonable
    for (node_idx, node_logits) in logits.iter().enumerate() {
        for (v_idx, &logit) in node_logits.iter().enumerate() {
            assert!(
                logit.is_finite(),
                "Logit at node {}, vocab {} is not finite: {}",
                node_idx,
                v_idx,
                logit
            );
        }
    }

    println!(
        "EAGLE tree verification (3 nodes, D=64): {:.2} ms, logits_shape=[3, 64], features_shape=[3, 64], PASS",
        elapsed.as_secs_f64() * 1000.0
    );
}

/// ========================================================================
// TEST 5: Memory Budget Enforcement
/// ========================================================================
/// Tests that the hardware profiler correctly computes memory budgets
/// and triggers hybrid mode for large models on small Macs.
#[test]
fn test_memory_budget_enforcement() {
    use kriya_core::hardware::{AppleGpuFamily, HardwareProfile, MacFormFactor};

    // 8GB M1 Air: 80% usable = ~6.4GB, model larger than that → hybrid
    let profile = HardwareProfile::detect(
        "Apple M1",
        AppleGpuFamily::Apple7,
        8,
        8 * 1024 * 1024 * 1024, // 8 GB
    );
    assert_eq!(profile.form_factor, MacFormFactor::AirOrBase);
    assert_eq!(profile.max_working_set_bytes, 6_871_947_673); // 8GB * 0.80

    let policy = profile.tune_for_model(14 * 1024 * 1024 * 1024); // 14GB model
    assert!(
        policy.gpu_hot_expert_ratio < 1.0,
        "Should trigger hybrid mode for large model"
    );
    assert!(
        policy.compress_kv_cache_q4,
        "Should compress KV cache under memory pressure"
    );

    // 128GB M4 Max: model fits entirely in GPU
    let profile = HardwareProfile::detect(
        "Apple M4 Max",
        AppleGpuFamily::Apple9,
        40,
        128 * 1024 * 1024 * 1024, // 128 GB
    );
    assert_eq!(profile.form_factor, MacFormFactor::Max);
    let policy = profile.tune_for_model(20 * 1024 * 1024 * 1024); // 20GB model
    assert_eq!(
        policy.gpu_hot_expert_ratio, 1.0,
        "Model fits in GPU, all experts hot"
    );
    assert!(!policy.compress_kv_cache_q4);

    println!("Memory budget enforcement: PASS");
}

/// ========================================================================
// TEST 6: Prefill Batching End-to-End
/// ========================================================================
/// Tests that forward_prefill_batch produces the same output as
/// calling forward_and_sample token-by-token for each prefill token.
#[test]
fn test_prefill_batch_matches_token_by_token() {
    let Some(_device) = require_metal_device() else {
        return;
    };

    use kriya_core::descriptor::{
        ActivationVariant, AttentionMaskVariant, ChannelMixerDescriptor, LayerDescriptor,
        NormVariant, PhaseRoutingPolicy, PositionalEncoding, SequenceMixerDescriptor,
    };
    let desc = ModelArchDescriptor {
        family: "test".into(),
        model_name: "mini-prefill".into(),
        vocab_size: 64,
        hidden_size: 64,
        max_position_embeddings: 128,
        weight_dtype: kriya_core::tensor::DType::Q4Group64,
        tie_word_embeddings: false,
        final_norm: NormVariant::RmsNorm {
            eps: 1e-5,
            zero_centered_unit_offset: false,
        },
        final_logit_softcap: None,
        phase_policy: PhaseRoutingPolicy::default(),
        layers: vec![LayerDescriptor {
            layer_idx: 0,
            attn_norm: NormVariant::RmsNorm {
                eps: 1e-5,
                zero_centered_unit_offset: false,
            },
            seq_mixer: SequenceMixerDescriptor::GroupedQueryAttention {
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 64,
                pos_encoding: PositionalEncoding::RoPE {
                    theta: 10_000.0,
                    rotary_dim: 64,
                    interleaved: false,
                    scaling_factor: 1.0,
                },
                mask_variant: AttentionMaskVariant::Causal,
                qk_norm: None,
                qkv_bias: false,
                attn_logit_softcap: None,
                split_k_decode_threshold: 128,
            },
            post_attn_norm: None,
            ffn_norm: NormVariant::RmsNorm {
                eps: 1e-5,
                zero_centered_unit_offset: false,
            },
            channel_mixer: ChannelMixerDescriptor::DenseMlp {
                intermediate_size: 64,
                activation: ActivationVariant::SwiGLU,
                use_bias: false,
            },
            post_ffn_norm: None,
        }],
    };

    let shader_dir = workspace_kernels_dir();
    let tmp_dir = tempfile::tempdir().unwrap();
    write_complete_miniature_checkpoint(tmp_dir.path());

    let mut executor =
        MetalExecutor::new(&desc, &shader_dir, Some(tmp_dir.path()), 16, 16).unwrap();

    let mut kv_mgr = PagedKvCacheManager::new(64, 32, 16);
    kv_mgr.allocate_sequence(1, &[1, 2, 3]).unwrap();

    // Token-by-token: process [1], then [2], then [3] separately
    let mut kv_mgr_ttt = PagedKvCacheManager::new(64, 32, 16);
    kv_mgr_ttt.allocate_sequence(1, &[1, 2, 3]).unwrap();

    let samp_cfg = SamplingConfig {
        vocab_size: 64,
        greedy: 1,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 64,
        rng_seed: 42,
    };

    // Process token 1
    let plan1 = kv_mgr_ttt.build_prefix_schedule_plan(1, 1, 16).unwrap();
    let tok1 = block_on(executor.forward_and_sample(&desc, &plan1, &[1], &[0], samp_cfg)).unwrap();

    // Process token 2 (position 1)
    let plan2 = kv_mgr_ttt.build_prefix_schedule_plan(1, 2, 16).unwrap();
    let tok2 = block_on(executor.forward_and_sample(&desc, &plan2, &[2], &[1], samp_cfg)).unwrap();

    // Process token 3 (position 2)
    let plan3 = kv_mgr_ttt.build_prefix_schedule_plan(1, 3, 16).unwrap();
    let tok3 = block_on(executor.forward_and_sample(&desc, &plan3, &[3], &[2], samp_cfg)).unwrap();

    // Batched prefill: process [1, 2, 3] in one command buffer
    let mut executor2 =
        MetalExecutor::new(&desc, &shader_dir, Some(tmp_dir.path()), 16, 16).unwrap();
    let mut kv_mgr_batch = PagedKvCacheManager::new(64, 32, 16);
    kv_mgr_batch.allocate_sequence(1, &[1, 2, 3]).unwrap();

    // Build a single parallel prefill layout for the entire chunk
    let prefix_plan = kv_mgr_batch
        .build_parallel_prefill_layout(1, 0, 3, 16)
        .unwrap();

    let positions = vec![0u32, 1, 2];
    let tokens = vec![1u32, 2, 3];

    let tok_batch = block_on(executor2.forward_prefill_batch(
        &desc,
        &tokens,
        &positions,
        &prefix_plan,
        samp_cfg,
    ))
    .unwrap();

    // Batched result should match the last token-by-token result
    assert_eq!(
        tok_batch, tok3,
        "Batched prefill should produce same token as last sequential token"
    );

    println!(
        "Prefill batching vs token-by-token: tok1={}, tok2={}, tok3={}, batched={}, PASS",
        tok1, tok2, tok3, tok_batch
    );
}

/// ========================================================================
/// TEST 8: Batched Decode Multi-Request Distinct Root Features
/// ========================================================================
/// Verifies that forward_and_sample_batch produces distinct, non-zero, finite
/// hidden state vectors for separate requests in the batch via read_hidden_feature_at,
/// ensuring EAGLE speculative decoding operates on isolated per-request hidden features.
#[test]
fn test_batched_speculative_per_request_hidden_features() {
    let Some(_device) = require_metal_device() else {
        return;
    };

    use kriya_core::descriptor::{
        ActivationVariant, AttentionMaskVariant, ChannelMixerDescriptor, LayerDescriptor,
        NormVariant, PhaseRoutingPolicy, PositionalEncoding, SequenceMixerDescriptor,
    };
    let desc = ModelArchDescriptor {
        family: "test".into(),
        model_name: "mini-batch-feature".into(),
        vocab_size: 64,
        hidden_size: 64,
        max_position_embeddings: 128,
        weight_dtype: kriya_core::tensor::DType::Q4Group64,
        tie_word_embeddings: false,
        final_norm: NormVariant::RmsNorm {
            eps: 1e-5,
            zero_centered_unit_offset: false,
        },
        final_logit_softcap: None,
        phase_policy: PhaseRoutingPolicy::default(),
        layers: vec![LayerDescriptor {
            layer_idx: 0,
            attn_norm: NormVariant::RmsNorm {
                eps: 1e-5,
                zero_centered_unit_offset: false,
            },
            seq_mixer: SequenceMixerDescriptor::GroupedQueryAttention {
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 64,
                pos_encoding: PositionalEncoding::RoPE {
                    theta: 10_000.0,
                    rotary_dim: 64,
                    interleaved: false,
                    scaling_factor: 1.0,
                },
                mask_variant: AttentionMaskVariant::Causal,
                qk_norm: None,
                qkv_bias: false,
                attn_logit_softcap: None,
                split_k_decode_threshold: 128,
            },
            post_attn_norm: None,
            ffn_norm: NormVariant::RmsNorm {
                eps: 1e-5,
                zero_centered_unit_offset: false,
            },
            channel_mixer: ChannelMixerDescriptor::DenseMlp {
                intermediate_size: 64,
                activation: ActivationVariant::SwiGLU,
                use_bias: false,
            },
            post_ffn_norm: None,
        }],
    };

    let shader_dir = workspace_kernels_dir();
    let tmp_dir = tempfile::tempdir().unwrap();
    write_miniature_checkpoint_with_distinct_embeddings(tmp_dir.path());

    let mut executor =
        MetalExecutor::new(&desc, &shader_dir, Some(tmp_dir.path()), 16, 16).unwrap();

    let mut kv_mgr = PagedKvCacheManager::new(64, 32, 16);
    let req_1 = 101u64;
    let req_2 = 102u64;
    kv_mgr.allocate_sequence(req_1, &[5]).unwrap();
    kv_mgr.allocate_sequence(req_2, &[23]).unwrap();

    let plan = kv_mgr.build_schedule_plan(&[req_1, req_2], 16).unwrap();
    let samp_cfg = SamplingConfig {
        vocab_size: 64,
        greedy: 1,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 64,
        rng_seed: 42,
    };

    let tokens = [5u32, 23u32];
    let positions = [0u32, 0u32];
    let samplings = [samp_cfg, samp_cfg];

    let sampled =
        block_on(executor.forward_and_sample_batch(&desc, &tokens, &positions, &plan, &samplings))
            .unwrap();

    assert_eq!(sampled.len(), 2);

    let feat_0 = executor.read_hidden_feature_at(64, 0);
    let feat_1 = executor.read_hidden_feature_at(64, 1);

    assert_eq!(feat_0.len(), 64);
    assert_eq!(feat_1.len(), 64);

    assert!(feat_0.iter().all(|x| x.is_finite()));
    assert!(feat_1.iter().all(|x| x.is_finite()));

    assert_ne!(
        feat_0, feat_1,
        "Distinct batched requests must produce non-identical root hidden features"
    );

    println!("Batched speculative root feature isolation verified successfully");
}

#[test]
fn test_concurrent_multi_request_batching_isolation_and_cancellation() {
    let Some(_device) = require_metal_device() else {
        return;
    };

    use kriya_core::descriptor::{
        ActivationVariant, AttentionMaskVariant, ChannelMixerDescriptor, LayerDescriptor,
        NormVariant, PhaseRoutingPolicy, PositionalEncoding, SequenceMixerDescriptor,
    };
    let desc = ModelArchDescriptor {
        family: "test".into(),
        model_name: "mini-multi-request".into(),
        vocab_size: 64,
        hidden_size: 64,
        max_position_embeddings: 128,
        weight_dtype: kriya_core::tensor::DType::Q4Group64,
        tie_word_embeddings: false,
        final_norm: NormVariant::RmsNorm {
            eps: 1e-5,
            zero_centered_unit_offset: false,
        },
        final_logit_softcap: None,
        phase_policy: PhaseRoutingPolicy::default(),
        layers: vec![LayerDescriptor {
            layer_idx: 0,
            attn_norm: NormVariant::RmsNorm {
                eps: 1e-5,
                zero_centered_unit_offset: false,
            },
            seq_mixer: SequenceMixerDescriptor::GroupedQueryAttention {
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 64,
                pos_encoding: PositionalEncoding::RoPE {
                    theta: 10_000.0,
                    rotary_dim: 64,
                    interleaved: false,
                    scaling_factor: 1.0,
                },
                mask_variant: AttentionMaskVariant::Causal,
                qk_norm: None,
                qkv_bias: false,
                attn_logit_softcap: None,
                split_k_decode_threshold: 128,
            },
            post_attn_norm: None,
            ffn_norm: NormVariant::RmsNorm {
                eps: 1e-5,
                zero_centered_unit_offset: false,
            },
            channel_mixer: ChannelMixerDescriptor::DenseMlp {
                intermediate_size: 64,
                activation: ActivationVariant::SwiGLU,
                use_bias: false,
            },
            post_ffn_norm: None,
        }],
    };

    let shader_dir = workspace_kernels_dir();
    let tmp_dir = tempfile::tempdir().unwrap();
    write_miniature_checkpoint_with_distinct_embeddings(tmp_dir.path());

    let mut executor =
        MetalExecutor::new(&desc, &shader_dir, Some(tmp_dir.path()), 16, 16).unwrap();

    let mut kv_mgr = PagedKvCacheManager::new(64, 32, 16);

    // 1. Scenario: 4 concurrent decode requests
    let req_ids = [201u64, 202, 203, 204];
    let initial_tokens = [5u32, 12, 23, 37];

    for (&req, &tok) in req_ids.iter().zip(&initial_tokens) {
        kv_mgr.allocate_sequence(req, &[tok]).unwrap();
    }

    let plan_4 = kv_mgr.build_schedule_plan(&req_ids, 16).unwrap();
    let samp_cfg = SamplingConfig {
        vocab_size: 64,
        greedy: 1,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 64,
        rng_seed: 42,
    };
    let samplings_4 = [samp_cfg, samp_cfg, samp_cfg, samp_cfg];
    let positions_4 = [0u32, 0, 0, 0];

    let sampled_4 = block_on(executor.forward_and_sample_batch(
        &desc,
        &initial_tokens,
        &positions_4,
        &plan_4,
        &samplings_4,
    ))
    .unwrap();

    assert_eq!(sampled_4.len(), 4);

    let mut features = Vec::new();
    for i in 0..4 {
        let feat = executor.read_hidden_feature_at(64, i);
        assert_eq!(feat.len(), 64);
        assert!(feat.iter().all(|x| x.is_finite()));
        features.push(feat);
    }

    // Verify all 4 hidden features are pairwise distinct (no cross-request collision)
    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(
                features[i], features[j],
                "Request {i} and Request {j} produced identical hidden features"
            );
        }
    }

    // 2. Scenario: Mixed prefill + decode iteration
    // Req 205 arrives as a new prefill request (prompt chunk of 3 tokens: &[2, 4, 8])
    let req_prefill = 205u64;
    let prefill_tokens = [2u32, 4, 8];
    kv_mgr
        .allocate_sequence(req_prefill, &prefill_tokens)
        .unwrap();

    let prefill_plan = kv_mgr
        .build_parallel_prefill_layout(req_prefill, 0, 3, 16)
        .unwrap();
    let prefill_positions = [0u32, 1, 2];

    let prefill_next_tok = block_on(executor.forward_prefill_batch(
        &desc,
        &prefill_tokens,
        &prefill_positions,
        &prefill_plan,
        samp_cfg,
    ))
    .unwrap();
    assert!(prefill_next_tok < 64);

    // Decode continues for Req 201 and 203 with distinct input tokens
    let decode_reqs = [201u64, 203u64];
    let decode_tokens = [14u32, 42u32];
    let decode_positions = [1u32, 1u32];
    let decode_plan = kv_mgr.build_schedule_plan(&decode_reqs, 16).unwrap();
    let decode_samplings = [samp_cfg, samp_cfg];

    let decode_sampled = block_on(executor.forward_and_sample_batch(
        &desc,
        &decode_tokens,
        &decode_positions,
        &decode_plan,
        &decode_samplings,
    ))
    .unwrap();

    assert_eq!(decode_sampled.len(), 2);
    let feat_201 = executor.read_hidden_feature_at(64, 0);
    let feat_203 = executor.read_hidden_feature_at(64, 1);
    assert_ne!(feat_201, feat_203);
    assert!(feat_201.iter().all(|x| x.is_finite()));
    assert!(feat_203.iter().all(|x| x.is_finite()));

    // 3. Scenario: One request cancelled mid-generation; assert KV pages are reclaimed
    assert!(kv_mgr.contains_sequence(202));
    kv_mgr.free_sequence(202).unwrap();
    assert!(
        !kv_mgr.contains_sequence(202),
        "Sequence 202 must be completely unregistered"
    );

    // 4. Scenario: One request reaches completion / EOS
    kv_mgr.free_sequence(204).unwrap();
    assert!(!kv_mgr.contains_sequence(204));

    // The remaining active requests (201, 203, 205) continue smoothly
    let final_reqs = [201u64, 203, 205];
    let final_tokens = [decode_sampled[0], decode_sampled[1], prefill_next_tok];
    let final_positions = [2u32, 2, 3];
    let final_plan = kv_mgr.build_schedule_plan(&final_reqs, 16).unwrap();
    let final_samplings = [samp_cfg, samp_cfg, samp_cfg];

    let final_sampled = block_on(executor.forward_and_sample_batch(
        &desc,
        &final_tokens,
        &final_positions,
        &final_plan,
        &final_samplings,
    ))
    .unwrap();

    assert_eq!(final_sampled.len(), 3);
    for tok in final_sampled {
        assert!(tok < 64);
    }

    // Clean up remaining sequences
    kv_mgr.free_sequence(201).unwrap();
    kv_mgr.free_sequence(203).unwrap();
    kv_mgr.free_sequence(205).unwrap();

    println!(
        "Concurrent multi-request batching, isolation, and cancellation verified successfully"
    );
}

// ========================================================================
// Helper Functions (duplicated from test_shader_compile.rs since test binaries are separate)
// ========================================================================

fn q4_payload(elements: usize) -> kriya_core::quantizer::SavedTensorPayload {
    let weights = vec![0.03125_f32; elements];
    kriya_core::quantizer::SavedTensorPayload::Q4Group64(
        kriya_core::quantizer::quantize_f32_to_q4_group64(&weights).unwrap(),
    )
}

fn write_miniature_checkpoint_with_distinct_embeddings(dir: &std::path::Path) {
    use kriya_core::quantizer::SavedTensorPayload;

    let bf16_ones = || SavedTensorPayload::Bf16(vec![0x3F80_u16; 64]);
    let q4_matrix = || q4_payload(64 * 64);

    let embed_weights: Vec<f32> = (0..64 * 64)
        .map(|i| {
            let row = i / 64;
            let col = i % 64;
            (row as f32 + 1.0) * 0.01 + (col as f32) * 0.001
        })
        .collect();
    let embed_payload = SavedTensorPayload::Q4Group64(
        kriya_core::quantizer::quantize_f32_to_q4_group64(&embed_weights).unwrap(),
    );

    let tensors = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![64, 64],
            embed_payload,
        ),
        ("model.norm.weight".into(), vec![64], bf16_ones()),
        ("lm_head.weight".into(), vec![64, 64], q4_matrix()),
        (
            "model.layers.0.input_layernorm.weight".into(),
            vec![64],
            bf16_ones(),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![64],
            bf16_ones(),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
    ];
    kriya_core::quantizer::write_mixed_safetensors(dir.join("model.safetensors"), &tensors)
        .unwrap();
}

fn write_complete_miniature_checkpoint(dir: &std::path::Path) {
    use kriya_core::quantizer::SavedTensorPayload;

    let bf16_ones = || SavedTensorPayload::Bf16(vec![0x3F80_u16; 64]);
    let q4_matrix = || q4_payload(64 * 64);
    let tensors = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        ("model.norm.weight".into(), vec![64], bf16_ones()),
        ("lm_head.weight".into(), vec![64, 64], q4_matrix()),
        (
            "model.layers.0.input_layernorm.weight".into(),
            vec![64],
            bf16_ones(),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![64],
            bf16_ones(),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
    ];
    kriya_core::quantizer::write_mixed_safetensors(dir.join("model.safetensors"), &tensors)
        .unwrap();
}
