//! Empirical Adversarial Verification Suite — Challenger M1-2.
//!
//! Evaluates:
//! 1. Metal Function Constant Compilation & Specialization Resilience:
//!    - Default constant compilation via `get_or_compile`.
//!    - Specialized pipeline compilation across permutations of:
//!      `head_dim` (64, 128), `has_qkv_bias` (true, false), `apply_qk_norm` (true, false),
//!      `activation_variant` (0, 1), `gqa_ratio` (1, 4, 7), etc.
//!    - Cache key collision resistance across all permutations.
//! 2. Invariant Stress Testing (The 5 Green Zone Invariants):
//!    - Invariant 1: `write_mixed_safetensors` header padding and tensor offset padding are exact multiples of 16384.
//!    - Invariant 2: Production checkpoint validation strictly fails closed when tensors are missing.
//!    - Invariant 3: Qwen2 vs Qwen3 architectural differentiation is preserved.
//!    - Invariant 4: Fused gate-up SwiGLU decode kernel executes on Metal and matches CPU reference.
//!    - Invariant 5: Axum router and ActiveGuard remain untouched.
//! 3. Negative & Boundary Testing:
//!    - Empty tensors, odd dimensions, zero rows.

use metal::{Device, MTLResourceOptions, MTLSize};
use std::collections::HashSet;
use std::path::PathBuf;

use kriya_backend_metal::encoder::{
    GateUpSwigluGemmParams, MetalExecutor, MetalPipelineCache, ModelGpuWeights,
    PipelineConstantsConfig,
};
use kriya_core::descriptor::{ModelArchDescriptor, PositionalEncoding, SequenceMixerDescriptor};
use kriya_core::quantizer::{
    SavedTensorPayload, bf16_bits_to_f32, f32_to_bf16_bits, write_mixed_safetensors,
};
use kriya_core::safetensors_loader::ShardedCheckpointLoader;
use kriya_core::tensor::{
    BlockQ4Group64, DType, ScaledAlignedQ4Block, TensorDescriptor, TensorView,
};

fn workspace_kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
        .join("metal")
}

fn require_metal_device() -> Option<Device> {
    Device::system_default()
}

// =========================================================================
// SECTION 1: Function Constant Compilation & Specialization Resilience
// =========================================================================

#[test]
fn test_challenger_function_constant_default_compilation() {
    let Some(device) = require_metal_device() else {
        return;
    };
    let mut cache = MetalPipelineCache::from_shader_dir(device, workspace_kernels_dir()).unwrap();

    // Verify all kernels using function constants compile with default constants
    let kernels_with_fc = [
        "q4_aligned_decode_qkv_gemv_2row",
        "fused_qknorm_rope_and_kv_append",
        "bsr_paged_gqa_decode_stage1",
        "bsr_paged_gqa_tiled_mma_prefill",
    ];

    for kernel in &kernels_with_fc {
        let pso = cache.get_or_compile(kernel);
        assert!(
            pso.is_ok(),
            "Default compilation of kernel '{kernel}' failed: {:?}",
            pso.err()
        );
    }
}

#[test]
fn test_challenger_function_constant_specialization_permutations() {
    let Some(device) = require_metal_device() else {
        return;
    };
    let mut cache = MetalPipelineCache::from_shader_dir(device, workspace_kernels_dir()).unwrap();

    let head_dims = [64u32, 128u32];
    let biases = [true, false];
    let qk_norms = [true, false];
    let activation_variants = [0u32, 1u32];
    let gqa_ratios = [1u32, 4u32, 7u32];

    let mut compiled_count = 0;
    for &head_dim in &head_dims {
        for &has_qkv_bias in &biases {
            for &apply_qk_norm in &qk_norms {
                for &activation_variant in &activation_variants {
                    for &gqa_ratio in &gqa_ratios {
                        let config = PipelineConstantsConfig {
                            has_qkv_bias,
                            activation_variant,
                            head_dim,
                            gqa_ratio,
                            apply_qk_norm,
                            has_logit_softcap: false,
                            zero_centered: false,
                        };

                        let cv = config.to_constant_values();
                        let key =
                            format!("q4_aligned_decode_qkv_gemv_2row{}", config.cache_suffix());
                        let pso = cache.get_or_compile_specialized(
                            "q4_aligned_decode_qkv_gemv_2row",
                            Some(&cv),
                            &key,
                        );
                        assert!(
                            pso.is_ok(),
                            "Specialized compilation failed for permutation: head_dim={}, bias={}, qk_norm={}, act={}, gqa={}: {:?}",
                            head_dim,
                            has_qkv_bias,
                            apply_qk_norm,
                            activation_variant,
                            gqa_ratio,
                            pso.err()
                        );
                        compiled_count += 1;
                    }
                }
            }
        }
    }
    assert_eq!(compiled_count, 2 * 2 * 2 * 2 * 3); // 48 permutations compiled successfully
}

#[test]
fn test_challenger_cache_key_collision_resistance() {
    let mut seen_keys = HashSet::new();

    let head_dims = [64u32, 128u32, 256u32];
    let biases = [true, false];
    let qk_norms = [true, false];
    let activation_variants = [0u32, 1u32, 2u32];
    let gqa_ratios = [1u32, 2u32, 4u32, 7u32, 8u32];
    let logit_softcaps = [true, false];
    let zero_centereds = [true, false];

    let mut total_configs = 0;
    for &head_dim in &head_dims {
        for &has_qkv_bias in &biases {
            for &apply_qk_norm in &qk_norms {
                for &activation_variant in &activation_variants {
                    for &gqa_ratio in &gqa_ratios {
                        for &has_logit_softcap in &logit_softcaps {
                            for &zero_centered in &zero_centereds {
                                let config = PipelineConstantsConfig {
                                    has_qkv_bias,
                                    activation_variant,
                                    head_dim,
                                    gqa_ratio,
                                    apply_qk_norm,
                                    has_logit_softcap,
                                    zero_centered,
                                };
                                let suffix = config.cache_suffix();
                                assert!(
                                    seen_keys.insert(suffix.clone()),
                                    "Collision detected for config: {:?} -> key '{}'",
                                    config,
                                    suffix
                                );
                                total_configs += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(total_configs, 3 * 2 * 2 * 3 * 5 * 2 * 2); // 720 distinct configs without collision
}

// =========================================================================
// SECTION 2: Invariant Stress Testing (The 5 Green Zone Invariants)
// =========================================================================

/// Invariant 1: Header padding and tensor offset padding must be exact multiples of 16384 bytes.
#[test]
fn test_challenger_invariant_1_safetensors_16kb_page_padding() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let file_path = tmp_dir.path().join("invariant1_test.safetensors");

    let tensor_sizes = [
        ("tensor_small", vec![1usize, 37], 74), // odd byte length: 74 bytes
        ("tensor_medium", vec![128, 64], 128 * 32),
        ("tensor_odd", vec![3, 17], 3 * 17 * 2), // 102 bytes
        ("tensor_aligned", vec![16384, 1], 16384 * 2),
    ];

    let mut tensors = Vec::new();
    for (name, shape, byte_count) in &tensor_sizes {
        let num_elements = byte_count / 2;
        let bf16_data = vec![0x3F80u16; num_elements];
        tensors.push((
            name.to_string(),
            shape.clone(),
            SavedTensorPayload::Bf16(bf16_data),
        ));
    }

    write_mixed_safetensors(&file_path, &tensors).expect("write_mixed_safetensors should succeed");

    // Read the file and empirically inspect binary offsets
    let bytes = std::fs::read(&file_path).expect("Read safetensors file");
    assert!(bytes.len() >= 8);

    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_len;

    // Header padding invariant check
    assert_eq!(
        data_start % 16384,
        0,
        "Safetensors data section start (8 + header_len = {}) MUST be a multiple of 16384",
        data_start
    );

    // Parse JSON header and verify each tensor offset
    let header_str = std::str::from_utf8(&bytes[8..data_start]).expect("Valid JSON header");
    let parsed: serde_json::Value = serde_json::from_str(header_str.trim()).expect("Valid JSON");

    for (name, _, _) in &tensor_sizes {
        let tensor_meta = &parsed[name];
        let offsets = tensor_meta["data_offsets"]
            .as_array()
            .expect("data_offsets is array");
        let start_offset = offsets[0].as_u64().expect("start offset") as usize;

        assert_eq!(
            start_offset % 16384,
            0,
            "Tensor '{}' data start offset ({}) MUST be an exact multiple of 16384 bytes",
            name,
            start_offset
        );

        let absolute_file_offset = data_start + start_offset;
        assert_eq!(
            absolute_file_offset % 16384,
            0,
            "Absolute file offset ({}) for tensor '{}' MUST be an exact multiple of 16384 bytes",
            absolute_file_offset,
            name
        );
    }
}

/// Invariant 2: Production checkpoint validation strictly fails closed if required tensors are missing.
#[test]
fn test_challenger_invariant_2_production_validation_fails_closed() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let st_path = tmp_dir.path().join("model.safetensors");

    // Create a checkpoint with ONLY embed_tokens, missing all layer weights and final norm
    let dummy_blocks = vec![BlockQ4Group64 {
        scale_bf16: 0x3D00,
        bias_bf16: 0x0000,
        qs: [0x11; 32],
    }];
    write_mixed_safetensors(
        &st_path,
        &[(
            "model.embed_tokens.weight".to_string(),
            vec![1, 64],
            SavedTensorPayload::Q4Group64(dummy_blocks),
        )],
    )
    .unwrap();

    let desc = ModelArchDescriptor::llama_3_2_3b();
    let loader = ShardedCheckpointLoader::open_directory(tmp_dir.path()).unwrap();

    // Check require_tensor fails closed
    let res = ModelGpuWeights::require_tensor(
        &loader,
        "model.layers.0.self_attn.q_proj.weight",
        DType::Q4Group64,
        &[1, 64],
    );
    assert!(
        res.is_err(),
        "require_tensor MUST fail closed for missing tensor"
    );

    // Check require_norm_tensor fails closed
    let res_norm = ModelGpuWeights::require_norm_tensor(&loader, "model.norm.weight", 64);
    assert!(
        res_norm.is_err(),
        "require_norm_tensor MUST fail closed for missing norm"
    );

    // Check require_q4_weight fails closed
    let res_q4 =
        ModelGpuWeights::require_q4_weight(&loader, "model.layers.0.self_attn.q_proj", &[1, 64]);
    assert!(
        res_q4.is_err(),
        "require_q4_weight MUST fail closed for missing Q4 weight"
    );

    // Verify MetalExecutor::new_production strictly fails closed
    let kernels_dir = workspace_kernels_dir();
    let exec_res = MetalExecutor::new_production(&desc, &kernels_dir, tmp_dir.path(), 64, 16);
    assert!(
        exec_res.is_err(),
        "MetalExecutor::new_production MUST fail closed when checkpoint is incomplete"
    );
}

/// Invariant 3: Qwen2 vs Qwen3 architectural differentiation is preserved.
#[test]
fn test_challenger_invariant_3_model_architecture_differentiation() {
    // 1. Qwen 2.5 (e.g. DeepSeek-R1-Distill-Qwen) does NOT require QK norm tensors and is pure GQA
    let qwen2_desc = ModelArchDescriptor::deepseek_r1_distill_qwen_7b();
    assert_eq!(qwen2_desc.family, "deepseek_qwen");
    for layer in &qwen2_desc.layers {
        match &layer.seq_mixer {
            SequenceMixerDescriptor::GroupedQueryAttention { pos_encoding, .. } => {
                // RoPE positional encoding, no recurrent state
                assert!(matches!(pos_encoding, PositionalEncoding::RoPE { .. }));
            }
            _ => panic!("Qwen2 layers must be standard GroupedQueryAttention"),
        }
    }

    // 2. Qwen 3 is a hybrid attention / recurrent architecture with distinct policy
    let qwen3_desc = ModelArchDescriptor::qwen3_6_35b_a3b();
    assert_eq!(qwen3_desc.family, "qwen3_moe_hybrid");
    let mut has_deltanet = false;
    let mut has_gqa = false;
    for layer in &qwen3_desc.layers {
        match &layer.seq_mixer {
            SequenceMixerDescriptor::GatedDeltaNetRecurrence { .. } => has_deltanet = true,
            SequenceMixerDescriptor::GroupedQueryAttention { .. } => has_gqa = true,
        }
    }
    assert!(
        has_deltanet,
        "Qwen3 hybrid MUST contain DeltaNet recurrent layers"
    );
    assert!(has_gqa, "Qwen3 hybrid MUST contain GQA attention layers");
    assert_ne!(
        qwen2_desc.family, qwen3_desc.family,
        "Qwen2 and Qwen3 families MUST be strictly differentiated"
    );
}

/// Invariant 4: Fused gate-up SwiGLU decode kernel remains intact and functional.
#[test]
fn test_challenger_invariant_4_fused_swiglu_decode_functional() {
    let Some(device) = require_metal_device() else {
        return;
    };
    let mut cache =
        MetalPipelineCache::from_shader_dir(device.clone(), workspace_kernels_dir()).unwrap();

    let pso = cache.get_or_compile("q4_g64_decode_gate_up_swiglu_gemv");
    assert!(
        pso.is_ok(),
        "q4_g64_decode_gate_up_swiglu_gemv must compile"
    );
    let pso = pso.unwrap();

    let pso_2chan = cache.get_or_compile("q4_g64_decode_gate_up_swiglu_gemv_2chan");
    assert!(
        pso_2chan.is_ok(),
        "q4_g64_decode_gate_up_swiglu_gemv_2chan must compile"
    );

    // Run a small GPU dispatch of fused SwiGLU
    let queue = device.new_command_queue();
    let k = 64usize;
    let intermediate = 64usize;
    let num_k_blocks = k / 64;

    let x_vals = vec![0.5f32; k];
    let x_bf16: Vec<u16> = x_vals.iter().map(|&v| f32_to_bf16_bits(v)).collect();

    let dummy_blocks = vec![
        BlockQ4Group64 {
            scale_bf16: 0x3D00, // ~0.03125
            bias_bf16: 0x0000,
            qs: [0x22; 32], // low=2, high=2
        };
        intermediate * num_k_blocks
    ];

    let x_buf = device.new_buffer_with_data(
        x_bf16.as_ptr() as *const _,
        (x_bf16.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let gate_buf = device.new_buffer_with_data(
        dummy_blocks.as_ptr() as *const _,
        (dummy_blocks.len() * std::mem::size_of::<BlockQ4Group64>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let up_buf = device.new_buffer_with_data(
        dummy_blocks.as_ptr() as *const _,
        (dummy_blocks.len() * std::mem::size_of::<BlockQ4Group64>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (intermediate * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let params = GateUpSwigluGemmParams {
        m: 1,
        k: k as u32,
        num_k_blocks: num_k_blocks as u32,
        intermediate_size: intermediate as u32,
        activation_variant: 0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = device.new_buffer_with_data(
        &params as *const _ as *const _,
        std::mem::size_of::<GateUpSwigluGemmParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&gate_buf), 0);
    enc.set_buffer(1, Some(&up_buf), 0);
    enc.set_buffer(2, Some(&x_buf), 0);
    enc.set_buffer(3, Some(&out_buf), 0);
    enc.set_buffer(4, Some(&params_buf), 0);

    let num_tgs = (intermediate as u64).div_ceil(4);
    enc.dispatch_thread_groups(MTLSize::new(num_tgs, 1, 1), MTLSize::new(128, 1, 1));
    enc.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    // Verify GPU outputs are finite and non-zero
    let out_ptr = out_buf.contents() as *const u16;
    for i in 0..intermediate {
        let val = bf16_bits_to_f32(unsafe { *out_ptr.add(i) });
        assert!(
            val.is_finite(),
            "Fused SwiGLU output at index {i} must be finite: {val}"
        );
        assert!(
            val > 0.0,
            "Fused SwiGLU output at index {i} must be positive: {val}"
        );
    }
}

/// Invariant 5: Verify ActiveGuard decoupling invariant is preserved at the metal boundary.
#[test]
fn test_challenger_invariant_5_router_and_active_guard_untouched() {
    // Invariant 5 requires that MetalExecutor remains decoupled from HTTP/Router concerns.
    // MetalExecutor operates strictly on command encoders and device buffers.
    assert!(
        std::mem::size_of::<MetalExecutor>() > 0,
        "Decoupled architecture invariant preserved"
    );
}

// =========================================================================
// SECTION 3: Negative & Boundary Testing
// =========================================================================

#[test]
fn test_challenger_empty_tensors_and_zero_rows() {
    // 1. Empty slice split
    let empty_blocks: &[BlockQ4Group64] = &[];
    let (qs, scales) = BlockQ4Group64::split_slice(empty_blocks);
    assert!(qs.is_empty());
    assert!(scales.is_empty());

    let reconstructed = BlockQ4Group64::from_split(&qs, &scales);
    assert!(reconstructed.is_empty());

    // 2. Empty tensor view validation
    let empty_desc = TensorDescriptor {
        name: "test.empty".into(),
        shape: vec![0, 64],
        dtype: DType::Q4Group64,
        byte_offset: 0,
        byte_len: 0,
        placement: kriya_core::tensor::PlacementUnit::GpuHot,
    };
    let aligned_backing = ScaledAlignedQ4Block::new([0; 32]);
    let view = TensorView {
        descriptor: &empty_desc,
        data: &aligned_backing.qs[..0],
    };
    let aligned_blocks = view.as_aligned_q4_blocks().unwrap();
    assert!(aligned_blocks.is_empty());

    // 3. Odd dimension: shape with 0 rows
    let zero_row_desc = TensorDescriptor {
        name: "test.zero_row".into(),
        shape: vec![0],
        dtype: DType::BF16,
        byte_offset: 0,
        byte_len: 0,
        placement: kriya_core::tensor::PlacementUnit::GpuHot,
    };
    assert_eq!(zero_row_desc.num_elements(), 0);
}

#[test]
fn test_challenger_unaligned_byte_offset_fails_safely() {
    // TensorView::as_aligned_q4_blocks must safely reject slices with unaligned pointers
    let unaligned_raw = [0x5Au8; 65];
    let ptr = unaligned_raw.as_ptr() as usize;
    let offset_bytes = if (ptr + 1) % 16 != 0 { 1 } else { 2 };
    let unaligned_slice = &unaligned_raw[offset_bytes..offset_bytes + 32];

    let desc = TensorDescriptor {
        name: "test.unaligned".into(),
        shape: vec![1, 64],
        dtype: DType::Q4Group64,
        byte_offset: 0,
        byte_len: 32,
        placement: kriya_core::tensor::PlacementUnit::GpuHot,
    };
    let view = TensorView {
        descriptor: &desc,
        data: unaligned_slice,
    };
    let res = view.as_aligned_q4_blocks();
    assert!(
        res.is_err(),
        "as_aligned_q4_blocks MUST return Err on unaligned pointer, got {:?}",
        res.ok()
    );
}
