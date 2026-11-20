//! Empirical Adversarial Verification Suite for Milestone 1.
//!
//! Evaluates:
//! 1. Memory alignment, sizing, and unaligned pointer safety.
//! 2. Exact bit-for-bit roundtrip identity on arbitrary buffers.
//! 3. Fused Add RMSNorm stress testing across extreme numerical regimes on Metal.
//! 4. Consolidated QKV output parity against separate projections.
//! 5. Dispatch count per layer == 5.

#![allow(clippy::useless_vec)]
#![allow(clippy::needless_range_loop)]

use metal::{Device, MTLResourceOptions, MTLSize};
use std::path::PathBuf;

use kriya_backend_metal::encoder::{
    MetalExecutor, MetalPipelineCache, QkvGemmParams, RmsNormParams,
};
use kriya_core::descriptor::ModelArchDescriptor;
use kriya_core::quantizer::{bf16_bits_to_f32, f32_to_bf16_bits};
use kriya_core::tensor::{
    BlockQ4Group64, BlockQ4ScaleBias, DType, PlacementUnit, ScaledAlignedQ4Block, TensorDescriptor,
    TensorError, TensorView,
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
// SECTION 1: Empirical Alignment, Sizing, and Roundtrip Verification
// =========================================================================

#[test]
fn test_challenger_alignment_and_sizing() {
    assert_eq!(
        std::mem::align_of::<ScaledAlignedQ4Block>(),
        16,
        "ScaledAlignedQ4Block MUST be 16-byte aligned"
    );
    assert_eq!(
        std::mem::size_of::<ScaledAlignedQ4Block>(),
        32,
        "ScaledAlignedQ4Block MUST be exactly 32 bytes"
    );
    assert_eq!(
        std::mem::size_of::<BlockQ4ScaleBias>(),
        4,
        "BlockQ4ScaleBias MUST be exactly 4 bytes"
    );
    assert!(
        std::mem::align_of::<BlockQ4ScaleBias>() >= 2,
        "BlockQ4ScaleBias alignment MUST be at least 2 bytes"
    );
}

#[test]
fn test_challenger_bit_for_bit_roundtrip_identity() {
    // 1. Empty buffer
    let empty_blocks: Vec<BlockQ4Group64> = Vec::new();
    let (empty_qs, empty_scales) = BlockQ4Group64::split_slice(&empty_blocks);
    assert!(empty_qs.is_empty());
    assert!(empty_scales.is_empty());
    let empty_reconstructed = BlockQ4Group64::from_split(&empty_qs, &empty_scales);
    assert!(empty_reconstructed.is_empty());

    // 2. Single block with boundary and extreme bit patterns
    let test_patterns: Vec<BlockQ4Group64> = vec![
        BlockQ4Group64 {
            scale_bf16: 0x0000,
            bias_bf16: 0x0000,
            qs: [0x00; 32],
        },
        BlockQ4Group64 {
            scale_bf16: 0xFFFF,
            bias_bf16: 0xFFFF,
            qs: [0xFF; 32],
        },
        BlockQ4Group64 {
            scale_bf16: 0x7F80, // +inf
            bias_bf16: 0xFF80,  // -inf
            qs: [0x5A; 32],
        },
        BlockQ4Group64 {
            scale_bf16: 0x0001, // subnormal
            bias_bf16: 0x8000,  // -0.0
            qs: [0xA5; 32],
        },
    ];

    for (idx, original) in test_patterns.iter().enumerate() {
        let (q, s) = original.to_split();
        let reconstructed = BlockQ4Group64::from_split_single(q, s);
        let orig_scale = original.scale_bf16;
        let recon_scale = reconstructed.scale_bf16;
        let orig_bias = original.bias_bf16;
        let recon_bias = reconstructed.bias_bf16;
        let orig_qs = original.qs;
        let recon_qs = reconstructed.qs;
        assert_eq!(orig_scale, recon_scale, "Pattern {idx}: scale mismatch");
        assert_eq!(orig_bias, recon_bias, "Pattern {idx}: bias mismatch");
        assert_eq!(orig_qs, recon_qs, "Pattern {idx}: qs nibbles mismatch");
    }

    // 3. 1000 arbitrary blocks: verify exact bit-for-bit identity
    const NUM_BLOCKS: usize = 1000;
    let mut large_blocks = Vec::with_capacity(NUM_BLOCKS);
    let mut rng_state: u64 = 0x123456789ABCDEF0;
    for _ in 0..NUM_BLOCKS {
        let mut qs = [0u8; 32];
        for b in qs.iter_mut() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng_state >> 33) as u8;
        }
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let scale_bf16 = (rng_state >> 16) as u16;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bias_bf16 = (rng_state >> 16) as u16;

        large_blocks.push(BlockQ4Group64 {
            scale_bf16,
            bias_bf16,
            qs,
        });
    }

    let (split_qs, split_scales) = BlockQ4Group64::split_slice(&large_blocks);
    assert_eq!(split_qs.len(), NUM_BLOCKS);
    assert_eq!(split_scales.len(), NUM_BLOCKS);

    let reconstructed_large = BlockQ4Group64::from_split(&split_qs, &split_scales);
    assert_eq!(reconstructed_large.len(), NUM_BLOCKS);

    let orig_bytes: &[u8] = bytemuck::cast_slice(&large_blocks);
    let recon_bytes: &[u8] = bytemuck::cast_slice(&reconstructed_large);
    assert_eq!(
        orig_bytes, recon_bytes,
        "1000 blocks roundtrip MUST be 100% bit-for-bit identical"
    );
}

#[test]
fn test_challenger_unaligned_pointer_fails_safely() {
    // Allocate 128 bytes of heap memory
    let buffer = vec![0u8; 128];

    // Test byte offsets 0 through 15
    for offset in 0..16 {
        let sub_slice = &buffer[offset..offset + 32];
        let sub_ptr = sub_slice.as_ptr() as usize;

        let desc = TensorDescriptor {
            name: "test_tensor".to_string(),
            dtype: DType::Q4Group64,
            shape: vec![64],
            byte_offset: offset,
            byte_len: 32,
            placement: PlacementUnit::GpuHot,
        };

        let view = TensorView {
            descriptor: &desc,
            data: sub_slice,
        };

        let result = view.as_aligned_q4_blocks();
        if sub_ptr % 16 == 0 {
            assert!(
                result.is_ok(),
                "Offset {offset} (ptr {sub_ptr:x}) is 16-byte aligned and should succeed"
            );
        } else {
            assert!(
                matches!(result, Err(TensorError::AlignmentError { align: 16, .. })),
                "Offset {offset} (ptr {sub_ptr:x}) is NOT 16-byte aligned and MUST return AlignmentError safely"
            );
        }
    }
}

// =========================================================================
// SECTION 2: Fused Add RMSNorm Stress Testing across Extreme Values
// =========================================================================

#[test]
fn test_challenger_fused_add_rmsnorm_extreme_numerical_stress() {
    let Some(device) = require_metal_device() else {
        return;
    };
    let shared = MTLResourceOptions::StorageModeShared;
    let mut pipelines =
        MetalPipelineCache::from_shader_dir(device.clone(), workspace_kernels_dir()).unwrap();
    let queue = device.new_command_queue();

    const NUM_TOKENS: usize = 2;
    const HIDDEN_SIZE: usize = 256;
    let total_elems = NUM_TOKENS * HIDDEN_SIZE;

    struct StressTestCase {
        name: &'static str,
        res_val: f32,
        proj_val: f32,
        weight_val: f32,
        eps: f32,
        zero_centered: u32,
        tolerance: f32,
    }

    let cases = vec![
        StressTestCase {
            name: "Standard scale",
            res_val: 1.5,
            proj_val: 0.5,
            weight_val: 1.0,
            eps: 1e-5,
            zero_centered: 0,
            tolerance: 1e-3,
        },
        StressTestCase {
            name: "Large activations (stress upper range)",
            res_val: 500.0,
            proj_val: 250.0,
            weight_val: 1.2,
            eps: 1e-5,
            zero_centered: 0,
            tolerance: 1e-2,
        },
        StressTestCase {
            name: "Near-zero activations",
            res_val: 1e-5,
            proj_val: 1e-5,
            weight_val: 0.9,
            eps: 1e-5,
            zero_centered: 0,
            tolerance: 1e-3,
        },
        StressTestCase {
            name: "Subnormal floats near bf16 minimum",
            res_val: 1e-38,
            proj_val: 1e-38,
            weight_val: 1.0,
            eps: 1e-6,
            zero_centered: 0,
            tolerance: 1e-3,
        },
        StressTestCase {
            name: "Exact cancellation to zero",
            res_val: 10.0,
            proj_val: -10.0,
            weight_val: 1.0,
            eps: 1e-5,
            zero_centered: 0,
            tolerance: 1e-3,
        },
        StressTestCase {
            name: "Small epsilon (1e-8)",
            res_val: 0.05,
            proj_val: 0.05,
            weight_val: 1.0,
            eps: 1e-8,
            zero_centered: 0,
            tolerance: 1e-3,
        },
        StressTestCase {
            name: "Zero centered offset enabled",
            res_val: 1.2,
            proj_val: 0.8,
            weight_val: 0.1, // weight + 1.0 = 1.1
            eps: 1e-5,
            zero_centered: 1,
            tolerance: 1e-3,
        },
    ];

    for tc in cases {
        let res_initial: Vec<u16> = (0..total_elems)
            .map(|i| {
                let factor = 1.0 + ((i % 17) as f32 - 8.0) * 0.02;
                f32_to_bf16_bits(tc.res_val * factor)
            })
            .collect();
        let proj_data: Vec<u16> = (0..total_elems)
            .map(|i| {
                let factor = 1.0 + ((i % 13) as f32 - 6.0) * 0.02;
                f32_to_bf16_bits(tc.proj_val * factor)
            })
            .collect();
        let weight_data: Vec<u16> = (0..HIDDEN_SIZE)
            .map(|i| {
                let factor = 1.0 + ((i % 7) as f32 - 3.0) * 0.01;
                f32_to_bf16_bits(tc.weight_val * factor)
            })
            .collect();

        let weight_buf = device.new_buffer_with_data(
            weight_data.as_ptr() as *const _,
            (weight_data.len() * 2) as u64,
            shared,
        );

        let norm_params = RmsNormParams {
            num_tokens: NUM_TOKENS as u32,
            hidden_size: HIDDEN_SIZE as u32,
            eps: tc.eps,
            zero_centered_offset: tc.zero_centered,
        };

        // --- Unfused Execution ---
        let res_unfused = device.new_buffer_with_data(
            res_initial.as_ptr() as *const _,
            (res_initial.len() * 2) as u64,
            shared,
        );
        let proj_unfused = device.new_buffer_with_data(
            proj_data.as_ptr() as *const _,
            (proj_data.len() * 2) as u64,
            shared,
        );
        let norm_out_unfused = device.new_buffer((total_elems * 2) as u64, shared);

        let cmd1 = queue.new_command_buffer();
        let enc1 = cmd1.new_compute_command_encoder();
        enc1.set_compute_pipeline_state(&pipelines.get_or_compile("residual_add_bf16").unwrap());
        enc1.set_buffer(0, Some(&res_unfused), 0);
        enc1.set_buffer(1, Some(&proj_unfused), 0);
        let total_u32 = total_elems as u32;
        enc1.set_bytes(2, 4, &total_u32 as *const _ as *const _);
        enc1.dispatch_thread_groups(
            MTLSize::new(total_elems.div_ceil(256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );

        enc1.set_compute_pipeline_state(&pipelines.get_or_compile("rms_norm_bf16").unwrap());
        enc1.set_buffer(0, Some(&res_unfused), 0);
        enc1.set_buffer(1, Some(&weight_buf), 0);
        enc1.set_buffer(2, Some(&norm_out_unfused), 0);
        enc1.set_bytes(
            3,
            std::mem::size_of_val(&norm_params) as u64,
            &norm_params as *const _ as *const _,
        );
        enc1.dispatch_thread_groups(
            MTLSize::new(NUM_TOKENS as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc1.end_encoding();
        cmd1.commit();
        cmd1.wait_until_completed();

        // --- Fused Execution ---
        let res_fused = device.new_buffer_with_data(
            res_initial.as_ptr() as *const _,
            (res_initial.len() * 2) as u64,
            shared,
        );
        let proj_fused = device.new_buffer_with_data(
            proj_data.as_ptr() as *const _,
            (proj_data.len() * 2) as u64,
            shared,
        );
        let norm_out_fused = device.new_buffer((total_elems * 2) as u64, shared);

        let cmd2 = queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        enc2.set_compute_pipeline_state(&pipelines.get_or_compile("fused_add_rmsnorm").unwrap());
        enc2.set_buffer(0, Some(&res_fused), 0);
        enc2.set_buffer(1, Some(&proj_fused), 0);
        enc2.set_buffer(2, Some(&weight_buf), 0);
        enc2.set_buffer(3, Some(&norm_out_fused), 0);
        enc2.set_bytes(
            4,
            std::mem::size_of_val(&norm_params) as u64,
            &norm_params as *const _ as *const _,
        );
        enc2.dispatch_thread_groups(
            MTLSize::new(NUM_TOKENS as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc2.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        let res_unfused_slice = unsafe {
            std::slice::from_raw_parts(res_unfused.contents() as *const u16, total_elems)
        };
        let res_fused_slice =
            unsafe { std::slice::from_raw_parts(res_fused.contents() as *const u16, total_elems) };
        let norm_unfused_slice = unsafe {
            std::slice::from_raw_parts(norm_out_unfused.contents() as *const u16, total_elems)
        };
        let norm_fused_slice = unsafe {
            std::slice::from_raw_parts(norm_out_fused.contents() as *const u16, total_elems)
        };

        assert_eq!(
            res_unfused_slice, res_fused_slice,
            "TestCase '{}': residual update mismatch",
            tc.name
        );

        for i in 0..total_elems {
            let u = bf16_bits_to_f32(norm_unfused_slice[i]);
            let f = bf16_bits_to_f32(norm_fused_slice[i]);
            let diff = (u - f).abs();
            assert!(
                diff <= tc.tolerance,
                "TestCase '{}' mismatch at index {i}: unfused={u}, fused={f}, diff={diff} > tolerance={}",
                tc.name,
                tc.tolerance
            );
        }
    }
}

// =========================================================================
// SECTION 3: Consolidated QKV Parity and Dispatch Count Verification
// =========================================================================

#[test]
fn test_challenger_dispatch_count_is_strictly_five() {
    let Some(device) = require_metal_device() else {
        return;
    };
    let mut desc = ModelArchDescriptor::llama_3_2_3b();
    desc.layers.truncate(1);

    let executor = MetalExecutor::allocate_for_descriptor(&device, &desc, workspace_kernels_dir())
        .expect("allocate_for_descriptor must succeed");

    let dispatches = executor.measure_dispatches_per_layer();
    assert_eq!(
        dispatches, 5,
        "executor.measure_dispatches_per_layer() MUST return exactly 5"
    );
}

#[test]
fn test_challenger_consolidated_qkv_activation_parity_against_separate() {
    let Some(device) = require_metal_device() else {
        return;
    };
    let queue = device.new_command_queue();
    let mut cache =
        MetalPipelineCache::from_shader_dir(device.clone(), workspace_kernels_dir()).unwrap();

    let k = 64u32;
    let q_dim = 64u32;
    let kv_dim = 64u32;
    let total_rows = q_dim + 2 * kv_dim;

    // Generate weights
    let weights_flat: Vec<f32> = (0..total_rows * k)
        .map(|idx| ((idx % 43) as f32 - 21.0) * 0.015)
        .collect();
    let blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&weights_flat).unwrap();
    let (aligned_qs, scale_biases) = BlockQ4Group64::split_slice(&blocks);

    let shared = MTLResourceOptions::StorageModeShared;
    let qs_bytes: &[u8] = bytemuck::cast_slice(&aligned_qs);
    let sb_bytes: &[u8] = bytemuck::cast_slice(&scale_biases);
    let qkv_weights_buf =
        device.new_buffer_with_data(qs_bytes.as_ptr() as *const _, qs_bytes.len() as u64, shared);
    let qkv_scales_buf =
        device.new_buffer_with_data(sb_bytes.as_ptr() as *const _, sb_bytes.len() as u64, shared);

    let input_f32: Vec<f32> = (0..k).map(|i| (i as f32 * 0.08).cos()).collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&x| f32_to_bf16_bits(x)).collect();
    let input_bytes: &[u8] = bytemuck::cast_slice(&input_bf16);
    let input_buf = device.new_buffer_with_data(
        input_bytes.as_ptr() as *const _,
        input_bytes.len() as u64,
        shared,
    );

    let out_q_buf = device.new_buffer((q_dim * 2) as u64, shared);
    let out_k_buf = device.new_buffer((kv_dim * 2) as u64, shared);
    let out_v_buf = device.new_buffer((kv_dim * 2) as u64, shared);
    let dummy_bias = device.new_buffer(4, shared);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    let pso = cache
        .get_or_compile("q4_aligned_decode_qkv_gemv_2row")
        .unwrap();
    let params = QkvGemmParams {
        m: 1,
        k,
        num_k_blocks: k.div_ceil(64),
        q_dim,
        k_dim: kv_dim,
        v_dim: kv_dim,
        has_bias: 0,
        _pad: 0,
    };
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&qkv_weights_buf), 0);
    enc.set_buffer(1, Some(&qkv_scales_buf), 0);
    enc.set_buffer(2, Some(&input_buf), 0);
    enc.set_buffer(3, Some(&out_q_buf), 0);
    enc.set_buffer(4, Some(&out_k_buf), 0);
    enc.set_buffer(5, Some(&out_v_buf), 0);
    enc.set_buffer(6, Some(&dummy_bias), 0);
    enc.set_bytes(
        7,
        std::mem::size_of::<QkvGemmParams>() as u64,
        &params as *const _ as *const _,
    );
    enc.dispatch_thread_groups(
        MTLSize::new((total_rows.div_ceil(8) as u64).min(128), 1, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out_q =
        unsafe { std::slice::from_raw_parts(out_q_buf.contents() as *const u16, q_dim as usize) };
    let out_k =
        unsafe { std::slice::from_raw_parts(out_k_buf.contents() as *const u16, kv_dim as usize) };
    let out_v =
        unsafe { std::slice::from_raw_parts(out_v_buf.contents() as *const u16, kv_dim as usize) };

    // Reference CPU dequantization and separate dot products for Q, K, V
    let cpu_dequant = kriya_core::quantizer::dequantize_q4_group64_to_f32(&blocks);

    // Verify Q
    for row in 0..q_dim as usize {
        let mut expected = 0.0f32;
        for c in 0..k as usize {
            expected += cpu_dequant[row * k as usize + c] * input_f32[c];
        }
        let actual = bf16_bits_to_f32(out_q[row]);
        assert!(
            (expected - actual).abs() < 0.05,
            "Q mismatch at row {row}: expected {expected}, got {actual}"
        );
    }

    // Verify K
    for row in 0..kv_dim as usize {
        let global_row = (q_dim as usize) + row;
        let mut expected = 0.0f32;
        for c in 0..k as usize {
            expected += cpu_dequant[global_row * k as usize + c] * input_f32[c];
        }
        let actual = bf16_bits_to_f32(out_k[row]);
        assert!(
            (expected - actual).abs() < 0.05,
            "K mismatch at row {row}: expected {expected}, got {actual}"
        );
    }

    // Verify V
    for row in 0..kv_dim as usize {
        let global_row = (q_dim as usize) + (kv_dim as usize) + row;
        let mut expected = 0.0f32;
        for c in 0..k as usize {
            expected += cpu_dequant[global_row * k as usize + c] * input_f32[c];
        }
        let actual = bf16_bits_to_f32(out_v[row]);
        assert!(
            (expected - actual).abs() < 0.05,
            "V mismatch at row {row}: expected {expected}, got {actual}"
        );
    }
}
