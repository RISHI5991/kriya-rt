//! Hardware Metal Shader Compilation & Live Apple Silicon GPU Execution Verification Suite.

use metal::{Device, MTLResourceOptions, MTLSize};
use std::path::PathBuf;
use std::process::Command;

fn workspace_kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
        .join("metal")
}

/// A deliberately small, but complete, dense-GQA model contract.  It lets the
/// production checkpoint validator be tested without treating a partial
/// checkpoint as evidence that a real model can run.
fn miniature_dense_gqa_descriptor() -> kriya_core::descriptor::ModelArchDescriptor {
    use kriya_core::descriptor::{
        ActivationVariant, AttentionMaskVariant, ChannelMixerDescriptor, LayerDescriptor,
        NormVariant, PhaseRoutingPolicy, PositionalEncoding, SequenceMixerDescriptor,
    };

    kriya_core::descriptor::ModelArchDescriptor {
        family: "test".into(),
        model_name: "miniature-dense-gqa".into(),
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
}

fn q4_payload(elements: usize) -> kriya_core::quantizer::SavedTensorPayload {
    let weights = vec![0.03125_f32; elements];
    kriya_core::quantizer::SavedTensorPayload::Q4Group64(
        kriya_core::quantizer::quantize_f32_to_q4_group64(&weights).unwrap(),
    )
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

#[test]
fn test_production_checkpoint_validation_fails_closed() {
    let tmp_dir = tempfile::tempdir().unwrap();

    // A production model cannot begin with an empty or partial checkpoint.
    let llama = kriya_core::descriptor::ModelArchDescriptor::llama_3_2_3b();
    assert!(
        kriya_backend_metal::encoder::ModelGpuWeights::validate_production_checkpoint(
            &llama,
            tmp_dir.path(),
        )
        .is_err()
    );

    // Sparse MoE is deliberately rejected until the executor binds every expert
    // tensor and implements the model's full topology.
    let qwen = kriya_core::descriptor::ModelArchDescriptor::qwen3_6_35b_a3b();
    assert!(
        kriya_backend_metal::encoder::ModelGpuWeights::validate_production_checkpoint(
            &qwen,
            tmp_dir.path(),
        )
        .is_err()
    );
}

#[test]
fn test_production_checkpoint_validation_accepts_complete_dense_q4_fixture() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let descriptor = miniature_dense_gqa_descriptor();
    write_complete_miniature_checkpoint(tmp_dir.path());

    kriya_backend_metal::encoder::ModelGpuWeights::validate_production_checkpoint(
        &descriptor,
        tmp_dir.path(),
    )
    .expect("complete canonical dense-GQA checkpoint should be production-valid");
}

#[test]
fn test_production_executor_runs_complete_dense_q4_fixture_on_metal() {
    let Some(_device) = Device::system_default() else {
        eprintln!("Skipping Metal execution: no Apple Silicon GPU is visible to this process");
        return;
    };

    let tmp_dir = tempfile::tempdir().unwrap();
    let descriptor = miniature_dense_gqa_descriptor();
    write_complete_miniature_checkpoint(tmp_dir.path());

    let mut executor = kriya_backend_metal::encoder::MetalExecutor::new_production(
        &descriptor,
        workspace_kernels_dir(),
        tmp_dir.path(),
        16,
        16,
    )
    .expect("complete checkpoint must initialize the strict production executor");

    let mut kv = kriya_kv::bsr_paged::PagedKvCacheManager::new(16, 16, 16);
    kv.allocate_sequence(7, &[1, 2]).unwrap();
    let schedule = kv.build_schedule_plan(&[7], 2).unwrap();
    let sampled = executor
        .forward_and_sample(
            &descriptor,
            &schedule,
            &[1, 2],
            &[0, 1],
            kriya_backend_metal::encoder::SamplingConfig {
                vocab_size: descriptor.vocab_size,
                greedy: 1,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 64,
                rng_seed: 123,
            },
        )
        .expect("strict production executor must complete its Metal forward pass");
    assert!(sampled < descriptor.vocab_size);
}

#[test]
fn test_q4_decode_gemv_matches_cpu_reference_on_metal() {
    let Some(device) = Device::system_default() else {
        eprintln!("Skipping Metal execution: no Apple Silicon GPU is visible to this process");
        return;
    };

    let rows = 5usize;
    let cols = 64usize;
    let source_weights: Vec<f32> = (0..rows * cols)
        .map(|idx| ((idx % 37) as f32 - 18.0) * 0.03125)
        .collect();
    let blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&source_weights).unwrap();
    let dequantized = kriya_core::quantizer::dequantize_q4_group64_to_f32(&blocks);
    let input: Vec<f32> = (0..cols)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
        .collect();
    let input_bf16: Vec<u16> = input
        .iter()
        .map(|&value| kriya_core::quantizer::f32_to_bf16_bits(value))
        .collect();

    let expected: Vec<f32> = dequantized
        .chunks_exact(cols)
        .map(|row| row.iter().zip(&input).map(|(w, x)| w * x).sum())
        .collect();

    let shared = MTLResourceOptions::StorageModeShared;
    let weight_buf = device.new_buffer_with_data(
        blocks.as_ptr() as *const _,
        (blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );
    let input_buf = device.new_buffer_with_data(
        input_bf16.as_ptr() as *const _,
        (input_bf16.len() * std::mem::size_of::<u16>()) as u64,
        shared,
    );
    let output_buf = device.new_buffer((rows * std::mem::size_of::<u16>()) as u64, shared);
    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device,
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
        n: rows as u32,
        k: cols as u32,
        num_k_blocks: 1,
    };
    encoder.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(rows.div_ceil(4) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let output = unsafe { std::slice::from_raw_parts(output_buf.contents() as *const u16, rows) };
    for (gpu_bits, cpu) in output.iter().zip(expected) {
        let gpu = kriya_core::quantizer::bf16_bits_to_f32(*gpu_bits);
        assert!((gpu - cpu).abs() < 0.05, "GPU={gpu}, CPU={cpu}");
    }
}

#[test]
fn test_paged_attention_matches_cpu_reference_on_metal() {
    let Some(device) = Device::system_default() else {
        eprintln!("Skipping Metal execution: no Apple Silicon GPU is visible to this process");
        return;
    };

    const D: usize = 64;
    const TOKENS: usize = 3;
    let q_raw: Vec<Vec<f32>> = (0..TOKENS)
        .map(|t| (0..D).map(|d| (t * 7 + d) as f32 * 0.01 - 0.2).collect())
        .collect();
    let k_raw: Vec<Vec<f32>> = (0..TOKENS)
        .map(|t| (0..D).map(|d| (t * 5 + d) as f32 * 0.008 - 0.15).collect())
        .collect();
    let v_raw: Vec<Vec<f32>> = (0..TOKENS)
        .map(|t| (0..D).map(|d| (t * 3 + d) as f32 * 0.006 - 0.1).collect())
        .collect();
    let to_bf16 = |values: &[f32]| -> Vec<u16> {
        values
            .iter()
            .map(|&value| kriya_core::quantizer::f32_to_bf16_bits(value))
            .collect()
    };
    let shared = MTLResourceOptions::StorageModeShared;
    let q_buf = device.new_buffer((D * std::mem::size_of::<u16>()) as u64, shared);
    let k_buf = device.new_buffer((D * std::mem::size_of::<u16>()) as u64, shared);
    let v_buf = device.new_buffer((D * std::mem::size_of::<u16>()) as u64, shared);
    let kv_bytes = (16 * D * std::mem::size_of::<u16>()) as u64;
    let kv_k = device.new_buffer(kv_bytes, shared);
    let kv_v = device.new_buffer(kv_bytes, shared);
    let indptr = device.new_buffer_with_data([0_i32, 1].as_ptr() as *const _, 8, shared);
    let indices = device.new_buffer_with_data([0_i32].as_ptr() as *const _, 4, shared);
    let last_len = device.new_buffer(4, shared);
    let position = device.new_buffer(4, shared);
    let tile = kriya_kv::bsr_paged::CtaWorkTile {
        request_idx: 0,
        kv_block_start: 0,
        kv_block_count: 1,
        is_split_k: 0,
    };
    let tiles = device.new_buffer_with_data(
        &tile as *const _ as *const _,
        std::mem::size_of_val(&tile) as u64,
        shared,
    );
    let split_idx = device.new_buffer(4, shared);
    let output = device.new_buffer((D * std::mem::size_of::<u16>()) as u64, shared);
    let partial_out = device.new_buffer((D * 4) as u64, shared);
    let partial_lse = device.new_buffer(4, shared);
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();
    let queue = device.new_command_queue();

    let rope_params = kriya_backend_metal::encoder::RopeKvWriteParams {
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: D as u32,
        rotary_dim: D as u32,
        block_size_tokens: 16,
        apply_qk_norm: 0,
        rms_eps: 1e-6,
        rope_theta: 10_000.0,
    };
    let gqa_params = kriya_backend_metal::encoder::BsrGqaParams {
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: D as u32,
        block_size_tokens: 16,
        max_split_k_tiles: 1,
        max_query_tokens: 1,
        sm_scale: 1.0 / (D as f32).sqrt(),
        logit_softcap: 0.0,
    };

    for token_idx in 0..TOKENS {
        let q = to_bf16(&q_raw[token_idx]);
        let k = to_bf16(&k_raw[token_idx]);
        let v = to_bf16(&v_raw[token_idx]);
        unsafe {
            std::ptr::copy_nonoverlapping(q.as_ptr(), q_buf.contents() as *mut u16, D);
            std::ptr::copy_nonoverlapping(k.as_ptr(), k_buf.contents() as *mut u16, D);
            std::ptr::copy_nonoverlapping(v.as_ptr(), v_buf.contents() as *mut u16, D);
            *(last_len.contents() as *mut i32) = (token_idx + 1) as i32;
            *(position.contents() as *mut u32) = token_idx as u32;
        }
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(
            &pipelines
                .get_or_compile("fused_qknorm_rope_and_kv_append")
                .unwrap(),
        );
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&k_buf), 0);
        enc.set_buffer(2, Some(&v_buf), 0);
        enc.set_buffer(3, None, 0);
        enc.set_buffer(4, None, 0);
        enc.set_buffer(5, Some(&kv_k), 0);
        enc.set_buffer(6, Some(&kv_v), 0);
        enc.set_buffer(7, Some(&indptr), 0);
        enc.set_buffer(8, Some(&indices), 0);
        enc.set_buffer(9, Some(&last_len), 0);
        enc.set_buffer(10, Some(&position), 0);
        enc.set_bytes(
            11,
            std::mem::size_of_val(&rope_params) as u64,
            &rope_params as *const _ as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(2, 1, 1), MTLSize::new(32, 1, 1));

        enc.set_compute_pipeline_state(
            &pipelines
                .get_or_compile("bsr_paged_gqa_decode_stage1")
                .unwrap(),
        );
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&kv_k), 0);
        enc.set_buffer(2, Some(&kv_v), 0);
        enc.set_buffer(3, Some(&indptr), 0);
        enc.set_buffer(4, Some(&indices), 0);
        enc.set_buffer(5, Some(&last_len), 0);
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
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let rotate = |values: &[f32], position: usize| -> Vec<f32> {
        let mut out = values.to_vec();
        for idx in 0..(D / 2) {
            let exponent = -((2 * idx) as f32) / D as f32;
            let angle = position as f32 * 10_000.0_f32.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            out[idx] = values[idx] * cos - values[idx + D / 2] * sin;
            out[idx + D / 2] = values[idx] * sin + values[idx + D / 2] * cos;
        }
        out
    };
    let q = rotate(&q_raw[TOKENS - 1], TOKENS - 1);
    let keys: Vec<Vec<f32>> = (0..TOKENS).map(|idx| rotate(&k_raw[idx], idx)).collect();
    let scores: Vec<f32> = keys
        .iter()
        .map(|key| q.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() / (D as f32).sqrt())
        .collect();
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalizer: f32 = scores.iter().map(|score| (score - max_score).exp()).sum();
    let expected: Vec<f32> = (0..D)
        .map(|d| {
            scores
                .iter()
                .enumerate()
                .map(|(idx, score)| ((score - max_score).exp() / normalizer) * v_raw[idx][d])
                .sum()
        })
        .collect();
    let actual = unsafe { std::slice::from_raw_parts(output.contents() as *const u16, D) };
    for (bits, cpu) in actual.iter().zip(expected) {
        let gpu = kriya_core::quantizer::bf16_bits_to_f32(*bits);
        assert!((gpu - cpu).abs() < 0.01, "GPU={gpu}, CPU={cpu}");
    }
}

#[test]
fn test_xcrun_metal_compiler_verification() {
    let kernels_dir = workspace_kernels_dir();
    let shader_files = [
        "q4_gemm_and_moe.metal",
        "paged_gqa_and_deltanet.metal",
        "transformer_glue.metal",
    ];

    // Check if xcrun is available on the host (macOS with Xcode Command Line Tools)
    let xcrun_check = Command::new("xcrun")
        .args(["-sdk", "macosx", "--find", "metal"])
        .output();
    match xcrun_check {
        Ok(out) if out.status.success() => {}
        _ => {
            eprintln!(
                "Skipping xcrun check: metal compiler not found (install Xcode Command Line Tools)"
            );
            return;
        }
    }

    let tmp_dir = std::env::temp_dir();
    for shader in &shader_files {
        let src_path = kernels_dir.join(shader);
        assert!(src_path.exists(), "Missing shader file: {:?}", src_path);

        let air_out = tmp_dir.join(format!("{shader}.air"));
        let output = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-std=metal3.1",
                "-O3",
                "-Werror",
                "-c",
                src_path.to_str().unwrap(),
                "-o",
                air_out.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to invoke xcrun metal compiler");

        assert!(
            output.status.success(),
            "Metal compilation failed for {}:\n{}",
            shader,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn test_kv_layer_stride_and_kriya_convert_q4_binding() {
    let Some(device) = Device::system_default() else {
        return;
    };

    let mut desc = kriya_core::descriptor::ModelArchDescriptor::llama_3_2_3b();
    desc.layers.truncate(4);

    // 1. Verify Blocker 3 Fix: Write a `kriya_convert` unified Q4_G64 safetensors checkpoint and load it via `from_checkpoint_dir`
    let tmp_dir = tempfile::tempdir().unwrap();
    let st_path = tmp_dir.path().join("model.safetensors");
    let dummy_weights = vec![0.25f32; desc.hidden_size as usize];
    let blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&dummy_weights).unwrap();
    kriya_core::quantizer::write_quantized_safetensors(
        &st_path,
        &[(
            "model.embed_tokens.weight".to_string(),
            vec![1, desc.hidden_size as usize],
            blocks,
        )],
    )
    .unwrap();

    let loaded_weights = kriya_backend_metal::encoder::ModelGpuWeights::from_checkpoint_dir(
        &device,
        &desc,
        tmp_dir.path(),
    )
    .expect("Failed to bind kriya_convert Q4_G64 checkpoint");
    assert!(loaded_weights.embed_tokens.length() >= 36);

    // 2. Verify Blocker 1 Fix: Ensure per-layer KV byte offsets are strictly non-overlapping across layers
    let kernels_dir = workspace_kernels_dir();
    let mut exec = kriya_backend_metal::encoder::MetalExecutor::new(
        &desc,
        &kernels_dir,
        Some(tmp_dir.path()),
        64,
        16,
    )
    .unwrap();
    assert!(exec.layer_kv_stride_bytes > 0);
    assert_ne!(exec.layer_kv_byte_offset(0), exec.layer_kv_byte_offset(1));
    assert_eq!(exec.layer_kv_byte_offset(2), 2 * exec.layer_kv_stride_bytes);

    // 3. Verify Blocker 2 & 4 Fixes: Run multi-token forward pass with explicit `seq_positions` and GPU EAGLE tree verification
    let mut kv_mgr = kriya_kv::bsr_paged::PagedKvCacheManager::new(64, 16, 16);
    kv_mgr.allocate_sequence(1, &[10, 20, 30]).unwrap();
    let plan = kv_mgr.build_schedule_plan(&[1], 8).unwrap();
    let tok = exec
        .forward_and_sample(
            &desc,
            &plan,
            &[10, 20, 30],
            &[0, 1, 2],
            kriya_backend_metal::encoder::SamplingConfig {
                vocab_size: desc.vocab_size,
                greedy: 1,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 50,
                rng_seed: 1,
            },
        )
        .unwrap();
    assert!(tok < desc.vocab_size);

    // The GPU tree kernel remains available for development, but production
    // speculative decoding must reject it until target-logit acceptance is wired.
    let verification = exec.verify_eagle_draft_tree_gpu(
        &desc,
        &plan,
        &[tok, (tok + 1) % 100, (tok + 2) % 100],
        &[true, false, false, true, true, false, true, false, true],
    );
    assert!(matches!(
        verification,
        Err(kriya_backend_metal::encoder::MetalError::SpeculativeVerificationNotImplemented)
    ));
}

#[test]
fn test_full_model_descriptors_and_weight_precision_policy() {
    // 1. 1D norm tensors are preserved in BF16 while 2D matrices use Q4_G64.
    assert!(kriya_core::quantizer::should_preserve_in_bf16(
        "model.norm.weight",
        &[3072]
    ));
    assert!(kriya_core::quantizer::should_preserve_in_bf16(
        "model.layers.0.input_layernorm.weight",
        &[3072]
    ));
    assert!(!kriya_core::quantizer::should_preserve_in_bf16(
        "model.layers.0.self_attn.q_proj.weight",
        &[3072, 3072]
    ));

    // 2. Validate the descriptors independently of a fabricated checkpoint.
    assert!(
        kriya_core::descriptor::ModelArchDescriptor::mistral_7b_v0_3()
            .validate()
            .is_ok()
    );
    assert!(kriya_core::descriptor::ModelArchDescriptor::gemma_2_9b()
        .validate()
        .is_ok());
    assert!(
        kriya_core::descriptor::ModelArchDescriptor::deepseek_r1_distill_qwen_7b()
            .validate()
            .is_ok()
    );

    let full_desc = kriya_core::descriptor::ModelArchDescriptor::llama_3_2_3b();
    assert_eq!(full_desc.layers.len(), 28);
}
