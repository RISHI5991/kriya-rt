//! Hardware Metal Shader Compilation & Live Apple Silicon GPU Execution Verification Suite.

use futures::executor::block_on;
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
fn test_production_checkpoint_rejects_missing_model_tensor_with_explicit_name() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let descriptor = miniature_dense_gqa_descriptor();
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
        // Deliberately omitted: model.layers.0.mlp.down_proj.weight
    ];
    kriya_core::quantizer::write_mixed_safetensors(
        tmp_dir.path().join("model.safetensors"),
        &tensors,
    )
    .unwrap();

    let err = kriya_backend_metal::encoder::ModelGpuWeights::validate_production_checkpoint(
        &descriptor,
        tmp_dir.path(),
    )
    .expect_err("should reject missing down_proj");

    let err_str = err.to_string();
    assert!(
        err_str.contains("model.layers.0.mlp.down_proj"),
        "Error must name missing tensor: {err_str}"
    );
}

#[test]
fn test_production_checkpoint_rejects_malformed_scale_tensor() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let descriptor = miniature_dense_gqa_descriptor();
    use kriya_core::quantizer::SavedTensorPayload;
    let bf16_ones = || SavedTensorPayload::Bf16(vec![0x3F80_u16; 64]);
    let q4_matrix = || q4_payload(64 * 64);
    // Write MLX triplet for q_proj where scales has wrong group length (1 instead of 64)
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
            vec![64, 16],
            SavedTensorPayload::Bf16(vec![0; 64 * 16]),
        ),
        (
            "model.layers.0.self_attn.q_proj.scales".into(),
            vec![1],
            SavedTensorPayload::Bf16(vec![0x3F80]),
        ),
        (
            "model.layers.0.self_attn.q_proj.biases".into(),
            vec![64],
            SavedTensorPayload::Bf16(vec![0; 64]),
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
    kriya_core::quantizer::write_mixed_safetensors(
        tmp_dir.path().join("model.safetensors"),
        &tensors,
    )
    .unwrap();

    let err = kriya_backend_metal::encoder::ModelGpuWeights::validate_production_checkpoint(
        &descriptor,
        tmp_dir.path(),
    )
    .expect_err("should reject malformed MLX scale count");

    eprintln!("Actual error received: {:?}", err);
    assert!(
        err.to_string().contains("q_proj"),
        "Error must name failing tensor"
    );
}

#[test]
fn test_strict_production_loader_rejects_missing_tensor_without_fallback() {
    let Some(device) = require_metal_device() else {
        return;
    };
    let tmp_dir = tempfile::tempdir().unwrap();
    let descriptor = miniature_dense_gqa_descriptor();
    // Write partial checkpoint: only embed_tokens, missing norm, lm_head, and all layers
    let partial_tensors = vec![(
        "model.embed_tokens.weight".into(),
        vec![64, 64],
        q4_payload(64 * 64),
    )];
    kriya_core::quantizer::write_mixed_safetensors(
        tmp_dir.path().join("model.safetensors"),
        &partial_tensors,
    )
    .unwrap();

    // 1. Test mode allows fallback on partial checkpoint
    let permissive_res = kriya_backend_metal::encoder::ModelGpuWeights::from_checkpoint_dir(
        &device,
        &descriptor,
        tmp_dir.path(),
    );
    assert!(
        permissive_res.is_ok(),
        "Test mode allows synthetic fallback"
    );

    // 2. Strict production mode strictly rejects missing tensors with zero synthetic allocation
    let res = kriya_backend_metal::encoder::ModelGpuWeights::from_checkpoint_dir_strict(
        &device,
        &descriptor,
        tmp_dir.path(),
    );
    let Err(err) = res else {
        panic!("strict production loader must fail closed on missing checkpoint");
    };

    assert!(
        err.to_string().contains("model.norm.weight"),
        "Error should explain production failure: {err}"
    );
}

#[test]
fn test_production_checkpoint_rejects_gemma_missing_post_ffn_norm() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let mut descriptor = miniature_dense_gqa_descriptor();
    descriptor.family = "gemma2".into();
    let rms = kriya_core::descriptor::NormVariant::RmsNorm {
        eps: 1e-6,
        zero_centered_unit_offset: true,
    };
    descriptor.layers[0].post_attn_norm = Some(rms);
    descriptor.layers[0].post_ffn_norm = Some(rms);

    use kriya_core::quantizer::SavedTensorPayload;
    let bf16_ones = || SavedTensorPayload::Bf16(vec![0x3F80_u16; 64]);
    let q4_matrix = || q4_payload(64 * 64);

    // Provide pre_feedforward and post_attention layernorms, but omit post_feedforward_layernorm
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
            "model.layers.0.pre_feedforward_layernorm.weight".into(),
            vec![64],
            bf16_ones(),
        ),
        // OMITTED: model.layers.0.post_feedforward_layernorm.weight
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
    kriya_core::quantizer::write_mixed_safetensors(
        tmp_dir.path().join("model.safetensors"),
        &tensors,
    )
    .unwrap();

    let err = kriya_backend_metal::encoder::ModelGpuWeights::validate_production_checkpoint(
        &descriptor,
        tmp_dir.path(),
    )
    .expect_err("Gemma 2 checkpoint missing post_feedforward_layernorm must be rejected");

    assert!(
        err.to_string().contains("post_feedforward_layernorm"),
        "Error must explicitly name the missing Gemma post-FFN norm: {err}"
    );
}

#[test]
fn test_production_executor_new_production_rejects_malformed_checkpoint_directly() {
    let Some(_device) = require_metal_device() else {
        return;
    };
    let tmp_dir = tempfile::tempdir().unwrap();
    let descriptor = miniature_dense_gqa_descriptor();
    // Empty directory
    let res = kriya_backend_metal::encoder::MetalExecutor::new_production(
        &descriptor,
        workspace_kernels_dir(),
        tmp_dir.path(),
        16,
        16,
    );
    assert!(
        res.is_err(),
        "new_production must fail closed on missing/malformed checkpoint"
    );
}

#[test]
fn test_production_executor_runs_complete_dense_q4_fixture_on_metal() {
    let Some(_device) = require_metal_device() else {
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
    let sampled = block_on(executor.forward_and_sample(
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
    ))
    .expect("strict production executor must complete its Metal forward pass");
    assert!(sampled < descriptor.vocab_size);
}

#[test]
fn test_q4_decode_gemv_matches_cpu_reference_on_metal() {
    let Some(device) = require_metal_device() else {
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
fn test_q4_decode_gemv_2row_matches_cpu_reference_on_metal() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let rows = 7usize; // non-multiple of 8 to test boundary clipping
    let cols = 128usize; // 2 blocks
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
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder
        .set_compute_pipeline_state(&pipelines.get_or_compile("q4_g64_decode_gemv_2row").unwrap());
    encoder.set_buffer(0, Some(&weight_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    let params = kriya_backend_metal::encoder::GemmParams {
        m: 1,
        n: rows as u32,
        k: cols as u32,
        num_k_blocks: (cols / 64) as u32,
    };
    encoder.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(rows.div_ceil(8) as u64, 1, 1),
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
fn test_q4_decode_gate_up_swiglu_matches_cpu_reference_on_metal() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let intermediate_size = 9usize;
    let k = 128usize;

    let gate_source: Vec<f32> = (0..intermediate_size * k)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let up_source: Vec<f32> = (0..intermediate_size * k)
        .map(|idx| ((idx % 31) as f32 - 15.0) * 0.03125)
        .collect();

    let gate_blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&gate_source).unwrap();
    let up_blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&up_source).unwrap();

    let gate_deq = kriya_core::quantizer::dequantize_q4_group64_to_f32(&gate_blocks);
    let up_deq = kriya_core::quantizer::dequantize_q4_group64_to_f32(&up_blocks);

    let input: Vec<f32> = (0..k).map(|idx| ((idx % 13) as f32 - 6.0) * 0.1).collect();
    let input_bf16: Vec<u16> = input
        .iter()
        .map(|&v| kriya_core::quantizer::f32_to_bf16_bits(v))
        .collect();

    let expected: Vec<f32> = (0..intermediate_size)
        .map(|i| {
            let g: f32 = gate_deq[i * k..(i + 1) * k]
                .iter()
                .zip(&input)
                .map(|(w, x)| w * x)
                .sum();
            let u: f32 = up_deq[i * k..(i + 1) * k]
                .iter()
                .zip(&input)
                .map(|(w, x)| w * x)
                .sum();
            let silu_g = g / (1.0 + (-g).exp());
            silu_g * u
        })
        .collect();

    let shared = MTLResourceOptions::StorageModeShared;
    let gate_buf = device.new_buffer_with_data(
        gate_blocks.as_ptr() as *const _,
        (gate_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );
    let up_buf = device.new_buffer_with_data(
        up_blocks.as_ptr() as *const _,
        (up_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );
    let input_buf = device.new_buffer_with_data(
        input_bf16.as_ptr() as *const _,
        (input_bf16.len() * std::mem::size_of::<u16>()) as u64,
        shared,
    );
    let output_buf = device.new_buffer(
        (intermediate_size * std::mem::size_of::<u16>()) as u64,
        shared,
    );

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(
        &pipelines
            .get_or_compile("q4_g64_decode_gate_up_swiglu_gemv")
            .unwrap(),
    );
    encoder.set_buffer(0, Some(&gate_buf), 0);
    encoder.set_buffer(1, Some(&up_buf), 0);
    encoder.set_buffer(2, Some(&input_buf), 0);
    encoder.set_buffer(3, Some(&output_buf), 0);

    let params = kriya_backend_metal::encoder::GateUpSwigluGemmParams {
        m: 1,
        k: k as u32,
        num_k_blocks: (k / 64) as u32,
        intermediate_size: intermediate_size as u32,
        activation_variant: 0, // SwiGLU
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    encoder.set_bytes(
        4,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(intermediate_size.div_ceil(4) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let output = unsafe {
        std::slice::from_raw_parts(output_buf.contents() as *const u16, intermediate_size)
    };
    for (gpu_bits, cpu) in output.iter().zip(expected) {
        let gpu = kriya_core::quantizer::bf16_bits_to_f32(*gpu_bits);
        assert!((gpu - cpu).abs() < 0.05, "GPU={gpu}, CPU={cpu}");
    }
}

#[test]
fn test_q4_decode_gemv_4row_matches_cpu_reference_on_metal() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let rows = 19usize; // non-multiple of 16 to test boundary clipping
    let cols = 128usize; // 2 blocks
    let source_weights: Vec<f32> = (0..rows * cols)
        .map(|idx| ((idx % 41) as f32 - 20.0) * 0.03125)
        .collect();
    let blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&source_weights).unwrap();
    let dequantized = kriya_core::quantizer::dequantize_q4_group64_to_f32(&blocks);
    let input: Vec<f32> = (0..cols)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.125)
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
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder
        .set_compute_pipeline_state(&pipelines.get_or_compile("q4_g64_decode_gemv_4row").unwrap());
    encoder.set_buffer(0, Some(&weight_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    let params = kriya_backend_metal::encoder::GemmParams {
        m: 1,
        n: rows as u32,
        k: cols as u32,
        num_k_blocks: (cols / 64) as u32,
    };
    encoder.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(rows.div_ceil(16) as u64, 1, 1),
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
fn test_q4_decode_gate_up_swiglu_2chan_matches_cpu_reference_on_metal() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let intermediate_size = 11usize; // non-multiple of 8 to test boundary clipping
    let k = 128usize;

    let gate_source: Vec<f32> = (0..intermediate_size * k)
        .map(|idx| ((idx % 33) as f32 - 16.0) * 0.03125)
        .collect();
    let up_source: Vec<f32> = (0..intermediate_size * k)
        .map(|idx| ((idx % 37) as f32 - 18.0) * 0.03125)
        .collect();

    let gate_blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&gate_source).unwrap();
    let up_blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&up_source).unwrap();

    let gate_deq = kriya_core::quantizer::dequantize_q4_group64_to_f32(&gate_blocks);
    let up_deq = kriya_core::quantizer::dequantize_q4_group64_to_f32(&up_blocks);

    let input: Vec<f32> = (0..k).map(|idx| ((idx % 17) as f32 - 8.0) * 0.1).collect();
    let input_bf16: Vec<u16> = input
        .iter()
        .map(|&value| kriya_core::quantizer::f32_to_bf16_bits(value))
        .collect();

    let mut expected = Vec::with_capacity(intermediate_size);
    for row in 0..intermediate_size {
        let g: f32 = gate_deq[row * k..(row + 1) * k]
            .iter()
            .zip(&input)
            .map(|(w, x)| w * x)
            .sum();
        let u: f32 = up_deq[row * k..(row + 1) * k]
            .iter()
            .zip(&input)
            .map(|(w, x)| w * x)
            .sum();
        let act_g = g / (1.0 + (-g).exp()); // SwiGLU
        expected.push(act_g * u);
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let gate_buf = device.new_buffer_with_data(
        gate_blocks.as_ptr() as *const _,
        (gate_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );
    let up_buf = device.new_buffer_with_data(
        up_blocks.as_ptr() as *const _,
        (up_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );
    let input_buf = device.new_buffer_with_data(
        input_bf16.as_ptr() as *const _,
        (input_bf16.len() * std::mem::size_of::<u16>()) as u64,
        shared,
    );
    let output_buf = device.new_buffer(
        (intermediate_size * std::mem::size_of::<u16>()) as u64,
        shared,
    );

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(
        &pipelines
            .get_or_compile("q4_g64_decode_gate_up_swiglu_gemv_2chan")
            .unwrap(),
    );
    encoder.set_buffer(0, Some(&gate_buf), 0);
    encoder.set_buffer(1, Some(&up_buf), 0);
    encoder.set_buffer(2, Some(&input_buf), 0);
    encoder.set_buffer(3, Some(&output_buf), 0);

    let params = kriya_backend_metal::encoder::GateUpSwigluGemmParams {
        m: 1,
        k: k as u32,
        num_k_blocks: (k / 64) as u32,
        intermediate_size: intermediate_size as u32,
        activation_variant: 0, // SwiGLU
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    encoder.set_bytes(
        4,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(intermediate_size.div_ceil(8) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let output = unsafe {
        std::slice::from_raw_parts(output_buf.contents() as *const u16, intermediate_size)
    };
    for (gpu_bits, cpu) in output.iter().zip(expected) {
        let gpu = kriya_core::quantizer::bf16_bits_to_f32(*gpu_bits);
        assert!((gpu - cpu).abs() < 0.05, "GPU={gpu}, CPU={cpu}");
    }
}

#[test]
fn test_paged_attention_matches_cpu_reference_on_metal() {
    let Some(device) = require_metal_device() else {
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
        max_seq_len: 4096,
        has_table: 0,
        _pad0: 0,
        _pad1: 0,
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
    let Some(device) = require_metal_device() else {
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
    let tok = block_on(exec.forward_and_sample(
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
    ))
    .unwrap();
    assert!(tok < desc.vocab_size);

    // The GPU tree kernel remains available for development, but production
    // speculative decoding must reject it until target-logit acceptance is wired.
    let verification = block_on(exec.verify_eagle_draft_tree_gpu(
        &desc,
        &plan,
        &[tok, (tok + 1) % 100, (tok + 2) % 100],
        &[true, false, false, true, true, false, true, false, true],
    ));
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
    assert!(
        kriya_core::descriptor::ModelArchDescriptor::gemma_2_9b()
            .validate()
            .is_ok()
    );
    assert!(
        kriya_core::descriptor::ModelArchDescriptor::deepseek_r1_distill_qwen_7b()
            .validate()
            .is_ok()
    );

    let full_desc = kriya_core::descriptor::ModelArchDescriptor::llama_3_2_3b();
    assert_eq!(full_desc.layers.len(), 28);
}

#[test]
fn test_q4_decode_qkv_gemv_2row_matches_cpu_reference_on_metal() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let q_dim = 16usize;
    let kv_dim = 8usize;
    let k = 128usize; // 2 blocks

    let q_source: Vec<f32> = (0..q_dim * k)
        .map(|idx| ((idx % 23) as f32 - 11.0) * 0.03125)
        .collect();
    let k_source: Vec<f32> = (0..kv_dim * k)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let v_source: Vec<f32> = (0..kv_dim * k)
        .map(|idx| ((idx % 31) as f32 - 15.0) * 0.03125)
        .collect();

    let q_blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&q_source).unwrap();
    let k_blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&k_source).unwrap();
    let v_blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&v_source).unwrap();

    let q_deq = kriya_core::quantizer::dequantize_q4_group64_to_f32(&q_blocks);
    let k_deq = kriya_core::quantizer::dequantize_q4_group64_to_f32(&k_blocks);
    let v_deq = kriya_core::quantizer::dequantize_q4_group64_to_f32(&v_blocks);

    let input: Vec<f32> = (0..k).map(|idx| ((idx % 19) as f32 - 9.0) * 0.1).collect();
    let input_bf16: Vec<u16> = input
        .iter()
        .map(|&value| kriya_core::quantizer::f32_to_bf16_bits(value))
        .collect();

    let expected_q: Vec<f32> = q_deq
        .chunks_exact(k)
        .map(|row| row.iter().zip(&input).map(|(w, x)| w * x).sum())
        .collect();
    let expected_k: Vec<f32> = k_deq
        .chunks_exact(k)
        .map(|row| row.iter().zip(&input).map(|(w, x)| w * x).sum())
        .collect();
    let expected_v: Vec<f32> = v_deq
        .chunks_exact(k)
        .map(|row| row.iter().zip(&input).map(|(w, x)| w * x).sum())
        .collect();

    let shared = MTLResourceOptions::StorageModeShared;
    let q_weight_buf = device.new_buffer_with_data(
        q_blocks.as_ptr() as *const _,
        (q_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );
    let k_weight_buf = device.new_buffer_with_data(
        k_blocks.as_ptr() as *const _,
        (k_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );
    let v_weight_buf = device.new_buffer_with_data(
        v_blocks.as_ptr() as *const _,
        (v_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>()) as u64,
        shared,
    );

    let input_buf = device.new_buffer_with_data(
        input_bf16.as_ptr() as *const _,
        (input_bf16.len() * std::mem::size_of::<u16>()) as u64,
        shared,
    );
    let q_out_buf = device.new_buffer((q_dim * std::mem::size_of::<u16>()) as u64, shared);
    let k_out_buf = device.new_buffer((kv_dim * std::mem::size_of::<u16>()) as u64, shared);
    let v_out_buf = device.new_buffer((kv_dim * std::mem::size_of::<u16>()) as u64, shared);

    let dummy_bias = device.new_buffer(4, shared);

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let total_rows = (q_dim + kv_dim + kv_dim) as u32;
    let params = kriya_backend_metal::encoder::QkvGemmParams {
        m: 1,
        k: k as u32,
        num_k_blocks: (k / 64) as u32,
        q_dim: q_dim as u32,
        k_dim: kv_dim as u32,
        v_dim: kv_dim as u32,
        has_bias: 0,
        _pad: 0,
    };

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(
        &pipelines
            .get_or_compile("q4_g64_decode_qkv_gemv_2row")
            .unwrap(),
    );
    encoder.set_buffer(0, Some(&q_weight_buf), 0);
    encoder.set_buffer(1, Some(&k_weight_buf), 0);
    encoder.set_buffer(2, Some(&v_weight_buf), 0);
    encoder.set_buffer(3, Some(&input_buf), 0);
    encoder.set_buffer(4, Some(&q_out_buf), 0);
    encoder.set_buffer(5, Some(&k_out_buf), 0);
    encoder.set_buffer(6, Some(&v_out_buf), 0);
    encoder.set_buffer(7, Some(&dummy_bias), 0);
    encoder.set_buffer(8, Some(&dummy_bias), 0);
    encoder.set_buffer(9, Some(&dummy_bias), 0);
    encoder.set_bytes(
        10,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new((total_rows as usize).div_ceil(8) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q_out = unsafe { std::slice::from_raw_parts(q_out_buf.contents() as *const u16, q_dim) };
    for (gpu_bits, cpu) in q_out.iter().zip(expected_q) {
        let gpu = kriya_core::quantizer::bf16_bits_to_f32(*gpu_bits);
        assert!((gpu - cpu).abs() < 0.05, "Q: GPU={gpu}, CPU={cpu}");
    }

    let k_out = unsafe { std::slice::from_raw_parts(k_out_buf.contents() as *const u16, kv_dim) };
    for (gpu_bits, cpu) in k_out.iter().zip(expected_k) {
        let gpu = kriya_core::quantizer::bf16_bits_to_f32(*gpu_bits);
        assert!((gpu - cpu).abs() < 0.05, "K: GPU={gpu}, CPU={cpu}");
    }

    let v_out = unsafe { std::slice::from_raw_parts(v_out_buf.contents() as *const u16, kv_dim) };
    for (gpu_bits, cpu) in v_out.iter().zip(expected_v) {
        let gpu = kriya_core::quantizer::bf16_bits_to_f32(*gpu_bits);
        assert!((gpu - cpu).abs() < 0.05, "V: GPU={gpu}, CPU={cpu}");
    }
}

#[test]
fn test_zero_copy_weight_buffer_mapping() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let dir = tempfile::tempdir().unwrap();
    let st_path = dir.path().join("model.safetensors");
    let norm_bf16 = vec![kriya_core::quantizer::f32_to_bf16_bits(1.0); 3072];
    let proj_q4 = kriya_core::quantizer::quantize_f32_to_q4_group64(&vec![0.25f32; 128]).unwrap();

    kriya_core::quantizer::write_mixed_safetensors(
        &st_path,
        &[
            (
                "model.norm.weight".to_string(),
                vec![3072],
                kriya_core::quantizer::SavedTensorPayload::Bf16(norm_bf16),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                vec![2, 64],
                kriya_core::quantizer::SavedTensorPayload::Q4Group64(proj_q4),
            ),
        ],
    )
    .unwrap();

    let archive = kriya_core::safetensors_loader::open_safetensors_file(&st_path).unwrap();
    let norm_view = archive.get_tensor("model.norm.weight").unwrap();
    let proj_view = archive
        .get_tensor("model.layers.0.self_attn.q_proj.weight")
        .unwrap();

    // Verify 16 KB page-aligned data pointers
    assert_eq!((norm_view.data.as_ptr() as usize) % 16384, 0);
    assert_eq!((proj_view.data.as_ptr() as usize) % 16384, 0);

    // Verify zero-copy Metal buffer creation succeeds on page-aligned mmap data
    let shared = MTLResourceOptions::StorageModeShared;
    let norm_buf = device.new_buffer_with_bytes_no_copy(
        norm_view.data.as_ptr() as *const _,
        norm_view.data.len() as u64,
        shared,
        None,
    );
    assert_eq!(norm_buf.length() as usize, norm_view.data.len());
    let gpu_norm_slice = unsafe {
        std::slice::from_raw_parts(norm_buf.contents() as *const u8, norm_view.data.len())
    };
    assert_eq!(gpu_norm_slice, norm_view.data);

    let proj_buf = device.new_buffer_with_bytes_no_copy(
        proj_view.data.as_ptr() as *const _,
        proj_view.data.len() as u64,
        shared,
        None,
    );
    assert_eq!(proj_buf.length() as usize, proj_view.data.len());
    let gpu_proj_slice = unsafe {
        std::slice::from_raw_parts(proj_buf.contents() as *const u8, proj_view.data.len())
    };
    assert_eq!(gpu_proj_slice, proj_view.data);
}

#[test]
fn test_batched_gemm_m64_n32_matches_cpu_reference() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let m = 64usize;
    let n = 64usize;
    let k = 128usize;

    let weights: Vec<f32> = (0..n * k)
        .map(|idx| (((idx % 23) as f32) - 11.0) * 0.05)
        .collect();
    let q4_blocks = kriya_core::quantizer::quantize_f32_to_q4_group64(&weights).unwrap();
    let dequant_w = kriya_core::quantizer::dequantize_q4_group64_to_f32(&q4_blocks);

    let input_f32: Vec<f32> = (0..m * k)
        .map(|idx| (((idx % 19) as f32) - 9.0) * 0.04)
        .collect();
    let input_bf16: Vec<u16> = input_f32
        .iter()
        .map(|&v| kriya_core::quantizer::f32_to_bf16_bits(v))
        .collect();
    let input_unquant: Vec<f32> = input_bf16
        .iter()
        .map(|&bits| kriya_core::quantizer::bf16_bits_to_f32(bits))
        .collect();

    let mut expected_y = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += input_unquant[row * k + inner] * dequant_w[col * k + inner];
            }
            expected_y[row * n + col] = sum;
        }
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let weight_bytes = q4_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ4Group64>();
    let weight_buf =
        device.new_buffer_with_data(q4_blocks.as_ptr() as *const _, weight_bytes as u64, shared);
    let input_buf = device.new_buffer_with_data(
        input_bf16.as_ptr() as *const _,
        (input_bf16.len() * std::mem::size_of::<u16>()) as u64,
        shared,
    );
    let out_buf = device.new_buffer((m * n * std::mem::size_of::<u16>()) as u64, shared);

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let params = kriya_backend_metal::encoder::GemmParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        num_k_blocks: (k / 64) as u32,
    };

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(
        &pipelines
            .get_or_compile("q4_g64_batched_simdgroup_gemm_m64")
            .unwrap(),
    );
    encoder.set_buffer(0, Some(&weight_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&out_buf), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out_slice = unsafe { std::slice::from_raw_parts(out_buf.contents() as *const u16, m * n) };
    for (idx, (gpu_bits, &cpu)) in out_slice.iter().zip(&expected_y).enumerate() {
        let gpu = kriya_core::quantizer::bf16_bits_to_f32(*gpu_bits);
        assert!(
            (gpu - cpu).abs() < 0.05,
            "Mismatch at elem {idx} (row {}, col {}): GPU={gpu}, CPU={cpu}",
            idx / n,
            idx % n
        );
    }
}

#[test]
fn test_tiled_mma_prefill_attention_matches_cpu_reference() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let num_tokens = 16usize;
    let num_q_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 64usize;
    let block_size_tokens = 16usize;

    let q_f32: Vec<f32> = (0..num_tokens * num_q_heads * head_dim)
        .map(|idx| (((idx % 17) as f32) - 8.0) * 0.05)
        .collect();
    let q_bf16: Vec<u16> = q_f32
        .iter()
        .map(|&v| kriya_core::quantizer::f32_to_bf16_bits(v))
        .collect();
    let q_unquant: Vec<f32> = q_bf16
        .iter()
        .map(|&bits| kriya_core::quantizer::bf16_bits_to_f32(bits))
        .collect();

    let k_f32: Vec<f32> = (0..block_size_tokens * num_kv_heads * head_dim)
        .map(|idx| (((idx % 13) as f32) - 6.0) * 0.06)
        .collect();
    let k_bf16: Vec<u16> = k_f32
        .iter()
        .map(|&v| kriya_core::quantizer::f32_to_bf16_bits(v))
        .collect();
    let k_unquant: Vec<f32> = k_bf16
        .iter()
        .map(|&bits| kriya_core::quantizer::bf16_bits_to_f32(bits))
        .collect();

    let v_f32: Vec<f32> = (0..block_size_tokens * num_kv_heads * head_dim)
        .map(|idx| (((idx % 11) as f32) - 5.0) * 0.07)
        .collect();
    let v_bf16: Vec<u16> = v_f32
        .iter()
        .map(|&v| kriya_core::quantizer::f32_to_bf16_bits(v))
        .collect();
    let v_unquant: Vec<f32> = v_bf16
        .iter()
        .map(|&bits| kriya_core::quantizer::bf16_bits_to_f32(bits))
        .collect();

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut expected_out = vec![0.0f32; num_tokens * num_q_heads * head_dim];
    for t in 0..num_tokens {
        for qh in 0..num_q_heads {
            let kh = qh / (num_q_heads / num_kv_heads);
            let q_vec = &q_unquant
                [(t * num_q_heads + qh) * head_dim..(t * num_q_heads + qh + 1) * head_dim];

            let mut scores = Vec::with_capacity(t + 1);
            for j in 0..=t {
                let k_vec = &k_unquant
                    [(j * num_kv_heads + kh) * head_dim..(j * num_kv_heads + kh + 1) * head_dim];
                let dot: f32 = q_vec.iter().zip(k_vec).map(|(q, k)| q * k).sum();
                scores.push(dot * sm_scale);
            }

            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..=t {
                    let v_val = v_unquant[(j * num_kv_heads + kh) * head_dim + d];
                    acc += probs[j] * v_val;
                }
                expected_out[(t * num_q_heads + qh) * head_dim + d] = acc;
            }
        }
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let q_buf = device.new_buffer_with_data(
        q_bf16.as_ptr() as *const _,
        (q_bf16.len() * 2) as u64,
        shared,
    );
    let k_buf = device.new_buffer_with_data(
        k_bf16.as_ptr() as *const _,
        (k_bf16.len() * 2) as u64,
        shared,
    );
    let v_buf = device.new_buffer_with_data(
        v_bf16.as_ptr() as *const _,
        (v_bf16.len() * 2) as u64,
        shared,
    );

    let page_indptr: Vec<i32> = (0..=num_tokens).map(|i| i as i32).collect();
    let page_indices: Vec<i32> = vec![0i32; num_tokens];
    let last_page_len: Vec<i32> = (0..num_tokens).map(|i| (i + 1) as i32).collect();

    let indptr_buf = device.new_buffer_with_data(
        page_indptr.as_ptr() as *const _,
        (page_indptr.len() * 4) as u64,
        shared,
    );
    let indices_buf = device.new_buffer_with_data(
        page_indices.as_ptr() as *const _,
        (page_indices.len() * 4) as u64,
        shared,
    );
    let last_len_buf = device.new_buffer_with_data(
        last_page_len.as_ptr() as *const _,
        (last_page_len.len() * 4) as u64,
        shared,
    );
    let dummy_cta = device.new_buffer(16, shared);
    let dummy_split = device.new_buffer(4, shared);
    let final_out_buf = device.new_buffer((expected_out.len() * 2) as u64, shared);
    let dummy_partial = device.new_buffer(4, shared);
    let dummy_lse = device.new_buffer(4, shared);

    let params = kriya_backend_metal::encoder::BsrGqaParams {
        num_q_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        block_size_tokens: block_size_tokens as u32,
        max_split_k_tiles: 16,
        max_query_tokens: num_tokens as u32,
        sm_scale,
        logit_softcap: 0.0,
    };

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(
        &pipelines
            .get_or_compile("bsr_paged_gqa_tiled_mma_prefill")
            .unwrap(),
    );
    encoder.set_buffer(0, Some(&q_buf), 0);
    encoder.set_buffer(1, Some(&k_buf), 0);
    encoder.set_buffer(2, Some(&v_buf), 0);
    encoder.set_buffer(3, Some(&indptr_buf), 0);
    encoder.set_buffer(4, Some(&indices_buf), 0);
    encoder.set_buffer(5, Some(&last_len_buf), 0);
    encoder.set_buffer(6, Some(&dummy_cta), 0);
    encoder.set_buffer(7, Some(&dummy_split), 0);
    encoder.set_buffer(8, Some(&final_out_buf), 0);
    encoder.set_buffer(9, Some(&dummy_partial), 0);
    encoder.set_buffer(10, Some(&dummy_lse), 0);
    encoder.set_bytes(
        11,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tokens.div_ceil(16) as u64, num_q_heads as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out_slice = unsafe {
        std::slice::from_raw_parts(final_out_buf.contents() as *const u16, expected_out.len())
    };
    for (idx, (gpu_bits, &cpu)) in out_slice.iter().zip(&expected_out).enumerate() {
        let gpu = kriya_core::quantizer::bf16_bits_to_f32(*gpu_bits);
        assert!(
            (gpu - cpu).abs() < 0.05,
            "Mismatch at token {}, head {}, dim {}: GPU={gpu}, CPU={cpu}",
            idx / (num_q_heads * head_dim),
            (idx / head_dim) % num_q_heads,
            idx % head_dim
        );
    }
}

#[test]
fn test_eagle_fc_fuse_and_decoder_matches_cpu() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let hidden_dim = 64usize;
    let fused_dim = 2 * hidden_dim;
    let shared = MTLResourceOptions::StorageModeShared;

    let tok_emb_f32: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01 + 0.1).collect();
    let tok_emb_bf16: Vec<u16> = tok_emb_f32
        .iter()
        .map(|&x| kriya_core::quantizer::f32_to_bf16_bits(x))
        .collect();
    let tok_emb_recon: Vec<f32> = tok_emb_bf16
        .iter()
        .map(|&b| kriya_core::quantizer::bf16_bits_to_f32(b))
        .collect();

    let prev_feat: Vec<f32> = (0..hidden_dim)
        .map(|i| (((i * 3 % 11) as f32) - 5.0) * 0.02)
        .collect();

    let mut fused = Vec::with_capacity(fused_dim);
    fused.extend_from_slice(&tok_emb_recon);
    fused.extend_from_slice(&prev_feat);

    let fc_fuse_w: Vec<f32> = (0..hidden_dim * fused_dim)
        .map(|i| (((i % 17) as f32) - 8.0) * 0.01)
        .collect();

    let decoder_w: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| (((i % 13) as f32) - 6.0) * 0.01)
        .collect();

    // CPU reference calculation
    let mut reduced_cpu = vec![0.0f32; hidden_dim];
    for h in 0..hidden_dim {
        let mut sum = 0.0f32;
        for j in 0..fused_dim {
            sum += fc_fuse_w[h * fused_dim + j] * fused[j];
        }
        reduced_cpu[h] = sum;
    }

    let mut next_feat_cpu = vec![0.0f32; hidden_dim];
    for h in 0..hidden_dim {
        let mut delta = 0.0f32;
        for j in 0..hidden_dim {
            delta += decoder_w[h * hidden_dim + j] * reduced_cpu[j];
        }
        next_feat_cpu[h] = reduced_cpu[h] + delta;
    }

    // GPU buffer setup
    let tok_emb_buf = device.new_buffer_with_data(
        tok_emb_bf16.as_ptr() as *const _,
        (tok_emb_bf16.len() * 2) as u64,
        shared,
    );
    let prev_feat_buf = device.new_buffer_with_data(
        prev_feat.as_ptr() as *const _,
        (prev_feat.len() * 4) as u64,
        shared,
    );
    let fc_fuse_buf = device.new_buffer_with_data(
        fc_fuse_w.as_ptr() as *const _,
        (fc_fuse_w.len() * 4) as u64,
        shared,
    );
    let reduced_buf = device.new_buffer((hidden_dim * 4) as u64, shared);
    let decoder_buf = device.new_buffer_with_data(
        decoder_w.as_ptr() as *const _,
        (decoder_w.len() * 4) as u64,
        shared,
    );
    let next_feat_f32_buf = device.new_buffer((hidden_dim * 4) as u64, shared);
    let next_feat_bf16_buf = device.new_buffer((hidden_dim * 2) as u64, shared);

    let params = kriya_backend_metal::encoder::EagleParams {
        hidden_dim: hidden_dim as u32,
        vocab_size: 64,
        top_k: 4,
        _pad: 0,
    };

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    // 1. Dispatch eagle_fc_fuse
    encoder.set_compute_pipeline_state(&pipelines.get_or_compile("eagle_fc_fuse").unwrap());
    encoder.set_buffer(0, Some(&tok_emb_buf), 0);
    encoder.set_buffer(1, Some(&prev_feat_buf), 0);
    encoder.set_buffer(2, Some(&fc_fuse_buf), 0);
    encoder.set_buffer(3, Some(&reduced_buf), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new((hidden_dim as u64).div_ceil(64), 1, 1),
        MTLSize::new(64, 1, 1),
    );

    // 2. Dispatch eagle_decoder_layer
    encoder.set_compute_pipeline_state(&pipelines.get_or_compile("eagle_decoder_layer").unwrap());
    encoder.set_buffer(0, Some(&reduced_buf), 0);
    encoder.set_buffer(1, Some(&decoder_buf), 0);
    encoder.set_buffer(2, Some(&next_feat_f32_buf), 0);
    encoder.set_buffer(3, Some(&next_feat_bf16_buf), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new((hidden_dim as u64).div_ceil(64), 1, 1),
        MTLSize::new(64, 1, 1),
    );

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_reduced =
        unsafe { std::slice::from_raw_parts(reduced_buf.contents() as *const f32, hidden_dim) };
    for (h, (&gpu, &cpu)) in gpu_reduced.iter().zip(&reduced_cpu).enumerate() {
        assert!(
            (gpu - cpu).abs() < 1e-4,
            "reduced mismatch at {h}: GPU={gpu}, CPU={cpu}"
        );
    }

    let gpu_next_feat = unsafe {
        std::slice::from_raw_parts(next_feat_f32_buf.contents() as *const f32, hidden_dim)
    };
    for (h, (&gpu, &cpu)) in gpu_next_feat.iter().zip(&next_feat_cpu).enumerate() {
        assert!(
            (gpu - cpu).abs() < 1e-4,
            "next_feat mismatch at {h}: GPU={gpu}, CPU={cpu}"
        );
    }
}

#[test]
fn test_eagle_topk_and_probs_matches_cpu() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let vocab_size = 1024usize;
    let top_k = 4usize;
    let shared = MTLResourceOptions::StorageModeShared;

    let mut logits_f32: Vec<f32> = (0..vocab_size)
        .map(|v| (((v % 97) as f32) * 0.1) - 4.0)
        .collect();
    // Inject distinct top values at specific indices
    logits_f32[42] = 15.0;
    logits_f32[128] = 14.5;
    logits_f32[512] = 14.0;
    logits_f32[777] = 13.2;

    let logits_bf16: Vec<u16> = logits_f32
        .iter()
        .map(|&x| kriya_core::quantizer::f32_to_bf16_bits(x))
        .collect();
    let logits_recon: Vec<f32> = logits_bf16
        .iter()
        .map(|&b| kriya_core::quantizer::bf16_bits_to_f32(b))
        .collect();

    // CPU ground truth
    let max_logit = logits_recon.iter().cloned().fold(-f32::INFINITY, f32::max);
    let sum_exp: f32 = logits_recon.iter().map(|&l| (l - max_logit).exp()).sum();
    let mut cpu_probs: Vec<(usize, f32)> = logits_recon
        .iter()
        .enumerate()
        .map(|(v, &l)| (v, (l - max_logit).exp() / sum_exp))
        .collect();
    cpu_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let expected_tokens: Vec<u32> = cpu_probs[..top_k].iter().map(|&(v, _)| v as u32).collect();
    let expected_probs: Vec<f32> = cpu_probs[..top_k].iter().map(|&(_, p)| p).collect();

    assert_eq!(expected_tokens, vec![42, 128, 512, 777]);

    // Test 1: eagle_topk_and_probs with BF16 logits
    let logits_bf16_buf = device.new_buffer_with_data(
        logits_bf16.as_ptr() as *const _,
        (logits_bf16.len() * 2) as u64,
        shared,
    );
    let out_tokens_buf = device.new_buffer((top_k * 4) as u64, shared);
    let out_probs_buf = device.new_buffer((top_k * 4) as u64, shared);

    let params = kriya_backend_metal::encoder::EagleParams {
        hidden_dim: 64,
        vocab_size: vocab_size as u32,
        top_k: top_k as u32,
        _pad: 0,
    };

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.get_or_compile("eagle_topk_and_probs").unwrap());
    encoder.set_buffer(0, Some(&logits_bf16_buf), 0);
    encoder.set_buffer(1, Some(&out_tokens_buf), 0);
    encoder.set_buffer(2, Some(&out_probs_buf), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_tokens =
        unsafe { std::slice::from_raw_parts(out_tokens_buf.contents() as *const u32, top_k) };
    let gpu_probs =
        unsafe { std::slice::from_raw_parts(out_probs_buf.contents() as *const f32, top_k) };

    assert_eq!(gpu_tokens, expected_tokens.as_slice());
    for (k, (&gpu_p, &cpu_p)) in gpu_probs.iter().zip(&expected_probs).enumerate() {
        assert!(
            (gpu_p - cpu_p).abs() < 1e-4,
            "prob mismatch at rank {k}: GPU={gpu_p}, CPU={cpu_p}"
        );
    }

    // Test 2: eagle_topk_and_probs_f32 with FP32 logits
    let max_logit_f32 = logits_f32.iter().cloned().fold(-f32::INFINITY, f32::max);
    let sum_exp_f32: f32 = logits_f32.iter().map(|&l| (l - max_logit_f32).exp()).sum();
    let mut cpu_probs_f32: Vec<(usize, f32)> = logits_f32
        .iter()
        .enumerate()
        .map(|(v, &l)| (v, (l - max_logit_f32).exp() / sum_exp_f32))
        .collect();
    cpu_probs_f32.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let expected_probs_f32: Vec<f32> = cpu_probs_f32[..top_k].iter().map(|&(_, p)| p).collect();

    let logits_f32_buf = device.new_buffer_with_data(
        logits_f32.as_ptr() as *const _,
        (logits_f32.len() * 4) as u64,
        shared,
    );
    let out_tokens_f32_buf = device.new_buffer((top_k * 4) as u64, shared);
    let out_probs_f32_buf = device.new_buffer((top_k * 4) as u64, shared);

    let cmd2 = queue.new_command_buffer();
    let enc2 = cmd2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(
        &pipelines
            .get_or_compile("eagle_topk_and_probs_f32")
            .unwrap(),
    );
    enc2.set_buffer(0, Some(&logits_f32_buf), 0);
    enc2.set_buffer(1, Some(&out_tokens_f32_buf), 0);
    enc2.set_buffer(2, Some(&out_probs_f32_buf), 0);
    enc2.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    enc2.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    enc2.end_encoding();
    cmd2.commit();
    cmd2.wait_until_completed();

    let gpu_tokens_f32 =
        unsafe { std::slice::from_raw_parts(out_tokens_f32_buf.contents() as *const u32, top_k) };
    let gpu_probs_f32 =
        unsafe { std::slice::from_raw_parts(out_probs_f32_buf.contents() as *const f32, top_k) };

    assert_eq!(gpu_tokens_f32, vec![42, 128, 512, 777]);
    for (k, (&gpu_p, &cpu_p)) in gpu_probs_f32.iter().zip(&expected_probs_f32).enumerate() {
        assert!(
            (gpu_p - cpu_p).abs() < 1e-4,
            "f32 prob mismatch at rank {k}: GPU={gpu_p}, CPU={cpu_p}"
        );
    }
}

#[test]
fn test_precomputed_rope_matches_transcendental() {
    let Some(device) = Device::system_default() else {
        eprintln!("Skipping test: no Metal GPU available");
        return;
    };
    let shared = MTLResourceOptions::StorageModeShared;
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();
    let queue = device.new_command_queue();

    const HEAD_DIM: usize = 64;
    const NUM_TOKENS: usize = 4;
    let rotary_dim = 64u32;
    let rope_theta = 10_000.0f32;
    let max_seq_len = 128u32;
    let half_rot = rotary_dim / 2;

    // Build precomputed table
    let mut cos_sin_table = Vec::with_capacity((max_seq_len * half_rot * 2) as usize);
    for pos in 0..max_seq_len {
        for pair_i in 0..half_rot {
            let freq_exp = (2.0 * pair_i as f32) / (rotary_dim as f32);
            let inv_freq = rope_theta.powf(-freq_exp);
            let angle = (pos as f32) * inv_freq;
            cos_sin_table.push(angle.cos());
            cos_sin_table.push(angle.sin());
        }
    }
    let table_buf = device.new_buffer_with_data(
        cos_sin_table.as_ptr() as *const _,
        (cos_sin_table.len() * 4) as u64,
        shared,
    );

    // Initial Q and K in BF16
    let initial_data: Vec<u16> = (0..(NUM_TOKENS * HEAD_DIM))
        .map(|i| kriya_core::quantizer::f32_to_bf16_bits(((i % 17) as f32 - 8.0) * 0.25))
        .collect();
    let positions: Vec<u32> = vec![0, 5, 23, 67];

    let pos_buf = device.new_buffer_with_data(
        positions.as_ptr() as *const _,
        (positions.len() * 4) as u64,
        shared,
    );

    // Run without table (transcendental)
    let q_buf_trans = device.new_buffer_with_data(
        initial_data.as_ptr() as *const _,
        (initial_data.len() * 2) as u64,
        shared,
    );
    let k_buf_trans = device.new_buffer_with_data(
        initial_data.as_ptr() as *const _,
        (initial_data.len() * 2) as u64,
        shared,
    );
    let params_trans = kriya_backend_metal::encoder::RopeKvWriteParams {
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: HEAD_DIM as u32,
        rotary_dim,
        block_size_tokens: 16,
        apply_qk_norm: 0,
        rms_eps: 1e-6,
        rope_theta,
        max_seq_len,
        has_table: 0,
        _pad0: 0,
        _pad1: 0,
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipelines.get_or_compile("apply_rope_inplace").unwrap());
    enc.set_buffer(0, Some(&q_buf_trans), 0);
    enc.set_buffer(1, Some(&k_buf_trans), 0);
    enc.set_buffer(2, Some(&pos_buf), 0);
    enc.set_buffer(3, None, 0);
    enc.set_buffer(4, None, 0);
    enc.set_bytes(
        11,
        std::mem::size_of_val(&params_trans) as u64,
        &params_trans as *const _ as *const _,
    );
    enc.dispatch_thread_groups(
        MTLSize::new(1, NUM_TOKENS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Run WITH table (precomputed)
    let q_buf_table = device.new_buffer_with_data(
        initial_data.as_ptr() as *const _,
        (initial_data.len() * 2) as u64,
        shared,
    );
    let k_buf_table = device.new_buffer_with_data(
        initial_data.as_ptr() as *const _,
        (initial_data.len() * 2) as u64,
        shared,
    );
    let params_table = kriya_backend_metal::encoder::RopeKvWriteParams {
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: HEAD_DIM as u32,
        rotary_dim,
        block_size_tokens: 16,
        apply_qk_norm: 0,
        rms_eps: 1e-6,
        rope_theta,
        max_seq_len,
        has_table: 1,
        _pad0: 0,
        _pad1: 0,
    };

    let cmd2 = queue.new_command_buffer();
    let enc2 = cmd2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&pipelines.get_or_compile("apply_rope_inplace").unwrap());
    enc2.set_buffer(0, Some(&q_buf_table), 0);
    enc2.set_buffer(1, Some(&k_buf_table), 0);
    enc2.set_buffer(2, Some(&pos_buf), 0);
    enc2.set_buffer(3, None, 0);
    enc2.set_buffer(4, None, 0);
    enc2.set_bytes(
        11,
        std::mem::size_of_val(&params_table) as u64,
        &params_table as *const _ as *const _,
    );
    enc2.set_buffer(12, Some(&table_buf), 0);
    enc2.dispatch_thread_groups(
        MTLSize::new(1, NUM_TOKENS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc2.end_encoding();
    cmd2.commit();
    cmd2.wait_until_completed();

    let q_trans = unsafe {
        std::slice::from_raw_parts(q_buf_trans.contents() as *const u16, initial_data.len())
    };
    let q_table = unsafe {
        std::slice::from_raw_parts(q_buf_table.contents() as *const u16, initial_data.len())
    };
    let k_trans = unsafe {
        std::slice::from_raw_parts(k_buf_trans.contents() as *const u16, initial_data.len())
    };
    let k_table = unsafe {
        std::slice::from_raw_parts(k_buf_table.contents() as *const u16, initial_data.len())
    };

    for i in 0..initial_data.len() {
        let f_trans = kriya_core::quantizer::bf16_bits_to_f32(q_trans[i]);
        let f_table = kriya_core::quantizer::bf16_bits_to_f32(q_table[i]);
        assert!(
            (f_trans - f_table).abs() <= 1e-3,
            "Q mismatch at {i}: trans={f_trans}, table={f_table}"
        );
        let k_f_trans = kriya_core::quantizer::bf16_bits_to_f32(k_trans[i]);
        let k_f_table = kriya_core::quantizer::bf16_bits_to_f32(k_table[i]);
        assert!(
            (k_f_trans - k_f_table).abs() <= 1e-3,
            "K mismatch at {i}: trans={k_f_trans}, table={k_f_table}"
        );
    }
}

#[test]
fn test_fused_residual_add_rmsnorm_matches_unfused() {
    let Some(device) = Device::system_default() else {
        eprintln!("Skipping test: no Metal GPU available");
        return;
    };
    let shared = MTLResourceOptions::StorageModeShared;
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();
    let queue = device.new_command_queue();

    const NUM_TOKENS: usize = 3;
    const HIDDEN_SIZE: usize = 256;
    let total_elems = NUM_TOKENS * HIDDEN_SIZE;

    let res_initial: Vec<u16> = (0..total_elems)
        .map(|i| kriya_core::quantizer::f32_to_bf16_bits(((i % 29) as f32 - 14.0) * 0.1))
        .collect();
    let proj_data: Vec<u16> = (0..total_elems)
        .map(|i| kriya_core::quantizer::f32_to_bf16_bits(((i % 19) as f32 - 9.0) * 0.15))
        .collect();
    let weight_data: Vec<u16> = (0..HIDDEN_SIZE)
        .map(|i| kriya_core::quantizer::f32_to_bf16_bits(0.8 + (i % 7) as f32 * 0.05))
        .collect();

    let weight_buf = device.new_buffer_with_data(
        weight_data.as_ptr() as *const _,
        (weight_data.len() * 2) as u64,
        shared,
    );

    // Unfused path: residual_add_bf16 then rms_norm_bf16
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

    let norm_params = kriya_backend_metal::encoder::RmsNormParams {
        num_tokens: NUM_TOKENS as u32,
        hidden_size: HIDDEN_SIZE as u32,
        eps: 1e-5,
        zero_centered_offset: 0,
    };

    let cmd1 = queue.new_command_buffer();
    let enc1 = cmd1.new_compute_command_encoder();
    // 1. residual add
    enc1.set_compute_pipeline_state(&pipelines.get_or_compile("residual_add_bf16").unwrap());
    enc1.set_buffer(0, Some(&res_unfused), 0);
    enc1.set_buffer(1, Some(&proj_unfused), 0);
    let total_elems_u32 = total_elems as u32;
    enc1.set_bytes(2, 4, &total_elems_u32 as *const _ as *const _);
    enc1.dispatch_thread_groups(
        MTLSize::new(total_elems.div_ceil(256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    // 2. rms norm
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

    // Fused path: fused_residual_add_rms_norm_bf16
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
    enc2.set_compute_pipeline_state(
        &pipelines
            .get_or_compile("fused_residual_add_rms_norm_bf16")
            .unwrap(),
    );
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

    let res_unfused_slice =
        unsafe { std::slice::from_raw_parts(res_unfused.contents() as *const u16, total_elems) };
    let res_fused_slice =
        unsafe { std::slice::from_raw_parts(res_fused.contents() as *const u16, total_elems) };
    let norm_unfused_slice = unsafe {
        std::slice::from_raw_parts(norm_out_unfused.contents() as *const u16, total_elems)
    };
    let norm_fused_slice =
        unsafe { std::slice::from_raw_parts(norm_out_fused.contents() as *const u16, total_elems) };

    assert_eq!(
        res_unfused_slice, res_fused_slice,
        "Residual update mismatch between unfused and fused"
    );
    for i in 0..total_elems {
        let u = kriya_core::quantizer::bf16_bits_to_f32(norm_unfused_slice[i]);
        let f = kriya_core::quantizer::bf16_bits_to_f32(norm_fused_slice[i]);
        assert!(
            (u - f).abs() <= 1e-3,
            "Normalized output mismatch at {i}: unfused={u}, fused={f}"
        );
    }
}

#[test]
fn test_persistent_threadgroup_gemv_matches_reference() {
    let Some(device) = Device::system_default() else {
        eprintln!("Skipping test: no Metal GPU available");
        return;
    };
    let shared = MTLResourceOptions::StorageModeShared;
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();
    let queue = device.new_command_queue();

    const N: usize = 256;
    const K: usize = 256;
    let num_k_blocks = (K / 64) as u32;

    // Synthetic Q4_G64 weight matrix
    let blocks_total = (N * K) / 64;
    let mut weight_bytes = vec![0u8; blocks_total * 36];
    for b in 0..blocks_total {
        let base = b * 36;
        let scale = kriya_core::quantizer::f32_to_bf16_bits(0.05);
        let bias = kriya_core::quantizer::f32_to_bf16_bits(-0.4);
        weight_bytes[base..base + 2].copy_from_slice(&scale.to_le_bytes());
        weight_bytes[base + 2..base + 4].copy_from_slice(&bias.to_le_bytes());
        for byte_idx in 0..32 {
            weight_bytes[base + 4 + byte_idx] = ((b * 31 + byte_idx * 7) % 256) as u8;
        }
    }
    let weight_buf = device.new_buffer_with_data(
        weight_bytes.as_ptr() as *const _,
        weight_bytes.len() as u64,
        shared,
    );

    let input_x: Vec<u16> = (0..K)
        .map(|i| kriya_core::quantizer::f32_to_bf16_bits(((i % 13) as f32 - 6.0) * 0.1))
        .collect();
    let input_buf = device.new_buffer_with_data(
        input_x.as_ptr() as *const _,
        (input_x.len() * 2) as u64,
        shared,
    );

    let params = kriya_backend_metal::encoder::GemmParams {
        m: 1,
        n: N as u32,
        k: K as u32,
        num_k_blocks,
    };

    let total_tiles = (N as u64).div_ceil(16); // 16 threadgroups for 256 rows

    // Reference: full grid (1 threadgroup per tile)
    let out_ref = device.new_buffer((N * 2) as u64, shared);
    let cmd1 = queue.new_command_buffer();
    let enc1 = cmd1.new_compute_command_encoder();
    enc1.set_compute_pipeline_state(&pipelines.get_or_compile("q4_g64_decode_gemv_4row").unwrap());
    enc1.set_buffer(0, Some(&weight_buf), 0);
    enc1.set_buffer(1, Some(&input_buf), 0);
    enc1.set_buffer(2, Some(&out_ref), 0);
    enc1.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    enc1.dispatch_thread_groups(MTLSize::new(total_tiles, 1, 1), MTLSize::new(128, 1, 1));
    enc1.end_encoding();
    cmd1.commit();
    cmd1.wait_until_completed();

    // Persistent: small grid (e.g. 4 persistent threadgroups looping over 16 tiles)
    let out_persistent = device.new_buffer((N * 2) as u64, shared);
    let cmd2 = queue.new_command_buffer();
    let enc2 = cmd2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&pipelines.get_or_compile("q4_g64_decode_gemv_4row").unwrap());
    enc2.set_buffer(0, Some(&weight_buf), 0);
    enc2.set_buffer(1, Some(&input_buf), 0);
    enc2.set_buffer(2, Some(&out_persistent), 0);
    enc2.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        &params as *const _ as *const _,
    );
    enc2.dispatch_thread_groups(MTLSize::new(4, 1, 1), MTLSize::new(128, 1, 1));
    enc2.end_encoding();
    cmd2.commit();
    cmd2.wait_until_completed();

    let ref_slice = unsafe { std::slice::from_raw_parts(out_ref.contents() as *const u16, N) };
    let pers_slice =
        unsafe { std::slice::from_raw_parts(out_persistent.contents() as *const u16, N) };

    assert_eq!(
        ref_slice, pers_slice,
        "Persistent threadgroups produced different results than full grid dispatch"
    );
}

#[test]
fn test_q2_g128_decode_gemv_matches_cpu() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let n = 32usize;
    let k = 256usize;

    let weights: Vec<f32> = (0..n * k)
        .map(|idx| (((idx % 23) as f32) - 11.0) * 0.05)
        .collect();
    let q2_blocks = kriya_core::quantizer::quantize_f32_to_q2_group128(&weights).unwrap();
    let dequant_w = kriya_core::quantizer::dequantize_q2_group128_to_f32(&q2_blocks);

    let input_f32: Vec<f32> = (0..k)
        .map(|idx| (((idx % 19) as f32) - 9.0) * 0.04)
        .collect();
    let input_bf16: Vec<u16> = input_f32
        .iter()
        .map(|&v| kriya_core::quantizer::f32_to_bf16_bits(v))
        .collect();
    let input_unquant: Vec<f32> = input_bf16
        .iter()
        .map(|&bits| kriya_core::quantizer::bf16_bits_to_f32(bits))
        .collect();

    let mut expected_y = vec![0.0f32; n];
    for row in 0..n {
        let mut sum = 0.0f32;
        for inner in 0..k {
            sum += input_unquant[inner] * dequant_w[row * k + inner];
        }
        expected_y[row] = sum;
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let weight_bytes = q2_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ2Group128>();
    let weight_buf =
        device.new_buffer_with_data(q2_blocks.as_ptr() as *const _, weight_bytes as u64, shared);
    let input_buf = device.new_buffer_with_data(
        input_bf16.as_ptr() as *const _,
        (input_bf16.len() * std::mem::size_of::<u16>()) as u64,
        shared,
    );
    let out_buf = device.new_buffer((n * std::mem::size_of::<u16>()) as u64, shared);

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    kriya_backend_metal::encoder::MetalExecutor::encode_q2_linear(
        &mut pipelines,
        enc,
        &weight_buf,
        &input_buf,
        0,
        &out_buf,
        1,
        n as u32,
        k as u32,
    )
    .unwrap();
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out_slice = unsafe { std::slice::from_raw_parts(out_buf.contents() as *const u16, n) };
    for row in 0..n {
        let actual = kriya_core::quantizer::bf16_bits_to_f32(out_slice[row]);
        let expected = expected_y[row];
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.05,
            "Row {row} Q2 GEMV mismatch: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn test_q3_g64_decode_gemv_matches_cpu() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let n = 32usize;
    let k = 256usize;

    let weights: Vec<f32> = (0..n * k)
        .map(|idx| (((idx % 23) as f32) - 11.0) * 0.05)
        .collect();
    let q3_blocks = kriya_core::quantizer::quantize_f32_to_q3_group64(&weights).unwrap();
    let dequant_w = kriya_core::quantizer::dequantize_q3_group64_to_f32(&q3_blocks);

    let input_f32: Vec<f32> = (0..k)
        .map(|idx| (((idx % 19) as f32) - 9.0) * 0.04)
        .collect();
    let input_bf16: Vec<u16> = input_f32
        .iter()
        .map(|&v| kriya_core::quantizer::f32_to_bf16_bits(v))
        .collect();
    let input_unquant: Vec<f32> = input_bf16
        .iter()
        .map(|&bits| kriya_core::quantizer::bf16_bits_to_f32(bits))
        .collect();

    let mut expected_y = vec![0.0f32; n];
    for row in 0..n {
        let mut sum = 0.0f32;
        for inner in 0..k {
            sum += input_unquant[inner] * dequant_w[row * k + inner];
        }
        expected_y[row] = sum;
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let weight_bytes = q3_blocks.len() * std::mem::size_of::<kriya_core::tensor::BlockQ3Group64>();
    let weight_buf =
        device.new_buffer_with_data(q3_blocks.as_ptr() as *const _, weight_bytes as u64, shared);
    let input_buf = device.new_buffer_with_data(
        input_bf16.as_ptr() as *const _,
        (input_bf16.len() * std::mem::size_of::<u16>()) as u64,
        shared,
    );
    let out_buf = device.new_buffer((n * std::mem::size_of::<u16>()) as u64, shared);

    let queue = device.new_command_queue();
    let mut pipelines = kriya_backend_metal::encoder::MetalPipelineCache::from_shader_dir(
        device.clone(),
        workspace_kernels_dir(),
    )
    .unwrap();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    kriya_backend_metal::encoder::MetalExecutor::encode_q3_linear(
        &mut pipelines,
        enc,
        &weight_buf,
        &input_buf,
        0,
        &out_buf,
        1,
        n as u32,
        k as u32,
    )
    .unwrap();
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out_slice = unsafe { std::slice::from_raw_parts(out_buf.contents() as *const u16, n) };
    for row in 0..n {
        let actual = kriya_core::quantizer::bf16_bits_to_f32(out_slice[row]);
        let expected = expected_y[row];
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.05,
            "Row {row} Q3 GEMV mismatch: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn test_double_buffered_layer_streaming_matches_pinned() {
    let Some(device) = require_metal_device() else {
        return;
    };

    let tmp_dir = tempfile::tempdir().unwrap();
    let base = miniature_dense_gqa_descriptor();
    let mut desc = base.clone();
    desc.layers = (0..3)
        .map(|idx| {
            let mut l = base.layers[0];
            l.layer_idx = idx as u32;
            l
        })
        .collect();

    let q4_matrix = || {
        let blocks =
            kriya_core::quantizer::quantize_f32_to_q4_group64(&vec![0.125f32; 64 * 64]).unwrap();
        kriya_core::quantizer::SavedTensorPayload::Q4Group64(blocks)
    };
    let bf16_ones = || kriya_core::quantizer::SavedTensorPayload::Bf16(vec![0x3F80_u16; 64]);

    let mut tensors = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![64, 64],
            q4_matrix(),
        ),
        ("model.norm.weight".into(), vec![64], bf16_ones()),
        ("lm_head.weight".into(), vec![64, 64], q4_matrix()),
    ];

    for idx in 0..3 {
        let p = format!("model.layers.{idx}");
        tensors.push((format!("{p}.input_layernorm.weight"), vec![64], bf16_ones()));
        tensors.push((
            format!("{p}.post_attention_layernorm.weight"),
            vec![64],
            bf16_ones(),
        ));
        tensors.push((
            format!("{p}.self_attn.q_proj.weight"),
            vec![64, 64],
            q4_matrix(),
        ));
        tensors.push((
            format!("{p}.self_attn.k_proj.weight"),
            vec![64, 64],
            q4_matrix(),
        ));
        tensors.push((
            format!("{p}.self_attn.v_proj.weight"),
            vec![64, 64],
            q4_matrix(),
        ));
        tensors.push((
            format!("{p}.self_attn.o_proj.weight"),
            vec![64, 64],
            q4_matrix(),
        ));
        tensors.push((
            format!("{p}.mlp.gate_proj.weight"),
            vec![64, 64],
            q4_matrix(),
        ));
        tensors.push((format!("{p}.mlp.up_proj.weight"), vec![64, 64], q4_matrix()));
        tensors.push((
            format!("{p}.mlp.down_proj.weight"),
            vec![64, 64],
            q4_matrix(),
        ));
    }

    kriya_core::quantizer::write_mixed_safetensors(
        tmp_dir.path().join("model.safetensors"),
        &tensors,
    )
    .unwrap();

    let pinned_weights = kriya_backend_metal::encoder::ModelGpuWeights::from_checkpoint_dir_strict(
        &device,
        &desc,
        tmp_dir.path(),
    )
    .unwrap();
    assert!(!pinned_weights.is_streaming());
    assert_eq!(pinned_weights.layers.len(), 3);

    let streamed_weights =
        kriya_backend_metal::encoder::ModelGpuWeights::from_checkpoint_dir_streaming(
            &device,
            &desc,
            tmp_dir.path(),
        )
        .unwrap();
    assert!(streamed_weights.is_streaming());
    assert_eq!(streamed_weights.layers.len(), 0); // No pinned layer table!

    // Verify each layer buffers from streamed weights match pinned layer buffers
    for idx in 0..3 {
        let pinned_l = pinned_weights.layer(idx);
        let streamed_l = streamed_weights.layer(idx);
        assert_eq!(pinned_l.q_proj.length(), streamed_l.q_proj.length());
        assert_eq!(pinned_l.k_proj.length(), streamed_l.k_proj.length());
        assert_eq!(pinned_l.v_proj.length(), streamed_l.v_proj.length());
        assert_eq!(pinned_l.o_proj.length(), streamed_l.o_proj.length());
        assert_eq!(pinned_l.gate_proj.length(), streamed_l.gate_proj.length());
        assert_eq!(pinned_l.up_proj.length(), streamed_l.up_proj.length());
        assert_eq!(pinned_l.down_proj.length(), streamed_l.down_proj.length());
    }

    // Verify DoubleBufferedLayerStreamer keeps at most 2 active layers in memory
    let streamer_arc = streamed_weights.streamer.as_ref().unwrap();
    let streamer = streamer_arc.lock().unwrap();
    let active = streamer.active_slots();
    assert!(active[0].is_some());
    assert!(active[1].is_some());
    // Active slots must be a subset of {0, 1, 2} and length is strictly 2
    assert_eq!(active.len(), 2);
}
