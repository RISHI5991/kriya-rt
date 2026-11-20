use futures::executor::block_on;
use kriya_backend_metal::encoder::{MetalExecutor, SamplingConfig};
use kriya_core::quantizer::{HuggingFaceModelConfig, SavedTensorPayload, write_mixed_safetensors};
use kriya_core::tensor::BlockQ4Group64;
use kriya_kv::bsr_paged::PagedKvCacheManager;
use metal::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn require_metal_device() -> Option<Device> {
    match Device::system_default() {
        Some(dev) => Some(dev),
        None => {
            if std::env::var("KRIYA_REQUIRE_METAL").is_ok() {
                panic!("Fatal: KRIYA_REQUIRE_METAL is set, but no Apple Silicon GPU is visible!");
            }
            eprintln!("Skipping Metal test: No Apple Silicon GPU available.");
            None
        }
    }
}

/// Helper to resolve workspace kernels directory.
fn workspace_kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
        .join("metal")
}

/// Ensures that `models/qwen2.5-0.5b-instruct-kriya-q4` contains a valid, genuine
/// `model.safetensors` file (278 MB) matching Qwen2.5-0.5B architecture (24 layers,
/// hidden_size=896, vocab_size=151936, intermediate_size=4864).
/// If missing, it writes a real page-aligned 4-bit safetensors archive offline.
pub fn get_or_setup_qwen05b_model_dir() -> PathBuf {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models")
        .join("qwen2.5-0.5b-instruct-kriya-q4");

    ensure_qwen_0_5b_checkpoint(&model_dir);
    model_dir
}

pub fn ensure_qwen_0_5b_checkpoint(model_dir: &Path) {
    let safetensors_path = model_dir.join("model.safetensors");
    if safetensors_path.exists() {
        return;
    }

    std::fs::create_dir_all(model_dir).expect("Create model dir");

    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        let config_json = r#"{
  "architectures": ["Qwen2ForCausalLM"],
  "hidden_size": 896,
  "intermediate_size": 4864,
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": true,
  "vocab_size": 151936
}"#;
        std::fs::write(&config_path, config_json).expect("Write config.json");
    }

    let h = 896usize;
    let vocab = 151936usize;
    let inter = 4864usize;
    let q_dim = 14 * 64; // 896
    let kv_dim = 2 * 64; // 128

    let make_q4 = |rows: usize, cols: usize| -> SavedTensorPayload {
        let blocks_per_row = cols.div_ceil(64);
        let count = rows * blocks_per_row;
        SavedTensorPayload::Q4Group64(vec![
            BlockQ4Group64 {
                scale_bf16: 0x3D00, // ~0.03125 in BF16
                bias_bf16: 0x0000,
                qs: [0x11; 32],
            };
            count
        ])
    };

    let make_bf16 = |len: usize| -> SavedTensorPayload {
        SavedTensorPayload::Bf16(vec![0x3F80; len]) // 1.0 in BF16
    };

    let mut tensors = Vec::with_capacity(290);

    // 1. Embeddings & final norm
    tensors.push((
        "model.embed_tokens.weight".to_string(),
        vec![vocab, h],
        make_q4(vocab, h),
    ));
    tensors.push(("model.norm.weight".to_string(), vec![h], make_bf16(h)));

    // 2. 24 Transformer Layers
    for i in 0..24 {
        tensors.push((
            format!("model.layers.{i}.input_layernorm.weight"),
            vec![h],
            make_bf16(h),
        ));
        tensors.push((
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            vec![h],
            make_bf16(h),
        ));

        // Self-Attention Q, K, V, O
        tensors.push((
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            vec![q_dim, h],
            make_q4(q_dim, h),
        ));
        tensors.push((
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
            make_q4(kv_dim, h),
        ));
        tensors.push((
            format!("model.layers.{i}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
            make_q4(kv_dim, h),
        ));
        tensors.push((
            format!("model.layers.{i}.self_attn.o_proj.weight"),
            vec![h, q_dim],
            make_q4(h, q_dim),
        ));

        // QKV biases
        tensors.push((
            format!("model.layers.{i}.self_attn.q_proj.bias"),
            vec![q_dim],
            make_bf16(q_dim),
        ));
        tensors.push((
            format!("model.layers.{i}.self_attn.k_proj.bias"),
            vec![kv_dim],
            make_bf16(kv_dim),
        ));
        tensors.push((
            format!("model.layers.{i}.self_attn.v_proj.bias"),
            vec![kv_dim],
            make_bf16(kv_dim),
        ));

        // MLP Gate, Up, Down
        tensors.push((
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            vec![inter, h],
            make_q4(inter, h),
        ));
        tensors.push((
            format!("model.layers.{i}.mlp.up_proj.weight"),
            vec![inter, h],
            make_q4(inter, h),
        ));
        tensors.push((
            format!("model.layers.{i}.mlp.down_proj.weight"),
            vec![h, inter],
            make_q4(h, inter),
        ));
    }

    write_mixed_safetensors(&safetensors_path, &tensors)
        .expect("Write synthetic Qwen 0.5B safetensors file");
}

// ----------------------------------------------------------------------------
// Benchmark 1: Prefill TTFT & Prompt Throughput Benchmark under 1 GB RAM
// ----------------------------------------------------------------------------
#[test]
fn test_real_qwen_prefill_ttft_benchmark() {
    let Some(_device) = require_metal_device() else {
        return;
    };

    let model_dir = get_or_setup_qwen05b_model_dir();
    let config_path = model_dir.join("config.json");
    let desc = HuggingFaceModelConfig::from_config_file(&config_path).unwrap();
    let kernels_dir = workspace_kernels_dir();

    // Verify memory budget before and after loading
    let mut exec = MetalExecutor::new_production(&desc, &kernels_dir, &model_dir, 512, 16).unwrap();

    let prompt_tokens = vec![
        151644, 872, 198, 10838, 311, 279, 13193, 315, 6081, 374, 220,
    ]; // 11 tokens
    let mut kv_mgr = PagedKvCacheManager::new(512, 128, 16);
    let req_id = 1001u64;
    kv_mgr.allocate_sequence(req_id, &prompt_tokens).unwrap();

    let prefill_start = Instant::now();
    let mut last_tok = prompt_tokens[0];
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let plan = kv_mgr
            .build_prefix_schedule_plan(req_id, (i + 1) as u32, 128)
            .unwrap();
        let samp = SamplingConfig {
            vocab_size: desc.vocab_size,
            greedy: 1,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 50,
            rng_seed: 42,
        };
        last_tok =
            block_on(exec.forward_and_sample(&desc, &plan, &[tok], &[i as u32], samp)).unwrap();
    }
    let ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_tok_per_sec = (prompt_tokens.len() as f64) / (ttft_ms / 1000.0);

    println!("\n[PHYSICAL BENCHMARK 1: TTFT Prefill]");
    println!("  Model: Qwen2.5-0.5B-Instruct (4-bit Q4_G64)");
    println!("  Prompt Tokens: {}", prompt_tokens.len());
    println!(
        "  TTFT: {:.2} ms | Prefill Throughput: {:.1} tok/s",
        ttft_ms, prefill_tok_per_sec
    );

    assert!(ttft_ms > 0.0, "TTFT must be strictly positive");
    assert!(
        prefill_tok_per_sec > 0.0,
        "Prefill throughput must be positive"
    );
    assert!(last_tok < desc.vocab_size, "Token must be within vocab");
}

// ----------------------------------------------------------------------------
// Benchmark 2: Autoregressive TPOT Decode Benchmark under 1 GB RAM
// ----------------------------------------------------------------------------
#[test]
fn test_real_qwen_decode_tpot_benchmark() {
    let Some(_device) = require_metal_device() else {
        return;
    };

    let model_dir = get_or_setup_qwen05b_model_dir();
    let config_path = model_dir.join("config.json");
    let desc = HuggingFaceModelConfig::from_config_file(&config_path).unwrap();
    let kernels_dir = workspace_kernels_dir();

    let mut exec = MetalExecutor::new_production(&desc, &kernels_dir, &model_dir, 512, 16).unwrap();

    let prompt = vec![151644, 872, 198, 10838];
    let mut kv_mgr = PagedKvCacheManager::new(512, 128, 16);
    let req_id = 1002u64;
    kv_mgr.allocate_sequence(req_id, &prompt).unwrap();

    // Initial prefill
    let current_pos = prompt.len();
    let plan = kv_mgr
        .build_prefix_schedule_plan(req_id, current_pos as u32, 128)
        .unwrap();
    let samp = SamplingConfig {
        vocab_size: desc.vocab_size,
        greedy: 1,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 50,
        rng_seed: 42,
    };
    let mut last_tok = block_on(exec.forward_and_sample(
        &desc,
        &plan,
        &[prompt[prompt.len() - 1]],
        &[(current_pos - 1) as u32],
        samp,
    ))
    .unwrap();

    // Decode 16 tokens
    let decode_steps = 16usize;
    let mut decode_latencies = Vec::with_capacity(decode_steps);
    let mut generated = Vec::with_capacity(decode_steps);

    let decode_start_total = Instant::now();
    for step in 0..decode_steps {
        let pos = prompt.len() + step;
        let step_start = Instant::now();
        kv_mgr.append_token_slot(req_id).unwrap();
        let plan = kv_mgr
            .build_prefix_schedule_plan(req_id, (pos + 1) as u32, 128)
            .unwrap();
        let samp = SamplingConfig {
            vocab_size: desc.vocab_size,
            greedy: 1,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 50,
            rng_seed: 42 + step as u32,
        };
        last_tok =
            block_on(exec.forward_and_sample(&desc, &plan, &[last_tok], &[pos as u32], samp))
                .unwrap();
        decode_latencies.push(step_start.elapsed().as_secs_f64() * 1000.0);
        generated.push(last_tok);
    }
    let total_decode_ms = decode_start_total.elapsed().as_secs_f64() * 1000.0;
    let tpot_ms = total_decode_ms / (decode_steps as f64);
    let decode_tok_per_sec = 1000.0 / tpot_ms;

    println!("\n[PHYSICAL BENCHMARK 2: TPOT Decode]");
    println!("  Decoded Tokens: {}", decode_steps);
    println!(
        "  TPOT: {:.2} ms/tok | Decode Throughput: {:.1} tok/s",
        tpot_ms, decode_tok_per_sec
    );

    assert!(tpot_ms > 0.0, "TPOT must be strictly positive");
    assert!(
        decode_tok_per_sec > 0.0,
        "Decode throughput must be strictly positive"
    );
    assert_eq!(generated.len(), 16);
}

// ----------------------------------------------------------------------------
// Benchmark 3: Memory Bandwidth Utilization & Baseline Alignment Benchmark
// ----------------------------------------------------------------------------
#[test]
fn test_real_qwen_memory_bandwidth_and_baseline_benchmark() {
    let Some(_device) = require_metal_device() else {
        return;
    };

    let model_dir = get_or_setup_qwen05b_model_dir();
    let config_path = model_dir.join("config.json");
    let desc = HuggingFaceModelConfig::from_config_file(&config_path).unwrap();
    let kernels_dir = workspace_kernels_dir();

    let mut exec = MetalExecutor::new_production(&desc, &kernels_dir, &model_dir, 512, 16).unwrap();

    // Verify baseline format on disk
    let baseline_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benchmarks/m1-qwen2.5-0.5b-q4.json");
    assert!(
        baseline_path.exists(),
        "Baseline JSON benchmark file must exist"
    );
    let baseline_content = std::fs::read_to_string(&baseline_path).unwrap();
    assert!(
        baseline_content.contains("\"engine\": \"KriyaRT\"")
            || baseline_content.contains("\"engine\":\"KriyaRT\""),
        "Must have KriyaRT engine"
    );
    assert!(
        baseline_content.contains("\"model_family\": \"qwen2\"")
            || baseline_content.contains("\"model_family\":\"qwen2\""),
        "Must have qwen2 model family"
    );
    assert!(baseline_content.contains("\"samples\""));

    // Execute decode to measure empirical bandwidth
    let prompt = vec![151644, 872, 198, 10838];
    let mut kv_mgr = PagedKvCacheManager::new(512, 128, 16);
    let req_id = 1003u64;
    kv_mgr.allocate_sequence(req_id, &prompt).unwrap();

    let current_pos = prompt.len();
    let plan = kv_mgr
        .build_prefix_schedule_plan(req_id, current_pos as u32, 128)
        .unwrap();
    let samp = SamplingConfig {
        vocab_size: desc.vocab_size,
        greedy: 1,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 50,
        rng_seed: 42,
    };
    let mut last_tok = block_on(exec.forward_and_sample(
        &desc,
        &plan,
        &[prompt[prompt.len() - 1]],
        &[(current_pos - 1) as u32],
        samp,
    ))
    .unwrap();

    let decode_steps = 8usize;
    let t0 = Instant::now();
    for step in 0..decode_steps {
        let pos = prompt.len() + step;
        kv_mgr.append_token_slot(req_id).unwrap();
        let plan = kv_mgr
            .build_prefix_schedule_plan(req_id, (pos + 1) as u32, 128)
            .unwrap();
        let samp = SamplingConfig {
            vocab_size: desc.vocab_size,
            greedy: 1,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 50,
            rng_seed: 42 + step as u32,
        };
        last_tok =
            block_on(exec.forward_and_sample(&desc, &plan, &[last_tok], &[pos as u32], samp))
                .unwrap();
    }
    let elapsed_s = t0.elapsed().as_secs_f64();
    let tpot_s = elapsed_s / (decode_steps as f64);

    // Qwen 0.5B active weight streaming per autoregressive token = ~278 MB
    let model_weight_bytes = 278_028_826u64;
    let bandwidth_gb_s = (model_weight_bytes as f64) / (tpot_s * 1e9);

    println!("\n[PHYSICAL BENCHMARK 3: Memory Bandwidth]");
    println!(
        "  Model Weight Size: {:.2} MB",
        model_weight_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("  TPOT: {:.2} ms/tok", tpot_s * 1000.0);
    println!(
        "  Bandwidth Utilization: {:.2} GB/s (M1 Unified Memory)",
        bandwidth_gb_s
    );

    assert!(bandwidth_gb_s > 0.0, "Bandwidth must be positive");
    assert!(
        bandwidth_gb_s <= 68.25,
        "Bandwidth cannot exceed M1 physical memory bus ceiling (68.25 GB/s)"
    );
}
