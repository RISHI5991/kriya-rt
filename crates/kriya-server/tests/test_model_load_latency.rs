//! Live Cold-Start & Model Loading Latency Benchmark for KriyaRT.

use kriya_backend_metal::encoder::MetalExecutor;
use kriya_core::quantizer::HuggingFaceModelConfig;
use kriya_core::safetensors_loader::ShardedCheckpointLoader;
use kriya_core::tokenizer::KriyaTokenizer;
use kriya_kv::bsr_paged::PagedKvCacheManager;
use kriya_scheduler::orca_engine::{InstanceRole, OrcaMachineScheduler, ServiceLevelObjective};
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn test_cold_start_model_load_latency() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let model_dir = root.join("models/qwen2.5-0.5b-instruct-kriya-q4");
    let kernels_dir = root.join("kernels/metal");

    let has_weights = model_dir.join("model.safetensors").exists()
        || model_dir.join("model.safetensors.index.json").exists();
    if !has_weights {
        println!(
            "Model checkpoint weights not found in {}, skipping.",
            model_dir.display()
        );
        return;
    }

    println!(
        "\n=========================================================================================="
    );
    println!(" KriyaRT Cold-Start & Model Load Latency Reality Check");
    println!(" Checkpoint: {}", model_dir.display());
    println!(
        "=========================================================================================="
    );

    let total_start = Instant::now();

    // 1. Config reading and parsing
    let t0 = Instant::now();
    let config_path = model_dir.join("config.json");
    let desc = HuggingFaceModelConfig::from_config_file(&config_path)
        .expect("Failed to parse config.json");
    let config_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[1/5] Model Config Parsing        : {:6.2} ms (Arch: {}, Layers: {}, Hidden: {})",
        config_ms,
        desc.model_name,
        desc.layers.len(),
        desc.hidden_size
    );

    // 2. Tokenizer loading (BPE tokenizer.json parsing)
    let t0 = Instant::now();
    let tokenizer =
        KriyaTokenizer::from_checkpoint_dir(&model_dir).expect("Failed to load tokenizer");
    let tok_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[2/5] HuggingFace BPE Tokenizer   : {:6.2} ms (EOS token ID: {})",
        tok_ms, tokenizer.eos_token_id
    );

    // 3. Checkpoint mmap & Safetensors Header Parsing
    let t0 = Instant::now();
    let loader =
        ShardedCheckpointLoader::open(&model_dir).expect("Failed to open safetensors checkpoint");
    let mmap_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let tensor_count = loader.tensor_to_shard.len();
    println!(
        "[3/5] Checkpoint mmap & Header    : {:6.2} ms (Indexed {} tensors)",
        mmap_ms, tensor_count
    );

    // 4. Metal Device, Shader Compilation & Unified Memory Buffer Allocation
    let t0 = Instant::now();
    let _executor = MetalExecutor::new_production(&desc, &kernels_dir, &model_dir, 512, 16)
        .expect("Failed to initialize production Metal executor");
    let metal_init_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[4/5] Metal GPU & Pipeline States : {:6.2} ms (23 PSOs + Weights + Ping-Pong Scratch)",
        metal_init_ms
    );

    // 5. Orca Continuous Scheduler & BSR Paged KV Cache Initialization
    let t0 = Instant::now();
    let kv_mgr = PagedKvCacheManager::new(512, 128, 16);
    let _scheduler = OrcaMachineScheduler::new(
        InstanceRole::ColocatedHybrid,
        desc.layers.len() as u32,
        4096,
        16,
        ServiceLevelObjective::default(),
        kv_mgr,
    );
    let sched_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[5/5] ORCA Scheduler & BSR Paged  : {:6.2} ms (512 KV blocks, 16 tok/blk)",
        sched_ms
    );

    let total_cold_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "------------------------------------------------------------------------------------------"
    );
    println!(
        "TOTAL COLD-START READY-TO-SERVE   : {:6.2} ms ({:.3} s)",
        total_cold_ms,
        total_cold_ms / 1000.0
    );
    println!(
        "=========================================================================================="
    );

    // Warm reload measurement (weights reload & PSO cache reuse)
    let warm_start = Instant::now();
    let _warm_exec = MetalExecutor::new_production(&desc, &kernels_dir, &model_dir, 512, 16)
        .expect("Warm reload failed");
    let warm_ms = warm_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "WARM RELOAD LATENCY               : {:6.2} ms ({:.3} s)",
        warm_ms,
        warm_ms / 1000.0
    );
    println!(
        "==========================================================================================\n"
    );

    let max_allowed = if cfg!(debug_assertions) {
        15000.0
    } else {
        5000.0
    };
    assert!(
        total_cold_ms < max_allowed,
        "Model loading took {:.2} ms, exceeding threshold",
        total_cold_ms
    );
}
