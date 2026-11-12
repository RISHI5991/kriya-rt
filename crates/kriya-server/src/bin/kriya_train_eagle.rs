//! Pure-Rust Native EAGLE Speculative Draft Head Trainer (`kriya_train_eagle`).
//!
//! Features:
//! 1. Zero Python dependencies: Runs entirely on macOS using Apple Silicon Metal & Rust.
//! 2. Feature Extraction: Runs the target model on Metal GPU to extract true hidden activations.
//! 3. Feature Perturbation: Injects U(-0.1, 0.1) noise into features during AdamW optimization.
//! 4. Acceptance Rate Telemetry: Measures draft token prediction accuracy before and after training.
//! 5. Turnkey Release Artifact: Directly outputs `kriya_eagle_draft_v1.bin` with valid `KRYAEGL1`
//!    header and updates `kriya-artifact-manifest.json` with the new SHA-256 digest.

use kriya_backend_metal::encoder::{MetalExecutor, SamplingConfig};
use kriya_core::quantizer::HuggingFaceModelConfig;
use kriya_core::tokenizer::KriyaTokenizer;
use kriya_kv::bsr_paged::PagedKvCacheManager;
use kriya_speculate::eagle_medusa::{DeterministicRng, EagleDraftEngine};
use kriya_speculate::trainer::{EagleTrainer, EagleTrainingConfig, FeatureStepSample};
use ring::digest::{Context, SHA256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|err| err.to_string())?;
    let mut context = Context::new(&SHA256);
    let mut chunk = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut chunk).map_err(|err| err.to_string())?;
        if read == 0 {
            break;
        }
        context.update(&chunk[..read]);
    }
    Ok(context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn update_manifest(manifest_path: &Path, draft_file_name: &str, byte_len: u64, sha256: &str) {
    if !manifest_path.is_file() {
        return;
    }
    let Ok(data) = std::fs::read(manifest_path) else {
        return;
    };
    let Ok(mut json) = serde_json::from_slice::<serde_json::Value>(&data) else {
        return;
    };

    let spec_file = serde_json::json!({
        "file_name": draft_file_name,
        "byte_len": byte_len,
        "sha256": sha256,
    });

    json["speculative_draft_file"] = spec_file;
    json["speculative_draft_head_bundled"] = serde_json::Value::Bool(true);

    if let Ok(pretty) = serde_json::to_vec_pretty(&json) {
        let _ = std::fs::write(manifest_path, pretty);
        println!(
            "[kriya-train-eagle] Updated manifest with new draft head SHA-256 at {}",
            manifest_path.display()
        );
    }
}

fn find_arg<'a>(args: &'a [String], flags: &[&str]) -> Option<&'a str> {
    args.windows(2)
        .find(|w| flags.contains(&w[0].as_str()))
        .map(|w| w[1].as_str())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "kriya_train_eagle 0.1.0\n\
             Pure-Rust Native EAGLE Speculative Draft Head Trainer\n\n\
             Usage: kriya_train_eagle --model <model_dir> [--steps <num_steps>] [--lr <learning_rate>] [--out <output_path>]\n\n\
             Options:\n  \
               --model, --input   Target model directory containing config.json and weights\n  \
               --steps            Training steps (default: 200)\n  \
               --lr               Learning rate (default: 1e-3)\n  \
               --out              Output draft artifact path (default: <model_dir>/kriya_eagle_draft_v1.bin)\n  \
               --version, -V      Print version and exit\n  \
               --help, -h         Print this help message and exit"
        );
        std::process::exit(0);
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("kriya_train_eagle 0.1.0");
        std::process::exit(0);
    }

    let model_dir = find_arg(&args, &["--model", "--input"])
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Usage: kriya_train_eagle --model <model_dir> [--steps <num_steps>] [--lr <learning_rate>] [--out <output_path>]");
            std::process::exit(2);
        });

    let steps = find_arg(&args, &["--steps"])
        .and_then(|w| w.parse::<usize>().ok())
        .unwrap_or(200);

    let lr = find_arg(&args, &["--lr"])
        .and_then(|w| w.parse::<f32>().ok())
        .unwrap_or(1e-3);

    let output_path = find_arg(&args, &["--out"])
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.join("kriya_eagle_draft_v1.bin"));

    println!(
        "=========================================================================================="
    );
    println!(" KriyaRT EAGLE Speculative Draft Head Distillation Trainer");
    println!(" Target Model Dir : {}", model_dir.display());
    println!(" Training Steps   : {}", steps);
    println!(" Learning Rate    : {}", lr);
    println!(" Output Artifact  : {}", output_path.display());
    println!(
        "=========================================================================================="
    );

    let config_path = model_dir.join("config.json");
    let desc = HuggingFaceModelConfig::from_config_file(&config_path)
        .expect("Valid config.json required in model directory");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let kernels_dir = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
        .join("metal");

    println!("[kriya-train-eagle] Initializing target model on Metal GPU...");
    let mut exec = MetalExecutor::new_production(&desc, &kernels_dir, &model_dir, 512, 16)
        .expect("Failed to initialize target MetalExecutor");

    let tokenizer = KriyaTokenizer::from_checkpoint_dir(&model_dir)
        .expect("Failed to load tokenizer from checkpoint directory");

    let h = desc.hidden_size as usize;
    let vocab = desc.vocab_size as usize;

    println!("[kriya-train-eagle] Extracting frozen target embeddings and LM head...");
    let target_embedding = exec
        .get_embedding_weights_f32()
        .expect("Failed to extract target embedding weights");
    let target_lm_head = exec
        .get_lm_head_weights_f32()
        .expect("Failed to extract target LM head weights");

    // Initialize or load draft head
    let mut fc_fuse = vec![0.0f32; h * (2 * h)];
    // Model-aligned identity skip initialization for prev_feature (offset h..2h)
    for i in 0..h {
        fc_fuse[i * (2 * h) + h + i] = 1.0;
    }
    let decoder_w = vec![0.0f32; h * h];

    let mut engine = EagleDraftEngine::new(
        h,
        vocab,
        target_embedding,
        target_lm_head,
        fc_fuse,
        decoder_w,
    )
    .expect("Failed to instantiate EagleDraftEngine");

    if output_path.is_file()
        && let Ok(existing_bytes) = std::fs::read(&output_path)
        && engine.load_weights(&existing_bytes).is_ok()
    {
        println!("[kriya-train-eagle] Loaded existing draft head weights for continued training.");
    }

    // Diverse text corpus for feature extraction
    let training_prompts = [
        "In computer science, algorithms and data structures form the foundational core of efficient software systems.",
        "Artificial intelligence and machine learning architectures rely heavily on parallel matrix operations and gradient descent.",
        "Apple Silicon unified memory provides high bandwidth and low latency across CPU and GPU cores without PCIe bus limits.",
        "The quick brown fox jumps over the lazy dog in a swift leap across the green meadow on a sunny morning.",
        "Rust programming language guarantees memory safety and thread concurrency without requiring a garbage collector runtime.",
        "Large language models utilize transformer blocks composed of multi-head self-attention and gated feed-forward neural networks.",
        "To optimize autoregressive inference, speculative decoding generates multiple candidate tokens using a lightweight draft head.",
        "Distributed computing frameworks leverage message passing, remote procedure calls, and consensus protocols for high availability.",
    ];

    println!("[kriya-train-eagle] Extracting target feature activations on Metal GPU...");
    let mut dataset_samples = Vec::new();
    let mut kv_mgr = PagedKvCacheManager::new(512, 128, 16);

    for (p_idx, &prompt) in training_prompts.iter().enumerate() {
        let input_tokens = tokenizer
            .encode(prompt, false)
            .unwrap_or_else(|_| vec![1, 2, 3]);
        if input_tokens.len() < 3 {
            continue;
        }

        let req_id = 5000u64 + p_idx as u64;
        let _ = kv_mgr.allocate_sequence(req_id, &input_tokens);

        let mut prev_feature: Option<Vec<f32>> = None;
        let mut prev_token: Option<u32> = None;

        for (t_idx, &tok) in input_tokens.iter().enumerate() {
            let plan = match kv_mgr.build_prefix_schedule_plan(req_id, (t_idx + 1) as u32, 128) {
                Ok(p) => p,
                Err(_) => break,
            };

            let samp = SamplingConfig {
                vocab_size: desc.vocab_size,
                greedy: 1,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 50,
                rng_seed: 42,
            };

            let _ = exec
                .forward_and_sample(&desc, &plan, &[tok], &[t_idx as u32], samp)
                .await;
            let curr_feature = exec.read_last_hidden_feature(h, 1);

            if let (Some(f_prev), Some(t_prev)) = (prev_feature, prev_token) {
                dataset_samples.push(FeatureStepSample {
                    prev_feature: f_prev,
                    advanced_token_id: t_prev,
                    target_next_feature: curr_feature.clone(),
                    target_next_token_id: tok,
                });
            }

            prev_feature = Some(curr_feature);
            prev_token = Some(tok);
        }

        let _ = kv_mgr.free_sequence(req_id);
    }

    println!(
        "[kriya-train-eagle] Collected {} transition samples across prompt corpus.",
        dataset_samples.len()
    );

    let (val_samples, train_samples) = dataset_samples.split_at(dataset_samples.len().min(10));
    let mut trainer = EagleTrainer::new(
        &engine,
        EagleTrainingConfig {
            learning_rate: lr,
            weight_decay: 1e-4,
            noise_bound: 0.1,
            cls_weight: 0.5,
            ..Default::default()
        },
    );

    let mut rng = DeterministicRng::new(0x1337_BEEF);

    let initial_acc = trainer
        .evaluate_acceptance_rate(&engine, val_samples)
        .unwrap_or(0.0);
    println!(
        "[kriya-train-eagle] Pre-training Validation Draft Token Match Rate: {:.2}%",
        initial_acc * 100.0
    );

    let train_start = Instant::now();
    let batch_size = 8.min(train_samples.len().max(1));

    for step in 1..=steps {
        // Sample random mini-batch
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = (rng.next_u64() as usize) % train_samples.len();
            batch.push(train_samples[idx].clone());
        }

        let (total_loss, reg_loss, acc) = trainer
            .train_batch(&mut engine, &batch, &mut rng)
            .expect("Training step failed");

        if step % 20 == 0 || step == steps {
            println!(
                "Step {:4}/{} | Total Loss: {:7.4} | Reg Loss (SmoothL1): {:7.4} | Train Match: {:5.1}%",
                step,
                steps,
                total_loss,
                reg_loss,
                acc * 100.0
            );
        }
    }

    let elapsed = train_start.elapsed();
    println!(
        "[kriya-train-eagle] Training finished in {:.2} seconds ({:.1} steps/sec)",
        elapsed.as_secs_f64(),
        (steps as f64) / elapsed.as_secs_f64()
    );

    let final_acc = trainer
        .evaluate_acceptance_rate(&engine, val_samples)
        .unwrap_or(0.0);
    println!(
        "[kriya-train-eagle] Post-training Validation Draft Token Match Rate: {:.2}% (vs pre-training {:.2}%)",
        final_acc * 100.0,
        initial_acc * 100.0
    );

    // Save trained weights
    let serialized = engine.save_weights();
    std::fs::write(&output_path, &serialized).expect("Failed to write trained draft artifact");
    println!(
        "[kriya-train-eagle] Saved {} bytes to {}",
        serialized.len(),
        output_path.display()
    );

    // Update manifest
    let manifest_path = model_dir.join("kriya-artifact-manifest.json");
    let file_name = output_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("kriya_eagle_draft_v1.bin");
    if let Ok(sha256) = sha256_file(&output_path) {
        update_manifest(&manifest_path, file_name, serialized.len() as u64, &sha256);
    }

    println!(
        "=========================================================================================="
    );
    println!(
        " Training Complete! Trained draft head is now ready for production speculative decoding."
    );
    println!(
        "=========================================================================================="
    );
}
