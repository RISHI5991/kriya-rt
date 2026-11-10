//! Production KriyaRT Server CLI & Binary Entrypoint.
//!
//! - Supports `--model <local_dir_or_hf_repo>` and `--arch <llama|qwen3|mistral|gemma2|deepseek>`
//! - Decoupled HTTP router creation and ORCA batch execution.

use kriya_core::descriptor::ModelArchDescriptor;
use kriya_core::quantizer::HuggingFaceModelConfig;
use kriya_server::hub_fetcher;
use kriya_server::runtime::{EngineRuntime, orca_batch_worker_loop};
use kriya_server::server::{ServerHandle, ServerMetrics, create_app};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("kriya-server {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("KriyaRT Server {}", env!("CARGO_PKG_VERSION"));
        println!("High-throughput native Apple Silicon LLM inference engine\n");
        println!("Usage: kriya-server --model <local_checkpoint_dir_or_hf_repo> [options]\n");
        println!("Options:");
        println!("  --model <path_or_repo>   Local model directory or Hugging Face Hub repository");
        println!("  --host <addr>            Host to bind HTTP/OpenAI server (default: 127.0.0.1)");
        println!("  --port <port>            Port to bind HTTP/OpenAI server (default: 8080)");
        println!("  --prompt <text>          Run prompt directly via CLI and exit");
        println!("  --draft-weights <path>   Enable EAGLE speculative decoding with draft head");
        println!("  --max-tokens <n>         Maximum tokens to generate (default: 256)");
        println!("  --temperature <val>      Sampling temperature (default: 0.7)");
        println!("  --top-p <val>            Nucleus sampling top-p (default: 0.95)");
        println!("  --version, -V            Print version and exit");
        println!("  --help, -h               Print this help message and exit");
        return;
    }

    // 1. Resolve `--model <local_path_or_hf_repo_id>` (downloads all shards via `hub_fetcher` if not local)
    let mut checkpoint_dir = args
        .windows(2)
        .find(|w| w[0] == "--model")
        .map(|w| PathBuf::from(&w[1]));

    if let Some(ckpt) = &checkpoint_dir
        && !ckpt.exists()
    {
        let repo_id = ckpt.to_str().unwrap();
        println!(
            "[KriyaRT Hub] Model path '{}' not found locally. Fetching from Hugging Face Hub...",
            repo_id
        );
        checkpoint_dir = Some(
            hub_fetcher::fetch_hf_model_files(repo_id)
                .await
                .expect("Failed to fetch model from Hugging Face Hub"),
        );
    }

    // 2. Auto-compile `config.json` from checkpoint directory if present, otherwise use `--arch` profile
    let arch_arg = args
        .windows(2)
        .find(|w| w[0] == "--arch")
        .map(|w| w[1].as_str())
        .unwrap_or("llama");

    let checkpoint_dir = checkpoint_dir.unwrap_or_else(|| {
        eprintln!("KriyaRT requires --model <local_checkpoint_dir_or_hf_repo> for serving or benchmarking. Synthetic execution is test-only.");
        std::process::exit(2);
    });

    let mut desc = HuggingFaceModelConfig::from_config_file(checkpoint_dir.join("config.json"))
        .ok()
        .unwrap_or_else(|| match arch_arg {
            a if a.contains("qwen") => ModelArchDescriptor::qwen3_6_35b_a3b(),
            a if a.contains("mistral") => ModelArchDescriptor::mistral_7b_v0_3(),
            a if a.contains("gemma") => ModelArchDescriptor::gemma_2_9b(),
            a if a.contains("deepseek") => ModelArchDescriptor::deepseek_r1_distill_qwen_7b(),
            _ => ModelArchDescriptor::llama_3_2_3b(),
        });

    if let Some(custom_layers) = args
        .windows(2)
        .find(|w| w[0] == "--layers")
        .and_then(|w| w[1].parse::<usize>().ok())
    {
        if !args.iter().any(|a| a == "--allow-partial-model") {
            eprintln!(
                "Partial-layer execution requires --allow-partial-model and is not a valid model benchmark."
            );
            std::process::exit(2);
        }
        desc.layers.truncate(custom_layers);
    }

    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
        .join("metal");

    let draft_weights_path = args
        .windows(2)
        .find(|w| w[0] == "--draft-weights")
        .map(|w| PathBuf::from(&w[1]));

    let mut runtime = EngineRuntime::new(
        desc.clone(),
        &shader_dir,
        &checkpoint_dir,
        draft_weights_path.as_deref(),
    )
    .unwrap_or_else(|err| {
        eprintln!("KriyaRT startup failed: {err}");
        std::process::exit(1);
    });

    if args.iter().any(|arg| arg == "--no-speculation") {
        runtime.draft_engine = None;
    }

    if runtime.draft_engine.is_some() {
        println!("KriyaRT: Speculative decoding ACTIVE with EAGLE draft head.");
    } else {
        println!(
            "KriyaRT: Speculative decoding INACTIVE (standard single-token autoregressive decode). Pass --draft-weights to enable EAGLE."
        );
    }

    let has_prompt = args.iter().any(|arg| arg == "--prompt");
    let has_bench_only = args.iter().any(|arg| arg == "--bench-only");

    let benchmark_report = if has_bench_only
        || (!args.iter().any(|arg| arg == "--skip-startup-benchmark") && !has_prompt)
    {
        Some(runtime.run_hardware_benchmark().await)
    } else {
        None
    };

    if let Some(prompt_str) = args
        .windows(2)
        .find(|window| window[0] == "--prompt")
        .map(|window| window[1].clone())
    {
        let max_tokens = args
            .windows(2)
            .find(|w| w[0] == "--max-tokens")
            .and_then(|w| w[1].parse::<usize>().ok())
            .unwrap_or(32);
        let temp = args
            .windows(2)
            .find(|w| w[0] == "--temp" || w[0] == "--temperature")
            .and_then(|w| w[1].parse::<f32>().ok())
            .unwrap_or(0.0);
        let top_p = args
            .windows(2)
            .find(|w| w[0] == "--top-p")
            .and_then(|w| w[1].parse::<f32>().ok())
            .unwrap_or(0.95);

        runtime
            .run_direct_generation(&prompt_str, max_tokens, temp, top_p)
            .await;
        return;
    }

    if let Some(report_path) = args
        .windows(2)
        .find(|window| window[0] == "--benchmark-json")
        .map(|window| PathBuf::from(&window[1]))
    {
        let Some(report) = benchmark_report.as_ref() else {
            eprintln!(
                "--benchmark-json requires the startup benchmark; remove --skip-startup-benchmark"
            );
            std::process::exit(2);
        };
        let report_bytes = serde_json::to_vec_pretty(report).unwrap_or_else(|err| {
            eprintln!("Could not serialize benchmark report: {err}");
            std::process::exit(1);
        });
        if let Err(err) = std::fs::write(&report_path, report_bytes) {
            eprintln!(
                "Could not write benchmark report {}: {err}",
                report_path.display()
            );
            std::process::exit(1);
        }
        println!(
            "KriyaRT benchmark report written to {}",
            report_path.display()
        );
    }

    if args.iter().any(|a| a == "--bench-only") {
        return;
    }

    let tokenizer = runtime.tokenizer.clone();
    let (work_tx, work_rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        orca_batch_worker_loop(runtime, work_rx).await;
    });

    let host = args
        .windows(2)
        .find(|w| w[0] == "--host")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = args
        .windows(2)
        .find(|w| w[0] == "--port")
        .and_then(|w| w[1].parse::<u16>().ok())
        .unwrap_or(8080);

    if host == "0.0.0.0" {
        eprintln!(
            "WARNING: Binding to 0.0.0.0 exposes KriyaRT to network interfaces without authentication. Default 127.0.0.1 is recommended for local deployments."
        );
    }

    let bind_addr = format!("{host}:{port}");
    let metrics = Arc::new(ServerMetrics::default());

    let handle = ServerHandle {
        work_tx,
        tokenizer,
        model_name: desc.model_name.clone(),
        metrics,
        max_prompt_tokens: 8192,
        max_completion_tokens: 4096,
        max_concurrent_requests: 128,
    };

    let app = create_app(handle);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .unwrap_or_else(|err| {
            eprintln!("Failed to bind server to {bind_addr}: {err}");
            std::process::exit(1);
        });
    println!(
        "KriyaRT Production Server (ORCA continuous batching) listening on http://{}",
        bind_addr
    );
    let shutdown_signal = async {
        let _ = tokio::signal::ctrl_c().await;
        println!("\nKriyaRT server received termination signal; shutting down gracefully...");
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await
        .unwrap();
    println!("KriyaRT server stopped successfully.");
}
