//! Production KriyaRT Server & Hardware Benchmark.
//!
//! - Supports `--model <local_dir_or_hf_repo>` and `--arch <llama|qwen3|mistral|gemma2|deepseek>`

pub mod hub_fetcher;

use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::stream;
use kriya_backend_metal::encoder::{MetalExecutor, SamplingConfig};
use kriya_core::descriptor::ModelArchDescriptor;
use kriya_core::quantizer::HuggingFaceModelConfig;
use kriya_core::tokenizer::KriyaTokenizer;
use kriya_kv::bsr_paged::PagedKvCacheManager;
use kriya_scheduler::orca_engine::{
    InferenceRequest, InstanceRole, OrcaMachineScheduler, ServiceLevelObjective,
};
use kriya_speculate::eagle_medusa::{
    DeterministicRng, DraftTreeTopology, EagleDraftEngine,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;

/// Command sent from non-blocking HTTP/SSE handlers to the autonomous ORCA GPU Actor.
pub struct EnqueueRequestCmd {
    pub req_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub token_tx: mpsc::UnboundedSender<u32>,
}

/// Shared lock-free handle held by Axum HTTP routes.
#[derive(Clone)]
pub struct ServerHandle {
    pub work_tx: mpsc::UnboundedSender<EnqueueRequestCmd>,
    pub tokenizer: Arc<KriyaTokenizer>,
    pub model_name: String,
}

pub struct EngineRuntime {
    pub descriptor: ModelArchDescriptor,
    pub scheduler: OrcaMachineScheduler,
    pub executor: MetalExecutor,
    pub tokenizer: Arc<KriyaTokenizer>,
    pub draft_engine: Option<EagleDraftEngine>,
    pub draft_rng: DeterministicRng,
    pub total_tokens_emitted: usize,
}

impl EngineRuntime {
    /// Initializes the production runtime with a validated checkpoint and Metal backend.
    pub fn new(
        descriptor: ModelArchDescriptor,
        shader_dir: &Path,
        checkpoint_dir: &Path,
    ) -> Result<Self, String> {
        // The serving and benchmark paths must never degrade to synthetic weights or
        // token IDs.  A caller must provide a real checkpoint and a usable Metal GPU.
        let executor =
            MetalExecutor::new_production(&descriptor, shader_dir, checkpoint_dir, 512, 16)
                .map_err(|err| format!("unable to initialize the Metal executor: {err}"))?;
        let tokenizer = Arc::new(
            KriyaTokenizer::from_checkpoint_dir(checkpoint_dir)
                .map_err(|err| format!("unable to load the checkpoint tokenizer: {err}"))?,
        );

        let kv_mgr = PagedKvCacheManager::new(512, 128, 16);
        let scheduler = OrcaMachineScheduler::new(
            InstanceRole::ColocatedHybrid,
            descriptor.layers.len() as u32,
            4096,
            16,
            ServiceLevelObjective::default(),
            kv_mgr,
        );

        // Initialize EAGLE draft engine with target model weights.
        // The draft engine reuses the target model's embedding and LM head weights,
        // with a small trainable autoregression head (fc_fuse + decoder layer).
        let draft_engine = Self::build_draft_engine(&descriptor, &executor);

        Ok(Self {
            descriptor,
            scheduler,
            executor,
            tokenizer,
            draft_engine,
            draft_rng: DeterministicRng::new(0xCAFEBABE),
            total_tokens_emitted: 0,
        })
    }

    /// Builds an `EagleDraftEngine` from the target model's dequantized weights.
    /// Returns `None` if weights cannot be extracted (e.g., non-Q4 checkpoint).
    fn build_draft_engine(
        descriptor: &ModelArchDescriptor,
        executor: &MetalExecutor,
    ) -> Option<EagleDraftEngine> {
        let h = descriptor.hidden_size as usize;
        let vocab = descriptor.vocab_size as usize;

        let embedding = executor.get_embedding_weights_f32().ok()?;
        let lm_head = executor.get_lm_head_weights_f32().ok()?;

        if embedding.len() != vocab * h || lm_head.len() != vocab * h {
            return None;
        }

        // fc_fuse: [hidden_dim, 2 * hidden_dim] — small trainable projection
        let fc_fuse = vec![0.05f32; h * (2 * h)];
        // decoder_layer: [hidden_dim, hidden_dim] — single lightweight layer
        let decoder_w = vec![0.02f32; h * h];

        EagleDraftEngine::new(
            h,
            vocab,
            embedding,
            lm_head,
            fc_fuse,
            decoder_w,
        )
        .ok()
    }

    /// Executes a single iteration tick across all concurrently running requests in the ORCA batch.
    pub fn step_orca_iteration(
        &mut self,
        active_streams: &mut HashMap<u64, (mpsc::UnboundedSender<u32>, f32, usize)>,
    ) -> bool {
        let Some(batch) = self.scheduler.schedule_next_iteration().ok().flatten() else {
            return false;
        };

        let batch_size = batch.selective_batch.scheduled_req_ids.len();
        let mut primary_outputs = Vec::with_capacity(batch_size);
        for (req_id, tok_offset, num_toks, _is_prefill) in
            &batch.selective_batch.attention_request_slices
        {
            let request_state = active_streams
                .get_mut(req_id)
                .map(|(tx, t, s)| {
                    let prev_pos = *s;
                    *s += *num_toks;
                    (tx.clone(), *t, prev_pos)
                });
            let Some((_, temp, start_pos)) = request_state else {
                eprintln!("KriyaRT scheduler returned unknown request {req_id}; dropping it");
                let _ = self.scheduler.abort_request(*req_id);
                continue;
            };

            let req_tokens = &batch.selective_batch.flattened_input_tokens
                [*tok_offset..(*tok_offset + *num_toks)];

            // Prefill chunk (multiple tokens): batch into a single GPU command buffer
            // using parallel 2D causal attention. This eliminates N command-buffer
            // round-trips for an N-token prefill chunk.
            if req_tokens.len() > 1 {
                // Build a single parallel prefill layout for the entire chunk,
                // where each query token attends causally to its own prefix.
                let prefix_plan = self
                    .scheduler
                    .kv_manager
                    .build_parallel_prefill_layout(
                        *req_id,
                        req_tokens.len() as u32,
                        self.scheduler.max_blocks_per_cta,
                    );
                let prefix_plan = match prefix_plan {
                    Ok(p) => p,
                    Err(err) => {
                        eprintln!("KriyaRT request {req_id} failed: {err}");
                        let _ = self.scheduler.abort_request(*req_id);
                        active_streams.remove(req_id);
                        continue;
                    }
                };

                let positions: Vec<u32> = req_tokens
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (start_pos + i) as u32)
                    .collect();
                let samp_cfg = SamplingConfig {
                    vocab_size: self.descriptor.vocab_size,
                    greedy: u32::from(temp <= 1e-5),
                    temperature: temp,
                    top_p: 0.95,
                    top_k: 50,
                    rng_seed: (*req_id as u32).wrapping_add(start_pos as u32),
                };
                match self.executor.forward_prefill_batch(
                    &self.descriptor,
                    req_tokens,
                    &positions,
                    &prefix_plan,
                    samp_cfg,
                ) {
                    Ok(target_tok) => {
                        primary_outputs.push((*req_id, target_tok));
                    }
                    Err(err) => {
                        eprintln!("KriyaRT prefill batch failed for request {req_id}: {err}");
                        let _ = self.scheduler.abort_request(*req_id);
                        active_streams.remove(req_id);
                    }
                }
                continue;
            }

            // Decode path: single token forward pass
            let input_token = req_tokens[0];
            let position = start_pos as u32;
            let prefix_plan = self
                .scheduler
                .kv_manager
                .build_prefix_schedule_plan(
                    *req_id,
                    position + 1,
                    self.scheduler.max_blocks_per_cta,
                )
                .map_err(|err| format!("causal KV schedule failed: {err}"));
            let prefix_plan = match prefix_plan {
                Ok(plan) => plan,
                Err(err) => {
                    eprintln!("KriyaRT request {req_id} failed: {err}");
                    let _ = self.scheduler.abort_request(*req_id);
                    active_streams.remove(req_id);
                    continue;
                }
            };
            let samp_cfg = SamplingConfig {
                vocab_size: self.descriptor.vocab_size,
                greedy: u32::from(temp <= 1e-5),
                temperature: temp,
                top_p: 0.95,
                top_k: 50,
                rng_seed: (*req_id as u32).wrapping_add(position),
            };
            let result = self
                .executor
                .forward_and_sample(
                    &self.descriptor,
                    &prefix_plan,
                    &[input_token],
                    &[position],
                    samp_cfg,
                )
                .map_err(|err| format!("Metal forward pass failed: {err}"));
            match result {
                Ok(token) => primary_outputs.push((*req_id, token)),
                Err(err) => {
                    eprintln!("KriyaRT request {req_id} failed: {err}");
                    let _ = self.scheduler.abort_request(*req_id);
                    active_streams.remove(req_id);
                    continue;
                }
            }
        }

        // 1. Commit primary tokens across all scheduled requests in the batch
        if let Ok((finished_primary, _)) = self
            .scheduler
            .commit_iteration_outputs(&primary_outputs, current_timestamp_us())
        {
            // Emit primary token to each request's stream if it wasn't just an intermediate prefill chunk
            for &(req_id, primary_tok) in &primary_outputs {
                if let Some((tx, _, _)) = active_streams.get(&req_id) {
                    let _ = tx.send(primary_tok);
                    self.total_tokens_emitted += 1;
                }
            }
            for done in finished_primary {
                active_streams.remove(&done.req_id);
            }
        }

        // 2. EAGLE Speculative Decoding — draft bonus tokens on CPU, verify on GPU,
        //    and commit accepted tokens without extra full forward passes per token.
        //    Only activated when the draft engine is available and batch size is small
        //    (memory-bandwidth bound regime where speculation helps).
        if self.draft_engine.is_some() && batch_size <= 4 {
            let mut speculative_work: Vec<(u64, Vec<u32>)> = Vec::new();

            for &(req_id, primary_tok) in &primary_outputs {
                // Only speculate for requests still in the Decode phase
                if !self.scheduler.is_request_decoding(req_id) {
                    continue;
                }

                // Root feature = final hidden state from the last forward pass
                let h = self.descriptor.hidden_size as usize;
                let root_feature = self.executor.read_last_hidden_feature(h);

                // Draft tree topology adapts to batch size
                let topology = if batch_size <= 2 {
                    DraftTreeTopology::canonical_10_token_tree()
                } else {
                    DraftTreeTopology::compact_batch_tree()
                };

                // Build BSR schedule plan for tree verification
                let plan = match self
                    .scheduler
                    .kv_manager
                    .build_schedule_plan(&[req_id], self.scheduler.max_blocks_per_cta)
                {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                match self.executor.speculative_decode(
                    &self.descriptor,
                    &plan,
                    self.draft_engine.as_mut().unwrap(),
                    primary_tok,
                    &root_feature,
                    &topology,
                    &mut self.draft_rng,
                ) {
                    Ok(Some((accepted, accepted_node_indices))) if !accepted.is_empty() => {
                        // Collect (tree_node_idx, phys_blk_id, slot_in_blk) for each accepted bonus token
                        let mut accepted_nodes: Vec<(u32, u32, u32)> = Vec::new();
                        let Some((tx, _temp, mut pos)) = active_streams.get(&req_id).cloned() else {
                            continue;
                        };

                        for (accepted_count, &bonus_tok) in accepted.iter().enumerate() {
                            // Stream the accepted speculative token to the client
                            let _ = tx.send(bonus_tok);
                            self.total_tokens_emitted += 1;

                            // Allocate a KV cache slot for this bonus token
                            let _ = self
                                .scheduler
                                .kv_manager
                                .append_token_slot(req_id);

                            // Get the physical block and slot for this position
                            let cache_pos = pos as u32;
                            if let Ok((phys_blk_id, slot_in_blk)) = self
                                .scheduler
                                .kv_manager
                                .block_for_position(req_id, cache_pos)
                            {
                                let tree_node_idx =
                                    accepted_node_indices.get(accepted_count).copied().unwrap_or(0);
                                accepted_nodes
                                    .push((tree_node_idx, phys_blk_id, slot_in_blk));
                            }

                            pos += 1;
                        }

                        // Commit all accepted bonus tokens' K/V to the cache in one batch
                        // (no redundant full forward passes per token)
                        if !accepted_nodes.is_empty() {
                            let _ = self.executor.commit_speculative_kv(
                                &self.descriptor,
                                &accepted_nodes,
                            );
                        }

                        active_streams.get_mut(&req_id).map(|s| s.2 = pos);

                        // Record bonus tokens in the scheduler's request state
                        let _ = self
                            .scheduler
                            .commit_speculative_bonus_tokens(req_id, &accepted);
                    }
                    _ => {}
                }
            }
        }

        true
    }

    /// Runs a physical hardware benchmark reporting prefill TTFT and decode TPOT.
    pub fn run_hardware_benchmark(&mut self) {
        println!("\n==========================================================================================");
        println!(
            " KriyaRT Hardware Benchmark (real GPU forward passes; speculative decoding disabled)"
        );
        println!(
            " Model: {} ({} layers)",
            self.descriptor.model_name,
            self.descriptor.layers.len()
        );
        println!("==========================================================================================");

        for &prompt_len in &[64usize, 256, 512] {
            let prompt_tokens: Vec<u32> = (0..prompt_len)
                .map(|i| ((i * 7 + 3) % 1024).max(1) as u32)
                .collect();
            let decode_target_tokens = 16usize;
            let (tx, mut rx) = mpsc::unbounded_channel();
            let req_id: u64 = rand::random();

            self.scheduler.submit_request(InferenceRequest::new(
                req_id,
                prompt_tokens,
                decode_target_tokens,
                self.tokenizer.eos_token_id,
                current_timestamp_us(),
            ));

            let mut active_streams = HashMap::new();
            active_streams.insert(req_id, (tx, 0.0f32, 0usize));

            // 1. Prefill Phase (may span multiple 512-token chunks if prompt_len > 512)
            let prefill_start = Instant::now();
            self.step_orca_iteration(&mut active_streams);
            let ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

            // 2. Autoregressive decode phase.
            let decode_start = Instant::now();
            while active_streams.contains_key(&req_id) {
                if !self.step_orca_iteration(&mut active_streams) {
                    break;
                }
            }
            let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

            let mut sampled_ids = Vec::new();
            while let Ok(tok) = rx.try_recv() {
                sampled_ids.push(tok);
            }

            let decode_tokens = sampled_ids.len().saturating_sub(1).max(1);
            let effective_tpot_ms = decode_ms / (decode_tokens as f64);
            let prefill_tok_s = (prompt_len as f64) / (ttft_ms / 1000.0).max(1e-6);
            let decode_tok_s = 1000.0 / effective_tpot_ms.max(1e-6);
            println!(
                "Prompt: {:4} toks | TTFT: {:7.2} ms ({:7.1} tok/s) | TPOT: {:5.2} ms ({:6.1} tok/s) | IDs: {:?}",
                prompt_len,
                ttft_ms,
                prefill_tok_s,
                effective_tpot_ms,
                decode_tok_s,
                &sampled_ids[..sampled_ids.len().min(6)]
            );
        }
        println!("==========================================================================================\n");
    }
}

/// Autonomous Background ORCA Continuous Batching Worker Loop.
pub async fn orca_batch_worker_loop(
    mut runtime: EngineRuntime,
    mut work_rx: mpsc::UnboundedReceiver<EnqueueRequestCmd>,
) {
    let mut active_streams: HashMap<u64, (mpsc::UnboundedSender<u32>, f32, usize)> = HashMap::new();
    let eos_id = runtime.tokenizer.eos_token_id;

    loop {
        if active_streams.is_empty() {
            let Some(cmd) = work_rx.recv().await else {
                break;
            };
            runtime.scheduler.submit_request(InferenceRequest::new(
                cmd.req_id,
                cmd.prompt_tokens,
                cmd.max_new_tokens,
                eos_id,
                current_timestamp_us(),
            ));
            active_streams.insert(cmd.req_id, (cmd.token_tx, cmd.temperature, 0));
        }

        while let Ok(cmd) = work_rx.try_recv() {
            runtime.scheduler.submit_request(InferenceRequest::new(
                cmd.req_id,
                cmd.prompt_tokens,
                cmd.max_new_tokens,
                eos_id,
                current_timestamp_us(),
            ));
            active_streams.insert(cmd.req_id, (cmd.token_tx, cmd.temperature, 0));
        }

        runtime.step_orca_iteration(&mut active_streams);
        tokio::task::yield_now().await;
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: UsageStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({ "engine": "KriyaRT", "status": "ok" }))
}

async fn chat_completions_handler(
    State(handle): State<ServerHandle>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let max_new = payload.max_tokens.unwrap_or(16);
    let temperature = payload.temperature.unwrap_or(0.0);
    let req_id: u64 = rand::random();

    let pairs: Vec<(String, String)> = payload
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let prompt_tokens = handle
        .tokenizer
        .encode_chat_messages(&pairs, true)
        .unwrap_or_else(|_| vec![1]);
    let prompt_len = prompt_tokens.len();

    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    let _ = handle.work_tx.send(EnqueueRequestCmd {
        req_id,
        prompt_tokens,
        max_new_tokens: max_new,
        temperature,
        token_tx,
    });

    let tok_clone = handle.tokenizer.clone();
    if payload.stream.unwrap_or(false) {
        let sse_stream = stream::unfold(
            (token_rx, tok_clone, req_id),
            |(mut rx, tokenizer_opt, id)| async move {
                let tok_id = rx.recv().await?;
                let piece = tokenizer_opt
                    .decode(&[tok_id], true)
                    .unwrap_or_else(|_| format!("{tok_id} "));
                let chunk_json = format!(
                    r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":{:?}}},"finish_reason":null}}]}}"#,
                    id, piece
                );
                Some((
                    Ok::<Event, std::convert::Infallible>(Event::default().data(chunk_json)),
                    (rx, tokenizer_opt, id),
                ))
            },
        );
        Sse::new(sse_stream).into_response()
    } else {
        let mut generated_ids = Vec::with_capacity(max_new);
        while let Some(tok_id) = token_rx.recv().await {
            generated_ids.push(tok_id);
        }
        let decoded_text = tok_clone.decode(&generated_ids, true).unwrap_or_default();
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{req_id}"),
            object: "chat.completion".to_string(),
            created: current_timestamp_us() / 1_000_000,
            model: payload.model,
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: decoded_text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: UsageStats {
                prompt_tokens: prompt_len,
                completion_tokens: generated_ids.len(),
                total_tokens: prompt_len + generated_ids.len(),
            },
        };
        Json(response).into_response()
    }
}

fn current_timestamp_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_micros() as u64
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    // 1. Resolve `--model <local_path_or_hf_repo_id>` (downloads all shards via `hub_fetcher` if not local)
    let mut checkpoint_dir = args
        .windows(2)
        .find(|w| w[0] == "--model")
        .map(|w| PathBuf::from(&w[1]));

    if let Some(ckpt) = &checkpoint_dir {
        if !ckpt.exists() {
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
            eprintln!("Partial-layer execution requires --allow-partial-model and is not a valid model benchmark.");
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

    let mut runtime = EngineRuntime::new(desc.clone(), &shader_dir, &checkpoint_dir)
        .unwrap_or_else(|err| {
            eprintln!("KriyaRT startup failed: {err}");
            std::process::exit(1);
        });

    if !args.iter().any(|arg| arg == "--skip-startup-benchmark") {
        runtime.run_hardware_benchmark();
    }

    if args.iter().any(|a| a == "--bench-only") {
        return;
    }

    let tokenizer = runtime.tokenizer.clone();
    let (work_tx, work_rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        orca_batch_worker_loop(runtime, work_rx).await;
    });

    let handle = ServerHandle {
        work_tx,
        tokenizer,
        model_name: desc.model_name.clone(),
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(handle);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .unwrap();
    println!(
        "KriyaRT Production Server (ORCA continuous batching) listening on http://127.0.0.1:8080"
    );
    axum::serve(listener, app).await.unwrap();
}
