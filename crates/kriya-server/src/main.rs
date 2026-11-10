//! Production KriyaRT Server & Hardware Benchmark.
//!
//! - Supports `--model <local_dir_or_hf_repo>` and `--arch <llama|qwen3|mistral|gemma2|deepseek>`

pub mod hub_fetcher;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
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
use kriya_speculate::eagle_medusa::{DeterministicRng, DraftTreeTopology, EagleDraftEngine};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;

/// State maintained for each concurrently running request stream.
#[derive(Clone)]
pub struct ActiveStreamState {
    pub tx: mpsc::UnboundedSender<u32>,
    pub temperature: f32,
    pub top_p: f32,
    pub start_pos: usize,
}

/// Command sent from non-blocking HTTP/SSE handlers to the autonomous ORCA GPU Actor.
pub struct EnqueueRequestCmd {
    pub req_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub token_tx: mpsc::UnboundedSender<u32>,
}

/// Live telemetry & admission metrics for the inference server.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    pub active_requests: AtomicUsize,
    pub total_requests: AtomicU64,
    pub total_tokens_emitted: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    pub rejected_requests: AtomicU64,
}

/// Shared lock-free handle held by Axum HTTP routes.
#[derive(Clone)]
pub struct ServerHandle {
    pub work_tx: mpsc::UnboundedSender<EnqueueRequestCmd>,
    pub tokenizer: Arc<KriyaTokenizer>,
    pub model_name: String,
    pub metrics: Arc<ServerMetrics>,
    pub max_prompt_tokens: usize,
    pub max_completion_tokens: usize,
    pub max_concurrent_requests: usize,
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

/// One reproducible point from the built-in hardware benchmark.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkSample {
    pub prompt_tokens: usize,
    /// Median Time-To-First-Token across benchmark runs (ms)
    pub ttft_ms: f64,
    /// P50 TTFT across benchmark runs (ms)
    pub ttft_p50_ms: f64,
    /// P95 TTFT across benchmark runs (ms)
    pub ttft_p95_ms: f64,
    /// Prefill tokens per second (based on median TTFT)
    pub prefill_tokens_per_second: f64,
    /// Median Time-Per-Output-Token across benchmark runs (ms)
    pub tpot_ms: f64,
    /// P50 TPOT across benchmark runs (ms)
    pub tpot_p50_ms: f64,
    /// P95 TPOT across benchmark runs (ms)
    pub tpot_p95_ms: f64,
    /// Decode tokens per second (based on median TPOT)
    pub decode_tokens_per_second: f64,
    pub sampled_token_ids: Vec<u32>,
}

/// Computes the p-th percentile from a sorted slice of f64 values.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = (rank.ceil() as usize).min(sorted.len() - 1);
    let frac = rank - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * frac
}

/// Machine-readable evidence for a single KriyaRT benchmark invocation.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub engine: &'static str,
    pub model_name: String,
    pub model_family: String,
    pub model_weight_format: &'static str,
    pub model_layers: usize,
    pub speculative_decoding_enabled: bool,
    pub measured_at_unix_s: u64,
    pub samples: Vec<BenchmarkSample>,
}

impl EngineRuntime {
    /// Initializes the production runtime with a validated checkpoint and Metal backend.
    pub fn new(
        descriptor: ModelArchDescriptor,
        shader_dir: &Path,
        checkpoint_dir: &Path,
        draft_weights_path: Option<&Path>,
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
            8, // 8 blocks * 16 tokens = 128 tokens threshold before split-K
            ServiceLevelObjective::default(),
            kv_mgr,
        );

        let draft_engine =
            Self::build_draft_engine(&descriptor, &executor, checkpoint_dir, draft_weights_path);

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

    /// Builds EAGLE only from an explicitly supplied, trained draft-head artifact.
    /// Missing, malformed, and incompatible artifacts leave speculation disabled.
    fn build_draft_engine(
        descriptor: &ModelArchDescriptor,
        executor: &MetalExecutor,
        checkpoint_dir: &Path,
        explicit_draft_path: Option<&Path>,
    ) -> Option<EagleDraftEngine> {
        let h = descriptor.hidden_size as usize;
        let vocab = descriptor.vocab_size as usize;

        let artifact_path = if let Some(p) = explicit_draft_path {
            p.to_path_buf()
        } else {
            let default_p = checkpoint_dir.join("kriya_eagle_draft_v1.bin");
            if !default_p.exists() {
                return None;
            }
            default_p
        };

        let draft_weights = match std::fs::read(&artifact_path) {
            Ok(data) => data,
            Err(err) => {
                eprintln!(
                    "KriyaRT: speculative decoding disabled; failed to read draft head artifact at {}: {err}",
                    artifact_path.display()
                );
                return None;
            }
        };

        // Release integrity check: If kriya-artifact-manifest.json exists, verify checksum
        let manifest_path = checkpoint_dir.join("kriya-artifact-manifest.json");
        if manifest_path.is_file()
            && let Ok(manifest_bytes) = std::fs::read(&manifest_path)
            && let Ok(manifest_json) = serde_json::from_slice::<serde_json::Value>(&manifest_bytes)
            && let Some(spec_file) = manifest_json.get("speculative_draft_file")
            && let Some(expected_hash) = spec_file.get("sha256").and_then(|h| h.as_str())
        {
            let mut context = ring::digest::Context::new(&ring::digest::SHA256);
            context.update(&draft_weights);
            let computed_hash: String = context
                .finish()
                .as_ref()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            if computed_hash != expected_hash {
                eprintln!(
                    "KriyaRT release integrity error: draft head checksum mismatch!\n  Expected: {}\n  Computed: {}",
                    expected_hash, computed_hash
                );
                return None;
            }
            println!(
                "KriyaRT: Verified EAGLE draft head artifact integrity against manifest (SHA-256: {}...)",
                &expected_hash[..12]
            );
        }

        let embedding = executor.get_embedding_weights_f32().ok()?;
        let lm_head = executor.get_lm_head_weights_f32().ok()?;

        if embedding.len() != vocab * h || lm_head.len() != vocab * h {
            return None;
        }

        // Zeroes are only construction storage. `load_weights` must succeed before
        // the engine can be returned, so they can never act as a draft model.
        let mut engine = EagleDraftEngine::new(
            h,
            vocab,
            embedding,
            lm_head,
            vec![0.0; h * (2 * h)],
            vec![0.0; h * h],
        )
        .ok()?;
        if let Err(err) = engine.load_weights(&draft_weights) {
            eprintln!(
                "KriyaRT: speculative decoding disabled; invalid trained draft head {}: {err}",
                artifact_path.display()
            );
            return None;
        }
        Some(engine)
    }

    /// Executes a single iteration tick across all concurrently running requests in the ORCA batch.
    ///
    /// Decode requests are batched into a single GPU forward pass via
    /// [`MetalExecutor::forward_and_sample_batch`], eliminating the per-request
    /// weight re-streaming that was the largest serving bottleneck (1Review.html
    /// Bottleneck 1). Prefill requests are executed individually via
    /// [`MetalExecutor::forward_prefill_batch`] since each has a distinct causal
    /// prefix and chunk boundary.
    pub async fn step_orca_iteration(
        &mut self,
        active_streams: &mut HashMap<u64, ActiveStreamState>,
    ) -> bool {
        let Some(batch) = self.scheduler.schedule_next_iteration().ok().flatten() else {
            return false;
        };

        let batch_size = batch.selective_batch.scheduled_req_ids.len();
        let mut primary_outputs = Vec::with_capacity(batch_size);
        let mut client_visible_tokens = HashMap::with_capacity(batch_size);

        // ---------------------------------------------------------------------
        // Separate scheduled requests into prefill and decode groups.
        // Decode requests will be executed in a single batched GPU forward pass.
        // ---------------------------------------------------------------------
        let mut decode_req_ids: Vec<u64> = Vec::new();
        let mut decode_tokens: Vec<u32> = Vec::new();
        let mut decode_positions: Vec<u32> = Vec::new();
        let mut decode_samplings: Vec<SamplingConfig> = Vec::new();
        let mut decode_req_to_output_idx: HashMap<u64, usize> = HashMap::new();

        // Prefill results: (req_id, target_token, should_emit_to_client)
        let mut prefill_results: Vec<(u64, u32, bool)> = Vec::new();

        for (req_id, tok_offset, num_toks, _is_prefill) in
            &batch.selective_batch.attention_request_slices
        {
            let should_emit = self
                .scheduler
                .will_emit_current_iteration_token(*req_id)
                .unwrap_or(false);
            let request_state = active_streams.get_mut(req_id).map(|state| {
                let prev_pos = state.start_pos;
                state.start_pos += *num_toks;
                (state.tx.clone(), state.temperature, state.top_p, prev_pos)
            });
            let Some((_, temp, top_p, start_pos)) = request_state else {
                eprintln!("KriyaRT scheduler returned unknown request {req_id}; dropping it");
                let _ = self.scheduler.abort_request(*req_id);
                active_streams.remove(req_id);
                continue;
            };

            let req_tokens = &batch.selective_batch.flattened_input_tokens
                [*tok_offset..(*tok_offset + *num_toks)];

            // Prefill chunk: multiple tokens, each with its own causal prefix.
            if req_tokens.len() > 1 {
                // Build a single parallel prefill layout for the entire chunk,
                // where each query token attends causally to its own prefix.
                let prefix_plan = self.scheduler.kv_manager.build_parallel_prefill_layout(
                    *req_id,
                    start_pos as u32,
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
                    top_p,
                    top_k: 50,
                    rng_seed: (*req_id as u32).wrapping_add(start_pos as u32),
                };
                match self
                    .executor
                    .forward_prefill_batch(
                        &self.descriptor,
                        req_tokens,
                        &positions,
                        &prefix_plan,
                        samp_cfg,
                    )
                    .await
                {
                    Ok(target_tok) => {
                        prefill_results.push((*req_id, target_tok, should_emit));
                    }
                    Err(err) => {
                        eprintln!("KriyaRT prefill batch failed for request {req_id}: {err}");
                        let _ = self.scheduler.abort_request(*req_id);
                        active_streams.remove(req_id);
                    }
                }
            } else {
                // Decode path: single token — collect for batched execution.
                let input_token = req_tokens[0];
                let position = start_pos as u32;
                let samp_cfg = SamplingConfig {
                    vocab_size: self.descriptor.vocab_size,
                    greedy: u32::from(temp <= 1e-5),
                    temperature: temp,
                    top_p,
                    top_k: 50,
                    rng_seed: (*req_id as u32).wrapping_add(position),
                };
                let out_idx = primary_outputs.len();
                primary_outputs.push((*req_id, 0)); // placeholder, filled after batch
                client_visible_tokens.insert(*req_id, should_emit);
                decode_req_to_output_idx.insert(*req_id, out_idx);
                decode_req_ids.push(*req_id);
                decode_tokens.push(input_token);
                decode_positions.push(position);
                decode_samplings.push(samp_cfg);
            }
        }

        // ---------------------------------------------------------------------
        // Batched decode: single GPU forward pass for ALL decode requests.
        // Model weights are streamed from Unified Memory once per tick instead
        // of once per request.
        // ---------------------------------------------------------------------
        if !decode_tokens.is_empty() {
            let decode_plan = self
                .scheduler
                .kv_manager
                .build_schedule_plan(&decode_req_ids, self.scheduler.max_blocks_per_cta);

            match decode_plan {
                Ok(plan) => {
                    match self
                        .executor
                        .forward_and_sample_batch(
                            &self.descriptor,
                            &decode_tokens,
                            &decode_positions,
                            &plan,
                            &decode_samplings,
                        )
                        .await
                    {
                        Ok(decoded_tokens) => {
                            for (i, token) in decoded_tokens.iter().enumerate() {
                                let req_id = decode_req_ids[i];
                                let out_idx = decode_req_to_output_idx[&req_id];
                                primary_outputs[out_idx] = (req_id, *token);
                            }
                        }
                        Err(err) => {
                            eprintln!("KriyaRT batched decode failed: {err}");
                            for req_id in &decode_req_ids {
                                let _ = self.scheduler.abort_request(*req_id);
                                active_streams.remove(req_id);
                            }
                        }
                    }
                }
                Err(err) => {
                    eprintln!("KriyaRT decode schedule plan failed: {err}");
                    for req_id in &decode_req_ids {
                        let _ = self.scheduler.abort_request(*req_id);
                        active_streams.remove(req_id);
                    }
                }
            }
        }

        // ---------- Combine prefill + decode results ----------
        for (req_id, target_tok, should_emit) in prefill_results {
            primary_outputs.push((req_id, target_tok));
            if should_emit {
                client_visible_tokens.insert(req_id, true);
            }
        }

        // 1. Commit primary tokens across all scheduled requests in the batch
        let mut disconnected_reqs = Vec::new();
        if let Ok((finished_primary, _)) = self
            .scheduler
            .commit_iteration_outputs(&primary_outputs, current_timestamp_us())
        {
            // Intermediate prefill chunks populate KV state only. Their sampled
            // value is discarded; only the final prompt chunk or a decode step
            // creates a client-visible completion token.
            for &(req_id, primary_tok) in &primary_outputs {
                if client_visible_tokens.get(&req_id) == Some(&true)
                    && let Some(state) = active_streams.get(&req_id)
                {
                    if state.tx.send(primary_tok).is_err() {
                        // Client disconnected (e.g. cancelled HTTP/SSE connection).
                        // Abort immediately so KV cache pages are reclaimed.
                        disconnected_reqs.push(req_id);
                    } else {
                        self.total_tokens_emitted += 1;
                    }
                }
            }
            for req_id in disconnected_reqs {
                eprintln!("KriyaRT: client disconnected for request {req_id}; reclaiming KV cache");
                let _ = self.scheduler.abort_request(req_id);
                active_streams.remove(&req_id);
            }
            for done in finished_primary {
                active_streams.remove(&done.req_id);
            }
        }
        //    and commit accepted tokens without extra full forward passes per token.
        //    Only activated when the draft engine is available and batch size is small
        //    (memory-bandwidth bound regime where speculation helps).
        #[allow(clippy::unnecessary_unwrap)]
        // guarded by is_some; avoids moving the engine out of runtime
        if self.draft_engine.is_some() && batch_size <= 4 {
            for &(req_id, primary_tok) in &primary_outputs {
                // Only speculate for requests still in the Decode phase
                if !self.scheduler.is_request_decoding(req_id) {
                    continue;
                }

                // Root feature = final hidden state from the last forward pass
                // TODO: Track per-request row index for multi-request batched decode.
                // For now, read from offset 0 (correct for single-request EAGLE).
                let h = self.descriptor.hidden_size as usize;
                let root_feature = self.executor.read_last_hidden_feature(h, 1);

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

                match self
                    .executor
                    .speculative_decode(
                        &self.descriptor,
                        &plan,
                        self.draft_engine.as_mut().unwrap(),
                        primary_tok,
                        &root_feature,
                        &topology,
                        &mut self.draft_rng,
                    )
                    .await
                {
                    Ok(Some((accepted, accepted_node_indices))) if !accepted.is_empty() => {
                        // The scheduler owns max-token/EOS enforcement and the
                        // KV-slot allocation. Commit before streaming so a draft
                        // branch can never leak tokens past a request limit or
                        // allocate every page twice.
                        let Ok((committed, finished)) = self
                            .scheduler
                            .commit_speculative_bonus_tokens(req_id, &accepted)
                        else {
                            continue;
                        };
                        let Some(state) = active_streams.get(&req_id).cloned() else {
                            continue;
                        };
                        let tx = state.tx;
                        let start_pos = state.start_pos;

                        // A terminal sequence has already released its KV cache;
                        // no future forward pass can observe its final cache row.
                        if finished.is_none() {
                            let mut accepted_nodes = Vec::with_capacity(committed.len());
                            for (index, _) in committed.iter().enumerate() {
                                let cache_pos = (start_pos + index) as u32;
                                if let Ok((phys_blk_id, slot_in_blk)) = self
                                    .scheduler
                                    .kv_manager
                                    .block_for_position(req_id, cache_pos)
                                {
                                    let tree_node_idx =
                                        accepted_node_indices.get(index).copied().unwrap_or(0);
                                    accepted_nodes.push((tree_node_idx, phys_blk_id, slot_in_blk));
                                }
                            }
                            if !accepted_nodes.is_empty()
                                && self
                                    .executor
                                    .commit_speculative_kv(&self.descriptor, &accepted_nodes)
                                    .is_err()
                            {
                                eprintln!(
                                    "KriyaRT could not commit speculative KV for request {req_id}"
                                );
                                let _ = self.scheduler.abort_request(req_id);
                                active_streams.remove(&req_id);
                                continue;
                            }
                        }

                        let mut client_disconnected = false;
                        for &bonus_tok in &committed {
                            if tx.send(bonus_tok).is_err() {
                                client_disconnected = true;
                                break;
                            }
                            self.total_tokens_emitted += 1;
                        }
                        if client_disconnected {
                            eprintln!(
                                "KriyaRT: client disconnected during speculative emission for request {req_id}; reclaiming KV cache"
                            );
                            let _ = self.scheduler.abort_request(req_id);
                            active_streams.remove(&req_id);
                            continue;
                        }
                        if let Some(state) = active_streams.get_mut(&req_id) {
                            state.start_pos += committed.len();
                        }
                        if finished.is_some() {
                            active_streams.remove(&req_id);
                        }
                    }
                    _ => {}
                }
            }
        }
        true
    }

    /// Runs a physical hardware benchmark reporting prefill TTFT and decode TPOT on real English prompts.
    pub async fn run_hardware_benchmark(&mut self) -> BenchmarkReport {
        let speculation_enabled = self.draft_engine.is_some();
        println!(
            "\n=========================================================================================="
        );
        println!(
            " KriyaRT Hardware Benchmark (real GPU forward passes; speculative decoding {})",
            if speculation_enabled {
                "enabled"
            } else {
                "disabled"
            }
        );
        println!(
            " Model: {} ({} layers)",
            self.descriptor.model_name,
            self.descriptor.layers.len()
        );
        println!(
            "=========================================================================================="
        );

        let benchmark_texts = [
            // ~64 tokens: Natural language prompt
            "The quick brown fox jumps over the lazy dog. In computer science and artificial intelligence, inference engines optimize tensor operations to achieve low latency and high throughput. What are the key architectural advantages of Apple Silicon unified memory for running large language models locally on macOS systems with high efficiency?",
            // ~256 tokens: Extended technical prompt
            "Artificial intelligence, machine learning, and neural network architectures have undergone a massive evolution over the last several decades. From early perceptrons and connectionist models to deep convolutional neural networks and modern transformer architectures, computation requirements have shifted dramatically from general-purpose CPUs toward specialized vector and matrix engines. On Apple Silicon systems, the GPU and Neural Engine share unified LPDDR memory with the high-performance CPU cores. This shared memory architecture allows large model weights to be mapped zero-copy without PCIe bus bottlenecks. Describe how modern inference runtime systems leverage FlashAttention, Block-Sparse Row paged key-value caches, and 4-bit block quantization to maximize hardware performance and memory bandwidth utilization.",
            // ~512 tokens: Long technical benchmark prompt
            "Over the past few years, the development of large language models has accelerated at an unprecedented pace. Beginning with the original Transformer architecture introduced in 2017, self-attention mechanisms have become the foundational building block for state-of-the-art natural language processing, code generation, and multimodal reasoning systems. As these models grew from hundreds of millions to hundreds of billions of parameters, the computational and memory demands created significant engineering challenges. In particular, the autoregressive decoding phase is inherently memory-bandwidth-bound: generating each new token requires streaming billions of weight parameters from DRAM into compute registers while performing relatively few arithmetic operations per loaded byte. To alleviate these bottlenecks on modern client hardware such as Apple Silicon Macs, modern inference engines deploy a combination of specialized techniques. First, weight-only quantization (such as 4-bit grouped affine quantization) reduces memory bandwidth consumption by fourfold compared to 16-bit floating point representations. Second, paged attention and block-sparse row data structures eliminate KV cache memory fragmentation by allocating physical memory in discrete pages rather than contiguous buffers. Third, speculative decoding algorithms leverage smaller draft heads or speculative trees to propose multiple candidate tokens per step, verifying them in parallel with a single forward pass of the larger target model. Explain the synergistic effects of these optimizations on Time To First Token and Time Per Output Token.",
        ];

        // Warmup round to eliminate GPU driver shader compilation and unified memory page-table cold starts
        {
            let warmup_tokens = self
                .tokenizer
                .encode("Warmup hello", false)
                .unwrap_or_else(|_| vec![1, 2, 3, 4]);
            let (warmup_tx, mut warmup_rx) = mpsc::unbounded_channel();
            let warmup_id: u64 = 0xDEADBEEF;
            self.scheduler.submit_request(InferenceRequest::new(
                warmup_id,
                warmup_tokens,
                2,
                self.tokenizer.eos_token_id,
                current_timestamp_us(),
            ));
            let mut warmup_streams = HashMap::new();
            warmup_streams.insert(
                warmup_id,
                ActiveStreamState {
                    tx: warmup_tx,
                    temperature: 0.0,
                    top_p: 0.95,
                    start_pos: 0,
                },
            );
            while warmup_streams.contains_key(&warmup_id) {
                if !self.step_orca_iteration(&mut warmup_streams).await {
                    break;
                }
            }
            while warmup_rx.try_recv().is_ok() {}
        }

        let mut samples = Vec::new();
        const NUM_BENCHMARK_RUNS: usize = 3;

        for (idx, &prompt_len) in [64usize, 256, 512].iter().enumerate() {
            let base_text = benchmark_texts[idx];

            // Run multiple passes and collect measurements for percentile reporting.
            let mut ttft_measurements = Vec::with_capacity(NUM_BENCHMARK_RUNS);
            let mut tpot_measurements = Vec::with_capacity(NUM_BENCHMARK_RUNS);
            let mut sampled_ids_final = Vec::new();

            for run in 0..NUM_BENCHMARK_RUNS {
                let mut prompt_tokens =
                    self.tokenizer.encode(base_text, false).unwrap_or_else(|_| {
                        (0..prompt_len)
                            .map(|i| ((i * 7 + 3) % 1024).max(1) as u32)
                            .collect()
                    });
                if prompt_tokens.len() >= prompt_len {
                    prompt_tokens.truncate(prompt_len);
                } else {
                    let filler = prompt_tokens.clone();
                    while prompt_tokens.len() < prompt_len {
                        prompt_tokens.extend_from_slice(&filler);
                    }
                    prompt_tokens.truncate(prompt_len);
                }

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
                active_streams.insert(
                    req_id,
                    ActiveStreamState {
                        tx,
                        temperature: 0.0,
                        top_p: 0.95,
                        start_pos: 0,
                    },
                );

                // 1. Prefill Phase
                let prefill_start = Instant::now();
                while active_streams.contains_key(&req_id)
                    && !self.scheduler.is_request_decoding(req_id)
                {
                    if !self.step_orca_iteration(&mut active_streams).await {
                        break;
                    }
                }
                let ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

                // 2. Autoregressive decode phase.
                let decode_start = Instant::now();
                while active_streams.contains_key(&req_id) {
                    if !self.step_orca_iteration(&mut active_streams).await {
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

                ttft_measurements.push(ttft_ms);
                tpot_measurements.push(effective_tpot_ms);
                if run == NUM_BENCHMARK_RUNS - 1 {
                    sampled_ids_final = sampled_ids;
                }
            }

            // Sort for percentile computation.
            ttft_measurements.sort_by(|a, b| a.partial_cmp(b).unwrap());
            tpot_measurements.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let ttft_median = ttft_measurements[NUM_BENCHMARK_RUNS / 2];
            let ttft_p50 = percentile(&ttft_measurements, 50.0);
            let ttft_p95 = percentile(&ttft_measurements, 95.0);
            let tpot_median = tpot_measurements[NUM_BENCHMARK_RUNS / 2];
            let tpot_p50 = percentile(&tpot_measurements, 50.0);
            let tpot_p95 = percentile(&tpot_measurements, 95.0);

            let prefill_tok_s = (prompt_len as f64) / (ttft_median / 1000.0).max(1e-6);
            let decode_tok_s = 1000.0 / tpot_median.max(1e-6);
            let decoded_text = self
                .tokenizer
                .decode(&sampled_ids_final, true)
                .unwrap_or_default();
            println!(
                "Prompt: {:4} toks | TTFT: {:7.2} ms (p50: {:7.2}, p95: {:7.2}) | TPOT: {:5.2} ms (p50: {:5.2}, p95: {:5.2})\n  -> Output: {:?}",
                prompt_len,
                ttft_median,
                ttft_p50,
                ttft_p95,
                tpot_median,
                tpot_p50,
                tpot_p95,
                decoded_text.trim()
            );
            samples.push(BenchmarkSample {
                prompt_tokens: prompt_len,
                ttft_ms: ttft_median,
                ttft_p50_ms: ttft_p50,
                ttft_p95_ms: ttft_p95,
                prefill_tokens_per_second: prefill_tok_s,
                tpot_ms: tpot_median,
                tpot_p50_ms: tpot_p50,
                tpot_p95_ms: tpot_p95,
                decode_tokens_per_second: decode_tok_s,
                sampled_token_ids: sampled_ids_final,
            });
        }
        println!(
            "==========================================================================================\n"
        );
        BenchmarkReport {
            schema_version: 1,
            engine: "KriyaRT",
            model_name: self.descriptor.model_name.clone(),
            model_family: self.descriptor.family.clone(),
            model_weight_format: "Q4_G64 + BF16 norms/biases",
            model_layers: self.descriptor.layers.len(),
            speculative_decoding_enabled: speculation_enabled,
            measured_at_unix_s: current_timestamp_us() / 1_000_000,
            samples,
        }
    }

    /// Runs direct interactive prompt generation on the GPU, streaming tokens to stdout.
    pub async fn run_direct_generation(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) {
        use std::io::Write;

        println!(
            "\n=========================================================================================="
        );
        println!(" KriyaRT Interactive Text Generation");
        println!(
            " Model: {} ({} layers)",
            self.descriptor.model_name,
            self.descriptor.layers.len()
        );
        println!(" Prompt: {:?}", prompt);
        println!(
            " Max tokens: {}, Temperature: {:.2}, Top-P: {:.2}",
            max_new_tokens, temperature, top_p
        );
        println!(
            "=========================================================================================="
        );

        let pairs = vec![("user".to_string(), prompt.to_string())];
        let prompt_tokens = match self.tokenizer.encode_chat_messages(&pairs, true) {
            Ok(toks) if !toks.is_empty() => toks,
            _ => self
                .tokenizer
                .encode(prompt, false)
                .unwrap_or_else(|_| vec![1]),
        };

        let prompt_len = prompt_tokens.len();
        println!("Encoded prompt: {} tokens", prompt_len);
        print!("\nResponse: ");
        let _ = std::io::stdout().flush();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let req_id: u64 = rand::random();

        self.scheduler.submit_request(InferenceRequest::new(
            req_id,
            prompt_tokens,
            max_new_tokens,
            self.tokenizer.eos_token_id,
            current_timestamp_us(),
        ));

        let mut active_streams = HashMap::new();
        active_streams.insert(
            req_id,
            ActiveStreamState {
                tx,
                temperature,
                top_p,
                start_pos: 0,
            },
        );

        let prefill_start = Instant::now();
        while active_streams.contains_key(&req_id) && !self.scheduler.is_request_decoding(req_id) {
            if !self.step_orca_iteration(&mut active_streams).await {
                break;
            }
        }
        let ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut generated_ids = Vec::new();
        let decode_start = Instant::now();

        if let Ok(tok) = rx.try_recv()
            && tok != self.tokenizer.eos_token_id
        {
            let piece = self.tokenizer.decode(&[tok], true).unwrap_or_default();
            print!("{}", piece);
            let _ = std::io::stdout().flush();
            generated_ids.push(tok);
        }

        while active_streams.contains_key(&req_id) {
            if !self.step_orca_iteration(&mut active_streams).await {
                break;
            }
            while let Ok(tok) = rx.try_recv() {
                if tok == self.tokenizer.eos_token_id {
                    break;
                }
                let piece = self.tokenizer.decode(&[tok], true).unwrap_or_default();
                print!("{}", piece);
                let _ = std::io::stdout().flush();
                generated_ids.push(tok);
            }
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        println!("\n");

        let gen_count = generated_ids.len().max(1);
        let tpot_ms = decode_ms / (gen_count as f64);
        let prefill_tok_s = (prompt_len as f64) / (ttft_ms / 1000.0).max(1e-6);
        let decode_tok_s = 1000.0 / tpot_ms.max(1e-6);

        println!(
            "------------------------------------------------------------------------------------------"
        );
        println!(
            "Prompt Eval : {:4} tokens | TTFT: {:7.2} ms ({:7.1} tok/s)",
            prompt_len, ttft_ms, prefill_tok_s
        );
        println!(
            "Generation  : {:4} tokens | Time: {:7.2} ms ({:7.1} tok/s, {:5.2} ms/tok)",
            gen_count, decode_ms, decode_tok_s, tpot_ms
        );
        println!(
            "------------------------------------------------------------------------------------------\n"
        );
    }
}

/// Autonomous Background ORCA Continuous Batching Worker Loop.
pub async fn orca_batch_worker_loop(
    mut runtime: EngineRuntime,
    mut work_rx: mpsc::UnboundedReceiver<EnqueueRequestCmd>,
) {
    let mut active_streams: HashMap<u64, ActiveStreamState> = HashMap::new();
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
            active_streams.insert(
                cmd.req_id,
                ActiveStreamState {
                    tx: cmd.token_tx,
                    temperature: cmd.temperature,
                    top_p: cmd.top_p,
                    start_pos: 0,
                },
            );
        }

        while let Ok(cmd) = work_rx.try_recv() {
            runtime.scheduler.submit_request(InferenceRequest::new(
                cmd.req_id,
                cmd.prompt_tokens,
                cmd.max_new_tokens,
                eos_id,
                current_timestamp_us(),
            ));
            active_streams.insert(
                cmd.req_id,
                ActiveStreamState {
                    tx: cmd.token_tx,
                    temperature: cmd.temperature,
                    top_p: cmd.top_p,
                    start_pos: 0,
                },
            );
        }

        runtime.step_orca_iteration(&mut active_streams).await;
        tokio::task::yield_now().await;
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
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

async fn metrics_handler(State(handle): State<ServerHandle>) -> impl IntoResponse {
    let uptime = current_timestamp_us() / 1_000_000;
    let active = handle.metrics.active_requests.load(Ordering::Relaxed);
    let total_reqs = handle.metrics.total_requests.load(Ordering::Relaxed);
    let total_emitted = handle.metrics.total_tokens_emitted.load(Ordering::Relaxed);
    let total_prompt = handle.metrics.total_prompt_tokens.load(Ordering::Relaxed);
    let rejected = handle.metrics.rejected_requests.load(Ordering::Relaxed);
    let metrics_text = format!(
        "# HELP kriya_engine_info Engine metadata\n\
         # TYPE kriya_engine_info gauge\n\
         kriya_engine_info{{engine=\"KriyaRT\",arch=\"macOS-Metal\"}} 1\n\
         # HELP kriya_server_uptime_seconds Server uptime in seconds\n\
         # TYPE kriya_server_uptime_seconds counter\n\
         kriya_server_uptime_seconds {}\n\
         # HELP kriya_active_requests Currently in-flight inference requests\n\
         # TYPE kriya_active_requests gauge\n\
         kriya_active_requests {}\n\
         # HELP kriya_requests_total Total number of received inference requests\n\
         # TYPE kriya_requests_total counter\n\
         kriya_requests_total {}\n\
         # HELP kriya_prompt_tokens_total Total prompt tokens processed\n\
         # TYPE kriya_prompt_tokens_total counter\n\
         kriya_prompt_tokens_total {}\n\
         # HELP kriya_completion_tokens_total Total completion tokens emitted\n\
         # TYPE kriya_completion_tokens_total counter\n\
         kriya_completion_tokens_total {}\n\
         # HELP kriya_rejected_requests_total Total requests rejected due to admission limits\n\
         # TYPE kriya_rejected_requests_total counter\n\
         kriya_rejected_requests_total {}\n\
         # HELP kriya_pipeline_depth2_active Depth-2 Lookahead pipelining flag\n\
         # TYPE kriya_pipeline_depth2_active gauge\n\
         kriya_pipeline_depth2_active 1\n",
        uptime, active, total_reqs, total_prompt, total_emitted, rejected
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        metrics_text,
    )
}

async fn chat_completions_handler(
    State(handle): State<ServerHandle>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    // 1. Validation: Messages must not be empty
    if payload.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "Missing required field: 'messages' array must not be empty",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "missing_required_field"
                }
            })),
        )
            .into_response();
    }

    // 2. Validation: Max completion tokens limit
    let max_new = payload.max_tokens.unwrap_or(16);
    if max_new > handle.max_completion_tokens {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "Requested max_tokens ({}) exceeds server maximum of {}",
                        max_new, handle.max_completion_tokens
                    ),
                    "type": "invalid_request_error",
                    "param": "max_tokens",
                    "code": "parameter_above_maximum"
                }
            })),
        )
            .into_response();
    }

    // 3. Admission Control: Check queue saturation
    let current_active = handle.metrics.active_requests.load(Ordering::Relaxed);
    if current_active >= handle.max_concurrent_requests {
        handle
            .metrics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "Server admission queue saturated ({} active requests >= limit {})",
                        current_active, handle.max_concurrent_requests
                    ),
                    "type": "rate_limit_error",
                    "code": "queue_full"
                }
            })),
        )
            .into_response();
    }

    let temperature = payload.temperature.unwrap_or(0.0);
    let top_p = payload.top_p.unwrap_or(0.95);
    let req_id: u64 = rand::random();

    let pairs: Vec<(String, String)> = payload
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let prompt_tokens = match handle.tokenizer.encode_chat_messages(&pairs, true) {
        Ok(t) if !t.is_empty() => t,
        _ => match handle.tokenizer.encode(
            payload
                .messages
                .last()
                .map(|m| m.content.as_str())
                .unwrap_or(""),
            false,
        ) {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!("Failed to encode chat prompt tokens: {e}"),
                            "type": "invalid_request_error",
                            "param": "messages",
                            "code": "tokenization_failed"
                        }
                    })),
                )
                    .into_response();
            }
        },
    };

    let prompt_len = prompt_tokens.len();
    if prompt_len > handle.max_prompt_tokens {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "Prompt length ({} tokens) exceeds server context limit of {}",
                        prompt_len, handle.max_prompt_tokens
                    ),
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "context_length_exceeded"
                }
            })),
        )
            .into_response();
    }

    handle
        .metrics
        .active_requests
        .fetch_add(1, Ordering::Relaxed);
    handle
        .metrics
        .total_requests
        .fetch_add(1, Ordering::Relaxed);
    handle
        .metrics
        .total_prompt_tokens
        .fetch_add(prompt_len as u64, Ordering::Relaxed);

    struct ActiveGuard(Arc<ServerMetrics>);
    impl Drop for ActiveGuard {
        fn drop(&mut self) {
            self.0.active_requests.fetch_sub(1, Ordering::Relaxed);
        }
    }
    let _guard = ActiveGuard(handle.metrics.clone());

    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    if handle
        .work_tx
        .send(EnqueueRequestCmd {
            req_id,
            prompt_tokens,
            max_new_tokens: max_new,
            temperature,
            top_p,
            token_tx,
        })
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "message": "KriyaRT inference worker is unavailable",
                    "type": "server_error"
                }
            })),
        )
            .into_response();
    }

    let tok_clone = handle.tokenizer.clone();
    let metrics_clone = handle.metrics.clone();
    if payload.stream.unwrap_or(false) {
        let sse_stream = stream::unfold(
            (token_rx, tok_clone, req_id, metrics_clone),
            |(mut rx, tokenizer_opt, id, metrics)| async move {
                let tok_id = rx.recv().await?;
                metrics.total_tokens_emitted.fetch_add(1, Ordering::Relaxed);
                let piece = tokenizer_opt
                    .decode(&[tok_id], true)
                    .unwrap_or_else(|_| format!("{tok_id} "));
                let chunk_json = format!(
                    r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":{:?}}},"finish_reason":null}}]}}"#,
                    id, piece
                );
                Some((
                    Ok::<Event, std::convert::Infallible>(Event::default().data(chunk_json)),
                    (rx, tokenizer_opt, id, metrics),
                ))
            },
        );
        Sse::new(sse_stream).into_response()
    } else {
        let mut generated_ids = Vec::with_capacity(max_new);
        while let Some(tok_id) = token_rx.recv().await {
            handle
                .metrics
                .total_tokens_emitted
                .fetch_add(1, Ordering::Relaxed);
            generated_ids.push(tok_id);
        }
        let decoded_text = tok_clone.decode(&generated_ids, true).unwrap_or_default();
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{req_id}"),
            object: "chat.completion".to_string(),
            created: current_timestamp_us() / 1_000_000,
            model: handle.model_name.clone(),
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

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(axum::extract::DefaultBodyLimit::max(8 * 1024 * 1024))
        .with_state(handle);

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
