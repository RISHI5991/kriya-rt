//! Inference Engine Runtime and ORCA Batching Worker.

use crate::server::admission::{ActiveStreamState, EnqueueRequestCmd};
use crate::server::metrics::current_timestamp_us;
use kriya_backend_metal::encoder::{MetalExecutor, SamplingConfig};
use kriya_core::descriptor::ModelArchDescriptor;
use kriya_core::tokenizer::KriyaTokenizer;
use kriya_kv::bsr_paged::PagedKvCacheManager;
use kriya_scheduler::orca_engine::{
    InferenceRequest, InstanceRole, OrcaMachineScheduler, ServiceLevelObjective,
};
use kriya_speculate::eagle_medusa::{DeterministicRng, DraftTreeTopology, EagleDraftEngine};
use serde::Serialize;
use std::{collections::HashMap, path::Path, sync::Arc, time::Instant};
use tokio::sync::mpsc;

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
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
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
        let mut executor =
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

        let draft_engine = Self::build_draft_engine(
            &descriptor,
            &mut executor,
            checkpoint_dir,
            draft_weights_path,
        );

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
    pub(crate) fn build_draft_engine(
        descriptor: &ModelArchDescriptor,
        executor: &mut MetalExecutor,
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

        // Release integrity check: If kriya-artifact-manifest.json exists, verify checksum strictly
        let manifest_path = checkpoint_dir.join("kriya-artifact-manifest.json");
        if manifest_path.is_file() {
            let manifest_bytes = match std::fs::read(&manifest_path) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!(
                        "KriyaRT integrity failure: failed to read manifest at {}: {err}",
                        manifest_path.display()
                    );
                    return None;
                }
            };
            let manifest_json: serde_json::Value = match serde_json::from_slice(&manifest_bytes) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!(
                        "KriyaRT integrity failure: malformed manifest JSON at {}: {err}",
                        manifest_path.display()
                    );
                    return None;
                }
            };
            let Some(spec_file) = manifest_json.get("speculative_draft_file") else {
                eprintln!(
                    "KriyaRT integrity failure: manifest missing 'speculative_draft_file' entry"
                );
                return None;
            };
            let Some(expected_hash) = spec_file.get("sha256").and_then(|h| h.as_str()) else {
                eprintln!(
                    "KriyaRT integrity failure: manifest missing 'sha256' string for draft file"
                );
                return None;
            };

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

        let mut engine =
            EagleDraftEngine::new_gpu_backed(h, vocab, vec![0.0; h * (2 * h)], vec![0.0; h * h])
                .ok()?;
        if let Err(err) = engine.load_weights(&draft_weights) {
            eprintln!(
                "KriyaRT: speculative decoding disabled; invalid trained draft head {}: {err}",
                artifact_path.display()
            );
            return None;
        }

        if !engine.is_trained {
            let allow_untrained = std::env::var("KRIYA_ALLOW_UNTRAINED_DRAFT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !allow_untrained {
                eprintln!(
                    "KriyaRT: speculative decoding disabled; draft head at {} is marked untrained (is_trained=0). Train it using kriya_train_eagle or set KRIYA_ALLOW_UNTRAINED_DRAFT=1 to override.",
                    artifact_path.display()
                );
                return None;
            }
            eprintln!(
                "KriyaRT: WARNING: Loading untrained EAGLE draft head from {} because KRIYA_ALLOW_UNTRAINED_DRAFT=1 is set.",
                artifact_path.display()
            );
        }

        if let Err(err) = executor
            .load_eagle_draft_weights(&engine.fc_fuse_weights, &engine.decoder_layer_weights)
        {
            eprintln!(
                "KriyaRT: speculative decoding disabled; failed to load draft head weights to GPU: {err}"
            );
            return None;
        }
        println!("KriyaRT: GPU-native EAGLE speculative draft head loaded successfully");
        Some(engine)
    }

    #[inline]
    fn make_sampling_config(&self, req_id: u64, pos: u32, temp: f32, top_p: f32) -> SamplingConfig {
        SamplingConfig {
            vocab_size: self.descriptor.vocab_size,
            greedy: u32::from(temp <= 1e-5),
            temperature: temp,
            top_p,
            top_k: 50,
            rng_seed: (req_id as u32).wrapping_add(pos),
        }
    }

    /// Executes a single iteration tick across all concurrently running requests in the ORCA batch.
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

        let mut decode_req_ids: Vec<u64> = Vec::new();
        let mut decode_tokens: Vec<u32> = Vec::new();
        let mut decode_positions: Vec<u32> = Vec::new();
        let mut decode_samplings: Vec<SamplingConfig> = Vec::new();
        let mut decode_req_to_output_idx: HashMap<u64, usize> = HashMap::new();
        let mut req_to_root_feature: HashMap<u64, Vec<f32>> = HashMap::new();

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

            if req_tokens.len() > 1 {
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

                let positions: Vec<u32> = (0..req_tokens.len())
                    .map(|i| (start_pos + i) as u32)
                    .collect();
                let samp_cfg = self.make_sampling_config(*req_id, start_pos as u32, temp, top_p);
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
                let input_token = req_tokens[0];
                let position = start_pos as u32;
                let samp_cfg = self.make_sampling_config(*req_id, position, temp, top_p);
                let out_idx = primary_outputs.len();
                primary_outputs.push((*req_id, 0));
                client_visible_tokens.insert(*req_id, should_emit);
                decode_req_to_output_idx.insert(*req_id, out_idx);
                decode_req_ids.push(*req_id);
                decode_tokens.push(input_token);
                decode_positions.push(position);
                decode_samplings.push(samp_cfg);
            }
        }

        if !decode_tokens.is_empty() {
            let decode_plan = self
                .scheduler
                .kv_manager
                .build_schedule_plan(&decode_req_ids, self.scheduler.max_blocks_per_cta);

            let decode_failed_err = match decode_plan {
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
                            let h = self.descriptor.hidden_size as usize;
                            for (i, token) in decoded_tokens.iter().enumerate() {
                                let req_id = decode_req_ids[i];
                                let out_idx = decode_req_to_output_idx[&req_id];
                                primary_outputs[out_idx] = (req_id, *token);
                                if self.draft_engine.is_some() {
                                    req_to_root_feature
                                        .insert(req_id, self.executor.read_hidden_feature_at(h, i));
                                }
                            }
                            None
                        }
                        Err(err) => Some(format!("KriyaRT batched decode failed: {err}")),
                    }
                }
                Err(err) => Some(format!("KriyaRT decode schedule plan failed: {err}")),
            };

            if let Some(msg) = decode_failed_err {
                eprintln!("{msg}");
                for req_id in &decode_req_ids {
                    let _ = self.scheduler.abort_request(*req_id);
                    active_streams.remove(req_id);
                }
            }
        }

        for (req_id, target_tok, should_emit) in prefill_results {
            primary_outputs.push((req_id, target_tok));
            if should_emit {
                client_visible_tokens.insert(req_id, true);
            }
        }

        let mut disconnected_reqs = Vec::new();
        if let Ok((finished_primary, _)) = self
            .scheduler
            .commit_iteration_outputs(&primary_outputs, current_timestamp_us())
        {
            for &(req_id, primary_tok) in &primary_outputs {
                if client_visible_tokens.get(&req_id) == Some(&true)
                    && let Some(state) = active_streams.get(&req_id)
                {
                    if state.tx.send(primary_tok).is_err() {
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

        if let Some(ref mut draft_engine) = self.draft_engine
            && batch_size <= 4
        {
            for &(req_id, primary_tok) in &primary_outputs {
                if !self.scheduler.is_request_decoding(req_id) {
                    continue;
                }

                let Some(root_feature) = req_to_root_feature.get(&req_id) else {
                    continue;
                };

                let topology = if batch_size <= 2 {
                    DraftTreeTopology::canonical_10_token_tree()
                } else {
                    DraftTreeTopology::compact_batch_tree()
                };

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
                        draft_engine,
                        primary_tok,
                        root_feature,
                        &topology,
                        &mut self.draft_rng,
                    )
                    .await
                {
                    Ok(Some((accepted, accepted_node_indices))) if !accepted.is_empty() => {
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

                        if finished.is_none() {
                            let accepted_nodes: Vec<_> = committed
                                .iter()
                                .enumerate()
                                .filter_map(|(index, _)| {
                                    let cache_pos = (start_pos + index) as u32;
                                    let (phys_blk_id, slot_in_blk) = self
                                        .scheduler
                                        .kv_manager
                                        .block_for_position(req_id, cache_pos)
                                        .ok()?;
                                    let tree_node_idx =
                                        accepted_node_indices.get(index).copied().unwrap_or(0);
                                    Some((tree_node_idx, phys_blk_id, slot_in_blk))
                                })
                                .collect();

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
        println!(
            "\n=========================================================================================="
        );
        println!(" KriyaRT Empirical Hardware Benchmark (Apple Silicon / Metal)");
        let num_heads = self
            .descriptor
            .layers
            .first()
            .and_then(|l| match &l.seq_mixer {
                kriya_core::descriptor::SequenceMixerDescriptor::GroupedQueryAttention {
                    num_q_heads,
                    ..
                } => Some(*num_q_heads),
                _ => None,
            })
            .unwrap_or(0);
        println!(
            " Model: {} ({} layers, hidden_size={}, num_heads={})",
            self.descriptor.model_name,
            self.descriptor.layers.len(),
            self.descriptor.hidden_size,
            num_heads
        );
        let speculation_enabled = self.draft_engine.is_some();
        println!(
            " Speculative Acceleration: {}",
            if speculation_enabled {
                "ENABLED (EAGLE Medusa Draft Tree)"
            } else {
                "DISABLED"
            }
        );
        println!(
            "=========================================================================================="
        );

        let prompts = [
            (
                64,
                "The quick brown fox jumps over the lazy dog. Artificial intelligence inference on Apple \
                 Silicon requires high memory bandwidth and low latency execution. Unified memory allows \
                 the GPU and CPU to share the same physical address space without PCIe bus transfers. \
                 Modern LLMs utilize multi-head attention and grouped-query attention to optimize cache efficiency.",
            ),
            (
                256,
                "In computer architecture, unified memory architecture (UMA) refers to a design where the \
                 CPU and GPU share a common system memory. Apple's Silicon processors, beginning with the M1, \
                 implement a wide, high-bandwidth unified memory subsystem directly integrated into the SoC package. \
                 This design provides up to 800 GB/s of bandwidth on Max and Ultra variants, significantly \
                 reducing the data transfer penalties traditionally associated with discrete GPUs connected \
                 via PCI Express. In deep learning inference, memory bandwidth is almost always the primary bottleneck \
                 during the autoregressive decoding phase, where weights must be read from DRAM for every single \
                 generated token. By contrast, the prefill phase is compute-bound, benefiting from high-density \
                 matrix multiplication units such as Apple's Neural Engine and AMX coprocessors. In this benchmark, \
                 we evaluate both phases systematically to establish real-world hardware throughput characteristics. \
                 The attention mechanism further utilizes flash-decoding and paged memory layouts to prevent \
                 fragmentation and sustain high token generation throughput across long sequence lengths.",
            ),
            (
                512,
                "Large language models represent a paradigm shift in artificial intelligence, transforming natural \
                 language understanding and generation through scale and architectural innovation. The core building \
                 block of modern transformer models is the multi-head self-attention mechanism, which computes pairwise \
                 token interactions across an input sequence. While quadratically complex with respect to sequence length, \
                 attention mechanisms can be efficiently evaluated using block-sparse and tiled computation strategies \
                 that keep intermediate activation matrices inside fast on-chip SRAM caches. When deploying these models \
                 on edge devices like MacBooks and Mac Studios, software engineers must carefully balance computational \
                 efficiency with memory constraints. 4-bit affine quantization compresses large weight matrices into \
                 compact representations, enabling 3-billion to 70-billion parameter models to reside entirely within \
                 the unified memory of modern Apple Silicon SoCs. During prefill, the entire prompt is ingested in parallel, \
                 saturating the GPU cores with matrix multiplications. Once the key-value cache is populated, generation \
                 proceeds autoregressively: the model consumes its own previous output token and emits the next one. \
                 Because each autoregressive step requires streaming the entire model's parameters from memory while performing \
                 only a single vector-matrix multiply per weight block, the maximum generation speed is mathematically bounded \
                 by memory bandwidth. Hardware-specific optimizations, including SIMDgroup matrix instructions and fused \
                 activation kernels, allow inference engines to achieve near-theoretical memory bandwidth saturation. \
                 This benchmark executes multiple iterations across multiple prompt lengths to compute reproducible \
                 P50 and P95 latency distributions, confirming that the engine meets strict production latency requirements.",
            ),
        ];

        let gen_tokens = 64usize;
        let mut samples = Vec::with_capacity(prompts.len());

        for (target_ctx, prompt_text) in prompts {
            let pairs = vec![("user".to_string(), prompt_text.to_string())];
            let full_prompt = match self.tokenizer.encode_chat_messages(&pairs, true) {
                Ok(toks) if !toks.is_empty() => toks,
                _ => self
                    .tokenizer
                    .encode(prompt_text, false)
                    .unwrap_or_else(|_| vec![1]),
            };

            let actual_prompt_len = full_prompt.len();
            let mut ttft_runs: Vec<f64> = Vec::with_capacity(5);
            let mut tpot_runs: Vec<f64> = Vec::with_capacity(5);
            let mut last_generated_ids = Vec::new();

            for run in 0..5 {
                let (tx, mut rx) = mpsc::unbounded_channel();
                let req_id: u64 = 1000 + run as u64;

                self.scheduler.submit_request(InferenceRequest::new(
                    req_id,
                    full_prompt.clone(),
                    gen_tokens,
                    self.tokenizer.eos_token_id,
                    current_timestamp_us(),
                ));

                let mut active_streams = HashMap::new();
                active_streams.insert(
                    req_id,
                    ActiveStreamState {
                        tx,
                        temperature: 0.0,
                        top_p: 1.0,
                        start_pos: 0,
                    },
                );

                let prefill_start = Instant::now();
                while active_streams.contains_key(&req_id)
                    && !self.scheduler.is_request_decoding(req_id)
                {
                    if !self.step_orca_iteration(&mut active_streams).await {
                        break;
                    }
                }
                let ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
                ttft_runs.push(ttft_ms);

                let mut gen_ids = Vec::with_capacity(gen_tokens);
                let decode_start = Instant::now();

                if let Ok(tok) = rx.try_recv()
                    && tok != self.tokenizer.eos_token_id
                {
                    gen_ids.push(tok);
                }

                while active_streams.contains_key(&req_id) {
                    if !self.step_orca_iteration(&mut active_streams).await {
                        break;
                    }
                    while let Ok(tok) = rx.try_recv() {
                        if tok == self.tokenizer.eos_token_id {
                            break;
                        }
                        gen_ids.push(tok);
                    }
                }
                let decode_duration_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
                let actual_gen = gen_ids.len().max(1);
                let tpot_ms = decode_duration_ms / (actual_gen as f64);
                tpot_runs.push(tpot_ms);

                if run == 4 {
                    last_generated_ids = gen_ids;
                }
            }

            ttft_runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            tpot_runs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median_ttft = percentile(&ttft_runs, 50.0);
            let p50_ttft = percentile(&ttft_runs, 50.0);
            let p95_ttft = percentile(&ttft_runs, 95.0);
            let prefill_tok_s = (actual_prompt_len as f64) / (median_ttft / 1000.0).max(1e-6);

            let median_tpot = percentile(&tpot_runs, 50.0);
            let p50_tpot = percentile(&tpot_runs, 50.0);
            let p95_tpot = percentile(&tpot_runs, 95.0);
            let decode_tok_s = 1000.0 / median_tpot.max(1e-6);

            println!(
                "\n[*] Context: {:3} tokens (target: {:3}) | Generated: {:2} tokens",
                actual_prompt_len, target_ctx, gen_tokens
            );
            println!(
                "    Prefill (TTFT) : Median {:7.2} ms | P50 {:7.2} ms | P95 {:7.2} ms | Rate: {:7.1} tok/s",
                median_ttft, p50_ttft, p95_ttft, prefill_tok_s
            );
            println!(
                "    Decode  (TPOT) : Median {:7.2} ms | P50 {:7.2} ms | P95 {:7.2} ms | Rate: {:7.1} tok/s",
                median_tpot, p50_tpot, p95_tpot, decode_tok_s
            );

            samples.push(BenchmarkSample {
                prompt_tokens: actual_prompt_len,
                ttft_ms: median_ttft,
                ttft_p50_ms: p50_ttft,
                ttft_p95_ms: p95_ttft,
                prefill_tokens_per_second: prefill_tok_s,
                tpot_ms: median_tpot,
                tpot_p50_ms: p50_tpot,
                tpot_p95_ms: p95_tpot,
                decode_tokens_per_second: decode_tok_s,
                sampled_token_ids: last_generated_ids,
            });
        }

        println!(
            "\n==========================================================================================\n"
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
        let mut decoded_prefix_len = 0usize;
        let decode_start = Instant::now();

        if let Ok(tok) = rx.try_recv()
            && tok != self.tokenizer.eos_token_id
        {
            generated_ids.push(tok);
            let full = self
                .tokenizer
                .decode(&generated_ids, true)
                .unwrap_or_default();
            let valid = if full.ends_with('\u{FFFD}') {
                full.trim_end_matches('\u{FFFD}')
            } else {
                &full
            };
            if valid.len() > decoded_prefix_len {
                let safe_idx = if valid.is_char_boundary(decoded_prefix_len) {
                    decoded_prefix_len
                } else {
                    valid
                        .char_indices()
                        .map(|(i, _)| i)
                        .find(|&i| i >= decoded_prefix_len)
                        .unwrap_or(valid.len())
                };
                print!("{}", &valid[safe_idx..]);
                let _ = std::io::stdout().flush();
                decoded_prefix_len = valid.len();
            }
        }

        while active_streams.contains_key(&req_id) {
            if !self.step_orca_iteration(&mut active_streams).await {
                break;
            }
            while let Ok(tok) = rx.try_recv() {
                if tok == self.tokenizer.eos_token_id {
                    break;
                }
                generated_ids.push(tok);
                let full = self
                    .tokenizer
                    .decode(&generated_ids, true)
                    .unwrap_or_default();
                let valid = if full.ends_with('\u{FFFD}') {
                    full.trim_end_matches('\u{FFFD}')
                } else {
                    &full
                };
                if valid.len() > decoded_prefix_len {
                    let safe_idx = if valid.is_char_boundary(decoded_prefix_len) {
                        decoded_prefix_len
                    } else {
                        valid
                            .char_indices()
                            .map(|(i, _)| i)
                            .find(|&i| i >= decoded_prefix_len)
                            .unwrap_or(valid.len())
                    };
                    print!("{}", &valid[safe_idx..]);
                    let _ = std::io::stdout().flush();
                    decoded_prefix_len = valid.len();
                }
            }
        }
        let final_text = self
            .tokenizer
            .decode(&generated_ids, true)
            .unwrap_or_default();
        if final_text.len() > decoded_prefix_len {
            let safe_idx = if final_text.is_char_boundary(decoded_prefix_len) {
                decoded_prefix_len
            } else {
                final_text
                    .char_indices()
                    .map(|(i, _)| i)
                    .find(|&i| i >= decoded_prefix_len)
                    .unwrap_or(final_text.len())
            };
            print!("{}", &final_text[safe_idx..]);
            let _ = std::io::stdout().flush();
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
