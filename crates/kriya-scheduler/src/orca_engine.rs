// File: crates/kriya-scheduler/src/orca_engine.rs
//! ORCA Iteration-Level Continuous Batching Scheduler with Selective Batching,
//! vLLM Preemptive Block Management, and DistServe/Splitwise Phase Disaggregation
//! with Layer-Wise One-Sided Contiguous KV-Cache Transfers.

use kriya_kv::bsr_paged::{FlashInferSchedulePlan, KvCacheError, PagedKvCacheManager};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    #[error("KV cache allocation error: {0}")]
    KvError(#[from] KvCacheError),
    #[error("Request {0} not found in machine scheduler")]
    RequestNotFound(u64),
    #[error("Invalid phase transition for request {req_id} on instance role {role:?}")]
    InvalidPhaseTransition { req_id: u64, role: InstanceRole },
}

/// Disaggregated or colocated role assigned to an engine instance (DistServe / Splitwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstanceRole {
    /// Dedicated Prompt / Prefill instance (computes first token and streams KV cache per layer).
    PrefillOnly,
    /// Dedicated Token / Decode instance (receives KV cache via semaphore and batches token steps).
    DecodeOnly,
    /// Colocated hybrid instance running both prefill and decode iterations.
    ColocatedHybrid,
}

/// Service-Level Objective (SLO) latency targets for maximizing per-GPU goodput.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ServiceLevelObjective {
    /// P90 Time-To-First-Token (TTFT) latency target in milliseconds.
    pub ttft_slo_ms: f32,
    /// P90 Time-Per-Output-Token (TPOT) / Time-Between-Tokens (TBT) target in milliseconds.
    pub tpot_slo_ms: f32,
}

impl Default for ServiceLevelObjective {
    fn default() -> Self {
        Self {
            ttft_slo_ms: 250.0,
            tpot_slo_ms: 25.0,
        }
    }
}

/// Execution phase of an individual request within an iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestPhase {
    /// Waiting in pending queue for KV block allocation.
    Pending,
    /// Running initial prompt computation (processes all prompt tokens in parallel).
    Prefill,
    /// Waiting on one-sided KV transfer semaphore completion at the Decode machine.
    AwaitingKvTransfer {
        semaphore_id: u64,
        layers_received: u32,
    },
    /// Running autoregressive token-by-token generation.
    Decode,
    /// Preempted back to CPU swap / pending queue due to GPU KV memory pressure.
    Preempted,
    /// Completed (`<EOS>` emitted or `max_new_tokens` reached); returned immediately to client.
    Completed,
}

/// Default maximum prefill chunk size (512 tokens) to protect ongoing Decode TPOT SLAs.
pub const DEFAULT_PREFILL_CHUNK_SIZE: usize = 512;

/// State of a client inference request managed at iteration and chunk granularity.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub req_id: u64,
    pub prompt_tokens: Vec<u32>,
    /// Number of prompt tokens already processed by bounded prefill chunks.
    pub prefill_tokens_processed: usize,
    /// Maximum tokens processed per prefill iteration step (default: 512).
    pub prefill_chunk_limit: usize,
    pub generated_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
    pub phase: RequestPhase,
    pub arrival_timestamp_us: u64,
    pub first_token_timestamp_us: Option<u64>,
    pub target_decode_node_id: Option<u32>,
}

impl InferenceRequest {
    pub fn new(
        req_id: u64,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: u32,
        arrival_timestamp_us: u64,
    ) -> Self {
        Self {
            req_id,
            prompt_tokens,
            prefill_tokens_processed: 0,
            prefill_chunk_limit: DEFAULT_PREFILL_CHUNK_SIZE,
            generated_tokens: Vec::new(),
            max_new_tokens,
            eos_token_id,
            phase: RequestPhase::Pending,
            arrival_timestamp_us,
            first_token_timestamp_us: None,
            target_decode_node_id: None,
        }
    }

    /// Remaining unprocessed prompt tokens.
    #[inline]
    pub fn remaining_prefill_tokens(&self) -> usize {
        self.prompt_tokens
            .len()
            .saturating_sub(self.prefill_tokens_processed)
    }

    /// Returns the slice of prompt tokens scheduled for the current bounded prefill chunk.
    #[inline]
    pub fn current_prefill_chunk(&self) -> &[u32] {
        let start = self.prefill_tokens_processed;
        let len = self
            .remaining_prefill_tokens()
            .min(self.prefill_chunk_limit);
        &self.prompt_tokens[start..start + len]
    }

    /// True if the current prefill chunk completes the entire prompt.
    #[inline]
    pub fn is_final_prefill_chunk(&self) -> bool {
        self.remaining_prefill_tokens() <= self.prefill_chunk_limit
    }

    /// Number of input tokens this request contributes to the current forward iteration:
    /// bounded to `<= prefill_chunk_limit` (512) during `Prefill`, or `1` during `Decode`.
    #[inline]
    pub fn iteration_token_count(&self) -> usize {
        match self.phase {
            RequestPhase::Prefill => self.current_prefill_chunk().len(),
            RequestPhase::Decode => 1,
            _ => 0,
        }
    }

    /// Checks whether the request has finished generation at the end of the current iteration.
    #[inline]
    pub fn is_finished(&self) -> bool {
        if self.generated_tokens.len() >= self.max_new_tokens {
            return true;
        }
        self.generated_tokens
            .last()
            .is_some_and(|&last_tok| last_tok == self.eos_token_id)
    }
}

/// ORCA Selective Batching Descriptor for a single iteration.
///
/// Non-Attention operations (LayerNorm, QKV Linear, Attn Out Linear, MLP) operate on a flattened
/// 2D tensor of shape `[total_flattened_tokens, hidden_size]`, whereas the Attention operation
/// splits execution per request using `attention_request_slices`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectiveBatchPlan {
    /// Ordered request IDs scheduled in this iteration.
    pub scheduled_req_ids: Vec<u64>,
    /// Total tokens across all scheduled requests (`sum(l_i)`), defining `[total_tokens, H]` for Linear/MLP.
    pub total_flattened_tokens: usize,
    /// Per-request `(req_id, token_offset_in_flat_tensor, num_tokens, is_prefill)` for split Attention.
    pub attention_request_slices: Vec<(u64, usize, usize, bool)>,
    /// Flattened input token IDs `[total_flattened_tokens]` fed into the embedding lookup.
    pub flattened_input_tokens: Vec<u32>,
}

/// Splitwise One-Sided Zero-Copy Contiguous KV-Cache Transfer Descriptor.
/// Coalesces adjacent physical KV blocks into a single RDMA / NVLink / Thunderbolt `put` operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContiguousKvPutOp {
    pub req_id: u64,
    pub layer_idx: u32,
    pub start_physical_block_id: u32,
    pub num_contiguous_blocks: u32,
    pub target_decode_node_id: u32,
    pub semaphore_id: u64,
}

/// Complete Iteration Execution Bundle produced by the Machine-Level Scheduler (MLS).
#[derive(Debug, Clone)]
pub struct ScheduledIterationBatch {
    pub selective_batch: SelectiveBatchPlan,
    pub kv_schedule_plan: FlashInferSchedulePlan,
    pub cow_block_copies: Vec<(u32, u32)>,
}

/// Machine-Level Scheduler (MLS) implementing ORCA Iteration-Level Scheduling,
/// vLLM Preemptive Paged Memory Management, and Splitwise/DistServe Phase Disaggregation.
pub struct OrcaMachineScheduler {
    pub role: InstanceRole,
    pub num_model_layers: u32,
    pub max_batch_tokens: usize,
    pub max_blocks_per_cta: u32,
    pub slo: ServiceLevelObjective,
    pub kv_manager: PagedKvCacheManager,
    pending_queue: VecDeque<u64>,
    running_requests: Vec<u64>,
    requests: HashMap<u64, InferenceRequest>,
    next_semaphore_id: u64,
}

impl OrcaMachineScheduler {
    pub fn new(
        role: InstanceRole,
        num_model_layers: u32,
        max_batch_tokens: usize,
        max_blocks_per_cta: u32,
        slo: ServiceLevelObjective,
        kv_manager: PagedKvCacheManager,
    ) -> Self {
        Self {
            role,
            num_model_layers,
            max_batch_tokens,
            max_blocks_per_cta,
            slo,
            kv_manager,
            pending_queue: VecDeque::new(),
            running_requests: Vec::new(),
            requests: HashMap::new(),
            next_semaphore_id: 1,
        }
    }

    /// Submits a new client request into the pending queue (can arrive at any time between iterations).
    pub fn submit_request(&mut self, mut req: InferenceRequest) {
        let req_id = req.req_id;
        req.phase = RequestPhase::Pending;
        self.requests.insert(req_id, req);
        self.pending_queue.push_back(req_id);
    }

    /// Removes a request that cannot continue because its execution backend
    /// failed. This is deliberately separate from normal completion: callers
    /// must not leave its page-table entries or KV blocks resident after an
    /// error, otherwise one bad request can eventually exhaust the cache.
    pub fn abort_request(&mut self, req_id: u64) -> Result<(), SchedulerError> {
        self.pending_queue.retain(|&id| id != req_id);
        self.running_requests.retain(|&id| id != req_id);
        if self.requests.remove(&req_id).is_some() {
            if self.kv_manager.contains_sequence(req_id) {
                self.kv_manager.free_sequence(req_id)?;
            }
        }
        Ok(())
    }

    /// Registers an incoming disaggregated request on a `DecodeOnly` machine together with
    /// its pre-allocated physical KV blocks and semaphore handle.
    pub fn register_incoming_disaggregated_decode(
        &mut self,
        mut req: InferenceRequest,
        semaphore_id: u64,
    ) -> Result<(), SchedulerError> {
        if self.role == InstanceRole::PrefillOnly {
            return Err(SchedulerError::InvalidPhaseTransition {
                req_id: req.req_id,
                role: self.role,
            });
        }
        let req_id = req.req_id;
        self.kv_manager
            .allocate_sequence(req_id, &req.prompt_tokens)?;
        req.phase = RequestPhase::AwaitingKvTransfer {
            semaphore_id,
            layers_received: 0,
        };
        self.requests.insert(req_id, req);
        Ok(())
    }

    /// Signals per-layer one-sided KV put completion on the Decode machine.
    /// Once all layers are received, transitions the request into the active `Decode` pool.
    pub fn notify_layer_kv_received(&mut self, req_id: u64) -> Result<bool, SchedulerError> {
        let req = self
            .requests
            .get_mut(&req_id)
            .ok_or(SchedulerError::RequestNotFound(req_id))?;

        if let RequestPhase::AwaitingKvTransfer {
            layers_received, ..
        } = &mut req.phase
        {
            *layers_received += 1;
            if *layers_received >= self.num_model_layers {
                req.phase = RequestPhase::Decode;
                if !self.running_requests.contains(&req_id) {
                    self.running_requests.push(req_id);
                }
                return Ok(true);
            }
            Ok(false)
        } else {
            Err(SchedulerError::InvalidPhaseTransition {
                req_id,
                role: self.role,
            })
        }
    }

    /// Coalesces a sequence's physical KV blocks into contiguous one-sided `put` operations
    /// for a completed layer during the Prompt phase (Splitwise Fig. 11b per-layer transfer).
    pub fn build_layer_kv_put_ops(
        &mut self,
        req_id: u64,
        layer_idx: u32,
        target_decode_node_id: u32,
    ) -> Result<Vec<ContiguousKvPutOp>, SchedulerError> {
        let plan = self
            .kv_manager
            .build_schedule_plan(&[req_id], self.max_blocks_per_cta)?;
        let indices = &plan.bsr_layout.page_indices;
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        let semaphore_id = self.next_semaphore_id;
        if layer_idx + 1 == self.num_model_layers {
            self.next_semaphore_id += 1;
        }

        let mut ops = Vec::new();
        let mut run_start = indices[0] as u32;
        let mut run_len = 1u32;

        for &blk in &indices[1..] {
            let blk_u32 = blk as u32;
            if blk_u32 == run_start + run_len {
                run_len += 1;
            } else {
                ops.push(ContiguousKvPutOp {
                    req_id,
                    layer_idx,
                    start_physical_block_id: run_start,
                    num_contiguous_blocks: run_len,
                    target_decode_node_id,
                    semaphore_id,
                });
                run_start = blk_u32;
                run_len = 1;
            }
        }

        ops.push(ContiguousKvPutOp {
            req_id,
            layer_idx,
            start_physical_block_id: run_start,
            num_contiguous_blocks: run_len,
            target_decode_node_id,
            semaphore_id,
        });

        Ok(ops)
    }

    /// Schedules the next single iteration across active Decode requests and newly admitted
    /// Prefill requests, building the ORCA `SelectiveBatchPlan` and FlashInfer `BsrKvLayout`.
    pub fn schedule_next_iteration(
        &mut self,
    ) -> Result<Option<ScheduledIterationBatch>, SchedulerError> {
        let mut cow_block_copies = Vec::new();
        let mut token_budget = self.max_batch_tokens;

        // 1. Reserve 1 token slot per active Decode request; preempt newest running request on KV OOM
        if self.role != InstanceRole::PrefillOnly {
            let mut idx = 0;
            while idx < self.running_requests.len() {
                let req_id = self.running_requests[idx];
                match self.kv_manager.append_token_slot(req_id) {
                    Ok((_blk, cow)) => {
                        if let Some(copy_pair) = cow {
                            cow_block_copies.push(copy_pair);
                        }
                        token_budget = token_budget.saturating_sub(1);
                        idx += 1;
                    }
                    Err(KvCacheError::OutOfBlocks { .. }) => {
                        // vLLM-style preemption: evict newest running sequence back to pending queue
                        let victim_id = self.running_requests.pop().unwrap();
                        self.kv_manager.free_sequence(victim_id)?;
                        if let Some(victim) = self.requests.get_mut(&victim_id) {
                            victim.phase = RequestPhase::Preempted;
                        }
                        self.pending_queue.push_front(victim_id);
                    }
                    Err(e) => return Err(SchedulerError::KvError(e)),
                }
            }
        }

        // 2. Admit or continue chunked-prefill requests within the remaining token budget
        let mut scheduled_req_ids = self.running_requests.clone();

        if self.role != InstanceRole::DecodeOnly {
            while let Some(&cand_id) = self.pending_queue.front() {
                let is_first_chunk = self.requests[&cand_id].prefill_tokens_processed == 0;
                let chunk_len = self.requests[&cand_id]
                    .remaining_prefill_tokens()
                    .min(self.requests[&cand_id].prefill_chunk_limit)
                    .min(token_budget);

                if chunk_len == 0 {
                    break;
                }

                if is_first_chunk {
                    let prompt_clone = self.requests[&cand_id].prompt_tokens.clone();
                    match self.kv_manager.allocate_sequence(cand_id, &prompt_clone) {
                        Ok(_matched_prefix_len) => {}
                        Err(KvCacheError::OutOfBlocks { .. }) => break,
                        Err(e) => return Err(SchedulerError::KvError(e)),
                    }
                }

                self.pending_queue.pop_front();
                if let Some(req) = self.requests.get_mut(&cand_id) {
                    req.phase = RequestPhase::Prefill;
                    req.prefill_chunk_limit = chunk_len;
                }
                scheduled_req_ids.push(cand_id);
                token_budget = token_budget.saturating_sub(chunk_len);
                if token_budget == 0 {
                    break;
                }
            }
        }

        if scheduled_req_ids.is_empty() {
            return Ok(None);
        }

        // 3. Construct ORCA Selective Batching Plan using bounded prefill chunks
        let mut total_flattened_tokens = 0usize;
        let mut attention_request_slices = Vec::with_capacity(scheduled_req_ids.len());
        let mut flattened_input_tokens = Vec::new();

        for &req_id in &scheduled_req_ids {
            let req = &self.requests[&req_id];
            let num_toks = req.iteration_token_count();
            let is_prefill = req.phase == RequestPhase::Prefill;

            attention_request_slices.push((req_id, total_flattened_tokens, num_toks, is_prefill));
            total_flattened_tokens += num_toks;

            if is_prefill {
                flattened_input_tokens.extend_from_slice(req.current_prefill_chunk());
            } else if let Some(&last_gen) = req.generated_tokens.last() {
                flattened_input_tokens.push(last_gen);
            }
        }

        // 4. Build FlashInfer Inspector-Executor BSR Schedule Plan across all scheduled sequences
        let kv_schedule_plan = self
            .kv_manager
            .build_schedule_plan(&scheduled_req_ids, self.max_blocks_per_cta)?;

        Ok(Some(ScheduledIterationBatch {
            selective_batch: SelectiveBatchPlan {
                scheduled_req_ids,
                total_flattened_tokens,
                attention_request_slices,
                flattened_input_tokens,
            },
            kv_schedule_plan,
            cow_block_copies,
        }))
    }

    /// Commits sampled output tokens at the end of a single iteration:
    /// - Immediately returns finished requests (`<EOS>` or `max_new_tokens`) and frees their KV blocks.
    /// - Transitions `Prefill` requests to `Decode` (on `ColocatedHybrid`) or dispatches them for KV handoff (on `PrefillOnly`).
    pub fn commit_iteration_outputs(
        &mut self,
        sampled_tokens: &[(u64, u32)],
        current_timestamp_us: u64,
    ) -> Result<(Vec<InferenceRequest>, Vec<u64>), SchedulerError> {
        let mut finished_requests = Vec::new();
        let mut handoff_to_decode_pool = Vec::new();

        for &(req_id, next_token) in sampled_tokens {
            let req = self
                .requests
                .get_mut(&req_id)
                .ok_or(SchedulerError::RequestNotFound(req_id))?;

            if req.phase == RequestPhase::Prefill {
                let chunk_done = req.current_prefill_chunk().len();
                req.prefill_tokens_processed += chunk_done;
                req.prefill_chunk_limit = DEFAULT_PREFILL_CHUNK_SIZE;

                // If more prefill chunks remain, preempt back to front of pending_queue
                // so ongoing decode requests never wait on the rest of a huge prompt!
                if req.remaining_prefill_tokens() > 0 {
                    self.pending_queue.push_front(req_id);
                    continue;
                }
            }

            let was_prefill = req.phase == RequestPhase::Prefill;
            if was_prefill && req.first_token_timestamp_us.is_none() {
                req.first_token_timestamp_us = Some(current_timestamp_us);
            }

            req.generated_tokens.push(next_token);

            if req.is_finished() {
                req.phase = RequestPhase::Completed;
                self.running_requests.retain(|&id| id != req_id);
                self.kv_manager.free_sequence(req_id)?;
                if let Some(done_req) = self.requests.remove(&req_id) {
                    finished_requests.push(done_req);
                }
            } else if was_prefill {
                match self.role {
                    InstanceRole::ColocatedHybrid => {
                        req.phase = RequestPhase::Decode;
                        if !self.running_requests.contains(&req_id) {
                            self.running_requests.push(req_id);
                        }
                    }
                    InstanceRole::PrefillOnly => {
                        // Prefill instance finishes after generating the first token and shipping KV cache
                        handoff_to_decode_pool.push(req_id);
                        self.kv_manager.free_sequence(req_id)?;
                    }
                    InstanceRole::DecodeOnly => {}
                }
            }
        }

        Ok((finished_requests, handoff_to_decode_pool))
    }

    /// Commits additional speculative bonus tokens accepted during an EAGLE tree verification pass.
    ///
    /// For each bonus token:
    /// 1. Checks if the request is already completed (stopping early if `max_new_tokens` or `eos_token_id` was hit).
    /// 2. Reserves a physical KV block slot in `PagedKvCacheManager`.
    /// 3. Appends the bonus token to `req.generated_tokens`.
    /// 4. Returns the list of bonus tokens actually committed and `Some(InferenceRequest)` if the sequence finished.
    pub fn commit_speculative_bonus_tokens(
        &mut self,
        req_id: u64,
        bonus_tokens: &[u32],
    ) -> Result<(Vec<u32>, Option<InferenceRequest>), SchedulerError> {
        // If the primary token already finished and removed the request, no bonus tokens are needed
        if !self.requests.contains_key(&req_id) {
            return Ok((Vec::new(), None));
        }

        let mut committed_bonus = Vec::with_capacity(bonus_tokens.len());

        for &bonus_tok in bonus_tokens {
            let is_done = {
                let req = self
                    .requests
                    .get(&req_id)
                    .ok_or(SchedulerError::RequestNotFound(req_id))?;
                req.is_finished() || req.phase != RequestPhase::Decode
            };

            if is_done {
                break;
            }

            // Reserve a KV cache slot for the accepted speculative token
            self.kv_manager.append_token_slot(req_id)?;

            let finished_now = {
                let req = self
                    .requests
                    .get_mut(&req_id)
                    .ok_or(SchedulerError::RequestNotFound(req_id))?;
                req.generated_tokens.push(bonus_tok);
                committed_bonus.push(bonus_tok);
                req.is_finished()
            };

            if finished_now {
                if let Some(req) = self.requests.get_mut(&req_id) {
                    req.phase = RequestPhase::Completed;
                }
                self.running_requests.retain(|&id| id != req_id);
                self.kv_manager.free_sequence(req_id)?;
                let done_req = self.requests.remove(&req_id);
                return Ok((committed_bonus, done_req));
            }
        }

        Ok((committed_bonus, None))
    }

    /// Reports current Machine-Level Scheduler (MLS) telemetry to the Cluster-Level Scheduler (CLS).
    pub fn pending_and_running_counts(&self) -> (usize, usize, usize) {
        (
            self.pending_queue.len(),
            self.running_requests.len(),
            self.kv_manager.gpu_allocator.num_free_blocks(),
        )
    }

    /// Returns true if the given request is currently in the `Decode` phase
    /// (eligible for speculative bonus token generation).
    pub fn is_request_decoding(&self, req_id: u64) -> bool {
        matches!(
            self.requests.get(&req_id).map(|r| r.phase),
            Some(RequestPhase::Decode)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orca_iteration_scheduling_and_selective_batching() {
        let kv_mgr = PagedKvCacheManager::new(64, 32, 4);
        let mut sched = OrcaMachineScheduler::new(
            InstanceRole::ColocatedHybrid,
            32,
            1024,
            8,
            ServiceLevelObjective::default(),
            kv_mgr,
        );

        // Submit Request 1 (3 prompt tokens, max 2 output tokens) and Request 2 (2 prompt tokens, max 5 output tokens)
        sched.submit_request(InferenceRequest::new(1, vec![10, 11, 12], 2, 999, 100));
        sched.submit_request(InferenceRequest::new(2, vec![20, 21], 5, 999, 110));

        // Iteration 1: Both requests run Prefill together -> Selective batch flattens 3 + 2 = 5 tokens
        let batch1 = sched.schedule_next_iteration().unwrap().unwrap();
        assert_eq!(batch1.selective_batch.total_flattened_tokens, 5);
        assert_eq!(batch1.selective_batch.attention_request_slices.len(), 2);

        let (done1, _) = sched
            .commit_iteration_outputs(&[(1, 101), (2, 201)], 200)
            .unwrap();
        assert!(done1.is_empty());

        // Inject a newly arrived Request 3 (4 prompt tokens) before Iteration 2!
        // ORCA schedules Request 1 (1 decode token) + Request 2 (1 decode token) + Request 3 (4 prefill tokens) = 6 flattened tokens
        sched.submit_request(InferenceRequest::new(3, vec![30, 31, 32, 33], 3, 999, 210));
        let batch2 = sched.schedule_next_iteration().unwrap().unwrap();
        assert_eq!(batch2.selective_batch.total_flattened_tokens, 6);
        assert_eq!(batch2.selective_batch.scheduled_req_ids, vec![1, 2, 3]);

        // Request 1 hits its max_new_tokens (2) on Iteration 2 and returns immediately without waiting for Requests 2 & 3!
        let (done2, _) = sched
            .commit_iteration_outputs(&[(1, 102), (2, 202), (3, 301)], 300)
            .unwrap();
        assert_eq!(done2.len(), 1);
        assert_eq!(done2[0].req_id, 1);
        assert_eq!(done2[0].generated_tokens, vec![101, 102]);
    }

    #[test]
    fn test_splitwise_contiguous_kv_block_transfer_and_layer_handshake() {
        let kv_mgr_prefill = PagedKvCacheManager::new(32, 16, 4);
        let mut prefill_node = OrcaMachineScheduler::new(
            InstanceRole::PrefillOnly,
            4, // 4 transformer layers
            1024,
            8,
            ServiceLevelObjective::default(),
            kv_mgr_prefill,
        );

        let req = InferenceRequest::new(77, vec![1, 2, 3, 4, 5, 6, 7, 8], 10, 999, 50);
        prefill_node.submit_request(req.clone());

        let _batch = prefill_node.schedule_next_iteration().unwrap().unwrap();

        // Contiguous blocks 0 and 1 should be coalesced into a single one-sided put per layer
        let put_ops = prefill_node.build_layer_kv_put_ops(77, 0, 2).unwrap();
        assert_eq!(put_ops.len(), 1);
        assert_eq!(put_ops[0].start_physical_block_id, 0);
        assert_eq!(put_ops[0].num_contiguous_blocks, 2);

        let (_, handoff_ids) = prefill_node
            .commit_iteration_outputs(&[(77, 42)], 120)
            .unwrap();
        assert_eq!(handoff_ids, vec![77]);

        // Decode machine receives the 4 layer transfers and activates the request on layer 4
        let kv_mgr_decode = PagedKvCacheManager::new(32, 16, 4);
        let mut decode_node = OrcaMachineScheduler::new(
            InstanceRole::DecodeOnly,
            4,
            1024,
            8,
            ServiceLevelObjective::default(),
            kv_mgr_decode,
        );
        decode_node
            .register_incoming_disaggregated_decode(req, put_ops[0].semaphore_id)
            .unwrap();

        for layer in 0..3 {
            let ready = decode_node.notify_layer_kv_received(77).unwrap();
            assert!(
                !ready,
                "Should not activate before final layer (layer {layer})"
            );
        }
        let ready_final = decode_node.notify_layer_kv_received(77).unwrap();
        assert!(ready_final);
    }

    #[test]
    fn test_dynamic_chunked_prefill_sla_preemption() {
        let kv_mgr = PagedKvCacheManager::new(256, 64, 16);
        let mut sched = OrcaMachineScheduler::new(
            InstanceRole::ColocatedHybrid,
            32,
            513, // Budget allows 1 active decode token + 512 prefill chunk tokens per step
            16,
            ServiceLevelObjective::default(),
            kv_mgr,
        );

        // Request 1 is already decoding; Request 2 arrives with a large 1200-token prompt
        sched.submit_request(InferenceRequest::new(1, vec![10, 11, 12], 5, 999, 100));
        let _ = sched.schedule_next_iteration().unwrap().unwrap();
        sched.commit_iteration_outputs(&[(1, 101)], 150).unwrap();

        let large_prompt: Vec<u32> = (0..1200).collect();
        sched.submit_request(InferenceRequest::new(2, large_prompt, 4, 999, 160));

        // Step A: Request 1 decodes (1 tok) + Request 2 runs Chunk 1 (512 toks) = 513 tokens
        let step_a = sched.schedule_next_iteration().unwrap().unwrap();
        assert_eq!(step_a.selective_batch.total_flattened_tokens, 513);
        sched
            .commit_iteration_outputs(&[(1, 102), (2, 0)], 200)
            .unwrap();

        // Step B: Request 1 decodes (1 tok) + Request 2 runs Chunk 2 (512 toks) = 513 tokens
        let step_b = sched.schedule_next_iteration().unwrap().unwrap();
        assert_eq!(step_b.selective_batch.total_flattened_tokens, 513);
        sched
            .commit_iteration_outputs(&[(1, 103), (2, 0)], 250)
            .unwrap();

        // Step C: Request 1 decodes (1 tok) + Request 2 finishes final Chunk 3 (176 toks) = 177 tokens
        let step_c = sched.schedule_next_iteration().unwrap().unwrap();
        assert_eq!(step_c.selective_batch.total_flattened_tokens, 177);
        sched
            .commit_iteration_outputs(&[(1, 104), (2, 201)], 300)
            .unwrap();

        // Step D: Both Request 1 and Request 2 are now in single-token Decode together (2 tokens)
        let step_d = sched.schedule_next_iteration().unwrap().unwrap();
        assert_eq!(step_d.selective_batch.total_flattened_tokens, 2);
    }

    #[test]
    fn test_fix4_multi_token_speculative_bonus_commit() {
        let kv_mgr = PagedKvCacheManager::new(64, 16, 4);
        let mut sched = OrcaMachineScheduler::new(
            InstanceRole::ColocatedHybrid,
            16,
            1024,
            8,
            ServiceLevelObjective::default(),
            kv_mgr,
        );

        // Request 900 wants 4 output tokens
        sched.submit_request(InferenceRequest::new(900, vec![1, 2, 3], 4, 999, 100));

        // Iteration 1 (Prefill): Generates 1st token (101)
        let _ = sched.schedule_next_iteration().unwrap().unwrap();
        let (done1, _) = sched.commit_iteration_outputs(&[(900, 101)], 150).unwrap();
        assert!(done1.is_empty());

        // Iteration 2 (Decode + EAGLE Speculative Verification):
        // Primary pass generates 2nd token (102), and EAGLE tree verification accepts 2 bonus tokens (103, 104)!
        // Total generated tokens reaches 4 (101, 102, 103, 104) in just ONE decode iteration!
        let _ = sched.schedule_next_iteration().unwrap().unwrap();
        let (done2, _) = sched.commit_iteration_outputs(&[(900, 102)], 200).unwrap();
        assert!(done2.is_empty());

        let (committed_bonus, finished_req) = sched
            .commit_speculative_bonus_tokens(900, &[103, 104, 105])
            .unwrap();

        // Only [103, 104] should be committed because max_new_tokens == 4 is reached at 104!
        assert_eq!(committed_bonus, vec![103, 104]);

        let completed = finished_req.expect("Request should complete after bonus token 104");
        assert_eq!(completed.generated_tokens, vec![101, 102, 103, 104]);
    }
}
