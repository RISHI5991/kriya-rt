// File: crates/kriya-hybrid/src/solver.rs
//! PowerInfer Offline 64-Neuron Batch Placement Solver, Adaptive Sparsity Predictors,
//! Dual GPU/CPU Neuron Tables, Hybrid DAG Execution Queue, and FlexGen Block Cost Model.

use kriya_core::tensor::NeuronPlacementTable;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;

/// Number of neurons with similar activation impact grouped into a single solver batch.
pub const NEURON_SOLVER_BATCH_SIZE: usize = 64;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum HybridError {
    #[error("Hardware memory budget invalid: GPU capacity {0} bytes")]
    InvalidHardwareBudget(usize),
    #[error("Layer {layer_idx} neuron count {count} must be non-zero")]
    EmptyLayer { layer_idx: u32, count: usize },
    #[error("DAG operator {op_id} references unknown prerequisite operator {prereq_id}")]
    InvalidDagDependency { op_id: usize, prereq_id: usize },
}

/// Hardware bandwidth and memory specification for the offline placement policy solver.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HardwareHierarchySpec {
    /// Available GPU memory budget in bytes for hot neurons (after reserving predictor/KV space).
    pub gpu_memory_budget_bytes: usize,
    /// Available CPU memory budget in bytes for cold neurons.
    pub cpu_memory_budget_bytes: usize,
    /// GPU HBM/Unified memory bandwidth in GB/s.
    pub gpu_bandwidth_gbps: f32,
    /// CPU DRAM memory bandwidth in GB/s.
    pub cpu_bandwidth_gbps: f32,
    /// PCIe / interconnect synchronization latency in microseconds for merging CPU results on GPU.
    pub sync_overhead_us: f32,
}

/// Profiled activation statistics for a single neuron (a row/column in a weight matrix).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NeuronActivationProfile {
    pub layer_idx: u32,
    pub original_neuron_idx: u32,
    /// Normalized activation frequency / impact metric collected from general datasets (0.0..=1.0).
    pub impact_metric: f32,
    /// Memory footprint of this single neuron (row + column weights) in bytes.
    pub byte_size: usize,
}

/// Aggregated batch of up to 64 neurons with similar impact metrics within the same layer.
#[derive(Debug, Clone)]
pub struct AggregatedNeuronBatch {
    pub layer_idx: u32,
    pub neuron_indices: Vec<u32>,
    pub total_impact: f32,
    pub total_bytes: usize,
}

/// Offline Policy Solver that groups neurons into 64-neuron batches and solves the
/// placement policy to maximize hot-neuron impact on the GPU under memory and bandwidth constraints.
pub struct OfflineNeuronPolicySolver {
    pub spec: HardwareHierarchySpec,
}

impl OfflineNeuronPolicySolver {
    pub fn new(spec: HardwareHierarchySpec) -> Result<Self, HybridError> {
        if spec.gpu_memory_budget_bytes == 0 {
            return Err(HybridError::InvalidHardwareBudget(0));
        }
        Ok(Self { spec })
    }

    /// Aggregates neurons within a layer into sorted batches of 64 neurons with similar impacts.
    pub fn aggregate_layer_batches(
        layer_idx: u32,
        mut profiles: Vec<NeuronActivationProfile>,
    ) -> Result<Vec<AggregatedNeuronBatch>, HybridError> {
        if profiles.is_empty() {
            return Err(HybridError::EmptyLayer {
                layer_idx,
                count: 0,
            });
        }

        // Sort descending by impact metric so similar-impact neurons are grouped together
        profiles.sort_by(|a, b| {
            b.impact_metric
                .partial_cmp(&a.impact_metric)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut batches = Vec::with_capacity(profiles.len().div_ceil(NEURON_SOLVER_BATCH_SIZE));

        for chunk in profiles.chunks(NEURON_SOLVER_BATCH_SIZE) {
            let neuron_indices = chunk.iter().map(|p| p.original_neuron_idx).collect();
            let total_impact = chunk.iter().map(|p| p.impact_metric).sum();
            let total_bytes = chunk.iter().map(|p| p.byte_size).sum();

            batches.push(AggregatedNeuronBatch {
                layer_idx,
                neuron_indices,
                total_impact,
                total_bytes,
            });
        }

        Ok(batches)
    }

    /// Solves the global hot/cold neuron placement policy across all layers,
    /// returning per-layer `NeuronPlacementTable` mappings for the online inference engine.
    pub fn solve_placement_policy(
        &self,
        layer_profiles: &[(u32, Vec<NeuronActivationProfile>)],
    ) -> Result<Vec<(u32, NeuronPlacementTable)>, HybridError> {
        let mut all_batches: Vec<AggregatedNeuronBatch> = Vec::new();
        let mut layer_totals: Vec<(u32, u32)> = Vec::with_capacity(layer_profiles.len());

        for (layer_idx, profiles) in layer_profiles {
            layer_totals.push((*layer_idx, profiles.len() as u32));
            let mut layer_batches = Self::aggregate_layer_batches(*layer_idx, profiles.clone())?;
            all_batches.append(&mut layer_batches);
        }

        // Rank 64-neuron batches by impact density (impact per byte weighted by GPU/CPU bandwidth ratio)
        let bw_ratio =
            (self.spec.gpu_bandwidth_gbps / self.spec.cpu_bandwidth_gbps.max(1.0)).max(1.0);
        let mut batch_order: Vec<usize> = (0..all_batches.len()).collect();
        batch_order.sort_by(|&a, &b| {
            let density_a = (all_batches[a].total_impact * bw_ratio)
                / (all_batches[a].total_bytes.max(1) as f32);
            let density_b = (all_batches[b].total_impact * bw_ratio)
                / (all_batches[b].total_bytes.max(1) as f32);
            density_b
                .partial_cmp(&density_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut remaining_gpu_bytes = self.spec.gpu_memory_budget_bytes;
        let mut batch_on_gpu = vec![false; all_batches.len()];

        for idx in batch_order {
            let b_bytes = all_batches[idx].total_bytes;
            if b_bytes <= remaining_gpu_bytes {
                batch_on_gpu[idx] = true;
                remaining_gpu_bytes -= b_bytes;
            }
        }

        // Construct the GPU and CPU neuron tables for each layer
        let mut tables = Vec::with_capacity(layer_totals.len());
        for (layer_idx, total_neurons) in layer_totals {
            let mut gpu_hot_indices = Vec::new();
            let mut cpu_cold_indices = Vec::new();

            for (b_idx, batch) in all_batches.iter().enumerate() {
                if batch.layer_idx == layer_idx {
                    if batch_on_gpu[b_idx] {
                        gpu_hot_indices.extend_from_slice(&batch.neuron_indices);
                    } else {
                        cpu_cold_indices.extend_from_slice(&batch.neuron_indices);
                    }
                }
            }

            // Sort indices to preserve deterministic row order inside GPU and CPU tables
            gpu_hot_indices.sort_unstable();
            cpu_cold_indices.sort_unstable();

            tables.push((
                layer_idx,
                NeuronPlacementTable {
                    total_neurons,
                    gpu_hot_indices,
                    cpu_cold_indices,
                },
            ));
        }

        Ok(tables)
    }
}

/// Adaptive MLP Sparsity Predictor sized according to layer activation sparsity and skewness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSparsityPredictor {
    pub layer_idx: u32,
    pub input_dim: usize,
    pub hidden_rank: usize,
    pub num_neurons: usize,
    pub activation_threshold: f32,
    /// Down-projection weights `[hidden_rank, input_dim]`
    w1: Vec<f32>,
    /// Up-projection weights `[num_neurons, hidden_rank]`
    w2: Vec<f32>,
}

impl AdaptiveSparsityPredictor {
    /// Adapts predictor hidden layer rank iteratively based on observed layer sparsity (0.0..1.0)
    /// and skewness so total predictor footprint stays compact (~6% of LLM parameters).
    pub fn new_adaptive(
        layer_idx: u32,
        input_dim: usize,
        num_neurons: usize,
        observed_sparsity: f32,
        observed_skewness: f32,
        activation_threshold: f32,
    ) -> Self {
        let base_rank = (input_dim / 8).max(32);
        let compression_factor =
            (observed_sparsity * 0.6 + observed_skewness.clamp(0.0, 1.0) * 0.4).clamp(0.1, 0.85);
        let hidden_rank = ((base_rank as f32) * (1.0 - compression_factor)).round() as usize;
        let hidden_rank = hidden_rank.clamp(16, base_rank);

        Self {
            layer_idx,
            input_dim,
            hidden_rank,
            num_neurons,
            activation_threshold,
            w1: vec![0.01; hidden_rank * input_dim],
            w2: vec![0.01; num_neurons * hidden_rank],
        }
    }

    /// Predicts active neurons for `input_x` and partitions them into GPU-hot and CPU-cold subsets
    /// using the layer's `NeuronPlacementTable`.
    pub fn predict_active_split(
        &self,
        input_x: &[f32],
        table: &NeuronPlacementTable,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut hidden = vec![0.0f32; self.hidden_rank];
        for (r, hidden_value) in hidden.iter_mut().enumerate() {
            let row = &self.w1[r * self.input_dim..(r + 1) * self.input_dim];
            let dot: f32 = row.iter().zip(input_x.iter()).map(|(w, x)| w * x).sum();
            *hidden_value = dot.max(0.0); // ReLU hidden activation
        }

        let mut active_gpu_hot = Vec::new();
        for (compact_gpu_row, &orig_idx) in table.gpu_hot_indices.iter().enumerate() {
            let row = &self.w2[(orig_idx as usize) * self.hidden_rank
                ..((orig_idx as usize) + 1) * self.hidden_rank];
            let score: f32 = row.iter().zip(hidden.iter()).map(|(w, h)| w * h).sum();
            if score > self.activation_threshold {
                active_gpu_hot.push(compact_gpu_row as u32);
            }
        }

        let mut active_cpu_cold = Vec::new();
        for (compact_cpu_row, &orig_idx) in table.cpu_cold_indices.iter().enumerate() {
            let row = &self.w2[(orig_idx as usize) * self.hidden_rank
                ..((orig_idx as usize) + 1) * self.hidden_rank];
            let score: f32 = row.iter().zip(hidden.iter()).map(|(w, h)| w * h).sum();
            if score > self.activation_threshold {
                active_cpu_cold.push(compact_cpu_row as u32);
            }
        }

        (active_gpu_hot, active_cpu_cold)
    }
}

/// Processing unit target for a computational operator node in the hybrid DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorUnit {
    GpuExecutor,
    CpuExecutor { num_threads: usize },
}

/// Single computational LLM operator stored in the global CPU DAG queue.
#[derive(Debug, Clone)]
pub struct DagOperatorNode {
    pub op_id: usize,
    pub layer_idx: u32,
    pub unit: ExecutorUnit,
    pub prerequisites: Vec<usize>,
    /// True if a CPU operator's parent node ran on the GPU, requiring a GPU completion barrier.
    pub requires_gpu_parent_barrier: bool,
    pub completed: bool,
}

/// Directed Acyclic Graph (DAG) Global Queue coordinating concurrent GPU and CPU executors.
pub struct HybridDagCoordinator {
    nodes: Vec<DagOperatorNode>,
    ready_queue: VecDeque<usize>,
    enqueued: Vec<bool>,
}

impl HybridDagCoordinator {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            ready_queue: VecDeque::new(),
            enqueued: Vec::new(),
        }
    }

    /// Enqueues an operator into the DAG and automatically detects cross-device barrier requirements.
    pub fn add_operator(
        &mut self,
        layer_idx: u32,
        unit: ExecutorUnit,
        prerequisites: Vec<usize>,
    ) -> Result<usize, HybridError> {
        let op_id = self.nodes.len();
        let mut requires_gpu_parent_barrier = false;

        for &prereq_id in &prerequisites {
            let parent = self
                .nodes
                .get(prereq_id)
                .ok_or(HybridError::InvalidDagDependency { op_id, prereq_id })?;

            if matches!(unit, ExecutorUnit::CpuExecutor { .. })
                && parent.unit == ExecutorUnit::GpuExecutor
            {
                requires_gpu_parent_barrier = true;
            }
        }

        let is_ready = prerequisites.is_empty();
        self.nodes.push(DagOperatorNode {
            op_id,
            layer_idx,
            unit,
            prerequisites,
            requires_gpu_parent_barrier,
            completed: false,
        });
        self.enqueued.push(is_ready);

        if is_ready {
            self.ready_queue.push_back(op_id);
        }

        Ok(op_id)
    }

    /// Pulls the next ready operator whose prerequisite operators have all completed.
    pub fn pop_ready_operator(&mut self) -> Option<DagOperatorNode> {
        let op_id = self.ready_queue.pop_front()?;
        Some(self.nodes[op_id].clone())
    }

    /// Marks an operator complete and releases dependent child operators into `ready_queue`.
    pub fn complete_operator(&mut self, completed_id: usize) {
        if let Some(node) = self.nodes.get_mut(completed_id) {
            node.completed = true;
        }

        for idx in 0..self.nodes.len() {
            if !self.nodes[idx].completed
                && !self.enqueued[idx]
                && self.nodes[idx]
                    .prerequisites
                    .iter()
                    .all(|&p| self.nodes[p].completed)
            {
                self.enqueued[idx] = true;
                self.ready_queue.push_back(idx);
            }
        }
    }

    /// Merges CPU cold-neuron partial outputs into the GPU hot-neuron layer output buffer.
    pub fn merge_cpu_cold_into_gpu_output(
        gpu_layer_output: &mut [f32],
        cpu_cold_results: &[f32],
        active_cpu_compact_rows: &[u32],
        table: &NeuronPlacementTable,
    ) {
        for (i, &compact_row) in active_cpu_compact_rows.iter().enumerate() {
            if let Some(&orig_row) = table.cpu_cold_indices.get(compact_row as usize) {
                if let Some(dst) = gpu_layer_output.get_mut(orig_row as usize) {
                    *dst += cpu_cold_results[i];
                }
            }
        }
    }
}

impl Default for HybridDagCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// FlexGen Analytical Block Cost & Peak Memory Estimator for 3-Tier Offloading (GPU, CPU, Disk).
#[derive(Debug, Clone, Copy)]
pub struct FlexGenBlockCostModel {
    pub num_layers: usize,
    pub prompt_len: usize,
    pub gen_len: usize,
    pub block_batch_size: usize,
    pub hidden_size: usize,
    pub layer_weight_bytes: usize,
}

impl FlexGenBlockCostModel {
    /// Estimates total latency for computing a zig-zag block: `T = T_pre * l + T_gen * (n - 1) * l`.
    #[inline]
    pub fn estimate_block_latency_sec(&self, t_pre_layer_sec: f64, t_gen_layer_sec: f64) -> f64 {
        let l = self.num_layers as f64;
        let n = self.gen_len.max(1) as f64;
        t_pre_layer_sec * l + t_gen_layer_sec * (n - 1.0) * l
    }

    /// Estimates FP16 peak memory in bytes: `w + 2*h1*bls + 4*h1*bls*l*(s + n)`.
    #[inline]
    pub fn estimate_peak_memory_bytes(&self) -> usize {
        let w = self.layer_weight_bytes;
        let h1 = self.hidden_size;
        let bls = self.block_batch_size;
        let l = self.num_layers;
        let s = self.prompt_len;
        let n = self.gen_len;

        w + (2 * h1 * bls) + (4 * h1 * bls * l * (s + n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_64_neuron_batch_solver_and_hot_cold_split() {
        // 128 neurons -> 2 batches of 64 neurons each
        let mut profiles = Vec::with_capacity(128);
        for i in 0..128u32 {
            let impact = if i < 64 { 0.90 } else { 0.05 };
            profiles.push(NeuronActivationProfile {
                layer_idx: 0,
                original_neuron_idx: i,
                impact_metric: impact,
                byte_size: 1024, // 1 KB per neuron -> 64 KB per batch
            });
        }

        // Budget allows exactly 1 batch (64 KB) on GPU
        let spec = HardwareHierarchySpec {
            gpu_memory_budget_bytes: 64 * 1024,
            cpu_memory_budget_bytes: 512 * 1024,
            gpu_bandwidth_gbps: 800.0,
            cpu_bandwidth_gbps: 64.0,
            sync_overhead_us: 5.0,
        };

        let solver = OfflineNeuronPolicySolver::new(spec).unwrap();
        let policy = solver.solve_placement_policy(&[(0, profiles)]).unwrap();
        let (_, table) = &policy[0];

        assert_eq!(table.gpu_hot_indices.len(), 64);
        assert_eq!(table.cpu_cold_indices.len(), 64);
        assert_eq!(table.gpu_hot_indices[0], 0);
        assert_eq!(table.cpu_cold_indices[0], 64);
    }

    #[test]
    fn test_hybrid_dag_dependencies_and_gpu_parent_barrier() {
        let mut dag = HybridDagCoordinator::new();
        // Node 0: Predictor on GPU
        let pred_op = dag
            .add_operator(0, ExecutorUnit::GpuExecutor, vec![])
            .unwrap();
        // Node 1: Hot neurons on GPU
        let gpu_op = dag
            .add_operator(0, ExecutorUnit::GpuExecutor, vec![pred_op])
            .unwrap();
        // Node 2: Cold neurons on CPU (depends on GPU predictor -> requires barrier!)
        let cpu_op = dag
            .add_operator(
                0,
                ExecutorUnit::CpuExecutor { num_threads: 8 },
                vec![pred_op],
            )
            .unwrap();
        // Node 3: Merge results on GPU (depends on both GPU hot and CPU cold)
        let merge_op = dag
            .add_operator(0, ExecutorUnit::GpuExecutor, vec![gpu_op, cpu_op])
            .unwrap();

        let first = dag.pop_ready_operator().unwrap();
        assert_eq!(first.op_id, pred_op);
        assert!(dag.pop_ready_operator().is_none());

        dag.complete_operator(pred_op);
        let next1 = dag.pop_ready_operator().unwrap();
        let next2 = dag.pop_ready_operator().unwrap();
        assert_eq!(next1.op_id, gpu_op);
        assert_eq!(next2.op_id, cpu_op);
        assert!(next2.requires_gpu_parent_barrier);

        dag.complete_operator(gpu_op);
        assert!(dag.pop_ready_operator().is_none());
        dag.complete_operator(cpu_op);

        let final_node = dag.pop_ready_operator().unwrap();
        assert_eq!(final_node.op_id, merge_op);
    }
}
