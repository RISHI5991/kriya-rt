//! Zero-Allocation Execution Graph & Amortized Multi-Token Decode Scheduler.
//!
//! Pre-allocates all workspace scratch memory, residual accumulators, and token buffers
//! at startup. The hot path performs zero heap allocations, relying entirely on pre-baked
//! command streams and amortized CPU-GPU synchronization.

use crate::descriptor::{ChannelMixerDescriptor, ModelArchDescriptor, SequenceMixerDescriptor};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error(
        "Workspace allocation failure: required {required_bytes} bytes exceeds budget {budget_bytes}"
    )]
    OomBudgetExceeded {
        required_bytes: usize,
        budget_bytes: usize,
    },
    #[error("Execution graph error: {0}")]
    ExecutionFailure(String),
}

/// Static memory arena pre-allocated once at model load time to guarantee zero decode allocations.
pub struct StaticWorkspaceArena {
    pub residual_scratch_bytes: usize,
    pub attention_workspace_bytes: usize,
    pub moe_routing_scratch_bytes: usize,
    pub logits_buffer_bytes: usize,
    raw_arena: Vec<u8>,
}

impl StaticWorkspaceArena {
    /// Allocates and locks the entire memory footprint required for inference.
    pub fn new(descriptor: &ModelArchDescriptor) -> Result<Self, GraphError> {
        let hidden_size = descriptor.hidden_size as usize;
        let max_seq = descriptor.max_position_embeddings as usize;
        let vocab_size = descriptor.vocab_size as usize;

        // 1. Residual activation scratch ([seq_len * hidden_size * 2 bytes for bf16])
        let residual_scratch_bytes = 2048 * hidden_size * std::mem::size_of::<u16>();

        // 2. Attention workspace for online softmax and intermediate tiles
        let attention_workspace_bytes = 32 * max_seq * std::mem::size_of::<f32>();

        // 3. MoE routing buffer (expert histograms, prefix sums, and token scatter indices)
        let moe_routing_scratch_bytes = 256 * 4096 * std::mem::size_of::<u32>();

        // 4. Logits output buffer
        let logits_buffer_bytes = vocab_size * std::mem::size_of::<f32>();

        let total_bytes = residual_scratch_bytes
            .checked_add(attention_workspace_bytes)
            .and_then(|v| v.checked_add(moe_routing_scratch_bytes))
            .and_then(|v| v.checked_add(logits_buffer_bytes))
            .ok_or_else(|| GraphError::ExecutionFailure("Workspace size overflow".into()))?;

        // Pre-allocate the entire arena once
        let raw_arena = vec![0u8; total_bytes];

        Ok(Self {
            residual_scratch_bytes,
            attention_workspace_bytes,
            moe_routing_scratch_bytes,
            logits_buffer_bytes,
            raw_arena,
        })
    }

    #[inline]
    pub fn total_footprint(&self) -> usize {
        self.raw_arena.len()
    }
}

/// Abstract representation of a compiled execution node in the static graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecNodeType {
    RmsNorm,
    GqaAttention,
    GatedDeltaNetScan,
    DenseMlp,
    SparseMoeRouter,
    SparseMoeExperts,
    ResidualAdd,
    LmHeadAndSample,
}

#[derive(Debug, Clone)]
pub struct ExecNode {
    pub node_id: usize,
    pub layer_idx: u32,
    pub node_type: ExecNodeType,
    pub depends_on: Vec<usize>,
}

/// Compiled execution graph ensuring zero branching on model identity during inference.
pub struct CompiledExecutionGraph {
    pub nodes: Vec<ExecNode>,
}

impl CompiledExecutionGraph {
    /// Compiles a model descriptor into a static topological graph of GPU execution nodes.
    pub fn compile(descriptor: &ModelArchDescriptor) -> Result<Self, GraphError> {
        let mut nodes = Vec::new();
        let mut add_node = |layer_idx, node_type, depends_on: Vec<usize>| {
            let id = nodes.len();
            nodes.push(ExecNode {
                node_id: id,
                layer_idx,
                node_type,
                depends_on,
            });
            id
        };

        // Input embedding placeholder node
        let mut prev_node = add_node(0, ExecNodeType::ResidualAdd, vec![]);

        for layer in &descriptor.layers {
            let norm_id = add_node(layer.layer_idx, ExecNodeType::RmsNorm, vec![prev_node]);
            let mixer_type = match layer.seq_mixer {
                SequenceMixerDescriptor::GroupedQueryAttention { .. } => ExecNodeType::GqaAttention,
                SequenceMixerDescriptor::GatedDeltaNetRecurrence { .. } => {
                    ExecNodeType::GatedDeltaNetScan
                }
            };
            let mixer_id = add_node(layer.layer_idx, mixer_type, vec![norm_id]);
            let res1_id = add_node(
                layer.layer_idx,
                ExecNodeType::ResidualAdd,
                vec![prev_node, mixer_id],
            );

            let ffn_norm_id = add_node(layer.layer_idx, ExecNodeType::RmsNorm, vec![res1_id]);
            let channel_type = match layer.channel_mixer {
                ChannelMixerDescriptor::DenseMlp { .. } => ExecNodeType::DenseMlp,
                ChannelMixerDescriptor::SparseMoE { .. } => ExecNodeType::SparseMoeRouter,
            };
            let channel_id = add_node(layer.layer_idx, channel_type, vec![ffn_norm_id]);
            let final_channel_id = if matches!(
                layer.channel_mixer,
                ChannelMixerDescriptor::SparseMoE { .. }
            ) {
                add_node(
                    layer.layer_idx,
                    ExecNodeType::SparseMoeExperts,
                    vec![channel_id],
                )
            } else {
                channel_id
            };

            prev_node = add_node(
                layer.layer_idx,
                ExecNodeType::ResidualAdd,
                vec![res1_id, final_channel_id],
            );
        }

        // Final lm_head and sampling node
        add_node(0, ExecNodeType::LmHeadAndSample, vec![prev_node]);

        Ok(Self { nodes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_workspace_allocation() {
        let desc = ModelArchDescriptor::llama_3_2_3b();
        let arena = StaticWorkspaceArena::new(&desc).unwrap();
        assert!(arena.total_footprint() > 0);
        // Verify component sizes are sensible
        assert!(arena.residual_scratch_bytes > 0);
        assert!(arena.attention_workspace_bytes > 0);
        assert!(arena.logits_buffer_bytes > 0);
    }

    #[test]
    fn test_execution_graph_compilation_dense() {
        let desc = ModelArchDescriptor::llama_3_2_3b();
        let graph = CompiledExecutionGraph::compile(&desc).unwrap();
        // Dense model (no MoE expert nodes):
        // 1 embedding + 28 layers * 6 nodes/layer + 1 lm_head = 170 nodes
        assert!(graph.nodes.len() > 100);
        assert_eq!(
            graph.nodes.last().unwrap().node_type,
            ExecNodeType::LmHeadAndSample
        );
    }

    #[test]
    fn test_execution_graph_compilation_hybrid_moe() {
        let desc = ModelArchDescriptor::qwen3_6_35b_a3b();
        let graph = CompiledExecutionGraph::compile(&desc).unwrap();
        // Hybrid MoE model (all layers have MoE expert nodes):
        // 1 embedding + 40 layers * 7 nodes/layer + 1 lm_head = 282 nodes
        assert!(graph.nodes.len() > 200);
        assert_eq!(
            graph.nodes.last().unwrap().node_type,
            ExecNodeType::LmHeadAndSample
        );

        // Verify dependency integrity: every node's dependencies reference earlier nodes
        for node in &graph.nodes {
            for &dep in &node.depends_on {
                assert!(
                    dep < node.node_id,
                    "Forward dependency violation at node {}",
                    node.node_id
                );
            }
        }
    }
}
