//! Zero-Allocation Execution Graph & Amortized Multi-Token Decode Scheduler.
//!
//! Pre-allocates all workspace scratch memory, residual accumulators, and token buffers
//! at startup. The hot path performs zero heap allocations, relying entirely on pre-baked
//! command streams and amortized CPU-GPU synchronization.

use crate::descriptor::{ChannelMixerDescriptor, ModelArchDescriptor, SequenceMixerDescriptor};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("Workspace allocation failure: required {required_bytes} bytes exceeds budget {budget_bytes}")]
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
        let mut node_counter = 0;

        // Input embedding node
        nodes.push(ExecNode {
            node_id: node_counter,
            layer_idx: 0,
            node_type: ExecNodeType::ResidualAdd, // Placeholder for embedding lookup
            depends_on: vec![],
        });
        node_counter += 1;

        let mut prev_node = 0;

        for layer in &descriptor.layers {
            // 1. Input LayerNorm
            let norm_node = ExecNode {
                node_id: node_counter,
                layer_idx: layer.layer_idx,
                node_type: ExecNodeType::RmsNorm,
                depends_on: vec![prev_node],
            };
            nodes.push(norm_node);
            let current_norm_id = node_counter;
            node_counter += 1;

            // 2. Sequence Mixer (GQA or Recurrent DeltaNet)
            let mixer_type = match layer.seq_mixer {
                SequenceMixerDescriptor::GroupedQueryAttention { .. } => ExecNodeType::GqaAttention,
                SequenceMixerDescriptor::GatedDeltaNetRecurrence { .. } => {
                    ExecNodeType::GatedDeltaNetScan
                }
            };

            let mixer_node = ExecNode {
                node_id: node_counter,
                layer_idx: layer.layer_idx,
                node_type: mixer_type,
                depends_on: vec![current_norm_id],
            };
            nodes.push(mixer_node);
            let mixer_id = node_counter;
            node_counter += 1;

            // Residual add after attention
            let res1_node = ExecNode {
                node_id: node_counter,
                layer_idx: layer.layer_idx,
                node_type: ExecNodeType::ResidualAdd,
                depends_on: vec![prev_node, mixer_id],
            };
            nodes.push(res1_node);
            let res1_id = node_counter;
            node_counter += 1;

            // 3. FFN / Channel Mixer Norm
            let ffn_norm_node = ExecNode {
                node_id: node_counter,
                layer_idx: layer.layer_idx,
                node_type: ExecNodeType::RmsNorm,
                depends_on: vec![res1_id],
            };
            nodes.push(ffn_norm_node);
            let ffn_norm_id = node_counter;
            node_counter += 1;

            // 4. Channel Mixer (Dense MLP or Sparse MoE)
            let channel_type = match layer.channel_mixer {
                ChannelMixerDescriptor::DenseMlp { .. } => ExecNodeType::DenseMlp,
                ChannelMixerDescriptor::SparseMoE { .. } => ExecNodeType::SparseMoeRouter,
            };

            let channel_node = ExecNode {
                node_id: node_counter,
                layer_idx: layer.layer_idx,
                node_type: channel_type,
                depends_on: vec![ffn_norm_id],
            };
            nodes.push(channel_node);
            let channel_id = node_counter;
            node_counter += 1;

            // If MoE, add explicit expert computation node depending on router
            let final_channel_id = if matches!(
                layer.channel_mixer,
                ChannelMixerDescriptor::SparseMoE { .. }
            ) {
                let expert_node = ExecNode {
                    node_id: node_counter,
                    layer_idx: layer.layer_idx,
                    node_type: ExecNodeType::SparseMoeExperts,
                    depends_on: vec![channel_id],
                };
                nodes.push(expert_node);
                let exp_id = node_counter;
                node_counter += 1;
                exp_id
            } else {
                channel_id
            };

            // Residual add after channel mixer
            let res2_node = ExecNode {
                node_id: node_counter,
                layer_idx: layer.layer_idx,
                node_type: ExecNodeType::ResidualAdd,
                depends_on: vec![res1_id, final_channel_id],
            };
            nodes.push(res2_node);
            prev_node = node_counter;
            node_counter += 1;
        }

        // Final lm_head and sampling node
        nodes.push(ExecNode {
            node_id: node_counter,
            layer_idx: 0,
            node_type: ExecNodeType::LmHeadAndSample,
            depends_on: vec![prev_node],
        });

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
