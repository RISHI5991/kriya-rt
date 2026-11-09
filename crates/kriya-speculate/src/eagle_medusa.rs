// File: crates/kriya-speculate/src/eagle_medusa.rs
//! EAGLE Feature-Level Speculative Decoding Pipeline, Context-Adaptive Draft Tree Builder,
//! Tree-Attention Mask Generator, and Recursive Speculative Verification Engine.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum SpeculateError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Invalid draft tree configuration: {0}")]
    InvalidTreeConfig(String),
    #[error("Verification distribution missing for draft node {node_id}")]
    MissingTargetDistribution { node_id: usize },
}

/// Target-verified speculative output: accepted token IDs, their tree-node
/// indices, and the corresponding target-model features.
pub type VerifiedDraftBranch = (Vec<u32>, Vec<usize>, Vec<Vec<f32>>);

/// Deterministic pseudo-random number generator (xorshift64*) for reproducible
/// speculative acceptance/rejection sampling and training noise injection.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABEu64
            } else {
                seed
            },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1Du64)
    }

    /// Returns a uniform float in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Samples uniform noise in `[-bound, bound]` (e.g., `U(-0.1, 0.1)` for EAGLE feature augmentation).
    #[inline]
    pub fn uniform_range(&mut self, bound: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * bound
    }
}

/// Context-dependent draft tree topology where higher-probability branches are deeper and wider,
/// and tree size shrinks automatically as batch size increases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftTreeTopology {
    /// Number of autoregressive draft forward passes `m`.
    pub depth: usize,
    /// Branching factor per node at each depth level (higher-probability roots branch wider).
    pub level_branching: Vec<Vec<usize>>,
    /// Total drafted token nodes across all `m` passes.
    pub total_draft_tokens: usize,
}

impl DraftTreeTopology {
    /// Canonical 3-pass, 10-token tree structure (branches of higher-probability tokens are deeper and wider):
    /// - Pass 1 (root): top-3 tokens -> [n0, n1, n2] (3 nodes)
    /// - Pass 2: n0 branches into 3, n1 into 2, n2 into 0 -> [n3, n4, n5, n6, n7] (5 nodes)
    /// - Pass 3: n3 (highest prob) branches into 2 -> [n8, n9] (2 nodes)
    ///   Total = 3 + 5 + 2 = 10 draft tokens in 3 forward passes.
    pub fn canonical_10_token_tree() -> Self {
        Self {
            depth: 3,
            level_branching: vec![
                vec![3],             // Pass 1: Root expands top-3 candidates
                vec![3, 2, 0], // Pass 2: Top branch gets 3 children, second gets 2, third gets 0
                vec![2, 0, 0, 0, 0], // Pass 3: Highest-prob grandchild expands 2 children
            ],
            total_draft_tokens: 10,
        }
    }

    /// Compact 2-pass, 4-token tree used when batch size increases and redundant compute shrinks.
    pub fn compact_batch_tree() -> Self {
        Self {
            depth: 2,
            level_branching: vec![
                vec![2],    // Pass 1: Root expands top-2 candidates
                vec![2, 0], // Pass 2: Top candidate expands 2 children
            ],
            total_draft_tokens: 4,
        }
    }

    /// Selects the optimal draft tree structure dynamically based on current batch size.
    pub fn select_for_batch_size(batch_size: usize, max_speculative_batch: usize) -> Option<Self> {
        if batch_size > max_speculative_batch {
            None
        } else if batch_size <= 2 {
            Some(Self::canonical_10_token_tree())
        } else {
            Some(Self::compact_batch_tree())
        }
    }
}

/// Single node inside the tree-structured draft.
#[derive(Debug, Clone)]
pub struct DraftTreeNode {
    /// Index of this node in the flattened draft tree (`0..=total_draft_tokens`, 0 is root).
    pub node_id: usize,
    /// Parent node index (`None` for root node 0).
    pub parent_id: Option<usize>,
    /// Depth level in the draft tree (`0` for root, `1..=m` for drafted tokens).
    pub depth: usize,
    /// Sampled or top-k drafted token ID at this node.
    pub token_id: u32,
    /// Drafted feature vector `\hat{f}` of length `hidden_dim` produced by the Autoregression Head.
    pub predicted_feature: Vec<f32>,
    /// Draft probability distribution `\hat{p}` over `vocab_size` at this node.
    pub draft_probs: Vec<f32>,
    /// Child node indices in the flattened tree array.
    pub children: Vec<usize>,
}

/// Complete tree-structured draft with its flattened causal tree-attention mask.
#[derive(Debug, Clone)]
pub struct DraftTree {
    pub nodes: Vec<DraftTreeNode>,
    /// Flattened boolean causal tree mask `[num_nodes * num_nodes]` where `mask[i * N + j] == true`
    /// iff node `j` is an ancestor of (or equal to) node `i`.
    pub tree_attention_mask: Vec<bool>,
}

impl DraftTree {
    /// Builds the 2D tree-attention mask so the target LLM can verify the entire tree in one forward pass.
    pub fn compute_tree_attention_mask(nodes: &[DraftTreeNode]) -> Vec<bool> {
        let n = nodes.len();
        let mut mask = vec![false; n * n];

        for i in 0..n {
            let mut curr = Some(i);
            while let Some(ancestor_idx) = curr {
                mask[i * n + ancestor_idx] = true;
                curr = nodes[ancestor_idx].parent_id;
            }
        }
        mask
    }
}

/// Numerically stable Softmax over a slice of logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    for v in &mut exps {
        *v /= sum;
    }
    exps
}

/// Returns the indices of the `k` largest probabilities in descending order.
fn top_k_indices(probs: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// EAGLE Autoregression Head & Frozen Target Parameter References.
///
/// The Embedding layer and LM Head reuse the target LLM's parameters without additional training.
/// The Autoregression Head consists of:
/// 1. An FC layer reducing the fused `(token_embedding || feature)` sequence from `2 * hidden_dim` to `hidden_dim`.
/// 2. A single lightweight decoder layer predicting the next feature `\hat{f}_{i+1}`.
pub struct EagleDraftEngine {
    pub hidden_dim: usize,
    pub vocab_size: usize,
    /// Frozen target LLM token embedding table `[vocab_size, hidden_dim]`.
    pub target_embedding_weights: Vec<f32>,
    /// Frozen target LLM language model head `[vocab_size, hidden_dim]`.
    pub target_lm_head_weights: Vec<f32>,
    /// Trainable FC dimensionality reduction matrix `[hidden_dim, 2 * hidden_dim]`.
    pub fc_fuse_weights: Vec<f32>,
    /// Trainable single decoder layer projection `[hidden_dim, hidden_dim]`.
    pub decoder_layer_weights: Vec<f32>,
}

impl EagleDraftEngine {
    pub fn new(
        hidden_dim: usize,
        vocab_size: usize,
        target_embedding_weights: Vec<f32>,
        target_lm_head_weights: Vec<f32>,
        fc_fuse_weights: Vec<f32>,
        decoder_layer_weights: Vec<f32>,
    ) -> Result<Self, SpeculateError> {
        if target_embedding_weights.len() != vocab_size * hidden_dim {
            return Err(SpeculateError::DimensionMismatch {
                expected: vocab_size * hidden_dim,
                actual: target_embedding_weights.len(),
            });
        }
        if target_lm_head_weights.len() != vocab_size * hidden_dim {
            return Err(SpeculateError::DimensionMismatch {
                expected: vocab_size * hidden_dim,
                actual: target_lm_head_weights.len(),
            });
        }
        if fc_fuse_weights.len() != hidden_dim * (2 * hidden_dim) {
            return Err(SpeculateError::DimensionMismatch {
                expected: hidden_dim * 2 * hidden_dim,
                actual: fc_fuse_weights.len(),
            });
        }
        if decoder_layer_weights.len() != hidden_dim * hidden_dim {
            return Err(SpeculateError::DimensionMismatch {
                expected: hidden_dim * hidden_dim,
                actual: decoder_layer_weights.len(),
            });
        }

        if hidden_dim == 0 || vocab_size == 0 {
            return Err(SpeculateError::InvalidTreeConfig(
                "hidden_dim and vocab_size must both be non-zero".into(),
            ));
        }

        Ok(Self {
            hidden_dim,
            vocab_size,
            target_embedding_weights,
            target_lm_head_weights,
            fc_fuse_weights,
            decoder_layer_weights,
        })
    }

    /// Predicts the next feature `\hat{f}_{i+1}` and next-token probability distribution `\hat{p}_{i+2}`
    /// from an advanced token `t_{i+1}` and previous feature `f_i`.
    pub fn forward_step(
        &self,
        advanced_token_id: u32,
        prev_feature: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), SpeculateError> {
        if prev_feature.len() != self.hidden_dim {
            return Err(SpeculateError::DimensionMismatch {
                expected: self.hidden_dim,
                actual: prev_feature.len(),
            });
        }

        let tok_idx = (advanced_token_id as usize).min(self.vocab_size - 1);
        let tok_emb = &self.target_embedding_weights
            [tok_idx * self.hidden_dim..(tok_idx + 1) * self.hidden_dim];

        // 1. Concatenate token embedding (hidden_dim) and feature (hidden_dim) -> fused (2 * hidden_dim)
        let fused_dim = 2 * self.hidden_dim;
        let mut fused = Vec::with_capacity(fused_dim);
        fused.extend_from_slice(tok_emb);
        fused.extend_from_slice(prev_feature);

        // 2. FC layer reduces dimensionality from `2 * hidden_dim` to `hidden_dim`
        let mut reduced = vec![0.0f32; self.hidden_dim];
        for (h, reduced_value) in reduced.iter_mut().enumerate() {
            let row = &self.fc_fuse_weights[h * fused_dim..(h + 1) * fused_dim];
            *reduced_value = row.iter().zip(fused.iter()).map(|(w, x)| w * x).sum();
        }

        // 3. Decoder layer predicts next feature `\hat{f}_{i+1}` (with residual connection)
        let mut next_feature = vec![0.0f32; self.hidden_dim];
        for h in 0..self.hidden_dim {
            let row = &self.decoder_layer_weights[h * self.hidden_dim..(h + 1) * self.hidden_dim];
            let delta: f32 = row.iter().zip(reduced.iter()).map(|(w, x)| w * x).sum();
            next_feature[h] = reduced[h] + delta;
        }

        // 4. Frozen Target LM Head computes logits & Softmax distribution from `\hat{f}_{i+1}`
        let probs = self.compute_lm_head_probs(&next_feature);

        Ok((next_feature, probs))
    }

    /// Projects a feature vector through the frozen target LM Head and applies Softmax.
    /// Unrolled 4-way to allow LLVM auto-vectorization into ARM NEON fmla.4s SIMD on Apple Silicon.
    pub fn compute_lm_head_probs(&self, feature: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.vocab_size];
        let h_dim = self.hidden_dim;
        for (v, logit) in logits.iter_mut().enumerate() {
            let row = &self.target_lm_head_weights[v * h_dim..(v + 1) * h_dim];
            let chunks4 = h_dim / 4;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;
            for i in 0..chunks4 {
                let idx = i * 4;
                acc0 += row[idx] * feature[idx];
                acc1 += row[idx + 1] * feature[idx + 1];
                acc2 += row[idx + 2] * feature[idx + 2];
                acc3 += row[idx + 3] * feature[idx + 3];
            }
            let mut sum = (acc0 + acc1) + (acc2 + acc3);
            for idx in (chunks4 * 4)..h_dim {
                sum += row[idx] * feature[idx];
            }
            *logit = sum;
        }
        softmax(&logits)
    }

    /// Builds a tree-structured draft across `m` forward passes according to `topology`.
    pub fn build_draft_tree(
        &self,
        root_token_id: u32,
        root_feature: &[f32],
        topology: &DraftTreeTopology,
    ) -> Result<DraftTree, SpeculateError> {
        let (root_pred_feat, root_probs) = self.forward_step(root_token_id, root_feature)?;

        let mut nodes = Vec::with_capacity(topology.total_draft_tokens + 1);
        nodes.push(DraftTreeNode {
            node_id: 0,
            parent_id: None,
            depth: 0,
            token_id: root_token_id,
            predicted_feature: root_pred_feat,
            draft_probs: root_probs,
            children: Vec::new(),
        });

        let mut current_level_indices = vec![0usize];

        for pass_idx in 0..topology.depth {
            let branching_spec = topology.level_branching.get(pass_idx).ok_or_else(|| {
                SpeculateError::InvalidTreeConfig(format!(
                    "Missing branching spec at pass {pass_idx}"
                ))
            })?;

            let mut next_level_indices = Vec::new();

            for (pos_in_level, &parent_node_idx) in current_level_indices.iter().enumerate() {
                let num_branches = branching_spec.get(pos_in_level).copied().unwrap_or(0);
                if num_branches == 0 {
                    continue;
                }

                let parent_probs = nodes[parent_node_idx].draft_probs.clone();
                let parent_feat = nodes[parent_node_idx].predicted_feature.clone();
                let top_tokens = top_k_indices(&parent_probs, num_branches);

                for tok in top_tokens {
                    let child_id = nodes.len();
                    let (child_feat, child_probs) = self.forward_step(tok as u32, &parent_feat)?;

                    nodes[parent_node_idx].children.push(child_id);
                    nodes.push(DraftTreeNode {
                        node_id: child_id,
                        parent_id: Some(parent_node_idx),
                        depth: pass_idx + 1,
                        token_id: tok as u32,
                        predicted_feature: child_feat,
                        draft_probs: child_probs,
                        children: Vec::new(),
                    });
                    next_level_indices.push(child_id);
                }
            }

            current_level_indices = next_level_indices;
        }

        let tree_attention_mask = DraftTree::compute_tree_attention_mask(&nodes);
        Ok(DraftTree {
            nodes,
            tree_attention_mask,
        })
    }

    /// Recursive Speculative Verification Phase:
    /// Evaluates the draft tree against the target LLM's single-pass probability distributions `target_probs`
    /// and target features `target_features`. Guarantees output distribution alignment with the target LLM
    /// while recording accepted tokens and their features for the next drafting phase.
    pub fn verify_draft_tree_recursive(
        &self,
        tree: &DraftTree,
        target_probs: &[Vec<f32>],
        target_features: &[Vec<f32>],
        rng: &mut DeterministicRng,
    ) -> Result<VerifiedDraftBranch, SpeculateError> {
        if target_probs.len() < tree.nodes.len() || target_features.len() < tree.nodes.len() {
            return Err(SpeculateError::MissingTargetDistribution {
                node_id: target_probs.len(),
            });
        }

        let mut accepted_tokens = Vec::new();
        let mut accepted_node_indices = Vec::new();
        let mut accepted_features = Vec::new();
        let mut current_node_id = 0usize;

        loop {
            let node = &tree.nodes[current_node_id];
            let p_target = &target_probs[current_node_id];
            let p_draft = &node.draft_probs;

            // Residual probability distribution tracking remaining target mass after rejections
            let mut residual_p = p_target.clone();
            let mut accepted_child: Option<usize> = None;

            for &child_id in &node.children {
                let candidate_tok = tree.nodes[child_id].token_id as usize;
                let q_x = p_draft[candidate_tok].max(1e-12);
                let p_x = residual_p[candidate_tok];

                let accept_prob = (p_x / q_x).min(1.0);
                if rng.next_f32() < accept_prob {
                    accepted_tokens.push(candidate_tok as u32);
                    accepted_node_indices.push(child_id);
                    accepted_features.push(target_features[child_id].clone());
                    accepted_child = Some(child_id);
                    break;
                } else {
                    // Adjust residual distribution after rejecting candidate_tok
                    for v in 0..self.vocab_size {
                        residual_p[v] = (residual_p[v] - p_draft[v]).max(0.0);
                    }
                    let sum_res: f32 = residual_p.iter().sum();
                    if sum_res > 1e-12 {
                        for v in &mut residual_p {
                            *v /= sum_res;
                        }
                    } else {
                        residual_p.clone_from(p_target);
                    }
                }
            }

            if let Some(next_id) = accepted_child {
                current_node_id = next_id;
                // If we reached a leaf of the draft tree, sample 1 bonus token from the leaf's target distribution
                if tree.nodes[current_node_id].children.is_empty() {
                    let bonus_tok =
                        Self::sample_from_distribution(&target_probs[current_node_id], rng);
                    accepted_tokens.push(bonus_tok);
                    accepted_node_indices.push(current_node_id);
                    accepted_features.push(target_features[current_node_id].clone());
                    break;
                }
            } else {
                // All drafted children at `current_node_id` were rejected (or node had no children);
                // sample a replacement token from the adjusted residual distribution.
                let replacement_tok = Self::sample_from_distribution(&residual_p, rng);
                accepted_tokens.push(replacement_tok);
                accepted_node_indices.push(current_node_id);
                accepted_features.push(target_features[current_node_id].clone());
                break;
            }
        }

        Ok((accepted_tokens, accepted_node_indices, accepted_features))
    }

    fn sample_from_distribution(probs: &[f32], rng: &mut DeterministicRng) -> u32 {
        let r = rng.next_f32();
        let mut cdf = 0.0f32;
        for (idx, &p) in probs.iter().enumerate() {
            cdf += p;
            if r <= cdf {
                return idx as u32;
            }
        }
        (probs.len().saturating_sub(1)) as u32
    }

    /// EAGLE Training Loss Calculator:
    /// Injects uniform noise `U(-0.1, 0.1)` into target features to prevent autoregressive error
    /// accumulation, and computes `L = L_reg (Smooth L1) + w_cls * L_cls (Cross-Entropy)`.
    pub fn compute_training_loss_with_noise(
        &self,
        advanced_token_id: u32,
        clean_prev_feature: &[f32],
        target_next_feature: &[f32],
        cls_weight: f32,
        rng: &mut DeterministicRng,
    ) -> Result<(f32, f32, f32), SpeculateError> {
        if target_next_feature.len() != self.hidden_dim {
            return Err(SpeculateError::DimensionMismatch {
                expected: self.hidden_dim,
                actual: target_next_feature.len(),
            });
        }

        // Add random noise sampled from uniform distribution U(-0.1, 0.1) to target features
        let noisy_prev_feature: Vec<f32> = clean_prev_feature
            .iter()
            .map(|&f| f + rng.uniform_range(0.1))
            .collect();

        let (pred_next_feature, pred_probs) =
            self.forward_step(advanced_token_id, &noisy_prev_feature)?;
        let target_probs = self.compute_lm_head_probs(target_next_feature);

        // 1. Smooth L1 regression loss between target `f_{i+1}` and predicted `\hat{f}_{i+1}`
        let mut l_reg = 0.0f32;
        for (&f_true, &f_hat) in target_next_feature.iter().zip(pred_next_feature.iter()) {
            let diff = (f_true - f_hat).abs();
            l_reg += if diff < 1.0 {
                0.5 * diff * diff
            } else {
                diff - 0.5
            };
        }
        l_reg /= self.hidden_dim as f32;

        // 2. Classification cross-entropy loss between `p_{i+2}` and `\hat{p}_{i+2}`
        let mut l_cls = 0.0f32;
        for (&p_true, &p_hat) in target_probs.iter().zip(pred_probs.iter()) {
            if p_true > 0.0 {
                l_cls -= p_true * p_hat.max(1e-12).ln();
            }
        }

        let total_loss = l_reg + cls_weight * l_cls;
        Ok((total_loss, l_reg, l_cls))
    }

    /// File signature for trained EAGLE draft-head weights.
    ///
    /// The payload is intentionally independent from target weights: deployments must
    /// train this head against the exact target checkpoint, then provide it separately.
    const WEIGHTS_MAGIC: [u8; 8] = *b"KRYAEGL1";
    const WEIGHTS_VERSION: u32 = 1;
    const WEIGHTS_HEADER_BYTES: usize = 32;

    /// Serializes trained FC and decoder weights with model dimensions and a versioned
    /// header. This prevents accidentally loading a draft head trained for another model.
    pub fn save_weights(&self) -> Vec<u8> {
        let payload_bytes = (self.fc_fuse_weights.len() + self.decoder_layer_weights.len()) * 4;
        let mut buf = Vec::with_capacity(Self::WEIGHTS_HEADER_BYTES + payload_bytes);
        buf.extend_from_slice(&Self::WEIGHTS_MAGIC);
        buf.extend_from_slice(&Self::WEIGHTS_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.hidden_dim as u64).to_le_bytes());
        buf.extend_from_slice(&(self.vocab_size as u64).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // Reserved for future format flags.
        for &weight in self
            .fc_fuse_weights
            .iter()
            .chain(self.decoder_layer_weights.iter())
        {
            buf.extend_from_slice(&weight.to_le_bytes());
        }
        buf
    }

    /// Loads a trained EAGLE head. Rejects raw, wrong-model, truncated, and non-finite
    /// artifacts rather than silently enabling speculation with invalid probabilities.
    pub fn load_weights(&mut self, data: &[u8]) -> Result<(), SpeculateError> {
        let payload_bytes = (self.fc_fuse_weights.len() + self.decoder_layer_weights.len()) * 4;
        let expected_len = Self::WEIGHTS_HEADER_BYTES + payload_bytes;
        if data.len() != expected_len {
            return Err(SpeculateError::InvalidTreeConfig(format!(
                "EAGLE weight artifact length mismatch: expected {expected_len}, got {}",
                data.len()
            )));
        }
        if data[..8] != Self::WEIGHTS_MAGIC {
            return Err(SpeculateError::InvalidTreeConfig(
                "invalid EAGLE weight artifact magic".into(),
            ));
        }
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let hidden_dim = u64::from_le_bytes(data[12..20].try_into().unwrap()) as usize;
        let vocab_size = u64::from_le_bytes(data[20..28].try_into().unwrap()) as usize;
        if version != Self::WEIGHTS_VERSION
            || hidden_dim != self.hidden_dim
            || vocab_size != self.vocab_size
        {
            return Err(SpeculateError::InvalidTreeConfig(format!(
                "EAGLE weight artifact is for version {version}, hidden_dim {hidden_dim}, vocab_size {vocab_size}; expected version {}, hidden_dim {}, vocab_size {}",
                Self::WEIGHTS_VERSION,
                self.hidden_dim,
                self.vocab_size
            )));
        }

        // Validate the complete payload before mutating this engine. A failed load
        // must not leave a partially overwritten draft head behind.
        let mut parsed = Vec::with_capacity(payload_bytes / 4);
        let mut offset = Self::WEIGHTS_HEADER_BYTES;
        while offset < data.len() {
            let value = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            if !value.is_finite() {
                return Err(SpeculateError::InvalidTreeConfig(
                    "EAGLE weight artifact contains a non-finite value".into(),
                ));
            }
            parsed.push(value);
            offset += 4;
        }
        let split = self.fc_fuse_weights.len();
        self.fc_fuse_weights.copy_from_slice(&parsed[..split]);
        self.decoder_layer_weights.copy_from_slice(&parsed[split..]);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_eagle_engine() -> EagleDraftEngine {
        let hidden_dim = 8;
        let vocab_size = 16;
        let target_emb = vec![0.05f32; vocab_size * hidden_dim];
        let target_lm_head = (0..vocab_size * hidden_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let fc_fuse = vec![0.05f32; hidden_dim * (2 * hidden_dim)];
        let decoder_w = vec![0.02f32; hidden_dim * hidden_dim];

        EagleDraftEngine::new(
            hidden_dim,
            vocab_size,
            target_emb,
            target_lm_head,
            fc_fuse,
            decoder_w,
        )
        .unwrap()
    }

    #[test]
    fn test_eagle_10_token_tree_in_3_forward_passes() {
        let engine = create_test_eagle_engine();
        let topology = DraftTreeTopology::canonical_10_token_tree();
        let root_feat = vec![0.1f32; 8];

        let tree = engine.build_draft_tree(1, &root_feat, &topology).unwrap();
        // 1 root node + 10 drafted nodes = 11 total nodes in tree
        assert_eq!(tree.nodes.len(), 11);
        assert_eq!(tree.tree_attention_mask.len(), 11 * 11);

        // Verify root (node 0) is an ancestor of every node in the tree attention mask
        for i in 0..11 {
            assert!(tree.tree_attention_mask[i * 11]);
        }
    }

    #[test]
    fn test_recursive_speculative_verification_exact_match() {
        let engine = create_test_eagle_engine();
        let topology = DraftTreeTopology::canonical_10_token_tree();
        let root_feat = vec![0.2f32; 8];
        let tree = engine.build_draft_tree(2, &root_feat, &topology).unwrap();

        // When target distributions match draft distributions identically, acceptance rate is 100% along the deepest path (depth 3 + 1 bonus token = 4 tokens)
        let target_probs: Vec<Vec<f32>> =
            tree.nodes.iter().map(|n| n.draft_probs.clone()).collect();
        let target_feats: Vec<Vec<f32>> = tree
            .nodes
            .iter()
            .map(|n| n.predicted_feature.clone())
            .collect();

        let mut rng = DeterministicRng::new(42);
        let (accepted_tokens, _accepted_node_indices, accepted_features) = engine
            .verify_draft_tree_recursive(&tree, &target_probs, &target_feats, &mut rng)
            .unwrap();

        assert_eq!(accepted_tokens.len(), 4);
        assert_eq!(accepted_features.len(), 4);
    }

    #[test]
    fn test_training_loss_with_uniform_noise_augmentation() {
        let engine = create_test_eagle_engine();
        let mut rng = DeterministicRng::new(123);
        let clean_feat = vec![0.1f32; 8];
        let target_feat = vec![0.15f32; 8];

        let (total, l_reg, l_cls) = engine
            .compute_training_loss_with_noise(3, &clean_feat, &target_feat, 0.1, &mut rng)
            .unwrap();

        assert!(total > 0.0);
        assert!(l_reg >= 0.0);
        assert!(l_cls > 0.0);
    }

    #[test]
    fn test_trained_weight_artifact_round_trip_and_dimension_check() {
        let source = create_test_eagle_engine();
        let artifact = source.save_weights();

        let mut loaded = create_test_eagle_engine();
        loaded.fc_fuse_weights.fill(0.0);
        loaded.decoder_layer_weights.fill(0.0);
        loaded.load_weights(&artifact).unwrap();
        assert_eq!(loaded.fc_fuse_weights, source.fc_fuse_weights);
        assert_eq!(loaded.decoder_layer_weights, source.decoder_layer_weights);

        let mut wrong_model = EagleDraftEngine::new(
            4,
            16,
            vec![0.0; 64],
            vec![0.0; 64],
            vec![0.0; 32],
            vec![0.0; 16],
        )
        .unwrap();
        assert!(wrong_model.load_weights(&artifact).is_err());
    }

    #[test]
    fn test_weight_artifact_rejects_non_finite_values() {
        let source = create_test_eagle_engine();
        let mut artifact = source.save_weights();
        artifact[32..36].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut destination = create_test_eagle_engine();
        let before = destination.fc_fuse_weights.clone();
        assert!(destination.load_weights(&artifact).is_err());
        assert_eq!(destination.fc_fuse_weights, before);
    }
}
