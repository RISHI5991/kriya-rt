//! Data-Driven Architecture Descriptors & Canonical Weight Schema Validator.
//!
//! Captures all per-architecture variation (activation/normalization variants,
//! MoE routing specifics, attention/positional-encoding conventions, and phase
//! hardware routing) as declarative data so the inference hot path never branches
//! on model identity.

use crate::hardware::TunedExecutionPolicy;
use crate::tensor::{DType, MappedWeightArchive, TensorError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DescriptorError {
    #[error("Invalid architecture configuration: {0}")]
    InvalidConfig(String),
    #[error("Weight validation error at layer {layer_idx} ('{tensor_name}'): {source}")]
    WeightValidation {
        layer_idx: u32,
        tensor_name: String,
        #[source]
        source: TensorError,
    },
}

/// Normalization operator variants across Transformer families.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormVariant {
    /// Standard Root Mean Square Layer Normalization.
    RmsNorm {
        eps: f32,
        zero_centered_unit_offset: bool,
    },
    /// Standard LayerNorm with mean centering and optional bias.
    LayerNorm { eps: f32, has_bias: bool },
}

/// Pointwise / gated non-linear activation variants inside feed-forward blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationVariant {
    /// SiLU-gated linear unit (LLaMA, Qwen3).
    SwiGLU,
    /// GELU-gated linear unit (Gemma family).
    GeGLU,
    /// Standard Rectified Linear Unit (OPT / sparse ReLU models).
    ReLU,
}

/// Positional encoding conventions applied to Query and Key projections.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PositionalEncoding {
    /// Rotary Position Embedding (RoPE) with configurable base frequency and partial rotary factor.
    RoPE {
        theta: f32,
        rotary_dim: u32,
        interleaved: bool,
        scaling_factor: f32,
    },
    /// No positional encoding on this layer (e.g., NoPE or recurrent layers).
    None,
}

/// Attention mask and score customization conventions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AttentionMaskVariant {
    /// Standard causal lower-triangular mask.
    Causal,
    /// Sliding-window local causal attention (e.g., Gemma alternating layers).
    SlidingWindowCausal { window_size: u32 },
}

/// Sequence-mixing operator specification for a single Transformer/Hybrid block.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SequenceMixerDescriptor {
    /// Grouped-Query / Multi-Head / Multi-Query Attention.
    GroupedQueryAttention {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        pos_encoding: PositionalEncoding,
        mask_variant: AttentionMaskVariant,
        /// Optional Q and K per-head RMSNorm prior to RoPE (used in Qwen3 / Gemma).
        qk_norm: Option<NormVariant>,
        /// Whether Q/K/V projection vectors have learned bias terms (Qwen2).
        qkv_bias: bool,
        /// Optional attention logit soft-capping (used in Gemma variants).
        attn_logit_softcap: Option<f32>,
        /// Context threshold (in tokens) where decode switches to a split-K parallel layout.
        split_k_decode_threshold: u32,
    },
    /// Hybrid Linear / Gated DeltaNet Recurrent State Mixer (used in Qwen3.6-35B-A3B).
    GatedDeltaNetRecurrence {
        num_heads: u32,
        head_key_dim: u32,
        head_val_dim: u32,
        short_conv_kernel_size: u32,
    },
}

/// Channel-mixing (Feed-Forward / MoE) operator specification for a single block.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ChannelMixerDescriptor {
    /// Standard Dense MLP block.
    DenseMlp {
        intermediate_size: u32,
        activation: ActivationVariant,
        use_bias: bool,
    },
    /// Sparse Mixture-of-Experts (MoE) block with GPU-resident routing and optional shared expert.
    SparseMoE {
        num_total_experts: u32,
        num_active_experts: u32,
        expert_intermediate_size: u32,
        activation: ActivationVariant,
        normalize_topk_weights: bool,
        /// Optional always-active shared expert executed concurrently alongside routed experts.
        shared_expert_intermediate_size: Option<u32>,
    },
}

/// Phase-asymmetric hardware routing policy (BaseRT M5 / Lily principle).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseRoutingPolicy {
    /// Route compute-bound prefill GEMMs and QK^T/PV products to hardware Tensor/Matrix cores.
    pub use_tensor_cores_for_prefill: bool,
    /// Keep memory-bandwidth-bound M=1 decode GEMVs on specialized vector ALU kernels.
    pub use_vector_alu_for_decode: bool,
    /// Maximum prompt tokens per bounded prefill chunk to bound scratch activation memory.
    pub prefill_chunk_size: u32,
    /// Number of autoregressive decode steps amortized per CPU-GPU synchronization barrier.
    pub decode_amortize_steps: u32,
}

impl Default for PhaseRoutingPolicy {
    fn default() -> Self {
        Self {
            use_tensor_cores_for_prefill: true,
            use_vector_alu_for_decode: true,
            prefill_chunk_size: 1024,
            decode_amortize_steps: 4,
        }
    }
}

/// Complete declarative specification for a single model layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LayerDescriptor {
    pub layer_idx: u32,
    pub attn_norm: NormVariant,
    pub seq_mixer: SequenceMixerDescriptor,
    pub post_attn_norm: Option<NormVariant>,
    pub ffn_norm: NormVariant,
    pub channel_mixer: ChannelMixerDescriptor,
    pub post_ffn_norm: Option<NormVariant>,
}

/// Canonical weight tensor names for a single layer, resolved purely from `LayerDescriptor`.
#[derive(Debug, Clone)]
pub struct LayerWeightNames {
    pub attn_norm: String,
    pub seq_mixer_tensors: Vec<String>,
    pub ffn_norm: String,
    pub channel_mixer_tensors: Vec<String>,
}

impl LayerDescriptor {
    /// Deterministically resolves the canonical `.base` / `.safetensors` tensor keys for this layer.
    pub fn canonical_weight_names(&self) -> LayerWeightNames {
        let p = format!("model.layers.{}", self.layer_idx);
        let mut seq_mixer_tensors = Vec::new();

        match self.seq_mixer {
            SequenceMixerDescriptor::GroupedQueryAttention {
                qk_norm, qkv_bias, ..
            } => {
                seq_mixer_tensors.push(format!("{p}.self_attn.q_proj.weight"));
                seq_mixer_tensors.push(format!("{p}.self_attn.k_proj.weight"));
                seq_mixer_tensors.push(format!("{p}.self_attn.v_proj.weight"));
                seq_mixer_tensors.push(format!("{p}.self_attn.o_proj.weight"));
                if qkv_bias {
                    seq_mixer_tensors.push(format!("{p}.self_attn.q_proj.bias"));
                    seq_mixer_tensors.push(format!("{p}.self_attn.k_proj.bias"));
                    seq_mixer_tensors.push(format!("{p}.self_attn.v_proj.bias"));
                }
                if qk_norm.is_some() {
                    seq_mixer_tensors.push(format!("{p}.self_attn.q_norm.weight"));
                    seq_mixer_tensors.push(format!("{p}.self_attn.k_norm.weight"));
                }
            }
            SequenceMixerDescriptor::GatedDeltaNetRecurrence { .. } => {
                seq_mixer_tensors.push(format!("{p}.linear_attn.qkv_proj.weight"));
                seq_mixer_tensors.push(format!("{p}.linear_attn.conv1d.weight"));
                seq_mixer_tensors.push(format!("{p}.linear_attn.alpha_beta_proj.weight"));
                seq_mixer_tensors.push(format!("{p}.linear_attn.out_proj.weight"));
            }
        }

        let mut channel_mixer_tensors = Vec::new();
        match self.channel_mixer {
            ChannelMixerDescriptor::DenseMlp { .. } => {
                channel_mixer_tensors.push(format!("{p}.mlp.gate_proj.weight"));
                channel_mixer_tensors.push(format!("{p}.mlp.up_proj.weight"));
                channel_mixer_tensors.push(format!("{p}.mlp.down_proj.weight"));
            }
            ChannelMixerDescriptor::SparseMoE {
                shared_expert_intermediate_size,
                ..
            } => {
                channel_mixer_tensors.push(format!("{p}.mlp.gate.weight"));
                channel_mixer_tensors.push(format!("{p}.mlp.experts.gate_up_proj.weight"));
                channel_mixer_tensors.push(format!("{p}.mlp.experts.down_proj.weight"));
                if shared_expert_intermediate_size.is_some() {
                    channel_mixer_tensors.push(format!("{p}.mlp.shared_expert.gate_proj.weight"));
                    channel_mixer_tensors.push(format!("{p}.mlp.shared_expert.up_proj.weight"));
                    channel_mixer_tensors.push(format!("{p}.mlp.shared_expert.down_proj.weight"));
                    channel_mixer_tensors.push(format!("{p}.mlp.shared_expert_gate.weight"));
                }
            }
        }

        LayerWeightNames {
            attn_norm: format!("{p}.input_layernorm.weight"),
            seq_mixer_tensors,
            ffn_norm: format!("{p}.post_attention_layernorm.weight"),
            channel_mixer_tensors,
        }
    }
}

/// FlashInfer-style Inspector-Executor Plan generated on CPU once per batch/context
/// configuration and passed to the GPU command buffer as a constant metadata buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AttentionExecutionPlan {
    pub context_len: u32,
    pub page_block_size: u32, // e.g., 16 tokens per BSR page
    pub num_active_pages: u32,
    pub split_k_partitions: u32,
    pub gqa_group_size: u32, // num_q_heads / num_kv_heads
}

/// Top-level Data-Driven Architecture Descriptor consumed by the core runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelArchDescriptor {
    pub family: String,
    pub model_name: String,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub max_position_embeddings: u32,
    pub weight_dtype: DType,
    pub tie_word_embeddings: bool,
    pub final_norm: NormVariant,
    pub final_logit_softcap: Option<f32>,
    pub phase_policy: PhaseRoutingPolicy,
    pub layers: Vec<LayerDescriptor>,
}

impl ModelArchDescriptor {
    /// Validates structural invariants of the descriptor before allocating runtime buffers.
    pub fn validate(&self) -> Result<(), DescriptorError> {
        if self.vocab_size == 0 || self.hidden_size == 0 || self.layers.is_empty() {
            return Err(DescriptorError::InvalidConfig(
                "vocab_size, hidden_size, and layers must be non-zero".into(),
            ));
        }

        for (expected_idx, layer) in self.layers.iter().enumerate() {
            if layer.layer_idx != expected_idx as u32 {
                return Err(DescriptorError::InvalidConfig(format!(
                    "Non-contiguous layer_idx: expected {}, found {}",
                    expected_idx, layer.layer_idx
                )));
            }

            if let SequenceMixerDescriptor::GroupedQueryAttention {
                num_q_heads,
                num_kv_heads,
                head_dim,
                ..
            } = layer.seq_mixer
            {
                if num_kv_heads == 0 || num_q_heads % num_kv_heads != 0 {
                    return Err(DescriptorError::InvalidConfig(format!(
                        "Layer {}: num_q_heads ({}) must be a multiple of num_kv_heads ({})",
                        layer.layer_idx, num_q_heads, num_kv_heads
                    )));
                }
                if head_dim == 0 || head_dim % 2 != 0 {
                    return Err(DescriptorError::InvalidConfig(format!(
                        "Layer {}: head_dim ({}) must be non-zero and even",
                        layer.layer_idx, head_dim
                    )));
                }
            }

            if let ChannelMixerDescriptor::SparseMoE {
                num_total_experts,
                num_active_experts,
                ..
            } = layer.channel_mixer
            {
                if num_active_experts == 0 || num_active_experts > num_total_experts {
                    return Err(DescriptorError::InvalidConfig(format!(
                        "Layer {}: invalid MoE active experts ({}/{})",
                        layer.layer_idx, num_active_experts, num_total_experts
                    )));
                }
            }
        }
        Ok(())
    }

    /// Verifies that every tensor required by this descriptor exists in the `MappedWeightArchive`.
    pub fn verify_archive_completeness(
        &self,
        archive: &MappedWeightArchive,
    ) -> Result<(), DescriptorError> {
        archive
            .get_tensor("model.embed_tokens.weight")
            .map_err(|e| DescriptorError::WeightValidation {
                layer_idx: 0,
                tensor_name: "model.embed_tokens.weight".into(),
                source: e,
            })?;

        for layer in &self.layers {
            let names = layer.canonical_weight_names();
            let all_keys = std::iter::once(names.attn_norm.as_str())
                .chain(names.seq_mixer_tensors.iter().map(String::as_str))
                .chain(std::iter::once(names.ffn_norm.as_str()))
                .chain(names.channel_mixer_tensors.iter().map(String::as_str));

            for key in all_keys {
                archive
                    .get_tensor(key)
                    .map_err(|e| DescriptorError::WeightValidation {
                        layer_idx: layer.layer_idx,
                        tensor_name: key.to_string(),
                        source: e,
                    })?;
            }
        }

        if !self.tie_word_embeddings {
            archive.get_tensor("lm_head.weight").map_err(|e| {
                DescriptorError::WeightValidation {
                    layer_idx: self.layers.len() as u32,
                    tensor_name: "lm_head.weight".into(),
                    source: e,
                }
            })?;
        }
        Ok(())
    }

    /// Inspector step (`plan`): computes the load-balanced BSR PagedAttention schedule.
    pub fn plan_attention(
        &self,
        layer_id: usize,
        context_len: u32,
        policy: &TunedExecutionPolicy,
    ) -> Option<AttentionExecutionPlan> {
        match self.layers[layer_id].seq_mixer {
            SequenceMixerDescriptor::GroupedQueryAttention {
                num_q_heads,
                num_kv_heads,
                ..
            } => {
                let page_block_size = 16;
                let num_active_pages = context_len.div_ceil(page_block_size);
                let split_k_partitions = if context_len >= policy.gqa_split_k_threshold {
                    policy.gqa_split_k_partitions
                } else {
                    1
                };
                Some(AttentionExecutionPlan {
                    context_len,
                    page_block_size,
                    num_active_pages,
                    split_k_partitions,
                    gqa_group_size: num_q_heads / num_kv_heads,
                })
            }
            SequenceMixerDescriptor::GatedDeltaNetRecurrence { .. } => None,
        }
    }

    /// Returns indices of layers using standard KV-cached attention (vs. recurrent state).
    pub fn kv_cached_layer_indices(&self) -> Vec<u32> {
        self.layers
            .iter()
            .filter_map(|l| match l.seq_mixer {
                SequenceMixerDescriptor::GroupedQueryAttention { .. } => Some(l.layer_idx),
                SequenceMixerDescriptor::GatedDeltaNetRecurrence { .. } => None,
            })
            .collect()
    }

    /// Declarative profile for `Qwen3.6-35B-A3B` (40 layers: 30 Gated DeltaNet + 10 GQA, 256 routed + 1 shared MoE).
    pub fn qwen3_6_35b_a3b() -> Self {
        let rms = NormVariant::RmsNorm {
            eps: 1e-6,
            zero_centered_unit_offset: false,
        };
        let layers = (0..40)
            .map(|idx| {
                let seq_mixer = if (idx + 1) % 4 == 0 {
                    SequenceMixerDescriptor::GroupedQueryAttention {
                        num_q_heads: 32,
                        num_kv_heads: 4,
                        head_dim: 128,
                        pos_encoding: PositionalEncoding::RoPE {
                            theta: 1_000_000.0,
                            rotary_dim: 64,
                            interleaved: false,
                            scaling_factor: 1.0,
                        },
                        mask_variant: AttentionMaskVariant::Causal,
                        qk_norm: Some(rms),
                        qkv_bias: false,
                        attn_logit_softcap: None,
                        split_k_decode_threshold: 32_768,
                    }
                } else {
                    SequenceMixerDescriptor::GatedDeltaNetRecurrence {
                        num_heads: 32,
                        head_key_dim: 128,
                        head_val_dim: 128,
                        short_conv_kernel_size: 4,
                    }
                };

                LayerDescriptor {
                    layer_idx: idx,
                    attn_norm: rms,
                    seq_mixer,
                    post_attn_norm: None,
                    ffn_norm: rms,
                    channel_mixer: ChannelMixerDescriptor::SparseMoE {
                        num_total_experts: 256,
                        num_active_experts: 8,
                        expert_intermediate_size: 1024,
                        activation: ActivationVariant::SwiGLU,
                        normalize_topk_weights: true,
                        shared_expert_intermediate_size: Some(2048),
                    },
                    post_ffn_norm: None,
                }
            })
            .collect();

        Self {
            family: "qwen3_moe_hybrid".into(),
            model_name: "Qwen3.6-35B-A3B".into(),
            vocab_size: 151_936,
            hidden_size: 2560,
            max_position_embeddings: 131_072,
            weight_dtype: DType::Q4Group64,
            tie_word_embeddings: false,
            final_norm: rms,
            final_logit_softcap: None,
            phase_policy: PhaseRoutingPolicy::default(),
            layers,
        }
    }

    /// Declarative profile for `Llama-3.2-3B` (28 dense layers with GQA + SwiGLU).
    pub fn llama_3_2_3b() -> Self {
        let rms = NormVariant::RmsNorm {
            eps: 1e-5,
            zero_centered_unit_offset: false,
        };
        let layers = (0..28)
            .map(|idx| LayerDescriptor {
                layer_idx: idx,
                attn_norm: rms,
                seq_mixer: SequenceMixerDescriptor::GroupedQueryAttention {
                    num_q_heads: 24,
                    num_kv_heads: 8,
                    head_dim: 128,
                    pos_encoding: PositionalEncoding::RoPE {
                        theta: 500_000.0,
                        rotary_dim: 128,
                        interleaved: false,
                        scaling_factor: 1.0,
                    },
                    mask_variant: AttentionMaskVariant::Causal,
                    qk_norm: None,
                    qkv_bias: false,
                    attn_logit_softcap: None,
                    split_k_decode_threshold: 32_768,
                },
                post_attn_norm: None,
                ffn_norm: rms,
                channel_mixer: ChannelMixerDescriptor::DenseMlp {
                    intermediate_size: 8192,
                    activation: ActivationVariant::SwiGLU,
                    use_bias: false,
                },
                post_ffn_norm: None,
            })
            .collect();

        Self {
            family: "llama".into(),
            model_name: "Llama-3.2-3B".into(),
            vocab_size: 128_256,
            hidden_size: 3072,
            max_position_embeddings: 131_072,
            weight_dtype: DType::Q4Group64,
            tie_word_embeddings: true,
            final_norm: rms,
            final_logit_softcap: None,
            phase_policy: PhaseRoutingPolicy::default(),
            layers,
        }
    }

    /// Declarative profile for `Mistral-7B-Instruct-v0.3` (32 layers, GQA 32/8, Sliding Window 4096).
    pub fn mistral_7b_v0_3() -> Self {
        let rms = NormVariant::RmsNorm {
            eps: 1e-5,
            zero_centered_unit_offset: false,
        };
        let layers = (0..32)
            .map(|idx| LayerDescriptor {
                layer_idx: idx,
                attn_norm: rms,
                seq_mixer: SequenceMixerDescriptor::GroupedQueryAttention {
                    num_q_heads: 32,
                    num_kv_heads: 8,
                    head_dim: 128,
                    pos_encoding: PositionalEncoding::RoPE {
                        theta: 1_000_000.0,
                        rotary_dim: 128,
                        interleaved: false,
                        scaling_factor: 1.0,
                    },
                    mask_variant: AttentionMaskVariant::SlidingWindowCausal { window_size: 4096 },
                    qk_norm: None,
                    qkv_bias: false,
                    attn_logit_softcap: None,
                    split_k_decode_threshold: 32_768,
                },
                post_attn_norm: None,
                ffn_norm: rms,
                channel_mixer: ChannelMixerDescriptor::DenseMlp {
                    intermediate_size: 14_336,
                    activation: ActivationVariant::SwiGLU,
                    use_bias: false,
                },
                post_ffn_norm: None,
            })
            .collect();
        Self {
            family: "mistral".into(),
            model_name: "Mistral-7B-Instruct-v0.3".into(),
            vocab_size: 32_768,
            hidden_size: 4096,
            max_position_embeddings: 32_768,
            weight_dtype: DType::Q4Group64,
            tie_word_embeddings: false,
            final_norm: rms,
            final_logit_softcap: None,
            phase_policy: PhaseRoutingPolicy::default(),
            layers,
        }
    }

    /// Declarative profile for `Gemma-2-9B-It` (42 layers, GeGLU, Logit Softcap 50.0, Zero-Centered RMSNorm).
    pub fn gemma_2_9b() -> Self {
        let rms = NormVariant::RmsNorm {
            eps: 1e-6,
            zero_centered_unit_offset: true,
        };
        let layers = (0..42)
            .map(|idx| {
                let mask_variant = if idx % 2 == 0 {
                    AttentionMaskVariant::SlidingWindowCausal { window_size: 4096 }
                } else {
                    AttentionMaskVariant::Causal
                };
                LayerDescriptor {
                    layer_idx: idx,
                    attn_norm: rms,
                    seq_mixer: SequenceMixerDescriptor::GroupedQueryAttention {
                        num_q_heads: 16,
                        num_kv_heads: 8,
                        head_dim: 256,
                        pos_encoding: PositionalEncoding::RoPE {
                            theta: 10_000.0,
                            rotary_dim: 256,
                            interleaved: false,
                            scaling_factor: 1.0,
                        },
                        mask_variant,
                        qk_norm: None,
                        qkv_bias: false,
                        attn_logit_softcap: Some(50.0),
                        split_k_decode_threshold: 32_768,
                    },
                    post_attn_norm: Some(rms),
                    ffn_norm: rms,
                    channel_mixer: ChannelMixerDescriptor::DenseMlp {
                        intermediate_size: 14_336,
                        activation: ActivationVariant::GeGLU,
                        use_bias: false,
                    },
                    post_ffn_norm: Some(rms),
                }
            })
            .collect();
        Self {
            family: "gemma2".into(),
            model_name: "Gemma-2-9B-It".into(),
            vocab_size: 256_000,
            hidden_size: 3584,
            max_position_embeddings: 8192,
            weight_dtype: DType::Q4Group64,
            tie_word_embeddings: true,
            final_norm: rms,
            final_logit_softcap: Some(30.0),
            phase_policy: PhaseRoutingPolicy::default(),
            layers,
        }
    }

    /// Declarative profile for `DeepSeek-R1-Distill-Qwen-7B` (28 layers, GQA 28/4, RoPE theta 1M).
    pub fn deepseek_r1_distill_qwen_7b() -> Self {
        let rms = NormVariant::RmsNorm {
            eps: 1e-6,
            zero_centered_unit_offset: false,
        };
        let layers = (0..28)
            .map(|idx| LayerDescriptor {
                layer_idx: idx,
                attn_norm: rms,
                seq_mixer: SequenceMixerDescriptor::GroupedQueryAttention {
                    num_q_heads: 28,
                    num_kv_heads: 4,
                    head_dim: 128,
                    pos_encoding: PositionalEncoding::RoPE {
                        theta: 1_000_000.0,
                        rotary_dim: 128,
                        interleaved: false,
                        scaling_factor: 1.0,
                    },
                    mask_variant: AttentionMaskVariant::Causal,
                    qk_norm: None,
                    qkv_bias: false,
                    attn_logit_softcap: None,
                    split_k_decode_threshold: 32_768,
                },
                post_attn_norm: None,
                ffn_norm: rms,
                channel_mixer: ChannelMixerDescriptor::DenseMlp {
                    intermediate_size: 18_944,
                    activation: ActivationVariant::SwiGLU,
                    use_bias: false,
                },
                post_ffn_norm: None,
            })
            .collect();
        Self {
            family: "deepseek_qwen".into(),
            model_name: "DeepSeek-R1-Distill-Qwen-7B".into(),
            vocab_size: 152_064,
            hidden_size: 3584,
            max_position_embeddings: 131_072,
            weight_dtype: DType::Q4Group64,
            tie_word_embeddings: false,
            final_norm: rms,
            final_logit_softcap: None,
            phase_policy: PhaseRoutingPolicy::default(),
            layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_hybrid_descriptor_invariants() {
        let desc = ModelArchDescriptor::qwen3_6_35b_a3b();
        assert!(desc.validate().is_ok());

        // 10 full GQA layers (every 4th layer: indices 3, 7, 11, ..., 39) and 30 recurrent layers
        let kv_layers = desc.kv_cached_layer_indices();
        assert_eq!(kv_layers.len(), 10);
        assert_eq!(kv_layers[0], 3);
        assert_eq!(kv_layers[9], 39);

        // Verify canonical weight names for a recurrent MoE layer vs a GQA MoE layer
        let l0_names = desc.layers[0].canonical_weight_names();
        assert!(l0_names
            .seq_mixer_tensors
            .contains(&"model.layers.0.linear_attn.conv1d.weight".to_string()));

        let l3_names = desc.layers[3].canonical_weight_names();
        assert!(l3_names
            .seq_mixer_tensors
            .contains(&"model.layers.3.self_attn.q_norm.weight".to_string()));
    }

    #[test]
    fn test_llama_dense_descriptor_validation() {
        let desc = ModelArchDescriptor::llama_3_2_3b();
        assert!(desc.validate().is_ok());
        assert_eq!(desc.kv_cached_layer_indices().len(), 28);
    }

    #[test]
    fn test_attention_plan_generation() {
        let desc = ModelArchDescriptor::qwen3_6_35b_a3b();
        let policy = TunedExecutionPolicy {
            use_bf16_kernels: true,
            use_metal4_tensor_prefill: false,
            prefill_chunk_size: 1024,
            decode_amortize_steps: 4,
            gqa_split_k_threshold: 32_768,
            gqa_split_k_partitions: 8,
            gpu_hot_expert_ratio: 1.0,
            compress_kv_cache_q4: false,
        };

        // Layer 3 is GQA — should produce a plan
        let plan = desc.plan_attention(3, 4096, &policy);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.gqa_group_size, 8); // 32 / 4
        assert_eq!(plan.split_k_partitions, 1); // below threshold

        // Layer 0 is DeltaNet — should produce None
        let plan = desc.plan_attention(0, 4096, &policy);
        assert!(plan.is_none());

        // Above split-K threshold
        let plan = desc.plan_attention(3, 65_536, &policy).unwrap();
        assert_eq!(plan.split_k_partitions, 8);
    }
}
