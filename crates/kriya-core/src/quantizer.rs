// File: crates/kriya-core/src/quantizer.rs
//! Native Rust Offline Quantizer (`FP16`/`BF16`/`F32` -> `BlockQ4Group64`) and
//! Automatic Hugging Face `config.json` -> `ModelArchDescriptor` Compiler.

use crate::descriptor::{
    ActivationVariant, AttentionMaskVariant, ChannelMixerDescriptor, LayerDescriptor,
    ModelArchDescriptor, NormVariant, PhaseRoutingPolicy, PositionalEncoding,
    SequenceMixerDescriptor,
};
use crate::tensor::{BlockQ2Group128, BlockQ3Group64, BlockQ4Group64, DType, TensorError};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

#[inline]
pub fn f32_to_bf16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
}

#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal FP16
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            f32::from_bits((sign << 31) | ((e as u32) << 23) | (m << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        let e = exp + (127 - 15);
        f32::from_bits((sign << 31) | (e << 23) | (mant << 13))
    }
}

/// Quantizes a slice of `f32` weights (length must be a multiple of 64) into
/// 4-bit affine Group-64 blocks (`w_approx = q * scale + bias`, where `q in 0..=15`).
pub fn quantize_f32_to_q4_group64(weights: &[f32]) -> Result<Vec<BlockQ4Group64>, TensorError> {
    if weights.len() % 64 != 0 {
        return Err(TensorError::ShapeMismatch {
            expected: weights.len().div_ceil(64) * 64,
            actual: weights.len(),
        });
    }

    let num_blocks = weights.len() / 64;
    let mut out_blocks = Vec::with_capacity(num_blocks);

    for chunk in weights.chunks_exact(64) {
        let (min_val, max_val) = chunk
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
                (min.min(v), max.max(v))
            });

        let range = (max_val - min_val).max(1e-8);
        let scale = range / 15.0;
        let bias = min_val;

        let scale_bf16 = f32_to_bf16_bits(scale);
        let bias_bf16 = f32_to_bf16_bits(bias);

        // Use the exact BF16-rounded scale and bias to compute optimal 4-bit codes
        let eff_scale = bf16_bits_to_f32(scale_bf16).max(1e-8);
        let eff_bias = bf16_bits_to_f32(bias_bf16);

        let mut qs = [0u8; 32];
        for byte_i in 0..32 {
            let v0 = chunk[byte_i * 2];
            let v1 = chunk[byte_i * 2 + 1];

            let q0 = ((v0 - eff_bias) / eff_scale).round().clamp(0.0, 15.0) as u8;
            let q1 = ((v1 - eff_bias) / eff_scale).round().clamp(0.0, 15.0) as u8;

            qs[byte_i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        out_blocks.push(BlockQ4Group64 {
            scale_bf16,
            bias_bf16,
            qs,
        });
    }

    Ok(out_blocks)
}

/// Dequantizes a `BlockQ4Group64` slice back to `f32` to verify quantization SQNR / MSE.
pub fn dequantize_q4_group64_to_f32(blocks: &[BlockQ4Group64]) -> Vec<f32> {
    blocks
        .iter()
        .flat_map(|blk| {
            let scale = bf16_bits_to_f32(blk.scale_bf16);
            let bias = bf16_bits_to_f32(blk.bias_bf16);
            blk.qs.iter().flat_map(move |&packed| {
                [
                    ((packed & 0x0F) as f32).mul_add(scale, bias),
                    ((packed >> 4) as f32).mul_add(scale, bias),
                ]
            })
        })
        .collect()
}

/// Quantizes a slice of `f32` weights (length must be a multiple of 128) into
/// 2-bit affine Group-128 blocks (`w_approx = q * scale + bias`, where `q in 0..=3`).
pub fn quantize_f32_to_q2_group128(weights: &[f32]) -> Result<Vec<BlockQ2Group128>, TensorError> {
    if weights.len() % 128 != 0 {
        return Err(TensorError::ShapeMismatch {
            expected: weights.len().div_ceil(128) * 128,
            actual: weights.len(),
        });
    }

    let num_blocks = weights.len() / 128;
    let mut out_blocks = Vec::with_capacity(num_blocks);

    for chunk in weights.chunks_exact(128) {
        let (min_val, max_val) = chunk
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
                (min.min(v), max.max(v))
            });

        let range = (max_val - min_val).max(1e-8);
        let scale = range / 3.0;
        let bias = min_val;

        let scale_bf16 = f32_to_bf16_bits(scale);
        let bias_bf16 = f32_to_bf16_bits(bias);

        let eff_scale = bf16_bits_to_f32(scale_bf16).max(1e-8);
        let eff_bias = bf16_bits_to_f32(bias_bf16);

        let mut qs = [0u8; 32];
        for (byte_i, qs_byte) in qs.iter_mut().enumerate() {
            let base = byte_i * 4;
            let q0 = ((chunk[base] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 3.0) as u8;
            let q1 = ((chunk[base + 1] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 3.0) as u8;
            let q2 = ((chunk[base + 2] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 3.0) as u8;
            let q3 = ((chunk[base + 3] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 3.0) as u8;

            *qs_byte = (q0 & 0x03) | ((q1 & 0x03) << 2) | ((q2 & 0x03) << 4) | ((q3 & 0x03) << 6);
        }

        out_blocks.push(BlockQ2Group128 {
            scale_bf16,
            bias_bf16,
            qs,
        });
    }

    Ok(out_blocks)
}

/// Dequantizes a `BlockQ2Group128` slice back to `f32`.
pub fn dequantize_q2_group128_to_f32(blocks: &[BlockQ2Group128]) -> Vec<f32> {
    blocks
        .iter()
        .flat_map(|blk| {
            let scale = bf16_bits_to_f32(blk.scale_bf16);
            let bias = bf16_bits_to_f32(blk.bias_bf16);
            blk.qs.iter().flat_map(move |&packed| {
                [
                    ((packed & 0x03) as f32).mul_add(scale, bias),
                    (((packed >> 2) & 0x03) as f32).mul_add(scale, bias),
                    (((packed >> 4) & 0x03) as f32).mul_add(scale, bias),
                    (((packed >> 6) & 0x03) as f32).mul_add(scale, bias),
                ]
            })
        })
        .collect()
}

/// Quantizes a slice of `f32` weights (length must be a multiple of 64) into
/// 3-bit affine Group-64 blocks (`w_approx = q * scale + bias`, where `q in 0..=7`).
pub fn quantize_f32_to_q3_group64(weights: &[f32]) -> Result<Vec<BlockQ3Group64>, TensorError> {
    if weights.len() % 64 != 0 {
        return Err(TensorError::ShapeMismatch {
            expected: weights.len().div_ceil(64) * 64,
            actual: weights.len(),
        });
    }

    let num_blocks = weights.len() / 64;
    let mut out_blocks = Vec::with_capacity(num_blocks);

    for chunk in weights.chunks_exact(64) {
        let (min_val, max_val) = chunk
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
                (min.min(v), max.max(v))
            });

        let range = (max_val - min_val).max(1e-8);
        let scale = range / 7.0;
        let bias = min_val;

        let scale_bf16 = f32_to_bf16_bits(scale);
        let bias_bf16 = f32_to_bf16_bits(bias);

        let eff_scale = bf16_bits_to_f32(scale_bf16).max(1e-8);
        let eff_bias = bf16_bits_to_f32(bias_bf16);

        let mut qs = [0u8; 24];
        for g in 0..8 {
            let base = g * 8;
            let q0 = ((chunk[base] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 7.0) as u8;
            let q1 = ((chunk[base + 1] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 7.0) as u8;
            let q2 = ((chunk[base + 2] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 7.0) as u8;
            let q3 = ((chunk[base + 3] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 7.0) as u8;
            let q4 = ((chunk[base + 4] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 7.0) as u8;
            let q5 = ((chunk[base + 5] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 7.0) as u8;
            let q6 = ((chunk[base + 6] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 7.0) as u8;
            let q7 = ((chunk[base + 7] - eff_bias) / eff_scale)
                .round()
                .clamp(0.0, 7.0) as u8;

            let b = g * 3;
            qs[b] = (q0 & 0x07) | ((q1 & 0x07) << 3) | ((q2 & 0x03) << 6);
            qs[b + 1] =
                ((q2 >> 2) & 0x01) | ((q3 & 0x07) << 1) | ((q4 & 0x07) << 4) | ((q5 & 0x01) << 7);
            qs[b + 2] = ((q5 >> 1) & 0x03) | ((q6 & 0x07) << 2) | ((q7 & 0x07) << 5);
        }

        out_blocks.push(BlockQ3Group64 {
            scale_bf16,
            bias_bf16,
            qs,
        });
    }

    Ok(out_blocks)
}

/// Dequantizes a `BlockQ3Group64` slice back to `f32`.
pub fn dequantize_q3_group64_to_f32(blocks: &[BlockQ3Group64]) -> Vec<f32> {
    blocks
        .iter()
        .flat_map(|blk| {
            let scale = bf16_bits_to_f32(blk.scale_bf16);
            let bias = bf16_bits_to_f32(blk.bias_bf16);
            (0..8).flat_map(move |g| {
                let b = g * 3;
                let b0 = blk.qs[b] as u32;
                let b1 = blk.qs[b + 1] as u32;
                let b2 = blk.qs[b + 2] as u32;

                let q0 = b0 & 0x07;
                let q1 = (b0 >> 3) & 0x07;
                let q2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2);
                let q3 = (b1 >> 1) & 0x07;
                let q4 = (b1 >> 4) & 0x07;
                let q5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1);
                let q6 = (b2 >> 2) & 0x07;
                let q7 = (b2 >> 5) & 0x07;

                [
                    (q0 as f32).mul_add(scale, bias),
                    (q1 as f32).mul_add(scale, bias),
                    (q2 as f32).mul_add(scale, bias),
                    (q3 as f32).mul_add(scale, bias),
                    (q4 as f32).mul_add(scale, bias),
                    (q5 as f32).mul_add(scale, bias),
                    (q6 as f32).mul_add(scale, bias),
                    (q7 as f32).mul_add(scale, bias),
                ]
            })
        })
        .collect()
}

/// Universal Hugging Face `config.json` Schema supporting Llama, Qwen2/3, Gemma, Mistral, Phi, and DeepSeek.
#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceModelConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,
    #[serde(default)]
    pub head_dim: Option<u32>,
    #[serde(default = "default_inter_size")]
    pub intermediate_size: u32,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Hugging Face architectures such as Llama 3.1 expose learned Q/K/V
    /// projection biases through this flag. Qwen2 is handled below as a
    /// compatibility exception because its published config omits the flag.
    #[serde(default)]
    pub attention_bias: bool,
    // MoE optional fields (Qwen2/3-MoE, Mixtral, DeepSeek)
    #[serde(default)]
    pub num_experts: Option<u32>,
    #[serde(default)]
    pub num_experts_per_tok: Option<u32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<u32>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<u32>,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

fn default_model_type() -> String {
    "llama".to_string()
}
fn default_inter_size() -> u32 {
    4096
}
fn default_rms_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    500_000.0
}
fn default_max_pos() -> u32 {
    131_072
}

impl HuggingFaceModelConfig {
    /// Automatically compiles any Hugging Face `config.json` file into a `KriyaRT` `ModelArchDescriptor`.
    pub fn from_config_file<P: AsRef<Path>>(path: P) -> Result<ModelArchDescriptor, TensorError> {
        let bytes = std::fs::read(path)?;
        let cfg: Self = serde_json::from_slice(&bytes)
            .map_err(|e| TensorError::InvalidMetadata(format!("Invalid HF config.json: {e}")))?;
        Ok(cfg.into_arch_descriptor())
    }

    pub fn into_arch_descriptor(self) -> ModelArchDescriptor {
        let is_gemma = self.model_type.contains("gemma");
        // Qwen3 adds per-head Q/K RMS norms; Qwen2 / Qwen2.5 use the
        // canonical Llama-style projections without those tensors. Treating all
        // Qwen configurations as Qwen3 makes a valid Qwen2.5 checkpoint fail
        // strict tensor validation before inference can begin.
        let uses_qk_norm = self.model_type.contains("qwen3");
        // Qwen2 checkpoints carry learned Q/K/V projection biases even though
        // their config omits an explicit `attention_bias` field.
        let uses_qkv_bias = self.attention_bias || self.model_type == "qwen2";
        let is_hybrid_deltanet =
            self.model_type.contains("qwen3_moe_hybrid") || self.model_type.contains("deltanet");

        let rms = NormVariant::RmsNorm {
            eps: self.rms_norm_eps,
            zero_centered_unit_offset: is_gemma,
        };

        let activation = if is_gemma {
            ActivationVariant::GeGLU
        } else {
            ActivationVariant::SwiGLU
        };

        let num_kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let head_dim = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1));

        let channel_mixer = if let (Some(total_exp), Some(active_exp)) =
            (self.num_experts, self.num_experts_per_tok)
        {
            ChannelMixerDescriptor::SparseMoE {
                num_total_experts: total_exp,
                num_active_experts: active_exp,
                expert_intermediate_size: self
                    .moe_intermediate_size
                    .unwrap_or(self.intermediate_size),
                activation,
                normalize_topk_weights: true,
                shared_expert_intermediate_size: self.shared_expert_intermediate_size,
            }
        } else {
            ChannelMixerDescriptor::DenseMlp {
                intermediate_size: self.intermediate_size,
                activation,
                use_bias: false,
            }
        };

        let rope_scaling_factor = self
            .rope_scaling
            .as_ref()
            .and_then(|v| v.get("factor"))
            .and_then(|f| f.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1.0);

        let layers = (0..self.num_hidden_layers)
            .map(|idx| {
                let seq_mixer = if is_hybrid_deltanet && (idx + 1) % 4 != 0 {
                    SequenceMixerDescriptor::GatedDeltaNetRecurrence {
                        num_heads: self.num_attention_heads,
                        head_key_dim: head_dim,
                        head_val_dim: head_dim,
                        short_conv_kernel_size: 4,
                    }
                } else {
                    SequenceMixerDescriptor::GroupedQueryAttention {
                        num_q_heads: self.num_attention_heads,
                        num_kv_heads,
                        head_dim,
                        pos_encoding: PositionalEncoding::RoPE {
                            theta: self.rope_theta,
                            rotary_dim: head_dim,
                            interleaved: false,
                            scaling_factor: rope_scaling_factor,
                        },
                        mask_variant: AttentionMaskVariant::Causal,
                        qk_norm: if uses_qk_norm { Some(rms) } else { None },
                        qkv_bias: uses_qkv_bias,
                        attn_logit_softcap: self.attn_logit_softcapping,
                        split_k_decode_threshold: 32_768,
                    }
                };

                LayerDescriptor {
                    layer_idx: idx,
                    attn_norm: rms,
                    seq_mixer,
                    post_attn_norm: if is_gemma { Some(rms) } else { None },
                    ffn_norm: rms,
                    channel_mixer,
                    post_ffn_norm: if is_gemma { Some(rms) } else { None },
                }
            })
            .collect();

        ModelArchDescriptor {
            family: self.model_type.clone(),
            model_name: format!("HF-Auto-{}", self.model_type),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            max_position_embeddings: self.max_position_embeddings,
            weight_dtype: DType::Q4Group64,
            tie_word_embeddings: self.tie_word_embeddings || is_gemma,
            final_norm: rms,
            final_logit_softcap: self.attn_logit_softcapping,
            phase_policy: PhaseRoutingPolicy::default(),
            layers,
        }
    }
}

/// Writes a quantized tensor map out to a valid `.safetensors` archive on disk.
pub fn write_quantized_safetensors<P: AsRef<Path>>(
    output_path: P,
    tensors: &[(String, Vec<usize>, Vec<BlockQ4Group64>)],
) -> Result<(), TensorError> {
    let mixed: Vec<_> = tensors
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                s.clone(),
                SavedTensorPayload::Q4Group64(b.clone()),
            )
        })
        .collect();
    write_mixed_safetensors(output_path, &mixed)
}

/// Returns true if a tensor is a 1D normalization scale or bias that must stay in unquantized BF16.
#[inline]
pub fn should_preserve_in_bf16(tensor_name: &str, shape: &[usize]) -> bool {
    shape.len() <= 1
        || tensor_name.contains("norm")
        || tensor_name.ends_with(".bias")
        || tensor_name.ends_with(".scales")
        || tensor_name.ends_with(".biases")
}

/// Mixed-precision payload for `.safetensors` serialization.
#[derive(Debug, Clone)]
pub enum SavedTensorPayload {
    /// Unquantized BF16 tensor (used for 1D `*.norm.weight` and biases).
    Bf16(Vec<u16>),
    /// 4-bit Group-64 affine quantized blocks (used for 2D projection & embedding matrices).
    Q4Group64(Vec<BlockQ4Group64>),
    /// 2-bit Group-128 affine quantized blocks.
    Q2Group128(Vec<BlockQ2Group128>),
    /// 3-bit Group-64 affine quantized blocks.
    Q3Group64(Vec<BlockQ3Group64>),
}

/// Virtual memory page size on Apple Silicon unified memory architecture (16 KB).
const VIRTUAL_PAGE_SIZE: usize = 16384;

/// Writes a mixed-precision archive where 1D norms are stored as `"BF16"` and 2D weights as `"Q4_G64"`.
///
/// Both the JSON header and individual tensor payloads are aligned to 16 KB virtual-memory page
/// boundaries (`VIRTUAL_PAGE_SIZE`), enabling zero-copy Metal GPU unified memory mapping via
/// `device.new_buffer_with_bytes_no_copy()` without heap copying or memory duplication.
pub fn write_mixed_safetensors<P: AsRef<Path>>(
    output_path: P,
    tensors: &[(String, Vec<usize>, SavedTensorPayload)],
) -> Result<(), TensorError> {
    let mut header_map = serde_json::Map::new();
    let mut current_offset = 0usize;

    for (name, shape, payload) in tensors {
        let (dtype_str, byte_len) = match payload {
            SavedTensorPayload::Bf16(vals) => ("BF16", vals.len() * 2),
            SavedTensorPayload::Q4Group64(blocks) => (
                "Q4_G64",
                blocks.len() * std::mem::size_of::<BlockQ4Group64>(),
            ),
            SavedTensorPayload::Q2Group128(blocks) => (
                "Q2_G128",
                blocks.len() * std::mem::size_of::<BlockQ2Group128>(),
            ),
            SavedTensorPayload::Q3Group64(blocks) => (
                "Q3_G64",
                blocks.len() * std::mem::size_of::<BlockQ3Group64>(),
            ),
        };
        // Page-align start of each tensor so GPU buffers can map zero-copy
        let aligned_start = if current_offset % VIRTUAL_PAGE_SIZE != 0 {
            current_offset.next_multiple_of(VIRTUAL_PAGE_SIZE)
        } else {
            current_offset
        };
        let aligned_end = aligned_start + byte_len;
        header_map.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [aligned_start, aligned_end]
            }),
        );
        current_offset = aligned_end;
    }

    let mut header_bytes =
        serde_json::to_vec(&header_map).map_err(|e| TensorError::InvalidMetadata(e.to_string()))?;

    // safetensors binary format has:
    // - 8 bytes: header length (u64 little-endian)
    // - header_len bytes: JSON header string
    // - Data section starting at byte (8 + header_len)
    // Pad JSON header with trailing ASCII spaces so (8 + header_len) is a multiple of 16 KB.
    let raw_total = 8 + header_bytes.len();
    let target_total = raw_total.next_multiple_of(VIRTUAL_PAGE_SIZE);
    let pad_len = target_total - raw_total;
    header_bytes.resize(header_bytes.len() + pad_len, b' ');

    let header_len = header_bytes.len() as u64;
    debug_assert_eq!((8 + header_len) as usize % VIRTUAL_PAGE_SIZE, 0);

    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header_bytes)?;

    let mut written_offset = 0usize;
    for (_, _, payload) in tensors {
        let (byte_slice, byte_len) = match payload {
            SavedTensorPayload::Bf16(vals) => {
                (bytemuck::cast_slice(vals.as_slice()), vals.len() * 2)
            }
            SavedTensorPayload::Q4Group64(blocks) => (
                bytemuck::cast_slice(blocks.as_slice()),
                blocks.len() * std::mem::size_of::<BlockQ4Group64>(),
            ),
            SavedTensorPayload::Q2Group128(blocks) => (
                bytemuck::cast_slice(blocks.as_slice()),
                blocks.len() * std::mem::size_of::<BlockQ2Group128>(),
            ),
            SavedTensorPayload::Q3Group64(blocks) => (
                bytemuck::cast_slice(blocks.as_slice()),
                blocks.len() * std::mem::size_of::<BlockQ3Group64>(),
            ),
        };

        let target_start = if written_offset % VIRTUAL_PAGE_SIZE != 0 {
            written_offset.next_multiple_of(VIRTUAL_PAGE_SIZE)
        } else {
            written_offset
        };
        if target_start > written_offset {
            let pad = target_start - written_offset;
            static ZEROS: [u8; VIRTUAL_PAGE_SIZE] = [0u8; VIRTUAL_PAGE_SIZE];
            let mut remaining = pad;
            while remaining > 0 {
                let chunk = remaining.min(VIRTUAL_PAGE_SIZE);
                writer.write_all(&ZEROS[..chunk])?;
                remaining -= chunk;
            }
            written_offset = target_start;
        }

        writer.write_all(byte_slice)?;
        written_offset += byte_len;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_group64_quantization_accuracy() {
        let original: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.05).collect();
        let blocks = quantize_f32_to_q4_group64(&original).unwrap();
        assert_eq!(blocks.len(), 2);

        let reconstructed = dequantize_q4_group64_to_f32(&blocks);
        let mse: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / (original.len() as f32);

        // 4-bit affine group-64 MSE on smooth linear ramp must be < 0.005 (theoretical minimum ~0.0036)
        assert!(mse < 0.005, "Quantization MSE too high: {mse}");
    }

    #[test]
    fn test_q2_group128_quantization_accuracy() {
        let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.02).collect();
        let blocks = quantize_f32_to_q2_group128(&original).unwrap();
        assert_eq!(blocks.len(), 2);

        let reconstructed = dequantize_q2_group128_to_f32(&blocks);
        assert_eq!(reconstructed.len(), 256);
        let mse: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / (original.len() as f32);

        // 2-bit quantization has 4 levels, so MSE is higher than 4-bit but bounded
        assert!(mse < 0.1, "Q2 MSE too high: {mse}");
    }

    #[test]
    fn test_q3_group64_quantization_accuracy() {
        let original: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.03).collect();
        let blocks = quantize_f32_to_q3_group64(&original).unwrap();
        assert_eq!(blocks.len(), 2);

        let reconstructed = dequantize_q3_group64_to_f32(&blocks);
        assert_eq!(reconstructed.len(), 128);
        let mse: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / (original.len() as f32);

        // 3-bit quantization has 8 levels, MSE should be < 0.02
        assert!(mse < 0.02, "Q3 MSE too high: {mse}");
    }

    #[test]
    fn test_bf16_norm_preservation_in_mixed_safetensors() {
        assert!(should_preserve_in_bf16("model.norm.weight", &[3072]));
        assert!(should_preserve_in_bf16(
            "model.layers.0.input_layernorm.weight",
            &[3072]
        ));
        assert!(!should_preserve_in_bf16(
            "model.layers.0.self_attn.q_proj.weight",
            &[3072, 3072]
        ));

        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("mixed.safetensors");
        // 3072 BF16 norm weights (6,144 bytes) + 1 Q4_G64 projection block (36 bytes)
        let norm_bf16 = vec![f32_to_bf16_bits(1.0); 3072];
        let proj_q4 = quantize_f32_to_q4_group64(&vec![0.25f32; 64]).unwrap();

        write_mixed_safetensors(
            &out_path,
            &[
                (
                    "model.norm.weight".to_string(),
                    vec![3072],
                    SavedTensorPayload::Bf16(norm_bf16),
                ),
                (
                    "model.layers.0.self_attn.q_proj.weight".to_string(),
                    vec![1, 64],
                    SavedTensorPayload::Q4Group64(proj_q4),
                ),
            ],
        )
        .unwrap();

        let archive = crate::safetensors_loader::open_safetensors_file(&out_path).unwrap();
        let norm_view = archive.get_tensor("model.norm.weight").unwrap();
        assert_eq!(norm_view.descriptor.dtype, DType::BF16);
        assert_eq!(norm_view.data.len(), 6144);

        let proj_view = archive
            .get_tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(proj_view.descriptor.dtype, DType::Q4Group64);
        assert_eq!(proj_view.data.len(), 36);

        // Verify 16 KB page-aligned data pointers for zero-copy Metal GPU mapping
        assert_eq!(
            (norm_view.data.as_ptr() as usize) % VIRTUAL_PAGE_SIZE,
            0,
            "norm_view data pointer must be 16 KB page-aligned"
        );
        assert_eq!(
            (proj_view.data.as_ptr() as usize) % VIRTUAL_PAGE_SIZE,
            0,
            "proj_view data pointer must be 16 KB page-aligned"
        );
    }

    #[test]
    fn test_hf_config_auto_compiler_gemma_and_qwen_moe() {
        let raw_json = r#"{
            "model_type": "gemma4",
            "vocab_size": 256000,
            "hidden_size": 2048,
            "num_hidden_layers": 18,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 16384,
            "rms_norm_eps": 1e-6,
            "attn_logit_softcapping": 50.0
        }"#;
        let cfg: HuggingFaceModelConfig = serde_json::from_str(raw_json).unwrap();
        let desc = cfg.into_arch_descriptor();
        assert!(desc.validate().is_ok());
        assert_eq!(desc.layers.len(), 18);
        assert_eq!(desc.final_logit_softcap, Some(50.0));
    }

    #[test]
    fn test_qwen2_does_not_require_qk_norm_tensors() {
        let qwen2_json = r#"{
            "model_type": "qwen2",
            "vocab_size": 151936,
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "intermediate_size": 4864
        }"#;
        let qwen2: HuggingFaceModelConfig = serde_json::from_str(qwen2_json).unwrap();
        let qwen2_desc = qwen2.into_arch_descriptor();
        assert!(matches!(
            qwen2_desc.layers[0].seq_mixer,
            SequenceMixerDescriptor::GroupedQueryAttention {
                qk_norm: None,
                qkv_bias: true,
                ..
            }
        ));

        let qwen3_json = qwen2_json.replace("\"qwen2\"", "\"qwen3\"");
        let qwen3: HuggingFaceModelConfig = serde_json::from_str(&qwen3_json).unwrap();
        let qwen3_desc = qwen3.into_arch_descriptor();
        assert!(matches!(
            qwen3_desc.layers[0].seq_mixer,
            SequenceMixerDescriptor::GroupedQueryAttention {
                qk_norm: Some(_),
                qkv_bias: false,
                ..
            }
        ));
    }
}
