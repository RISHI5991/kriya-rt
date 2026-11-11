//! Real binary `.safetensors` header parser and sharded checkpoint loader.
//!
//! Supports both single-file and multi-shard (indexed) safetensors checkpoints.
//! Provides zero-copy memory-mapped loading via [`MappedWeightArchive`] and
//! MLX 4-bit affine Q4 group-64 repacking into [`BlockQ4Group64`].

use crate::tensor::{
    BlockQ4Group64, DType, MappedWeightArchive, PlacementUnit, TensorDescriptor, TensorError,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Maximum safetensors header size we accept (100 MB).
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// A single entry parsed from a safetensors JSON header.
#[derive(Debug, Clone, Deserialize)]
pub struct RawSafetensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Index JSON structure for sharded safetensors checkpoints
/// (e.g. `model.safetensors.index.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct SafetensorsIndexJson {
    pub weight_map: HashMap<String, String>,
}

impl DType {
    /// Converts a safetensors dtype string (e.g. `"F32"`, `"BF16"`) to our internal [`DType`].
    pub fn from_safetensors_str(s: &str) -> Result<Self, TensorError> {
        match s {
            "F32" => Ok(DType::F32),
            "F16" => Ok(DType::F16),
            "BF16" => Ok(DType::BF16),
            "F8_E4M3" => Ok(DType::FP8E4M3),
            "Q4_G64" => Ok(DType::Q4Group64),
            "U32" => Ok(DType::U32),
            "U8" => Ok(DType::U8),
            "I32" => Ok(DType::I32),
            "I8" => Ok(DType::I8),
            other => Err(TensorError::InvalidMetadata(format!(
                "Unsupported safetensors dtype: '{other}'"
            ))),
        }
    }
}

/// Opens a single `.safetensors` file, parses the binary header, and returns
/// a zero-copy [`MappedWeightArchive`].
///
/// # File format
/// - Bytes 0..8: little-endian u64 `header_byte_len`
/// - Bytes 8..8+header_byte_len: UTF-8 JSON header mapping tensor names to
///   `{ dtype, shape, data_offsets }`.
/// - Bytes 8+header_byte_len..: raw tensor data section.
///
/// The `__metadata__` key is skipped.
pub fn open_safetensors_file<P: AsRef<Path>>(path: P) -> Result<MappedWeightArchive, TensorError> {
    let path = path.as_ref();
    let mut file = fs::File::open(path)?;

    // Read 8-byte LE u64 header length.
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_byte_len = u64::from_le_bytes(len_buf);

    if header_byte_len == 0 || header_byte_len > MAX_HEADER_BYTES {
        return Err(TensorError::InvalidMetadata(format!(
            "Invalid safetensors header_byte_len: {header_byte_len} (must be 1..{MAX_HEADER_BYTES})"
        )));
    }

    // Read JSON header.
    let mut header_buf = vec![0u8; header_byte_len as usize];
    file.read_exact(&mut header_buf)?;

    let header_map: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)
        .map_err(|e| {
            TensorError::InvalidMetadata(format!("Safetensors header JSON parse error: {e}"))
        })?;

    let data_section_offset = 8 + header_byte_len as usize;

    let mut descriptors = Vec::new();
    for (name, value) in &header_map {
        // Skip __metadata__ key.
        if name == "__metadata__" {
            continue;
        }

        let entry: RawSafetensorEntry = serde_json::from_value(value.clone()).map_err(|e| {
            TensorError::InvalidMetadata(format!("Failed to parse safetensors entry '{name}': {e}"))
        })?;

        let dtype = DType::from_safetensors_str(&entry.dtype)?;
        let byte_offset = entry.data_offsets[0];
        let byte_len = entry.data_offsets[1] - entry.data_offsets[0];

        descriptors.push(TensorDescriptor {
            name: name.clone(),
            dtype,
            shape: entry.shape,
            byte_offset,
            byte_len,
            placement: PlacementUnit::GpuHot,
        });
    }

    MappedWeightArchive::open(path, data_section_offset, descriptors)
}

/// Handles both single-file and sharded (indexed) safetensors checkpoints.
pub struct ShardedCheckpointLoader {
    /// Root directory of the checkpoint.
    pub checkpoint_dir: PathBuf,
    /// Shard archives keyed by shard filename.
    pub shards: HashMap<String, MappedWeightArchive>,
    /// Maps tensor name → shard filename.
    pub tensor_to_shard: HashMap<String, String>,
}

impl ShardedCheckpointLoader {
    /// Opens a checkpoint directory. Detects sharded vs single-file layout automatically.
    ///
    /// If `model.safetensors.index.json` exists, it is treated as a sharded checkpoint;
    /// otherwise, `model.safetensors` is opened as a single file.
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self, TensorError> {
        let dir = dir.as_ref().to_path_buf();
        let index_path = dir.join("model.safetensors.index.json");

        if index_path.exists() {
            Self::open_sharded(&dir, &index_path)
        } else {
            Self::open_single(&dir)
        }
    }

    /// Alias for `open` matching File #12 spec.
    pub fn open_directory<P: AsRef<Path>>(dir: P) -> Result<Self, TensorError> {
        Self::open(dir)
    }

    fn open_single(dir: &Path) -> Result<Self, TensorError> {
        let model_path = dir.join("model.safetensors");
        let archive = open_safetensors_file(&model_path)?;

        let shard_name = "model.safetensors".to_string();
        let tensor_to_shard: HashMap<String, String> = archive
            .tensor_names()
            .map(|n| (n.to_string(), shard_name.clone()))
            .collect();

        let mut shards = HashMap::new();
        shards.insert(shard_name, archive);

        Ok(Self {
            checkpoint_dir: dir.to_path_buf(),
            shards,
            tensor_to_shard,
        })
    }

    fn open_sharded(dir: &Path, index_path: &Path) -> Result<Self, TensorError> {
        let index_bytes = fs::read(index_path)?;
        let index: SafetensorsIndexJson = serde_json::from_slice(&index_bytes).map_err(|e| {
            TensorError::InvalidMetadata(format!("Failed to parse index JSON: {e}"))
        })?;

        // Collect unique shard filenames.
        let unique_shards: std::collections::HashSet<&String> = index.weight_map.values().collect();

        let mut shards = HashMap::new();
        for shard_filename in unique_shards {
            let shard_path = dir.join(shard_filename);
            let archive = open_safetensors_file(&shard_path)?;
            shards.insert(shard_filename.clone(), archive);
        }

        Ok(Self {
            checkpoint_dir: dir.to_path_buf(),
            tensor_to_shard: index.weight_map,
            shards,
        })
    }

    /// Retrieves a tensor view from the correct shard by name.
    pub fn get_tensor(&self, name: &str) -> Result<crate::tensor::TensorView<'_>, TensorError> {
        let shard_name = self
            .tensor_to_shard
            .get(name)
            .ok_or_else(|| TensorError::TensorNotFound(name.to_string()))?;
        let archive = self
            .shards
            .get(shard_name)
            .ok_or_else(|| TensorError::TensorNotFound(name.to_string()))?;
        archive.get_tensor(name)
    }

    /// Retrieves a zero-copy tensor slice from the appropriate memory-mapped shard.
    pub fn get_tensor_bytes(&self, name: &str) -> Result<&[u8], TensorError> {
        let shard_name = self
            .tensor_to_shard
            .get(name)
            .ok_or_else(|| TensorError::TensorNotFound(name.to_string()))?;
        let archive = self.shards.get(shard_name).unwrap();
        Ok(archive.get_tensor(name)?.data)
    }

    /// Packs MLX-LM 4-bit affine triplets (`<prefix>.weight`, `<prefix>.scales`, `<prefix>.biases`)
    /// into contiguous 36-byte `BlockQ4Group64` blocks ready for `KriyaRT` Metal kernels.
    /// Handles both IEEE 754 float16 (F16) and bfloat16 (BF16) scale/bias tensor formats.
    pub fn pack_mlx_affine_q4_group64(
        &self,
        prefix: &str,
    ) -> Result<Vec<BlockQ4Group64>, TensorError> {
        let s_name = format!("{prefix}.scales");
        let s_shard = self
            .tensor_to_shard
            .get(&s_name)
            .ok_or_else(|| TensorError::TensorNotFound(s_name.clone()))?;
        let s_view = self.shards.get(s_shard).unwrap().get_tensor(&s_name)?;
        let s_is_f16 = s_view.descriptor.dtype == DType::F16;
        let s_bytes = s_view.data;

        let b_name = format!("{prefix}.biases");
        let b_shard = self
            .tensor_to_shard
            .get(&b_name)
            .ok_or_else(|| TensorError::TensorNotFound(b_name.clone()))?;
        let b_view = self.shards.get(b_shard).unwrap().get_tensor(&b_name)?;
        let b_is_f16 = b_view.descriptor.dtype == DType::F16;
        let b_bytes = b_view.data;

        let w_bytes = self.get_tensor_bytes(&format!("{prefix}.weight"))?;

        if w_bytes.len() % 32 != 0 {
            return Err(TensorError::InvalidMetadata(format!(
                "Weight tensor '{prefix}.weight' byte length {} is not a multiple of 32",
                w_bytes.len()
            )));
        }

        let num_blocks = w_bytes.len() / 32;
        if s_bytes.len() != num_blocks * 2 || b_bytes.len() != num_blocks * 2 {
            return Err(TensorError::ShapeMismatch {
                expected: num_blocks * 2,
                actual: s_bytes.len(),
            });
        }

        let packed_blocks = (0..num_blocks)
            .map(|i| {
                let mut qs = [0u8; 32];
                qs.copy_from_slice(&w_bytes[i * 32..(i + 1) * 32]);
                BlockQ4Group64 {
                    scale_bf16: parse_half_endian(&s_bytes[i * 2..i * 2 + 2], s_is_f16),
                    bias_bf16: parse_half_endian(&b_bytes[i * 2..i * 2 + 2], b_is_f16),
                    qs,
                }
            })
            .collect();

        Ok(packed_blocks)
    }
}

#[inline]
fn parse_half_endian(bytes: &[u8], is_f16: bool) -> u16 {
    let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
    if is_f16 {
        crate::quantizer::f32_to_bf16_bits(crate::quantizer::f16_bits_to_f32(raw))
    } else {
        raw
    }
}

/// Packs MLX-style 4-bit affine quantized data (separate weight/scale/bias arrays)
/// into contiguous [`BlockQ4Group64`] blocks.
pub fn pack_mlx_affine_q4_group64(
    weight_nibbles: &[u8],
    scales_bf16_raw: &[u16],
    biases_bf16_raw: &[u16],
) -> Vec<BlockQ4Group64> {
    assert_eq!(scales_bf16_raw.len(), biases_bf16_raw.len());
    assert_eq!(weight_nibbles.len(), scales_bf16_raw.len() * 32);

    scales_bf16_raw
        .iter()
        .zip(biases_bf16_raw)
        .zip(weight_nibbles.chunks_exact(32))
        .map(|((&scale_bf16, &bias_bf16), chunk)| {
            let mut qs = [0u8; 32];
            qs.copy_from_slice(chunk);
            BlockQ4Group64 {
                scale_bf16,
                bias_bf16,
                qs,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a minimal valid `.safetensors` binary file in memory and verifies parsing.
    #[test]
    fn test_real_safetensors_binary_parser_and_mlx_q4_repack() {
        // Build a header with one BF16 tensor of shape [2, 4] = 8 elements = 16 bytes.
        let header_obj = serde_json::json!({
            "test_tensor": {
                "dtype": "BF16",
                "shape": [2, 4],
                "data_offsets": [0, 16]
            },
            "__metadata__": {
                "format": "pt"
            }
        });
        let header_json = serde_json::to_string(&header_obj).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        // Build the binary file: 8-byte LE length + header + 16 bytes data.
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        // 16 bytes of BF16 data (8 elements * 2 bytes).
        file_bytes.extend_from_slice(&[
            0x00, 0x3F, 0x00, 0x3F, 0x00, 0x3F, 0x00, 0x3F, 0x00, 0x3F, 0x00, 0x3F, 0x00, 0x3F,
            0x00, 0x3F,
        ]); // bf16 ~0.5

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let archive = open_safetensors_file(tmp.path()).unwrap();
        let view = archive.get_tensor("test_tensor").unwrap();
        assert_eq!(view.descriptor.dtype, DType::BF16);
        assert_eq!(view.descriptor.shape, vec![2, 4]);
        assert_eq!(view.data.len(), 16);

        // Test MLX Q4 repacking.
        let weight_nibbles = vec![0xABu8; 64]; // 2 groups * 32 bytes
        let scales: Vec<u16> = vec![0x3F80, 0x4000]; // 2 groups
        let biases: Vec<u16> = vec![0x0000, 0x3C00]; // 2 groups
        let blocks = pack_mlx_affine_q4_group64(&weight_nibbles, &scales, &biases);

        assert_eq!(blocks.len(), 2);
        let s0 = { blocks[0].scale_bf16 };
        let b0 = { blocks[0].bias_bf16 };
        assert_eq!(s0, 0x3F80);
        assert_eq!(b0, 0x0000);
        assert_eq!(blocks[0].qs[0], 0xAB);
        let s1 = { blocks[1].scale_bf16 };
        let b1 = { blocks[1].bias_bf16 };
        assert_eq!(s1, 0x4000);
        assert_eq!(b1, 0x3C00);
    }
}
