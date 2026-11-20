//! Zero-Copy Tensor Views, 4-Bit Group-64 Quantized Blocks, and Hybrid Neuron Tables.
//!
//! Provides the foundational memory layer for KriyaRT:
//! - Memory-mapped weight loading with zero heap duplication
//! - Q4_G64 block quantization for in-kernel threadgroup tile dequantization
//! - PowerInfer-inspired hot/cold neuron placement tables for hybrid CPU-GPU execution

use bytemuck::{Pod, Zeroable};
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Apple Silicon / Metal page alignment requirement for zero-copy buffer binding (16 KB).
pub const METAL_PAGE_ALIGNMENT: usize = 16_384;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("I/O error while mapping model weights: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid header or metadata format: {0}")]
    InvalidMetadata(String),
    #[error("Tensor '{0}' not found in mapped weight archive")]
    TensorNotFound(String),
    #[error("Shape mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },
    #[error("Byte slice alignment violation: offset {offset} is not aligned to {align} bytes")]
    AlignmentError { offset: usize, align: usize },
}

/// Supported scalar and block-quantized data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DType {
    /// 32-bit IEEE 754 floating point.
    F32,
    /// 16-bit IEEE 754 floating point.
    F16,
    /// 16-bit Brain Floating Point (native to modern Tensor/Matrix cores).
    BF16,
    /// 8-bit floating point (E4M3 format).
    FP8E4M3,
    /// Lily-style 4-bit group-64 quantization: 64 4-bit weights (32 bytes) + bf16 scale (2B) + bf16 bias (2B) = 36 bytes.
    Q4Group64,
    /// 2-bit group-128 quantization: 128 2-bit weights (32 bytes) + bf16 scale (2B) + bf16 bias (2B) = 36 bytes.
    Q2Group128,
    /// 3-bit group-64 quantization: 64 3-bit weights (24 bytes) + bf16 scale (2B) + bf16 bias (2B) = 28 bytes.
    Q3Group64,
    /// FlexGen-style 4-bit KV cache / activation group-32 compression: 16 bytes payload + f16 scale/min.
    Q4KvGroup32,
    /// 32-bit unsigned integer (used for MLX packed 4-bit weights).
    U32,
    /// 8-bit unsigned integer (used for quantized byte weights).
    U8,
    /// 32-bit signed integer.
    I32,
    /// 8-bit signed integer.
    I8,
}

impl DType {
    /// Returns the number of elements packed inside a single storage block.
    #[inline]
    pub const fn block_elements(&self) -> usize {
        match self {
            DType::F32
            | DType::F16
            | DType::BF16
            | DType::FP8E4M3
            | DType::U32
            | DType::U8
            | DType::I32
            | DType::I8 => 1,
            DType::Q4Group64 | DType::Q3Group64 => 64,
            DType::Q2Group128 => 128,
            DType::Q4KvGroup32 => 32,
        }
    }

    /// Returns the byte size of a single storage block.
    #[inline]
    pub const fn block_byte_size(&self) -> usize {
        match self {
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::FP8E4M3 | DType::U8 | DType::I8 => 1,
            DType::Q4Group64 => 36, // 32B nibbles + 2B bf16 scale + 2B bf16 zero_bias
            DType::Q2Group128 => 36, // 32B 2-bit codes + 2B bf16 scale + 2B bf16 zero_bias
            DType::Q3Group64 => 28, // 24B 3-bit codes + 2B bf16 scale + 2B bf16 zero_bias
            DType::Q4KvGroup32 => 20, // 16B nibbles + 2B f16 scale + 2B f16 min
        }
    }

    /// Computes total required bytes for a given element count.
    #[inline]
    pub const fn byte_size_for_elements(&self, num_elements: usize) -> usize {
        let block_elems = self.block_elements();
        let num_blocks = num_elements.div_ceil(block_elems);
        num_blocks * self.block_byte_size()
    }
}

/// 4-bit Group-64 Quantized Block (`Q4_G64`) used for in-kernel threadgroup tile dequantization.
/// Stores 64 4-bit unsigned codes alongside shared `bf16` scale and bias (36 bytes total).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlockQ4Group64 {
    /// Raw `bf16` bits for the group multiplicative scale.
    pub scale_bf16: u16,
    /// Raw `bf16` bits for the group additive bias / zero-point.
    pub bias_bf16: u16,
    /// 64 4-bit quantized weights packed as 32 bytes (low nibble = even idx, high nibble = odd idx).
    pub qs: [u8; 32],
}

/// 16-byte aligned 32-byte quantized weight block containing 64 4-bit unsigned nibbles.
/// Guarantees 100% coalesced 128-bit vector loads (`packed_uint4` / `simdgroup_load`)
/// on Apple Silicon unified memory controllers without crossing cache line boundaries.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct ScaledAlignedQ4Block {
    pub qs: [u8; 32],
}

impl ScaledAlignedQ4Block {
    #[inline]
    pub const fn new(qs: [u8; 32]) -> Self {
        Self { qs }
    }
}

/// 4-byte packed multiplicative scale and additive zero-point bias for a 64-element Q4 block.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct BlockQ4ScaleBias {
    /// Raw `bf16` bits for the group multiplicative scale.
    pub scale_bf16: u16,
    /// Raw `bf16` bits for the group additive bias / zero-point.
    pub bias_bf16: u16,
}

impl BlockQ4ScaleBias {
    #[inline]
    pub const fn new(scale_bf16: u16, bias_bf16: u16) -> Self {
        Self {
            scale_bf16,
            bias_bf16,
        }
    }
}

impl BlockQ4Group64 {
    /// Splits a slice of legacy 36-byte `BlockQ4Group64` blocks into separate 16-byte aligned
    /// 32-byte weight blocks and 4-byte scale/bias pairs.
    pub fn split_slice(blocks: &[Self]) -> (Vec<ScaledAlignedQ4Block>, Vec<BlockQ4ScaleBias>) {
        let mut qs = Vec::with_capacity(blocks.len());
        let mut scales = Vec::with_capacity(blocks.len());
        for b in blocks {
            let scale_bf16 = b.scale_bf16;
            let bias_bf16 = b.bias_bf16;
            let block_qs = b.qs;
            qs.push(ScaledAlignedQ4Block { qs: block_qs });
            scales.push(BlockQ4ScaleBias {
                scale_bf16,
                bias_bf16,
            });
        }
        (qs, scales)
    }

    /// Reconstructs a vector of `BlockQ4Group64` blocks from split weight and scale/bias slices.
    pub fn from_split(qs: &[ScaledAlignedQ4Block], scales: &[BlockQ4ScaleBias]) -> Vec<Self> {
        assert_eq!(
            qs.len(),
            scales.len(),
            "Weight and scale/bias slices must have matching lengths"
        );
        qs.iter()
            .zip(scales.iter())
            .map(|(q, s)| Self {
                scale_bf16: s.scale_bf16,
                bias_bf16: s.bias_bf16,
                qs: q.qs,
            })
            .collect()
    }

    #[inline]
    pub fn to_split(&self) -> (ScaledAlignedQ4Block, BlockQ4ScaleBias) {
        let scale_bf16 = self.scale_bf16;
        let bias_bf16 = self.bias_bf16;
        let block_qs = self.qs;
        (
            ScaledAlignedQ4Block { qs: block_qs },
            BlockQ4ScaleBias {
                scale_bf16,
                bias_bf16,
            },
        )
    }

    #[inline]
    pub fn from_split_single(q: ScaledAlignedQ4Block, s: BlockQ4ScaleBias) -> Self {
        Self {
            scale_bf16: s.scale_bf16,
            bias_bf16: s.bias_bf16,
            qs: q.qs,
        }
    }
}

/// 2-bit Group-128 Quantized Block (`Q2_G128`) used for extreme compression inference.
/// Stores 128 2-bit unsigned codes packed into 32 bytes alongside shared `bf16` scale and bias (36 bytes total).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlockQ2Group128 {
    /// Raw `bf16` bits for the group multiplicative scale.
    pub scale_bf16: u16,
    /// Raw `bf16` bits for the group additive bias / zero-point.
    pub bias_bf16: u16,
    /// 128 2-bit quantized weights packed as 32 bytes (4 weights per byte, LSB first).
    pub qs: [u8; 32],
}

/// 3-bit Group-64 Quantized Block (`Q3_G64`) used for balanced sub-4-bit compression.
/// Stores 64 3-bit unsigned codes packed into 24 bytes alongside shared `bf16` scale and bias (28 bytes total).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlockQ3Group64 {
    /// Raw `bf16` bits for the group multiplicative scale.
    pub scale_bf16: u16,
    /// Raw `bf16` bits for the group additive bias / zero-point.
    pub bias_bf16: u16,
    /// 64 3-bit quantized weights packed as 24 bytes (8 triplets of 3 bytes for 8 weights).
    pub qs: [u8; 24],
}

/// Target hardware execution unit for hybrid CPU-GPU placement (PowerInfer model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlacementUnit {
    /// Hot-activated neurons or full layers pinned in GPU / Wired Unified Memory.
    GpuHot,
    /// Cold-activated neurons or offloaded blocks managed by CPU vector executors.
    CpuCold,
}

/// Metadata descriptor for an individual tensor stored inside a mapped model file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorDescriptor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub byte_offset: usize,
    pub byte_len: usize,
    pub placement: PlacementUnit,
}

impl TensorDescriptor {
    /// Total scalar elements across all dimensions.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().copied().product()
    }

    /// Validates that the byte length matches the shape and quantization block format.
    pub fn validate(&self) -> Result<(), TensorError> {
        let expected_bytes = self.dtype.byte_size_for_elements(self.num_elements());
        if self.byte_len != expected_bytes {
            return Err(TensorError::ShapeMismatch {
                expected: expected_bytes,
                actual: self.byte_len,
            });
        }
        Ok(())
    }
}

/// Zero-copy view over a memory-mapped tensor slice.
#[derive(Clone)]
pub struct TensorView<'a> {
    pub descriptor: &'a TensorDescriptor,
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Returns raw pointer suitable for `MTLDevice::newBufferWithBytesNoCopy`.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Checks if the underlying memory pointer is 16KB page-aligned for zero-copy Metal binding.
    #[inline]
    pub fn is_page_aligned(&self) -> bool {
        (self.data.as_ptr() as usize) % METAL_PAGE_ALIGNMENT == 0
    }

    /// Zero-copy cast into a slice of `BlockQ4Group64` blocks for `Q4Group64` tensors.
    pub fn as_q4_g64_blocks(&self) -> Result<&'a [BlockQ4Group64], TensorError> {
        if self.descriptor.dtype != DType::Q4Group64 {
            return Err(TensorError::InvalidMetadata(format!(
                "Expected Q4Group64 dtype, found {:?}",
                self.descriptor.dtype
            )));
        }
        Ok(bytemuck::cast_slice(self.data))
    }

    /// Safe zero-copy cast into a slice of 16-byte aligned `ScaledAlignedQ4Block` blocks.
    /// Fails safely with `TensorError::AlignmentError` if pointer is not 16-byte aligned.
    pub fn as_aligned_q4_blocks(&self) -> Result<&'a [ScaledAlignedQ4Block], TensorError> {
        let align = std::mem::align_of::<ScaledAlignedQ4Block>();
        let offset = self.data.as_ptr() as usize;
        if offset % align != 0 {
            return Err(TensorError::AlignmentError { offset, align });
        }
        bytemuck::try_cast_slice(self.data).map_err(|e| {
            TensorError::InvalidMetadata(format!("Failed to cast ScaledAlignedQ4Block slice: {e}"))
        })
    }

    /// Alias for `as_aligned_q4_blocks`.
    #[inline]
    pub fn as_scaled_aligned_q4_blocks(&self) -> Result<&'a [ScaledAlignedQ4Block], TensorError> {
        self.as_aligned_q4_blocks()
    }

    /// Safe zero-copy cast into a slice of `BlockQ4ScaleBias` blocks.
    /// Fails safely with `TensorError::AlignmentError` if pointer is not 2-byte aligned.
    pub fn as_q4_scale_bias_blocks(&self) -> Result<&'a [BlockQ4ScaleBias], TensorError> {
        let align = std::mem::align_of::<BlockQ4ScaleBias>();
        let offset = self.data.as_ptr() as usize;
        if offset % align != 0 {
            return Err(TensorError::AlignmentError { offset, align });
        }
        bytemuck::try_cast_slice(self.data).map_err(|e| {
            TensorError::InvalidMetadata(format!("Failed to cast BlockQ4ScaleBias slice: {e}"))
        })
    }

    /// Zero-copy cast into a slice of `BlockQ2Group128` blocks for `Q2Group128` tensors.
    pub fn as_q2_g128_blocks(&self) -> Result<&'a [BlockQ2Group128], TensorError> {
        if self.descriptor.dtype != DType::Q2Group128 {
            return Err(TensorError::InvalidMetadata(format!(
                "Expected Q2Group128 dtype, found {:?}",
                self.descriptor.dtype
            )));
        }
        Ok(bytemuck::cast_slice(self.data))
    }

    /// Zero-copy cast into a slice of `BlockQ3Group64` blocks for `Q3Group64` tensors.
    pub fn as_q3_g64_blocks(&self) -> Result<&'a [BlockQ3Group64], TensorError> {
        if self.descriptor.dtype != DType::Q3Group64 {
            return Err(TensorError::InvalidMetadata(format!(
                "Expected Q3Group64 dtype, found {:?}",
                self.descriptor.dtype
            )));
        }
        Ok(bytemuck::cast_slice(self.data))
    }
}

/// Zero-Copy Memory-Mapped Weight Archive (`.base` / `.safetensors` compatible).
pub struct MappedWeightArchive {
    mmap: Arc<Mmap>,
    data_section_offset: usize,
    tensors: HashMap<String, TensorDescriptor>,
}

impl MappedWeightArchive {
    /// Opens a weight file using OS virtual memory mapping (`mmap`) with zero heap allocation for weights.
    pub fn open<P: AsRef<Path>>(
        path: P,
        data_section_offset: usize,
        descriptors: Vec<TensorDescriptor>,
    ) -> Result<Self, TensorError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let mut tensors = HashMap::with_capacity(descriptors.len());
        for desc in descriptors {
            desc.validate()?;
            let end = data_section_offset
                .checked_add(desc.byte_offset)
                .and_then(|o| o.checked_add(desc.byte_len))
                .ok_or_else(|| TensorError::InvalidMetadata("Offset overflow".into()))?;

            if end > mmap.len() {
                return Err(TensorError::InvalidMetadata(format!(
                    "Tensor '{}' exceeds file size ({} > {})",
                    desc.name,
                    end,
                    mmap.len()
                )));
            }
            tensors.insert(desc.name.clone(), desc);
        }

        Ok(Self {
            mmap: Arc::new(mmap),
            data_section_offset,
            tensors,
        })
    }

    /// Returns an iterator over all tensor names in this archive.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Retrieves a zero-copy `TensorView` directly borrowing from the memory-mapped pages.
    pub fn get_tensor(&self, name: &str) -> Result<TensorView<'_>, TensorError> {
        let desc = self
            .tensors
            .get(name)
            .ok_or_else(|| TensorError::TensorNotFound(name.to_string()))?;

        let start = self.data_section_offset + desc.byte_offset;
        let end = start + desc.byte_len;
        Ok(TensorView {
            descriptor: desc,
            data: &self.mmap[start..end],
        })
    }

    /// Base pointer to the entire memory-mapped file for single-allocation GPU buffer binding.
    #[inline]
    pub fn base_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    /// Total mapped byte length.
    #[inline]
    pub fn total_mapped_bytes(&self) -> usize {
        self.mmap.len()
    }

    /// Issues an asynchronous OS page cache prefetch hint (`MADV_WILLNEED`) for a tensor from NVMe SSD.
    pub fn prefetch_tensor(&self, name: &str) -> Result<(), TensorError> {
        let desc = self
            .tensors
            .get(name)
            .ok_or_else(|| TensorError::TensorNotFound(name.to_string()))?;
        let start = self.data_section_offset + desc.byte_offset;
        let _ = self
            .mmap
            .advise_range(memmap2::Advice::WillNeed, start, desc.byte_len);
        Ok(())
    }

    /// Resets tensor page advice to Normal when a layer has finished execution.
    pub fn release_tensor(&self, name: &str) -> Result<(), TensorError> {
        let desc = self
            .tensors
            .get(name)
            .ok_or_else(|| TensorError::TensorNotFound(name.to_string()))?;
        let start = self.data_section_offset + desc.byte_offset;
        let _ = self
            .mmap
            .advise_range(memmap2::Advice::Normal, start, desc.byte_len);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_q4_group64_sizing_and_zero_copy_cast() {
        // 256 elements = 4 blocks of 64 elements = 4 * 36 = 144 bytes
        assert_eq!(DType::Q4Group64.byte_size_for_elements(256), 144);

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let raw_bytes = vec![0xABu8; 144];
        tmp.write_all(&raw_bytes).unwrap();
        tmp.flush().unwrap();

        let desc = TensorDescriptor {
            name: "layers.0.moe.experts.0.w_up".to_string(),
            dtype: DType::Q4Group64,
            shape: vec![4, 64],
            byte_offset: 0,
            byte_len: 144,
            placement: PlacementUnit::GpuHot,
        };

        let archive = MappedWeightArchive::open(tmp.path(), 0, vec![desc]).unwrap();
        let view = archive.get_tensor("layers.0.moe.experts.0.w_up").unwrap();
        let blocks = view.as_q4_g64_blocks().unwrap();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].qs[0], 0xAB);
    }

    #[test]
    fn test_dtype_element_sizes() {
        assert_eq!(DType::F32.byte_size_for_elements(100), 400);
        assert_eq!(DType::BF16.byte_size_for_elements(100), 200);
        assert_eq!(DType::FP8E4M3.byte_size_for_elements(100), 100);
        // Q4KvGroup32: 100 elements = ceil(100/32) = 4 blocks * 20 bytes = 80 bytes
        assert_eq!(DType::Q4KvGroup32.byte_size_for_elements(100), 80);
    }

    #[test]
    fn test_scaled_aligned_q4_block_memory_alignment() {
        assert_eq!(std::mem::size_of::<ScaledAlignedQ4Block>(), 32);
        assert_eq!(std::mem::align_of::<ScaledAlignedQ4Block>(), 16);
        assert_eq!(std::mem::size_of::<BlockQ4ScaleBias>(), 4);
        assert_eq!(std::mem::align_of::<BlockQ4ScaleBias>(), 2);

        let blocks: Vec<BlockQ4Group64> = (0..8)
            .map(|i| BlockQ4Group64 {
                scale_bf16: (i * 10) as u16,
                bias_bf16: (i * 5) as u16,
                qs: [i as u8; 32],
            })
            .collect();

        let (split_qs, split_sb) = BlockQ4Group64::split_slice(&blocks);
        assert_eq!(split_qs.len(), 8);
        assert_eq!(split_sb.len(), 8);
        assert_eq!(split_qs[3].qs[0], 3);
        assert_eq!(split_sb[3].scale_bf16, 30);
        assert_eq!(split_sb[3].bias_bf16, 15);

        let reconstructed = BlockQ4Group64::from_split(&split_qs, &split_sb);
        assert_eq!(reconstructed.len(), 8);
        for i in 0..8 {
            let rec_s = reconstructed[i].scale_bf16;
            let rec_b = reconstructed[i].bias_bf16;
            let rec_qs = reconstructed[i].qs;
            let orig_s = blocks[i].scale_bf16;
            let orig_b = blocks[i].bias_bf16;
            let orig_qs = blocks[i].qs;
            assert_eq!(rec_s, orig_s);
            assert_eq!(rec_b, orig_b);
            assert_eq!(rec_qs, orig_qs);
        }

        let single = blocks[4].to_split();
        assert_eq!(single.0.qs[0], 4);
        assert_eq!(single.1.scale_bf16, 40);
        let roundtrip_single = BlockQ4Group64::from_split_single(single.0, single.1);
        let rs_s = roundtrip_single.scale_bf16;
        let rs_b = roundtrip_single.bias_bf16;
        let rs_qs = roundtrip_single.qs;
        let b4_s = blocks[4].scale_bf16;
        let b4_b = blocks[4].bias_bf16;
        let b4_qs = blocks[4].qs;
        assert_eq!(rs_s, b4_s);
        assert_eq!(rs_b, b4_b);
        assert_eq!(rs_qs, b4_qs);
    }
}
