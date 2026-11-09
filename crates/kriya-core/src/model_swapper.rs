// File: crates/kriya-core/src/model_swapper.rs
//! Dynamic Multi-Model Unified Memory Swapper (`MADV_WILLNEED` / `MADV_DONTNEED`).
//!
//! Enables instant zero-restart hot-swapping across multiple `.safetensors` models
//! in Apple Silicon Unified Memory.

use crate::descriptor::ModelArchDescriptor;
use crate::safetensors_loader::open_safetensors_file;
use crate::tensor::{MappedWeightArchive, TensorError};
use std::collections::HashMap;
use std::path::Path;

#[cfg(unix)]
unsafe extern "C" {
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
}

#[cfg(unix)]
const POSIX_MADV_WILLNEED: i32 = 3;
#[cfg(unix)]
const POSIX_MADV_DONTNEED: i32 = 4;

/// Residency state of a registered model in Apple Silicon Unified Memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelResidencyState {
    /// Pages are prefetched and pinned in active working set (`MADV_WILLNEED`).
    ActiveHot,
    /// Virtual mapping retained, physical pages released to OS on pressure (`MADV_DONTNEED`).
    ParkedCold,
}

pub struct RegisteredModelSlot {
    pub model_id: String,
    pub descriptor: ModelArchDescriptor,
    pub archive: MappedWeightArchive,
    pub state: ModelResidencyState,
}

/// Multi-Model Registry & Zero-Restart Unified Memory Page Swapper.
pub struct DynamicModelSwapper {
    models: HashMap<String, RegisteredModelSlot>,
    active_model_id: Option<String>,
}

impl DynamicModelSwapper {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            active_model_id: None,
        }
    }

    /// Registers a `.safetensors` model file into the swapper without forcing immediate page faults.
    pub fn register_model<P: AsRef<Path>>(
        &mut self,
        model_id: impl Into<String>,
        descriptor: ModelArchDescriptor,
        safetensors_path: P,
    ) -> Result<(), TensorError> {
        let id = model_id.into();
        let archive = open_safetensors_file(safetensors_path)?;
        self.models.insert(
            id,
            RegisteredModelSlot {
                model_id: descriptor.model_name.clone(),
                descriptor,
                archive,
                state: ModelResidencyState::ParkedCold,
            },
        );
        Ok(())
    }

    /// Hot-swaps the active model in Unified Memory:
    /// 1. Issues `MADV_DONTNEED` on the outgoing model's mapped pages.
    /// 2. Issues `MADV_WILLNEED` on the incoming model's mapped pages to trigger hardware readahead.
    pub fn activate_model(&mut self, target_id: &str) -> Result<&RegisteredModelSlot, TensorError> {
        if !self.models.contains_key(target_id) {
            return Err(TensorError::TensorNotFound(target_id.to_string()));
        }

        if let Some(prev_id) = self.active_model_id.take()
            && prev_id != target_id
            && let Some(prev_slot) = self.models.get_mut(&prev_id)
        {
            Self::advise_pages(&prev_slot.archive, ModelResidencyState::ParkedCold);
            prev_slot.state = ModelResidencyState::ParkedCold;
        }

        let slot = self.models.get_mut(target_id).unwrap();
        Self::advise_pages(&slot.archive, ModelResidencyState::ActiveHot);
        slot.state = ModelResidencyState::ActiveHot;
        self.active_model_id = Some(target_id.to_string());

        Ok(slot)
    }

    #[inline]
    pub fn active_model_id(&self) -> Option<&str> {
        self.active_model_id.as_deref()
    }

    fn advise_pages(archive: &MappedWeightArchive, target_state: ModelResidencyState) {
        #[cfg(unix)]
        unsafe {
            let ptr = archive.base_ptr() as *mut std::ffi::c_void;
            let len = archive.total_mapped_bytes();
            let advice = match target_state {
                ModelResidencyState::ActiveHot => POSIX_MADV_WILLNEED,
                ModelResidencyState::ParkedCold => POSIX_MADV_DONTNEED,
            };
            if len > 0 {
                let _ = madvise(ptr, len, advice);
            }
        }
        #[cfg(not(unix))]
        let _ = (archive, target_state);
    }
}

impl Default for DynamicModelSwapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantizer::{quantize_f32_to_q4_group64, write_quantized_safetensors};

    #[test]
    fn test_dynamic_multi_model_hot_swap() {
        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("model_a.safetensors");
        let path_b = dir.path().join("model_b.safetensors");

        let blocks = quantize_f32_to_q4_group64(&vec![0.5f32; 64]).unwrap();
        write_quantized_safetensors(
            &path_a,
            &[(
                "model.embed_tokens.weight".into(),
                vec![1, 64],
                blocks.clone(),
            )],
        )
        .unwrap();
        write_quantized_safetensors(
            &path_b,
            &[("model.embed_tokens.weight".into(), vec![1, 64], blocks)],
        )
        .unwrap();

        let mut swapper = DynamicModelSwapper::new();
        swapper
            .register_model("llama-3.2", ModelArchDescriptor::llama_3_2_3b(), &path_a)
            .unwrap();
        swapper
            .register_model("qwen-3.6", ModelArchDescriptor::qwen3_6_35b_a3b(), &path_b)
            .unwrap();

        let slot_a = swapper.activate_model("llama-3.2").unwrap();
        assert_eq!(slot_a.state, ModelResidencyState::ActiveHot);

        let slot_b = swapper.activate_model("qwen-3.6").unwrap();
        assert_eq!(slot_b.state, ModelResidencyState::ActiveHot);
        assert_eq!(swapper.active_model_id(), Some("qwen-3.6"));
        assert_eq!(
            swapper.models["llama-3.2"].state,
            ModelResidencyState::ParkedCold
        );
    }
}
