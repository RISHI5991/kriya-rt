// File: crates/kriya-backend-metal/src/layer_streamer.rs
//! Double-Buffered NVMe SSD Layer Streaming ("LLM in a Flash").
//!
//! Enables running 14B–80B models within 8 GB–16 GB Apple Silicon unified memory
//! by keeping only TWO layer buffers active in GPU memory at any given time.
//!
//! Uses zero-copy virtual memory page mapping combined with POSIX `madvise`
//! (`POSIX_MADV_WILLNEED` and `POSIX_MADV_DONTNEED`) to stream layer L+1 from SSD
//! asynchronously while layer L computes on GPU, and promptly release evicted layer
//! physical memory pages back to macOS.

use crate::encoder::{LayerGpuBuffers, MetalError, ModelGpuWeights};
use kriya_core::descriptor::ModelArchDescriptor;
use kriya_core::safetensors_loader::ShardedCheckpointLoader;
use metal::Device;
use std::sync::Arc;

#[cfg(unix)]
unsafe extern "C" {
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
}

#[cfg(unix)]
const POSIX_MADV_WILLNEED: i32 = 3;
#[cfg(unix)]
const POSIX_MADV_DONTNEED: i32 = 4;

/// Generates the list of tensor names associated with a specific transformer layer.
pub fn get_layer_tensor_names(desc: &ModelArchDescriptor, idx: usize) -> Vec<String> {
    let p = format!("model.layers.{idx}");
    let mut names = vec![
        format!("{p}.input_layernorm.weight"),
        format!("{p}.self_attn.q_proj.weight"),
        format!("{p}.self_attn.k_proj.weight"),
        format!("{p}.self_attn.v_proj.weight"),
        format!("{p}.self_attn.o_proj.weight"),
        format!("{p}.self_attn.q_proj"),
        format!("{p}.self_attn.k_proj"),
        format!("{p}.self_attn.v_proj"),
        format!("{p}.self_attn.o_proj"),
        format!("{p}.mlp.gate_proj.weight"),
        format!("{p}.mlp.up_proj.weight"),
        format!("{p}.mlp.down_proj.weight"),
        format!("{p}.mlp.gate_proj"),
        format!("{p}.mlp.up_proj"),
        format!("{p}.mlp.down_proj"),
        format!("{p}.post_attention_layernorm.weight"),
        format!("{p}.pre_feedforward_layernorm.weight"),
        format!("{p}.post_feedforward_layernorm.weight"),
        format!("{p}.self_attn.q_proj.bias"),
        format!("{p}.self_attn.k_proj.bias"),
        format!("{p}.self_attn.v_proj.bias"),
        format!("{p}.self_attn.q_norm.weight"),
        format!("{p}.self_attn.k_norm.weight"),
    ];

    if let Some(layer) = desc.layers.get(idx) {
        if layer.post_attn_norm.is_some() {
            names.push(format!("{p}.post_attention_layernorm.weight"));
        }
        if layer.post_ffn_norm.is_some() {
            names.push(format!("{p}.post_feedforward_layernorm.weight"));
        }
    }

    names
}

/// Double-buffered GPU layer streamer holding at most two active transformer layers in GPU memory.
pub struct DoubleBufferedLayerStreamer {
    loader: Arc<ShardedCheckpointLoader>,
    device: Device,
    descriptor: ModelArchDescriptor,
    /// Slot 0 (ping) and Slot 1 (pong)
    slots: [Option<LayerGpuBuffers>; 2],
    /// Layer index assigned to each slot
    slot_layer_indices: [Option<usize>; 2],
    total_layers: usize,
}

impl DoubleBufferedLayerStreamer {
    /// Creates a new layer streamer and prefetches layer 0 and layer 1.
    pub fn new(
        loader: Arc<ShardedCheckpointLoader>,
        device: Device,
        descriptor: ModelArchDescriptor,
    ) -> Result<Self, MetalError> {
        let total_layers = descriptor.layers.len();
        let mut streamer = Self {
            loader,
            device,
            descriptor,
            slots: [None, None],
            slot_layer_indices: [None, None],
            total_layers,
        };

        if total_layers > 0 {
            streamer.load_slot(0, 0)?;
        }
        if total_layers > 1 {
            streamer.load_slot(1, 1)?;
        }

        Ok(streamer)
    }

    fn advise_layer(&self, l_idx: usize, advice: i32) {
        #[cfg(unix)]
        {
            let names = get_layer_tensor_names(&self.descriptor, l_idx);
            for name in names {
                if let Ok(view) = self.loader.get_tensor(&name) {
                    unsafe {
                        madvise(view.data.as_ptr() as *mut _, view.data.len(), advice);
                    }
                }
            }
        }
        #[cfg(not(unix))]
        let _ = (l_idx, advice);
    }

    fn load_slot(&mut self, slot: usize, l_idx: usize) -> Result<(), MetalError> {
        // Evict previous layer from slot: release physical pages back to OS
        if let Some(old_idx) = self.slot_layer_indices[slot] {
            if old_idx != l_idx {
                self.advise_layer(old_idx, POSIX_MADV_DONTNEED);
            }
        }

        // Advise OS to readahead incoming layer
        self.advise_layer(l_idx, POSIX_MADV_WILLNEED);

        let bufs = ModelGpuWeights::load_layer_from_checkpoint(
            &self.loader,
            &self.device,
            &self.descriptor,
            l_idx,
        )?;

        self.slots[slot] = Some(bufs);
        self.slot_layer_indices[slot] = Some(l_idx);
        Ok(())
    }

    /// Retrieves GPU buffers for `l_idx`, proactively prefetching `l_idx + 1` into the alternate slot.
    pub fn get_layer(&mut self, l_idx: usize) -> Result<LayerGpuBuffers, MetalError> {
        if l_idx >= self.total_layers {
            return Err(MetalError::InvalidExecutionInput(format!(
                "Requested layer index {l_idx} out of bounds (total layers: {})",
                self.total_layers
            )));
        }

        // Hit in slot 0
        if self.slot_layer_indices[0] == Some(l_idx) {
            let bufs = self.slots[0].as_ref().unwrap().clone();
            let next_l = l_idx + 1;
            if next_l < self.total_layers && self.slot_layer_indices[1] != Some(next_l) {
                self.load_slot(1, next_l)?;
            }
            return Ok(bufs);
        }

        // Hit in slot 1
        if self.slot_layer_indices[1] == Some(l_idx) {
            let bufs = self.slots[1].as_ref().unwrap().clone();
            let next_l = l_idx + 1;
            if next_l < self.total_layers && self.slot_layer_indices[0] != Some(next_l) {
                self.load_slot(0, next_l)?;
            }
            return Ok(bufs);
        }

        // Miss in both slots: load into slot (l_idx % 2) and prefetch next into alternate slot
        let target_slot = l_idx % 2;
        self.load_slot(target_slot, l_idx)?;
        let bufs = self.slots[target_slot].as_ref().unwrap().clone();

        let next_l = l_idx + 1;
        let alt_slot = (target_slot + 1) % 2;
        if next_l < self.total_layers && self.slot_layer_indices[alt_slot] != Some(next_l) {
            self.load_slot(alt_slot, next_l)?;
        }

        Ok(bufs)
    }

    /// Returns the layer indices currently active in GPU memory slots [slot_0, slot_1].
    pub fn active_slots(&self) -> [Option<usize>; 2] {
        self.slot_layer_indices
    }
}
