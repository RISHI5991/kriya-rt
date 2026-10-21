//! Universal Apple Silicon Hardware Profiler & Auto-Tuner (M1 → M5, Air → Ultra).
//!
//! Dynamically configures SIMD matrix paths, Split-K thresholds, ANE offload,
//! and PowerInfer hot/cold expert budgets based on exact chip capabilities.

/// Apple GPU family classification spanning M1 through M5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AppleGpuFamily {
    /// M1 / M1 Pro / M1 Max / M1 Ultra (FP16 simdgroup_matrix, no native BF16)
    Apple7,
    /// M2 / M2 Pro / M2 Max / M2 Ultra (Native BF16 + SIMD shift/fill)
    Apple8,
    /// M3 & M4 series (Dynamic Caching + 64-bit Atomics)
    Apple9,
    /// M5 series (Metal 4 Neural Accelerators / Tensor Ops)
    Apple10,
}

/// Mac hardware form factor determining GPU core count and thermal envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacFormFactor {
    /// 8-10 GPU cores, passive/light cooling, 100-153 GB/s bandwidth
    AirOrBase,
    /// 14-20 GPU cores, active cooling, 200-273 GB/s bandwidth
    Pro,
    /// 30-40 GPU cores, 400-546 GB/s bandwidth
    Max,
    /// 60-80 GPU cores, 800+ GB/s bandwidth
    Ultra,
}

/// Detected hardware capabilities for the current Apple Silicon chip.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub chip_name: String,
    pub gpu_family: AppleGpuFamily,
    pub form_factor: MacFormFactor,
    pub gpu_core_count: u32,
    pub unified_memory_bytes: u64,
    pub max_working_set_bytes: u64,
    pub supports_native_bf16: bool,
    pub supports_dynamic_caching: bool,
    pub supports_metal4_tensor_ops: bool,
}

/// Tuned execution policy generated from hardware profiling and model size analysis.
#[derive(Debug, Clone)]
pub struct TunedExecutionPolicy {
    pub use_bf16_kernels: bool,
    pub use_metal4_tensor_prefill: bool,
    pub prefill_chunk_size: u32,
    pub decode_amortize_steps: u32,
    pub gqa_split_k_threshold: u32,
    pub gqa_split_k_partitions: u32,
    /// Fraction of MoE experts pinned in wired GPU memory (1.0 = all in GPU, <1.0 = PowerInfer CPU-GPU hybrid)
    pub gpu_hot_expert_ratio: f32,
    pub compress_kv_cache_q4: bool,
}

impl HardwareProfile {
    /// Constructs a hardware profile from detected chip properties.
    pub fn detect(
        chip_name: &str,
        gpu_family: AppleGpuFamily,
        gpu_core_count: u32,
        unified_memory_bytes: u64,
    ) -> Self {
        let form_factor = match gpu_core_count {
            0..=12 => MacFormFactor::AirOrBase,
            13..=24 => MacFormFactor::Pro,
            25..=48 => MacFormFactor::Max,
            _ => MacFormFactor::Ultra,
        };

        // Reserve 20% on Air/Base or 12% on Pro/Max/Ultra for macOS WindowServer & OS overhead
        let usable_ratio = match form_factor {
            MacFormFactor::AirOrBase => 0.80,
            _ => 0.88,
        };
        let max_working_set_bytes = (unified_memory_bytes as f64 * usable_ratio) as u64;

        Self {
            chip_name: chip_name.to_string(),
            gpu_family,
            form_factor,
            gpu_core_count,
            unified_memory_bytes,
            max_working_set_bytes,
            supports_native_bf16: gpu_family >= AppleGpuFamily::Apple8,
            supports_dynamic_caching: gpu_family >= AppleGpuFamily::Apple9,
            supports_metal4_tensor_ops: gpu_family >= AppleGpuFamily::Apple10,
        }
    }

    /// Generates an optimal execution policy for any combination of Mac hardware and model size.
    pub fn tune_for_model(&self, model_weight_bytes: u64) -> TunedExecutionPolicy {
        // On Max/Ultra (40-80 cores), trigger Split-K attention much earlier (4K context)
        // so all GPU cores participate in single-sequence decoding.
        let (gqa_split_k_threshold, gqa_split_k_partitions) = match self.form_factor {
            MacFormFactor::AirOrBase => (32_768, 4),
            MacFormFactor::Pro => (16_384, 8),
            MacFormFactor::Max => (8_192, 16),
            MacFormFactor::Ultra => (4_096, 32),
        };

        // Bounded prefill chunk size keeps intermediate activations inside L2/SLC cache
        let prefill_chunk_size = match self.form_factor {
            MacFormFactor::AirOrBase => 512,
            MacFormFactor::Pro => 1024,
            MacFormFactor::Max | MacFormFactor::Ultra => 2048,
        };

        // If the model weights exceed 75% of the safe GPU working set (e.g., running a 26B/35B
        // MoE model on a 16GB/24GB MacBook Air), activate PowerInfer hot/cold expert splitting
        // and FlexGen 4-bit KV cache compression.
        let memory_budget_after_kv = (self.max_working_set_bytes as f64 * 0.82) as u64;
        let (gpu_hot_expert_ratio, compress_kv_cache_q4) =
            if model_weight_bytes > memory_budget_after_kv {
                let ratio =
                    (memory_budget_after_kv as f32 / model_weight_bytes as f32).clamp(0.25, 0.95);
                (ratio, true)
            } else {
                (1.0, false)
            };

        TunedExecutionPolicy {
            use_bf16_kernels: self.supports_native_bf16,
            use_metal4_tensor_prefill: self.supports_metal4_tensor_ops,
            prefill_chunk_size,
            decode_amortize_steps: if self.form_factor == MacFormFactor::AirOrBase {
                2
            } else {
                4
            },
            gqa_split_k_threshold,
            gqa_split_k_partitions,
            gpu_hot_expert_ratio,
            compress_kv_cache_q4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m1_air_profile() {
        let profile = HardwareProfile::detect(
            "Apple M1",
            AppleGpuFamily::Apple7,
            8,
            8 * 1024 * 1024 * 1024, // 8 GB
        );
        assert_eq!(profile.form_factor, MacFormFactor::AirOrBase);
        assert!(!profile.supports_native_bf16);
        assert!(!profile.supports_dynamic_caching);
        assert!(!profile.supports_metal4_tensor_ops);

        let policy = profile.tune_for_model(4 * 1024 * 1024 * 1024); // 4 GB model
        assert!(!policy.use_bf16_kernels);
        assert_eq!(policy.prefill_chunk_size, 512);
        assert_eq!(policy.decode_amortize_steps, 2);
        assert_eq!(policy.gqa_split_k_threshold, 32_768);
    }

    #[test]
    fn test_m4_max_profile() {
        let profile = HardwareProfile::detect(
            "Apple M4 Max",
            AppleGpuFamily::Apple9,
            40,
            128 * 1024 * 1024 * 1024, // 128 GB
        );
        assert_eq!(profile.form_factor, MacFormFactor::Max);
        assert!(profile.supports_native_bf16);
        assert!(profile.supports_dynamic_caching);
        assert!(!profile.supports_metal4_tensor_ops);

        let policy = profile.tune_for_model(20 * 1024 * 1024 * 1024); // 20 GB model
        assert!(policy.use_bf16_kernels);
        assert_eq!(policy.prefill_chunk_size, 2048);
        assert_eq!(policy.decode_amortize_steps, 4);
        assert_eq!(policy.gqa_split_k_threshold, 8_192);
        assert_eq!(policy.gpu_hot_expert_ratio, 1.0);
        assert!(!policy.compress_kv_cache_q4);
    }

    #[test]
    fn test_memory_pressure_triggers_hybrid() {
        let profile = HardwareProfile::detect(
            "Apple M2 Air",
            AppleGpuFamily::Apple8,
            10,
            16 * 1024 * 1024 * 1024, // 16 GB
        );
        // Model weights larger than usable GPU budget
        let policy = profile.tune_for_model(14 * 1024 * 1024 * 1024); // 14 GB model
        assert!(policy.gpu_hot_expert_ratio < 1.0);
        assert!(policy.compress_kv_cache_q4);
    }
}
