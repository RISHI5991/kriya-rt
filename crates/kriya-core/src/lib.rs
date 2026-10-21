//! KriyaRT Core — Zero-Framework LLM Inference Engine Foundation.
//!
//! Provides the foundational types for the KriyaRT inference engine:
//! - `tensor`: Zero-copy mmap weight loader, 4-bit quantization blocks, and neuron placement tables
//! - `hardware`: Apple Silicon hardware profiler and auto-tuner (M1→M5, Air→Ultra)
//! - `descriptor`: Data-driven architecture descriptors for zero-branch inference
//! - `graph`: Zero-allocation execution graph and amortized multi-token decode scheduler

pub mod descriptor;
pub mod graph;
pub mod hardware;
pub mod model_swapper;
pub mod quantizer;
pub mod safetensors_loader;
pub mod tensor;
pub mod tokenizer;
