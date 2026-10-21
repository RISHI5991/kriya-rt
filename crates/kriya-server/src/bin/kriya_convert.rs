//! Python-Free Native Rust Quantization CLI (`kriya_convert` — Closes Blocker 2).
//!
//! Preserves 1D `*.norm.weight` tensors in exact BF16 while quantizing 2D weight
//! matrices into 36-byte `BlockQ4Group64` blocks.

use kriya_core::quantizer::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, quantize_f32_to_q4_group64,
    should_preserve_in_bf16, write_mixed_safetensors, HuggingFaceModelConfig, SavedTensorPayload,
};
use kriya_core::safetensors_loader::open_safetensors_file;
use kriya_core::tensor::DType;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args
        .windows(2)
        .find(|w| w[0] == "--input-dir")
        .map(|w| PathBuf::from(&w[1]));
    let output_path = args
        .windows(2)
        .find(|w| w[0] == "--output-dir")
        .map(|w| PathBuf::from(&w[1]));
    let config_path = args
        .windows(2)
        .find(|w| w[0] == "--config")
        .map(|w| PathBuf::from(&w[1]));

    if let Some(cfg_file) = config_path {
        let desc = HuggingFaceModelConfig::from_config_file(&cfg_file)
            .expect("Failed to parse Hugging Face config.json");
        println!(
            "[kriya-convert] Auto-compiled HF config '{}' -> {} layers, hidden_size={}, vocab={}",
            desc.model_name,
            desc.layers.len(),
            desc.hidden_size,
            desc.vocab_size
        );
    }

    let (Some(in_file), Some(out_file)) = (input_path, output_path) else {
        println!("Usage: kriya_convert --input <raw.safetensors> --output <q4_g64.safetensors> [--config <config.json>]");
        return;
    };

    let start = Instant::now();
    let archive = open_safetensors_file(&in_file).expect("Failed to open input .safetensors");
    let mut names: Vec<String> = archive.tensor_names().map(String::from).collect();
    names.sort();

    let mut output_tensors = Vec::with_capacity(names.len());
    let mut num_bf16_preserved = 0usize;
    let mut total_q4_blocks = 0usize;

    for name in names {
        let view = archive.get_tensor(&name).unwrap();
        let f32_vals: Vec<f32> = match view.descriptor.dtype {
            DType::F32 => bytemuck::cast_slice::<u8, f32>(view.data).to_vec(),
            DType::BF16 => bytemuck::cast_slice::<u8, u16>(view.data)
                .iter()
                .map(|&b| bf16_bits_to_f32(b))
                .collect(),
            DType::F16 => bytemuck::cast_slice::<u8, u16>(view.data)
                .iter()
                .map(|&h| f16_bits_to_f32(h))
                .collect(),
            _ => continue,
        };

        // Blocker 2 Fix: Keep 1D normalization scales (`*.norm.weight`) in exact BF16!
        if should_preserve_in_bf16(&name, &view.descriptor.shape) {
            let bf16_vals: Vec<u16> = f32_vals.iter().map(|&v| f32_to_bf16_bits(v)).collect();
            num_bf16_preserved += 1;
            output_tensors.push((
                name,
                view.descriptor.shape.clone(),
                SavedTensorPayload::Bf16(bf16_vals),
            ));
        } else {
            let mut padded = f32_vals;
            if !padded.len().is_multiple_of(64) {
                padded.resize(padded.len().div_ceil(64) * 64, 0.0);
            }
            let blocks = quantize_f32_to_q4_group64(&padded).unwrap();
            total_q4_blocks += blocks.len();
            output_tensors.push((
                name,
                view.descriptor.shape.clone(),
                SavedTensorPayload::Q4Group64(blocks),
            ));
        }
    }

    write_mixed_safetensors(&out_file, &output_tensors)
        .expect("Failed to write mixed-precision .safetensors archive");

    println!(
        "[kriya-convert] Saved {} tensors ({} BF16 norm tensors preserved, {} Q4_G64 blocks) to {:?} in {:.2} ms",
        output_tensors.len(),
        num_bf16_preserved,
        total_q4_blocks,
        out_file,
        start.elapsed().as_secs_f64() * 1000.0
    );
}
