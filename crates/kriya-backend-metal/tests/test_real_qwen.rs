use futures::executor::block_on;
use kriya_backend_metal::encoder::{MetalExecutor, SamplingConfig};
use kriya_core::quantizer::HuggingFaceModelConfig;
use kriya_core::tokenizer::KriyaTokenizer;
use kriya_kv::bsr_paged::PagedKvCacheManager;
use metal::Device;
use std::path::PathBuf;

#[test]
fn test_real_qwen_inference() {
    let Some(_device) = Device::system_default() else {
        return;
    };

    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models")
        .join("qwen2.5-0.5b-instruct-kriya-q4");

    if !model_dir.exists() {
        println!("Model dir does not exist, skipping");
        return;
    }

    let config_path = model_dir.join("config.json");
    let desc = HuggingFaceModelConfig::from_config_file(&config_path).unwrap();

    let kernels_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
        .join("metal");

    let tokenizer = KriyaTokenizer::from_checkpoint_dir(&model_dir).unwrap();
    let mut exec = MetalExecutor::new(&desc, &kernels_dir, Some(&model_dir), 512, 16).unwrap();

    let prompt = "The capital of France is";
    let input_ids = tokenizer.encode(prompt, false).unwrap();
    println!("Prompt: '{}' -> IDs: {:?}", prompt, input_ids);

    let mut kv_mgr = PagedKvCacheManager::new(512, 128, 16);
    let req_id = 42u64;
    kv_mgr.allocate_sequence(req_id, &input_ids).unwrap();

    let mut generated = Vec::new();
    let mut current_pos = 0usize;

    // Prefill: process prompt tokens one by one
    let mut last_tok = input_ids[0];
    for (i, &tok) in input_ids.iter().enumerate() {
        let plan = kv_mgr
            .build_prefix_schedule_plan(req_id, (i + 1) as u32, 128)
            .unwrap();
        let samp = SamplingConfig {
            vocab_size: desc.vocab_size,
            greedy: 1,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 50,
            rng_seed: 42,
        };
        last_tok =
            block_on(exec.forward_and_sample(&desc, &plan, &[tok], &[i as u32], samp)).unwrap();
        current_pos = i + 1;
    }

    generated.push(last_tok);

    // Decode 8 tokens
    for step in 0..8 {
        kv_mgr.append_token_slot(req_id).unwrap();
        let plan = kv_mgr
            .build_prefix_schedule_plan(req_id, (current_pos + 1) as u32, 128)
            .unwrap();
        let samp = SamplingConfig {
            vocab_size: desc.vocab_size,
            greedy: 1,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 50,
            rng_seed: 42 + step as u32,
        };
        let next_tok = block_on(exec.forward_and_sample(
            &desc,
            &plan,
            &[generated.last().copied().unwrap()],
            &[current_pos as u32],
            samp,
        ))
        .unwrap();
        generated.push(next_tok);
        current_pos += 1;
    }

    println!("Generated token IDs: {:?}", generated);
    let decoded = tokenizer.decode(&generated, false).unwrap();
    println!("Decoded output: '{}'", decoded);
}
