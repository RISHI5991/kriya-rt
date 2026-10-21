//! Complete Multi-Shard Hugging Face Hub Model Fetcher for KriyaRT (Closes Blocker 1).

use hf_hub::api::tokio::Api;
use kriya_core::safetensors_loader::SafetensorsIndexJson;
use std::collections::BTreeSet;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HubError {
    #[error("Hugging Face API download error: {0}")]
    Api(String),
    #[error("Failed to read or parse model.safetensors.index.json: {0}")]
    IndexParse(String),
}

/// Downloads `config.json`, `tokenizer.json`, and either single-file `model.safetensors`
/// or ALL referenced shard files from `model.safetensors.index.json`.
pub async fn fetch_hf_model_files(repo_id: &str) -> Result<PathBuf, HubError> {
    let api = Api::new().map_err(|e| HubError::Api(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    let config_path = repo
        .get("config.json")
        .await
        .map_err(|e| HubError::Api(format!("config.json: {e}")))?;

    let _tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .map_err(|e| HubError::Api(format!("tokenizer.json: {e}")))?;

    // 1. Try single-file checkpoint first
    if let Ok(single_weights) = repo.get("model.safetensors").await {
        return Ok(single_weights
            .parent()
            .unwrap_or(config_path.parent().unwrap())
            .to_path_buf());
    }

    // 2. Multi-shard checkpoint: fetch `model.safetensors.index.json` and download every unique shard
    let index_path = repo
        .get("model.safetensors.index.json")
        .await
        .map_err(|e| HubError::Api(format!("model.safetensors.index.json: {e}")))?;

    let index_bytes = std::fs::read(&index_path)
        .map_err(|e| HubError::IndexParse(format!("IO error reading index: {e}")))?;

    let index_json: SafetensorsIndexJson = serde_json::from_slice(&index_bytes)
        .map_err(|e| HubError::IndexParse(format!("Invalid JSON in index: {e}")))?;

    let unique_shards: BTreeSet<String> = index_json.weight_map.into_values().collect();

    println!(
        "[KriyaRT Hub] Detected sharded checkpoint with {} shard files. Downloading...",
        unique_shards.len()
    );

    for (idx, shard_filename) in unique_shards.iter().enumerate() {
        println!(
            "[KriyaRT Hub] Fetching shard [{}/{}]: {}",
            idx + 1,
            unique_shards.len(),
            shard_filename
        );
        repo.get(shard_filename)
            .await
            .map_err(|e| HubError::Api(format!("Shard '{shard_filename}': {e}")))?;
    }

    Ok(index_path.parent().unwrap().to_path_buf())
}
