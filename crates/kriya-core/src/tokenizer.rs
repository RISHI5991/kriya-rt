//! Hugging Face BPE tokenizer wrapper for KriyaRT.
//!
//! Provides a thin wrapper around the `tokenizers` crate for loading
//! `tokenizer.json` files from model checkpoint directories, encoding
//! chat messages, and decoding token IDs back to text.

use std::path::Path;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Failed to load Hugging Face tokenizer.json: {0}")]
    LoadError(String),
    #[error("Token encoding failed: {0}")]
    EncodeError(String),
    #[error("Token decoding failed: {0}")]
    DecodeError(String),
}

pub struct KriyaTokenizer {
    inner: Tokenizer,
    pub eos_token_id: u32,
}

impl KriyaTokenizer {
    pub fn from_checkpoint_dir<P: AsRef<Path>>(dir: P) -> Result<Self, TokenizerError> {
        let tokenizer_path = dir.as_ref().join("tokenizer.json");
        Self::from_file(tokenizer_path)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let inner =
            Tokenizer::from_file(path).map_err(|e| TokenizerError::LoadError(e.to_string()))?;

        let eos_token_id = inner
            .token_to_id("<|im_end|>")
            .or_else(|| inner.token_to_id("<|eot_id|>"))
            .or_else(|| inner.token_to_id("<|endoftext|>"))
            .or_else(|| inner.token_to_id("</s>"))
            .unwrap_or(151_645);

        Ok(Self {
            inner,
            eos_token_id,
        })
    }

    pub fn encode_chat_messages(
        &self,
        messages: &[(String, String)],
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        let mut formatted = String::new();
        for (role, content) in messages {
            formatted.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        if add_generation_prompt {
            formatted.push_str("<|im_start|>assistant\n");
        }
        self.encode(&formatted, false)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::EncodeError(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| TokenizerError::DecodeError(e.to_string()))
    }
}
