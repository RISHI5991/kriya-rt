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

        let eos_token_id = ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "</s>"]
            .into_iter()
            .find_map(|t| inner.token_to_id(t))
            .unwrap_or(151_645);

        Ok(Self {
            inner,
            eos_token_id,
        })
    }

    pub fn format_chat_prompt(
        &self,
        messages: &[(String, String)],
        add_generation_prompt: bool,
    ) -> String {
        let is_llama3 = self.inner.token_to_id("<|start_header_id|>").is_some();
        let is_qwen = self.inner.token_to_id("<|im_start|>").is_some();
        let is_gemma = self.inner.token_to_id("<start_of_turn>").is_some();
        let is_mistral = self.inner.token_to_id("[INST]").is_some();

        let mut formatted = String::new();
        if is_llama3 {
            if self.inner.token_to_id("<|begin_of_text|>").is_some() {
                formatted.push_str("<|begin_of_text|>");
            }
            for (role, content) in messages {
                formatted.push_str(&format!(
                    "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                ));
            }
            if add_generation_prompt {
                formatted.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        } else if is_qwen {
            for (role, content) in messages {
                formatted.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            if add_generation_prompt {
                formatted.push_str("<|im_start|>assistant\n");
            }
        } else if is_gemma {
            if self.inner.token_to_id("<bos>").is_some() {
                formatted.push_str("<bos>");
            }
            for (role, content) in messages {
                let role_display = if role == "assistant" {
                    "model"
                } else {
                    role.as_str()
                };
                formatted.push_str(&format!(
                    "<start_of_turn>{role_display}\n{content}<end_of_turn>\n"
                ));
            }
            if add_generation_prompt {
                formatted.push_str("<start_of_turn>model\n");
            }
        } else if is_mistral {
            if self.inner.token_to_id("<s>").is_some() {
                formatted.push_str("<s>");
            }
            for (role, content) in messages {
                if role == "user" {
                    formatted.push_str(&format!("[INST] {content} [/INST]"));
                } else if role == "assistant" {
                    formatted.push_str(&format!(" {content}</s>"));
                } else {
                    formatted.push_str(&format!("[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"));
                }
            }
        } else {
            for (role, content) in messages {
                formatted.push_str(&format!("{role}: {content}\n"));
            }
            if add_generation_prompt {
                formatted.push_str("assistant: ");
            }
        }
        formatted
    }

    pub fn encode_chat_messages(
        &self,
        messages: &[(String, String)],
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        let formatted = self.format_chat_prompt(messages, add_generation_prompt);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_tokenizer(special_tokens: &[&str]) -> KriyaTokenizer {
        let vocab: serde_json::Value = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": special_tokens.iter().enumerate().map(|(id, t)| {
                serde_json::json!({
                    "id": id + 1,
                    "content": t,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                })
            }).collect::<Vec<_>>(),
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {
                    "hello": 100,
                    "world": 101,
                    "<|im_end|>": 151645
                },
                "merges": []
            }
        });

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(serde_json::to_string(&vocab).unwrap().as_bytes())
            .unwrap();
        KriyaTokenizer::from_file(file.path()).unwrap()
    }

    #[test]
    fn test_chatml_prompt_formatting() {
        let tok = create_test_tokenizer(&["<|im_start|>", "<|im_end|>"]);
        let messages = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello".to_string()),
        ];
        let formatted = tok.format_chat_prompt(&messages, true);
        assert_eq!(
            formatted,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!formatted.contains("elson"));
    }

    #[test]
    fn test_llama3_prompt_formatting() {
        let tok = create_test_tokenizer(&[
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ]);
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let formatted = tok.format_chat_prompt(&messages, true);
        assert_eq!(
            formatted,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn test_fallback_prompt_formatting() {
        let tok = create_test_tokenizer(&[]);
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let formatted = tok.format_chat_prompt(&messages, true);
        assert_eq!(formatted, "user: Hello\nassistant: ");
        assert!(!formatted.contains("elson"));
    }

    #[test]
    fn test_gemma_prompt_formatting() {
        let tok = create_test_tokenizer(&["<bos>", "<start_of_turn>", "<end_of_turn>"]);
        let messages = vec![
            ("user".to_string(), "Hello".to_string()),
            ("assistant".to_string(), "Hi!".to_string()),
        ];
        let formatted = tok.format_chat_prompt(&messages, true);
        assert_eq!(
            formatted,
            "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi!<end_of_turn>\n<start_of_turn>model\n"
        );
        assert!(!formatted.contains("elson"));
    }

    #[test]
    fn test_mistral_prompt_formatting() {
        let tok = create_test_tokenizer(&["<s>", "[INST]"]);
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let formatted = tok.format_chat_prompt(&messages, false);
        assert_eq!(formatted, "<s>[INST] Hello [/INST]");
        assert!(!formatted.contains("elson"));
    }
}
