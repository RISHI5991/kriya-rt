//! Admission control, request queues, and concurrency handles.

use crate::server::metrics::ServerMetrics;
use kriya_core::tokenizer::{KriyaTokenizer, TokenizerError};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::mpsc;

/// State maintained for each concurrently running request stream.
#[derive(Clone)]
pub struct ActiveStreamState {
    pub tx: mpsc::UnboundedSender<u32>,
    pub temperature: f32,
    pub top_p: f32,
    pub start_pos: usize,
}

/// Command sent from non-blocking HTTP/SSE handlers to the autonomous ORCA GPU Actor.
pub struct EnqueueRequestCmd {
    pub req_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub token_tx: mpsc::UnboundedSender<u32>,
}

/// Abstraction for tokenizer operations to allow deterministic test mocks
/// without loading full model directories.
pub trait TokenizerService: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError>;
    fn encode_chat_messages(
        &self,
        messages: &[(String, String)],
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, TokenizerError>;
    fn eos_token_id(&self) -> u32;
}

impl TokenizerService for KriyaTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        KriyaTokenizer::encode(self, text, add_special_tokens)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        KriyaTokenizer::decode(self, ids, skip_special_tokens)
    }

    fn encode_chat_messages(
        &self,
        messages: &[(String, String)],
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        KriyaTokenizer::encode_chat_messages(self, messages, add_generation_prompt)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

/// Deterministic mock tokenizer for fast unit and integration tests.
#[derive(Clone, Default)]
pub struct SimpleMockTokenizer {
    pub eos_id: u32,
}

impl SimpleMockTokenizer {
    pub fn new(eos_id: u32) -> Self {
        Self { eos_id }
    }
}

impl TokenizerService for SimpleMockTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        if text.is_empty() {
            Ok(vec![])
        } else {
            Ok(text.split_whitespace().map(|_| 1u32).collect())
        }
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String, TokenizerError> {
        let text = ids
            .iter()
            .map(|id| format!("tok{id}"))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(text)
    }

    fn encode_chat_messages(
        &self,
        messages: &[(String, String)],
        _add_generation_prompt: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        let mut tokens = Vec::new();
        for (_, content) in messages {
            tokens.extend(content.split_whitespace().map(|_| 1u32));
        }
        if tokens.is_empty() {
            tokens.push(1);
        }
        Ok(tokens)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_id
    }
}

/// RAII guard that decrements active requests on drop (e.g. on client disconnect or completion).
pub struct ActiveGuard(pub Arc<ServerMetrics>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Shared lock-free handle held by Axum HTTP routes.
#[derive(Clone)]
pub struct ServerHandle {
    pub work_tx: mpsc::UnboundedSender<EnqueueRequestCmd>,
    pub tokenizer: Arc<dyn TokenizerService>,
    pub model_name: String,
    pub metrics: Arc<ServerMetrics>,
    pub max_prompt_tokens: usize,
    pub max_completion_tokens: usize,
    pub max_concurrent_requests: usize,
}
