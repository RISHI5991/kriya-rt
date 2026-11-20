//! HTTP Handlers for Health, Metrics, and OpenAI-Compatible Chat Completions.

use crate::server::admission::{ActiveGuard, EnqueueRequestCmd, ServerHandle, TokenizerService};
use crate::server::metrics::current_timestamp_us;
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: UsageStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({ "engine": "KriyaRT", "status": "ok" }))
}

pub async fn metrics_handler(State(handle): State<ServerHandle>) -> impl IntoResponse {
    let uptime = handle.metrics.start_time.elapsed().as_secs();
    let active = handle.metrics.active_requests.load(Ordering::Relaxed);
    let total_reqs = handle.metrics.total_requests.load(Ordering::Relaxed);
    let total_emitted = handle.metrics.total_tokens_emitted.load(Ordering::Relaxed);
    let total_prompt = handle.metrics.total_prompt_tokens.load(Ordering::Relaxed);
    let rejected = handle.metrics.rejected_requests.load(Ordering::Relaxed);
    let depth2_active = if handle
        .metrics
        .pipeline_depth2_active
        .load(Ordering::Relaxed)
    {
        1
    } else {
        0
    };
    let metrics_text = format!(
        "# HELP kriya_engine_info Engine metadata\n\
         # TYPE kriya_engine_info gauge\n\
         kriya_engine_info{{engine=\"KriyaRT\",arch=\"macOS-Metal\"}} 1\n\
         # HELP kriya_server_uptime_seconds Server uptime in seconds\n\
         # TYPE kriya_server_uptime_seconds counter\n\
         kriya_server_uptime_seconds {}\n\
         # HELP kriya_active_requests Currently in-flight inference requests\n\
         # TYPE kriya_active_requests gauge\n\
         kriya_active_requests {}\n\
         # HELP kriya_requests_total Total number of received inference requests\n\
         # TYPE kriya_requests_total counter\n\
         kriya_requests_total {}\n\
         # HELP kriya_prompt_tokens_total Total prompt tokens processed\n\
         # TYPE kriya_prompt_tokens_total counter\n\
         kriya_prompt_tokens_total {}\n\
         # HELP kriya_completion_tokens_total Total completion tokens emitted\n\
         # TYPE kriya_completion_tokens_total counter\n\
         kriya_completion_tokens_total {}\n\
         # HELP kriya_rejected_requests_total Total requests rejected due to admission limits\n\
         # TYPE kriya_rejected_requests_total counter\n\
         kriya_rejected_requests_total {}\n\
         # HELP kriya_pipeline_depth2_active Depth-2 Lookahead pipelining flag\n\
         # TYPE kriya_pipeline_depth2_active gauge\n\
         kriya_pipeline_depth2_active {}\n",
        uptime, active, total_reqs, total_prompt, total_emitted, rejected, depth2_active
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        metrics_text,
    )
}

#[inline]
fn api_error(
    status: StatusCode,
    message: impl Into<String>,
    err_type: &str,
    code: Option<&str>,
    param: Option<&str>,
) -> Response {
    let mut err = serde_json::json!({
        "message": message.into(),
        "type": err_type,
    });
    if let Some(c) = code {
        err["code"] = serde_json::Value::String(c.to_string());
    }
    if let Some(p) = param {
        err["param"] = serde_json::Value::String(p.to_string());
    }
    (status, Json(serde_json::json!({ "error": err }))).into_response()
}

/// StreamingDetokenizer provides O(1) sliding-window detokenization for SSE streaming.
///
/// It maintains a sliding window of recent tokens (typically 4..=6 tokens),
/// detecting and holding back incomplete multi-byte UTF-8 sequences (indicated by trailing U+FFFD),
/// suppressing window sliding while multi-byte characters are pending, and using UTF-8 char boundary
/// slicing to emit clean incremental string deltas.
#[derive(Debug, Clone)]
pub struct StreamingDetokenizer {
    tokens: Vec<u32>,
    window_emitted_bytes: usize,
    window_size: usize,
}

impl StreamingDetokenizer {
    /// Creates a new `StreamingDetokenizer` with the specified sliding window size.
    /// Recommended window size is 6 (clamped between 4 and 6).
    pub fn new(window_size: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(12),
            window_emitted_bytes: 0,
            window_size: window_size.clamp(4, 6),
        }
    }

    /// Number of tokens currently held in the sliding window.
    pub fn window_tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Configured sliding window size W.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Ingests a new token ID, returning the incremental UTF-8 string delta.
    pub fn step(&mut self, tok_id: u32, tokenizer: &dyn TokenizerService) -> String {
        self.tokens.push(tok_id);

        let raw_text = tokenizer.decode(&self.tokens, true).unwrap_or_default();
        let ends_with_replacement = raw_text.ends_with('\u{FFFD}');
        let valid_text = if ends_with_replacement {
            raw_text.trim_end_matches('\u{FFFD}')
        } else {
            &raw_text
        };

        let piece = if valid_text.len() > self.window_emitted_bytes {
            let safe_idx = if valid_text.is_char_boundary(self.window_emitted_bytes) {
                self.window_emitted_bytes
            } else {
                valid_text
                    .char_indices()
                    .map(|(i, _)| i)
                    .find(|&i| i >= self.window_emitted_bytes)
                    .unwrap_or(valid_text.len())
            };
            let delta = valid_text[safe_idx..].to_string();
            self.window_emitted_bytes = valid_text.len();
            delta
        } else {
            String::new()
        };

        // Slide window only when no incomplete UTF-8 sequence is pending,
        // so that multi-byte code points are never split across window boundaries.
        if !ends_with_replacement && self.tokens.len() > self.window_size {
            let drop_count = self.tokens.len() - self.window_size;
            self.tokens.drain(0..drop_count);
            let new_text = tokenizer.decode(&self.tokens, true).unwrap_or_default();
            let new_valid = if new_text.ends_with('\u{FFFD}') {
                new_text.trim_end_matches('\u{FFFD}')
            } else {
                &new_text
            };
            self.window_emitted_bytes = new_valid.len();
        }

        piece
    }

    /// Flushes any remaining un-emitted text on stream completion.
    pub fn flush(&mut self, tokenizer: &dyn TokenizerService) -> String {
        if self.tokens.is_empty() {
            return String::new();
        }
        let full_text = tokenizer.decode(&self.tokens, true).unwrap_or_default();
        if full_text.len() > self.window_emitted_bytes {
            let safe_idx = if full_text.is_char_boundary(self.window_emitted_bytes) {
                self.window_emitted_bytes
            } else {
                full_text
                    .char_indices()
                    .map(|(i, _)| i)
                    .find(|&i| i >= self.window_emitted_bytes)
                    .unwrap_or(full_text.len())
            };
            let piece = full_text[safe_idx..].to_string();
            self.window_emitted_bytes = full_text.len();
            piece
        } else {
            String::new()
        }
    }
}

pub async fn chat_completions_handler(
    State(handle): State<ServerHandle>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    // 1. Validation: Messages must not be empty
    if payload.messages.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "Missing required field: 'messages' array must not be empty",
            "invalid_request_error",
            Some("missing_required_field"),
            Some("messages"),
        );
    }

    // 2. Validation: Max completion tokens limit
    let max_new = payload.max_tokens.unwrap_or(16);
    if max_new > handle.max_completion_tokens {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "Requested max_tokens ({max_new}) exceeds server maximum of {}",
                handle.max_completion_tokens
            ),
            "invalid_request_error",
            Some("parameter_above_maximum"),
            Some("max_tokens"),
        );
    }

    // 3. Admission Control: Atomic compare-exchange ticket reservation to prevent TOCTOU race conditions
    let mut current = handle.metrics.active_requests.load(Ordering::Relaxed);
    loop {
        if current >= handle.max_concurrent_requests {
            handle
                .metrics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "Server admission queue saturated ({current} active requests >= limit {})",
                    handle.max_concurrent_requests
                ),
                "rate_limit_error",
                Some("queue_full"),
                None,
            );
        }
        match handle.metrics.active_requests.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }

    let _guard = ActiveGuard(handle.metrics.clone());

    let temperature = payload.temperature.unwrap_or(0.0);
    let top_p = payload.top_p.unwrap_or(0.95);
    let req_id: u64 = rand::random();

    let pairs: Vec<(String, String)> = payload
        .messages
        .into_iter()
        .map(|m| (m.role, m.content))
        .collect();
    let prompt_tokens = match handle.tokenizer.encode_chat_messages(&pairs, true) {
        Ok(t) if !t.is_empty() => t,
        _ => match handle
            .tokenizer
            .encode(pairs.last().map(|m| m.1.as_str()).unwrap_or(""), false)
        {
            Ok(t) => t,
            Err(e) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    format!("Failed to encode chat prompt tokens: {e}"),
                    "invalid_request_error",
                    Some("tokenization_failed"),
                    Some("messages"),
                );
            }
        },
    };

    let prompt_len = prompt_tokens.len();
    if prompt_len > handle.max_prompt_tokens {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "Prompt length ({prompt_len} tokens) exceeds server context limit of {}",
                handle.max_prompt_tokens
            ),
            "invalid_request_error",
            Some("context_length_exceeded"),
            Some("messages"),
        );
    }

    handle
        .metrics
        .total_requests
        .fetch_add(1, Ordering::Relaxed);
    handle
        .metrics
        .total_prompt_tokens
        .fetch_add(prompt_len as u64, Ordering::Relaxed);

    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    if handle
        .work_tx
        .send(EnqueueRequestCmd {
            req_id,
            prompt_tokens,
            max_new_tokens: max_new,
            temperature,
            top_p,
            token_tx,
        })
        .is_err()
    {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "KriyaRT inference worker is unavailable",
            "server_error",
            None,
            None,
        );
    }

    let tok_clone = handle.tokenizer.clone();
    let metrics_clone = handle.metrics.clone();
    if payload.stream.unwrap_or(false) {
        let detok = StreamingDetokenizer::new(6);
        let sse_stream = stream::unfold(
            (
                token_rx,
                tok_clone,
                req_id,
                metrics_clone,
                _guard,
                detok,
                false,
                false,
            ),
            |(mut rx, tokenizer_opt, id, metrics, guard, mut detok, mut flushed, finished)| async move {
                if finished {
                    return None;
                }
                match rx.recv().await {
                    Some(tok_id) => {
                        metrics.total_tokens_emitted.fetch_add(1, Ordering::Relaxed);
                        let new_piece = detok.step(tok_id, &*tokenizer_opt);
                        let chunk_json = format!(
                            r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":{:?}}},"finish_reason":null}}]}}"#,
                            id, new_piece
                        );
                        Some((
                            Ok::<Event, std::convert::Infallible>(
                                Event::default().data(chunk_json),
                            ),
                            (rx, tokenizer_opt, id, metrics, guard, detok, flushed, false),
                        ))
                    }
                    None => {
                        if !flushed {
                            flushed = true;
                            let piece = detok.flush(&*tokenizer_opt);
                            if !piece.is_empty() {
                                let chunk_json = format!(
                                    r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":{:?}}},"finish_reason":null}}]}}"#,
                                    id, piece
                                );
                                return Some((
                                    Ok::<Event, std::convert::Infallible>(
                                        Event::default().data(chunk_json),
                                    ),
                                    (rx, tokenizer_opt, id, metrics, guard, detok, flushed, false),
                                ));
                            }
                        }
                        Some((
                            Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]")),
                            (rx, tokenizer_opt, id, metrics, guard, detok, flushed, true),
                        ))
                    }
                }
            },
        );
        Sse::new(sse_stream).into_response()
    } else {
        let mut generated_ids = Vec::with_capacity(max_new);
        while let Some(tok_id) = token_rx.recv().await {
            handle
                .metrics
                .total_tokens_emitted
                .fetch_add(1, Ordering::Relaxed);
            generated_ids.push(tok_id);
        }
        let decoded_text = tok_clone.decode(&generated_ids, true).unwrap_or_default();
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{req_id}"),
            object: "chat.completion".to_string(),
            created: current_timestamp_us() / 1_000_000,
            model: handle.model_name.clone(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: decoded_text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: UsageStats {
                prompt_tokens: prompt_len,
                completion_tokens: generated_ids.len(),
                total_tokens: prompt_len + generated_ids.len(),
            },
        };
        Json(response).into_response()
    }
}
