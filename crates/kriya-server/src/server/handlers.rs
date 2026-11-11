//! HTTP Handlers for Health, Metrics, and OpenAI-Compatible Chat Completions.

use crate::server::admission::{ActiveGuard, EnqueueRequestCmd, ServerHandle};
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
    let uptime = current_timestamp_us() / 1_000_000;
    let active = handle.metrics.active_requests.load(Ordering::Relaxed);
    let total_reqs = handle.metrics.total_requests.load(Ordering::Relaxed);
    let total_emitted = handle.metrics.total_tokens_emitted.load(Ordering::Relaxed);
    let total_prompt = handle.metrics.total_prompt_tokens.load(Ordering::Relaxed);
    let rejected = handle.metrics.rejected_requests.load(Ordering::Relaxed);
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
         kriya_pipeline_depth2_active 1\n",
        uptime, active, total_reqs, total_prompt, total_emitted, rejected
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

    // 3. Admission Control: Check queue saturation
    let current_active = handle.metrics.active_requests.load(Ordering::Relaxed);
    if current_active >= handle.max_concurrent_requests {
        handle
            .metrics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        return api_error(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "Server admission queue saturated ({current_active} active requests >= limit {})",
                handle.max_concurrent_requests
            ),
            "rate_limit_error",
            Some("queue_full"),
            None,
        );
    }

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
        .active_requests
        .fetch_add(1, Ordering::Relaxed);
    handle
        .metrics
        .total_requests
        .fetch_add(1, Ordering::Relaxed);
    handle
        .metrics
        .total_prompt_tokens
        .fetch_add(prompt_len as u64, Ordering::Relaxed);

    let _guard = ActiveGuard(handle.metrics.clone());

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
        let sse_stream = stream::unfold(
            (token_rx, tok_clone, req_id, metrics_clone, _guard),
            |(mut rx, tokenizer_opt, id, metrics, guard)| async move {
                let tok_id = rx.recv().await?;
                metrics.total_tokens_emitted.fetch_add(1, Ordering::Relaxed);
                let piece = tokenizer_opt
                    .decode(&[tok_id], true)
                    .unwrap_or_else(|_| format!("{tok_id} "));
                let chunk_json = format!(
                    r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":{:?}}},"finish_reason":null}}]}}"#,
                    id, piece
                );
                Some((
                    Ok::<Event, std::convert::Infallible>(Event::default().data(chunk_json)),
                    (rx, tokenizer_opt, id, metrics, guard),
                ))
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
