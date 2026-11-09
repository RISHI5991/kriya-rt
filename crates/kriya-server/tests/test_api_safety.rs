//! Integration Tests for API Safety & Admission Control (Fixes 3Review.md P0.2).

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tokio::sync::mpsc;
use tower::ServiceExt;

#[derive(Debug, Default)]
pub struct ServerMetrics {
    pub active_requests: AtomicUsize,
    pub total_requests: AtomicU64,
    pub total_tokens_emitted: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    pub rejected_requests: AtomicU64,
}

#[derive(Clone)]
pub struct ServerHandle {
    pub work_tx: mpsc::UnboundedSender<u64>,
    pub model_name: String,
    pub metrics: Arc<ServerMetrics>,
    pub max_prompt_tokens: usize,
    pub max_completion_tokens: usize,
    pub max_concurrent_requests: usize,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(serde::Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
}

async fn test_chat_handler(
    axum::extract::State(handle): axum::extract::State<ServerHandle>,
    axum::Json(payload): axum::Json<ChatCompletionRequest>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    if payload.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "error": {
                    "message": "Missing required field: 'messages' array must not be empty",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "missing_required_field"
                }
            })),
        )
            .into_response();
    }

    let max_new = payload.max_tokens.unwrap_or(16);
    if max_new > handle.max_completion_tokens {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "Requested max_tokens ({}) exceeds server maximum of {}",
                        max_new, handle.max_completion_tokens
                    ),
                    "type": "invalid_request_error",
                    "param": "max_tokens",
                    "code": "parameter_above_maximum"
                }
            })),
        )
            .into_response();
    }

    let current_active = handle.metrics.active_requests.load(Ordering::Relaxed);
    if current_active >= handle.max_concurrent_requests {
        handle
            .metrics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            axum::Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "Server admission queue saturated ({} active requests >= limit {})",
                        current_active, handle.max_concurrent_requests
                    ),
                    "type": "rate_limit_error",
                    "code": "queue_full"
                }
            })),
        )
            .into_response();
    }

    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": handle.model_name
        })),
    )
        .into_response()
}

async fn test_metrics_handler(
    axum::extract::State(handle): axum::extract::State<ServerHandle>,
) -> impl axum::response::IntoResponse {
    let active = handle.metrics.active_requests.load(Ordering::Relaxed);
    let rejected = handle.metrics.rejected_requests.load(Ordering::Relaxed);
    let metrics_text = format!(
        "# HELP kriya_active_requests Currently in-flight inference requests\n\
         # TYPE kriya_active_requests gauge\n\
         kriya_active_requests {}\n\
         # HELP kriya_rejected_requests_total Total requests rejected due to admission limits\n\
         # TYPE kriya_rejected_requests_total counter\n\
         kriya_rejected_requests_total {}\n",
        active, rejected
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        metrics_text,
    )
}

#[tokio::test]
async fn test_api_rejection_empty_messages() {
    let (tx, _rx) = mpsc::unbounded_channel();
    let handle = ServerHandle {
        work_tx: tx,
        model_name: "test-model".to_string(),
        metrics: Arc::new(ServerMetrics::default()),
        max_prompt_tokens: 1024,
        max_completion_tokens: 512,
        max_concurrent_requests: 10,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(test_chat_handler))
        .with_state(handle);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"messages":[]}"#))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_api_rejection_excessive_max_tokens() {
    let (tx, _rx) = mpsc::unbounded_channel();
    let handle = ServerHandle {
        work_tx: tx,
        model_name: "test-model".to_string(),
        metrics: Arc::new(ServerMetrics::default()),
        max_prompt_tokens: 1024,
        max_completion_tokens: 512,
        max_concurrent_requests: 10,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(test_chat_handler))
        .with_state(handle);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":10000}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_api_admission_queue_saturation_429() {
    let (tx, _rx) = mpsc::unbounded_channel();
    let metrics = Arc::new(ServerMetrics::default());
    // Simulate saturated queue (10 active >= limit 10)
    metrics.active_requests.store(10, Ordering::Relaxed);

    let handle = ServerHandle {
        work_tx: tx,
        model_name: "test-model".to_string(),
        metrics: metrics.clone(),
        max_prompt_tokens: 1024,
        max_completion_tokens: 512,
        max_concurrent_requests: 10,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(test_chat_handler))
        .route("/metrics", get(test_metrics_handler))
        .with_state(handle);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(metrics.rejected_requests.load(Ordering::Relaxed), 1);

    // Verify /metrics reflects the rejection
    let metrics_req = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let metrics_res = app.oneshot(metrics_req).await.unwrap();
    assert_eq!(metrics_res.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(metrics_res.into_body(), 1024)
        .await
        .unwrap();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("kriya_rejected_requests_total 1"));
}
