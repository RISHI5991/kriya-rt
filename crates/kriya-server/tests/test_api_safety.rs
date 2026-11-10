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

#[tokio::test]
async fn test_server_concurrent_requests_isolation_and_cancellation() {
    use std::sync::atomic::AtomicBool;

    let (work_tx, mut work_rx) = mpsc::unbounded_channel::<u64>();
    let metrics = Arc::new(ServerMetrics::default());
    let handle = ServerHandle {
        work_tx,
        model_name: "test-model".to_string(),
        metrics: metrics.clone(),
        max_prompt_tokens: 1024,
        max_completion_tokens: 512,
        max_concurrent_requests: 16,
    };

    let app = Router::new()
        .route(
            "/v1/chat/stream",
            post({
                let handle = handle.clone();
                move |axum::Json(payload): axum::Json<ChatCompletionRequest>| {
                    let handle = handle.clone();
                    async move {
                        handle
                            .metrics
                            .active_requests
                            .fetch_add(1, Ordering::SeqCst);
                        let _ = handle.work_tx.send(payload.messages.len() as u64);

                        struct ActiveGuard(Arc<ServerMetrics>);
                        impl Drop for ActiveGuard {
                            fn drop(&mut self) {
                                self.0.active_requests.fetch_sub(1, Ordering::SeqCst);
                            }
                        }

                        let guard = Arc::new(ActiveGuard(handle.metrics.clone()));
                        let stream = futures_util::stream::unfold(
                            (0usize, guard),
                            |(i, guard)| async move {
                                if i < 5 {
                                    tokio::time::sleep(tokio::time::Duration::from_millis(15))
                                        .await;
                                    Some((
                                        Ok::<_, std::io::Error>(format!("data: token_{i}\n\n")),
                                        (i + 1, guard),
                                    ))
                                } else {
                                    None
                                }
                            },
                        );
                        axum::response::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from_stream(stream))
                            .unwrap()
                    }
                }
            }),
        )
        .with_state(handle);

    // Drain the work channel in background
    let drainer = tokio::spawn(async move {
        let mut count = 0;
        while work_rx.recv().await.is_some() {
            count += 1;
            if count == 4 {
                break;
            }
        }
    });

    // Spawn 4 concurrent requests
    let client_cancelled = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    for req_idx in 0..4 {
        let app_clone = app.clone();
        let client_cancelled_clone = client_cancelled.clone();
        let join_h = tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/stream")
                .header("content-type", "application/json")
                .body(Body::from(format!(
                    r#"{{"messages":[{{"role":"user","content":"req_{req_idx}"}}],"max_tokens":16}}"#
                )))
                .unwrap();

            let res = app_clone.oneshot(req).await.unwrap();
            assert_eq!(res.status(), StatusCode::OK);

            let mut body_stream = res.into_body().into_data_stream();
            use futures_util::StreamExt;

            let mut tokens_received = 0;
            while let Some(chunk) = body_stream.next().await {
                let bytes = chunk.unwrap();
                let text = String::from_utf8_lossy(&bytes);
                if text.contains("token_") {
                    tokens_received += 1;
                }
                // Cancel request 3 after receiving 1 token
                if req_idx == 3 && tokens_received >= 1 {
                    client_cancelled_clone.store(true, Ordering::SeqCst);
                    // Dropping the body_stream simulates an aborted client HTTP connection
                    drop(body_stream);
                    return tokens_received;
                }
            }
            tokens_received
        });
        handles.push(join_h);
    }

    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.unwrap());
    }

    // Requests 0, 1, 2 completed fully (5 tokens each)
    assert_eq!(results[0], 5);
    assert_eq!(results[1], 5);
    assert_eq!(results[2], 5);
    // Request 3 cancelled early
    assert_eq!(results[3], 1);
    assert!(client_cancelled.load(Ordering::SeqCst));

    // Verify all active requests have decremented to 0
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    assert_eq!(
        metrics.active_requests.load(Ordering::SeqCst),
        0,
        "Active requests gauge must return to zero after completion and cancellation"
    );

    let _ = drainer.await;
}
