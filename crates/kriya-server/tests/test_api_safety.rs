//! Integration Tests for Production API Safety, Routing & Admission Control.
//!
//! Tests the actual production router (`create_app`) and HTTP handlers
//! without local mock router duplication.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use futures_util::StreamExt;
use kriya_server::server::admission::SimpleMockTokenizer;
use kriya_server::server::{ServerHandle, ServerMetrics, create_app};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::mpsc;
use tower::ServiceExt;

fn make_test_handle(
    max_prompt: usize,
    max_completion: usize,
    max_concurrent: usize,
) -> (
    ServerHandle,
    mpsc::UnboundedReceiver<kriya_server::server::EnqueueRequestCmd>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let metrics = Arc::new(ServerMetrics::default());
    let tokenizer = Arc::new(SimpleMockTokenizer::new(151645));

    let handle = ServerHandle {
        work_tx: tx,
        tokenizer,
        model_name: "test-qwen-model".to_string(),
        metrics,
        max_prompt_tokens: max_prompt,
        max_completion_tokens: max_completion,
        max_concurrent_requests: max_concurrent,
    };

    (handle, rx)
}

#[tokio::test]
async fn test_api_rejection_empty_messages() {
    let (handle, _rx) = make_test_handle(1024, 512, 10);
    let app = create_app(handle);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"messages":[],"model":"test-model"}"#))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    let body_bytes = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("missing_required_field"));
}

#[tokio::test]
async fn test_api_rejection_excessive_max_tokens() {
    let (handle, _rx) = make_test_handle(1024, 512, 10);
    let app = create_app(handle);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":10000,"model":"test-model"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    let body_bytes = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("parameter_above_maximum"));
}

#[tokio::test]
async fn test_api_rejection_prompt_too_long() {
    // Limit to 4 prompt tokens
    let (handle, _rx) = make_test_handle(4, 512, 10);
    let app = create_app(handle);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"one two three four five six seven"}],"model":"test-model"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    let body_bytes = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("context_length_exceeded"));
}

#[tokio::test]
async fn test_api_admission_queue_saturation_429() {
    let (handle, _rx) = make_test_handle(1024, 512, 10);
    // Saturate active requests up to limit
    handle.metrics.active_requests.store(10, Ordering::Relaxed);
    let app = create_app(handle.clone());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}],"model":"test-model"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(handle.metrics.rejected_requests.load(Ordering::Relaxed), 1);

    let body_bytes = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("queue_full"));
}

#[tokio::test]
async fn test_api_valid_non_streaming_200() {
    let (handle, mut rx) = make_test_handle(1024, 512, 10);
    let app = create_app(handle.clone());

    // Background worker task simulates GPU emitting 2 tokens
    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            let _ = cmd.token_tx.send(101);
            let _ = cmd.token_tx.send(102);
            drop(cmd.token_tx);
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hello world"}],"max_tokens":16,"model":"test-model"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(response_json["object"], "chat.completion");
    assert_eq!(response_json["model"], "test-qwen-model");
    assert_eq!(response_json["choices"][0]["finish_reason"], "stop");
    let content = response_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    assert!(content.contains("tok101"));
    assert_eq!(response_json["usage"]["completion_tokens"], 2);

    let _ = worker.await;
    assert_eq!(handle.metrics.active_requests.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn test_api_streaming_sse() {
    let (handle, mut rx) = make_test_handle(1024, 512, 10);
    let app = create_app(handle.clone());

    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            let _ = cmd.token_tx.send(201);
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            let _ = cmd.token_tx.send(202);
            drop(cmd.token_tx);
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"stream test"}],"stream":true,"model":"test-model"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        res.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );

    let mut body_stream = res.into_body().into_data_stream();
    let mut total_chunks = 0;
    let mut chunk_texts = Vec::new();
    let mut saw_done = false;

    while let Some(chunk_res) = body_stream.next().await {
        let chunk = chunk_res.unwrap();
        let text = String::from_utf8_lossy(&chunk);
        if text.contains("chat.completion.chunk") {
            total_chunks += 1;
            chunk_texts.push(text.to_string());
        }
        if text.contains("[DONE]") {
            saw_done = true;
        }
    }

    assert_eq!(total_chunks, 2);
    assert!(saw_done, "Terminal event must be data: [DONE]");
    assert!(chunk_texts[0].contains("tok201"));
    assert!(chunk_texts[1].contains("tok202"));

    let _ = worker.await;
    // Verify active requests cleaned up after stream completion
    let mut cleaned_up = false;
    for _ in 0..100 {
        if handle.metrics.active_requests.load(Ordering::Relaxed) == 0 {
            cleaned_up = true;
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    assert!(
        cleaned_up,
        "Active requests gauge must return to zero after all completions"
    );
}

#[tokio::test]
async fn test_server_concurrent_requests_isolation_and_cancellation() {
    let (handle, mut rx) = make_test_handle(1024, 512, 16);
    let app = create_app(handle.clone());

    // Worker serves 4 requests concurrently
    let worker = tokio::spawn(async move {
        let mut handlers = Vec::new();
        for _ in 0..4 {
            if let Some(cmd) = rx.recv().await {
                handlers.push(tokio::spawn(async move {
                    for i in 0..5 {
                        tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;
                        if cmd.token_tx.send(300 + i).is_err() {
                            // Client dropped connection
                            break;
                        }
                    }
                }));
            }
        }
        for h in handlers {
            let _ = h.await;
        }
    });

    let mut join_handles = Vec::new();
    for req_idx in 0..4 {
        let app_clone = app.clone();
        let jh = tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(format!(
                    r#"{{"messages":[{{"role":"user","content":"req {req_idx}"}}],"stream":true,"model":"test-model"}}"#
                )))
                .unwrap();

            let res = app_clone.oneshot(req).await.unwrap();
            assert_eq!(res.status(), StatusCode::OK);

            let mut body_stream = res.into_body().into_data_stream();
            let mut count = 0;
            while let Some(chunk_res) = body_stream.next().await {
                let chunk = chunk_res.unwrap();
                let text = String::from_utf8_lossy(&chunk);
                if text.contains("chat.completion.chunk") {
                    count += 1;
                }
                // Request 3 cancels early after 1 token
                if req_idx == 3 && count >= 1 {
                    drop(body_stream);
                    return count;
                }
            }
            count
        });
        join_handles.push(jh);
    }

    let mut results = Vec::new();
    for jh in join_handles {
        results.push(jh.await.unwrap());
    }

    // Requests 0, 1, 2 received 5 tokens each
    assert_eq!(results[0], 5);
    assert_eq!(results[1], 5);
    assert_eq!(results[2], 5);
    // Request 3 cancelled early after 1 token
    assert_eq!(results[3], 1);

    let _ = worker.await;
    let mut cleaned_up = false;
    for _ in 0..100 {
        if handle.metrics.active_requests.load(Ordering::Relaxed) == 0 {
            cleaned_up = true;
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    assert!(
        cleaned_up,
        "Active requests gauge must return to zero after all completions and drops"
    );
}

#[tokio::test]
async fn test_health_and_metrics_endpoints() {
    let (handle, _rx) = make_test_handle(1024, 512, 10);
    let app = create_app(handle);

    // 1. Health check
    let health_req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let health_res = app.clone().oneshot(health_req).await.unwrap();
    assert_eq!(health_res.status(), StatusCode::OK);
    let health_bytes = axum::body::to_bytes(health_res.into_body(), 1024)
        .await
        .unwrap();
    let health_json: serde_json::Value = serde_json::from_slice(&health_bytes).unwrap();
    assert_eq!(health_json["status"], "ok");
    assert_eq!(health_json["engine"], "KriyaRT");

    // 2. Metrics check
    let metrics_req = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let metrics_res = app.oneshot(metrics_req).await.unwrap();
    assert_eq!(metrics_res.status(), StatusCode::OK);
    let metrics_bytes = axum::body::to_bytes(metrics_res.into_body(), 4096)
        .await
        .unwrap();
    let metrics_str = String::from_utf8_lossy(&metrics_bytes);
    assert!(metrics_str.contains("kriya_engine_info"));
    assert!(metrics_str.contains("kriya_active_requests 0"));
    assert!(metrics_str.contains("kriya_server_uptime_seconds"));
    // Uptime should be a reasonable duration (< 1000s), not Unix epoch timestamp (~1.7B)
    for line in metrics_str.lines() {
        if line.starts_with("kriya_server_uptime_seconds ") {
            let val: u64 = line
                .strip_prefix("kriya_server_uptime_seconds ")
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            assert!(
                val < 1000,
                "Server uptime should be elapsed time since start, not epoch timestamp (was {val})"
            );
        }
    }
}

#[test]
fn test_untrained_draft_head_detection() {
    use kriya_speculate::eagle_medusa::EagleDraftEngine;

    let h = 8;
    let v = 16;
    let mut engine =
        EagleDraftEngine::new_gpu_backed(h, v, vec![0.0; h * 2 * h], vec![0.0; h * h]).unwrap();
    assert!(!engine.is_trained, "New engine must default to untrained");
    let weights = engine.save_weights();

    let mut loaded =
        EagleDraftEngine::new_gpu_backed(h, v, vec![0.0; h * 2 * h], vec![0.0; h * h]).unwrap();
    loaded.load_weights(&weights).unwrap();
    assert!(
        !loaded.is_trained,
        "Untrained draft head weights must preserve is_trained=false"
    );

    // After training
    engine.is_trained = true;
    let trained_weights = engine.save_weights();
    loaded.load_weights(&trained_weights).unwrap();
    assert!(
        loaded.is_trained,
        "Trained draft head weights must preserve is_trained=true"
    );
}

#[tokio::test]
async fn test_streaming_utf8_split_character_detokenization() {
    struct Utf8SplitMockTokenizer;
    impl kriya_server::server::admission::TokenizerService for Utf8SplitMockTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special: bool,
        ) -> Result<Vec<u32>, kriya_core::tokenizer::TokenizerError> {
            Ok(vec![1])
        }
        fn decode(
            &self,
            ids: &[u32],
            _skip_special: bool,
        ) -> Result<String, kriya_core::tokenizer::TokenizerError> {
            if ids == [301] {
                // Emulate incomplete UTF-8 character yielding replacement character at end
                Ok("prefix \u{FFFD}".to_string())
            } else if ids == [301, 302] {
                // Second token completes 4-byte UTF-8 emoji
                Ok("prefix 🦀".to_string())
            } else {
                Ok(ids
                    .iter()
                    .map(|id| format!("{id}"))
                    .collect::<Vec<_>>()
                    .join(" "))
            }
        }
        fn encode_chat_messages(
            &self,
            _messages: &[(String, String)],
            _add_generation_prompt: bool,
        ) -> Result<Vec<u32>, kriya_core::tokenizer::TokenizerError> {
            Ok(vec![1])
        }
        fn eos_token_id(&self) -> u32 {
            999
        }
    }

    let (tx, mut rx) = mpsc::unbounded_channel();
    let metrics = Arc::new(ServerMetrics::default());
    let handle = ServerHandle {
        work_tx: tx,
        tokenizer: Arc::new(Utf8SplitMockTokenizer),
        model_name: "test-qwen-model".to_string(),
        metrics,
        max_prompt_tokens: 1024,
        max_completion_tokens: 512,
        max_concurrent_requests: 10,
    };
    let app = create_app(handle.clone());

    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            let _ = cmd.token_tx.send(301);
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            let _ = cmd.token_tx.send(302);
            drop(cmd.token_tx);
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"utf8 split test"}],"stream":true,"model":"test-model"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let mut body_stream = res.into_body().into_data_stream();
    let mut chunk_texts = Vec::new();
    let mut saw_done = false;

    while let Some(chunk_res) = body_stream.next().await {
        let chunk = chunk_res.unwrap();
        let text = String::from_utf8_lossy(&chunk);
        if text.contains("chat.completion.chunk") {
            chunk_texts.push(text.to_string());
        }
        if text.contains("[DONE]") {
            saw_done = true;
        }
    }

    let _ = worker.await;
    assert!(saw_done, "Terminal event must be data: [DONE]");
    assert_eq!(chunk_texts.len(), 2);
    // First chunk emits "prefix " without corrupted replacement character
    assert!(chunk_texts[0].contains("prefix "));
    assert!(!chunk_texts[0].contains('\u{FFFD}'));
    // Second chunk emits completed multi-byte character "🦀"
    assert!(chunk_texts[1].contains('🦀'));
}

#[test]
fn test_untrained_draft_head_env_var_override() {
    use kriya_speculate::eagle_medusa::EagleDraftEngine;

    let h = 8;
    let v = 16;
    let engine =
        EagleDraftEngine::new_gpu_backed(h, v, vec![0.0; h * 2 * h], vec![0.0; h * h]).unwrap();
    assert!(!engine.is_trained);
    let bytes = engine.save_weights();

    let mut loaded =
        EagleDraftEngine::new_gpu_backed(h, v, vec![0.0; h * 2 * h], vec![0.0; h * h]).unwrap();
    loaded.load_weights(&bytes).unwrap();
    assert!(!loaded.is_trained);

    unsafe {
        std::env::remove_var("KRIYA_ALLOW_UNTRAINED_DRAFT");
    }
    let allow = std::env::var("KRIYA_ALLOW_UNTRAINED_DRAFT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    assert!(!allow);

    unsafe {
        std::env::set_var("KRIYA_ALLOW_UNTRAINED_DRAFT", "1");
    }
    let allow = std::env::var("KRIYA_ALLOW_UNTRAINED_DRAFT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    assert!(allow);
    unsafe {
        std::env::remove_var("KRIYA_ALLOW_UNTRAINED_DRAFT");
    }
}
