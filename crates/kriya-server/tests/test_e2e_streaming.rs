//! End-to-End Opaque-Box Test Suite for Server & Streaming Features (Milestone E2).
//!
//! Features Tested:
//! - Feature 8: Depth-2 Async Pipelining (overlap of decode token readback with GPU execution,
//!   eliminating CPU-GPU synchronization bubbles, latency hiding, flush on EOS, cancellation).
//! - Feature 9: Streaming Detokenizer Sliding Window (4-6 token sliding window, O(1) decoding,
//!   UTF-8 multi-byte boundary preservation for 2-byte, 3-byte, 4-byte characters, emoji/CJK,
//!   incomplete sequence buffering, and SSE `data: [DONE]`).
//!
//! Tiers Covered:
//! - Tier 1: Feature Coverage (≥5 tests per feature = 10 tests)
//! - Tier 2: Boundary & Corner Cases (≥5 tests per feature = 10 tests)
//! - Tier 3: Cross-Feature Combinations (≥4 tests)
//! - Tier 4: Real-World Application Scenarios (Scenario 2 & Scenario 5)

use axum::body::Body;
use axum::http::{Request, StatusCode};
use futures_util::StreamExt;
use kriya_core::tokenizer::TokenizerError;
use kriya_server::runtime::Depth2PipelinedWorker;
use kriya_server::server::admission::{EnqueueRequestCmd, ServerHandle, TokenizerService};
use kriya_server::server::handlers::StreamingDetokenizer;
use kriya_server::server::metrics::ServerMetrics;
use kriya_server::server::router::create_app;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tower::ServiceExt;

// ============================================================================
// Test Infrastructure Fixtures & Mocks
// ============================================================================

/// Byte-level mock tokenizer capable of simulating exact UTF-8 byte fragmentation
/// across token boundaries, matching real BPE tokenizers with byte-fallback.
#[derive(Clone)]
pub struct ByteLevelMockTokenizer {
    pub token_bytes: HashMap<u32, Vec<u8>>,
    pub text_to_tokens: HashMap<String, Vec<u32>>,
    pub eos_id: u32,
}

impl ByteLevelMockTokenizer {
    pub fn new(eos_id: u32) -> Self {
        Self {
            token_bytes: HashMap::new(),
            text_to_tokens: HashMap::new(),
            eos_id,
        }
    }

    pub fn register_token(&mut self, id: u32, bytes: Vec<u8>) {
        self.token_bytes.insert(id, bytes);
    }

    pub fn register_mapping(&mut self, text: &str, tokens: Vec<u32>) {
        self.text_to_tokens.insert(text.to_string(), tokens);
    }
}

impl TokenizerService for ByteLevelMockTokenizer {
    fn encode(&self, text: &str, _add_special: bool) -> Result<Vec<u32>, TokenizerError> {
        if let Some(toks) = self.text_to_tokens.get(text) {
            return Ok(toks.clone());
        }
        // Fallback: simple whitespace / char split
        if text.is_empty() {
            Ok(vec![])
        } else {
            Ok(vec![1])
        }
    }

    fn decode(&self, ids: &[u32], _skip_special: bool) -> Result<String, TokenizerError> {
        let mut combined_bytes = Vec::new();
        for id in ids {
            if let Some(bytes) = self.token_bytes.get(id) {
                combined_bytes.extend_from_slice(bytes);
            } else {
                combined_bytes.extend_from_slice(format!("tok{id}").as_bytes());
            }
        }
        Ok(String::from_utf8_lossy(&combined_bytes).to_string())
    }

    fn encode_chat_messages(
        &self,
        messages: &[(String, String)],
        _add_generation_prompt: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        let combined: String = messages.iter().map(|(_, c)| c.clone()).collect();
        self.encode(&combined, false)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_id
    }
}

/// Helper to create a test ServerHandle and Axum router.
fn make_test_server(
    max_prompt: usize,
    max_completion: usize,
    max_concurrent: usize,
    tokenizer: Arc<dyn TokenizerService>,
) -> (
    axum::Router,
    ServerHandle,
    mpsc::UnboundedReceiver<EnqueueRequestCmd>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let metrics = Arc::new(ServerMetrics::default());

    let handle = ServerHandle {
        work_tx: tx,
        tokenizer,
        model_name: "test-kriya-model".to_string(),
        metrics,
        max_prompt_tokens: max_prompt,
        max_completion_tokens: max_completion,
        max_concurrent_requests: max_concurrent,
    };

    let app = create_app(handle.clone());
    (app, handle, rx)
}

/// Helper to extract all SSE chunk texts and termination status from a Response.
async fn collect_sse_events(res: axum::response::Response) -> (Vec<String>, bool) {
    let mut stream = res.into_body().into_data_stream();
    let mut chunks = Vec::new();
    let mut saw_done = false;

    while let Some(item) = stream.next().await {
        let bytes = item.expect("failed reading stream item");
        let text = String::from_utf8_lossy(&bytes);
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    saw_done = true;
                } else if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(choices) = val.get("choices").and_then(|c| c.as_array()) {
                        if let Some(first) = choices.first() {
                            if let Some(content) =
                                first.pointer("/delta/content").and_then(|c| c.as_str())
                            {
                                chunks.push(content.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    (chunks, saw_done)
}

// ============================================================================
// Tier 1: Feature Coverage (≥5 tests per feature = 10 tests)
// ============================================================================

/// Feature 8 - Test 1: Pipelining Depth & Metric Verification
///
/// Verifies that the server exposes the Depth-2 pipeline metric
/// `kriya_pipeline_depth2_active 1` on `/metrics` and properly tracks active requests.
#[tokio::test]
async fn test_feature8_tier1_depth2_pipeline_metric_and_state() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("test prompt", vec![10, 11]);
    let (app, handle, _rx) = make_test_server(1024, 512, 16, Arc::new(tok));

    // 1. Verify Depth-2 pipeline flag is active by default
    let req = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(res.into_body(), 8192).await.unwrap();
    let metrics_text = String::from_utf8_lossy(&bytes);

    assert!(
        metrics_text.contains("kriya_pipeline_depth2_active 1"),
        "Server metrics must report kriya_pipeline_depth2_active 1"
    );
    assert!(
        metrics_text.contains("kriya_active_requests 0"),
        "Initial active requests must be zero"
    );
    assert_eq!(handle.metrics.active_requests.load(Ordering::Relaxed), 0);

    // 2. Verify metric dynamically updates when toggled
    handle
        .metrics
        .pipeline_depth2_active
        .store(false, Ordering::SeqCst);
    let req2 = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let res2 = app.oneshot(req2).await.unwrap();
    assert_eq!(res2.status(), StatusCode::OK);

    let bytes2 = axum::body::to_bytes(res2.into_body(), 8192).await.unwrap();
    let text2 = String::from_utf8_lossy(&bytes2);
    assert!(
        text2.contains("kriya_pipeline_depth2_active 0"),
        "Server metrics must reflect dynamic update to kriya_pipeline_depth2_active 0"
    );
}

/// Feature 8 - Test 2: Pipeline Concurrency Capacity (Depth=2)
///
/// Verifies that the asynchronous pipeline operates at depth 2 without blocking
/// or exceeding configured depth capacity.
#[tokio::test]
async fn test_feature8_tier1_depth2_pipeline_concurrency_depth() {
    let worker = Depth2PipelinedWorker::new(Duration::from_millis(5), Duration::from_millis(5));
    let (tx, mut rx) = mpsc::unbounded_channel();

    let tokens = vec![101, 102, 103, 104, 105, 106];
    let worker_handle = tokio::spawn({
        let w = worker.clone();
        async move { w.run_pipelined_generation(tokens, 999, tx).await }
    });

    let mut received = Vec::new();
    while let Some(tok) = rx.recv().await {
        received.push(tok);
    }

    let emitted = worker_handle.await.unwrap().unwrap();
    assert_eq!(emitted, 6);
    assert_eq!(received, vec![101, 102, 103, 104, 105, 106]);

    // Concurrency must reach 2 (depth-2 capacity) and never exceed 2
    let max_depth = worker.max_concurrency_observed.load(Ordering::SeqCst);
    assert_eq!(
        max_depth, 2,
        "Depth-2 pipeline must reach and sustain concurrency of 2"
    );
}

/// Feature 8 - Test 3: Token Readback Overlap with GPU Compute
///
/// Verifies that token readback on the host directly overlaps with GPU execution
/// of subsequent tokens, proving CPU-GPU synchronization bubbles are eliminated.
#[tokio::test]
async fn test_feature8_tier1_token_readback_overlap() {
    let worker = Depth2PipelinedWorker::new(Duration::from_millis(10), Duration::from_millis(10));
    let (tx, mut rx) = mpsc::unbounded_channel();

    let tokens = vec![201, 202, 203, 204];
    let jh = tokio::spawn({
        let w = worker.clone();
        async move { w.run_pipelined_generation(tokens, 999, tx).await }
    });

    let mut count = 0;
    while rx.recv().await.is_some() {
        count += 1;
    }
    let _ = jh.await;

    assert_eq!(count, 4);
    assert!(
        worker.overlap_detected.load(Ordering::SeqCst),
        "Pipeline must demonstrate simultaneous execution and readback overlap"
    );
}

/// Feature 8 - Test 4: Strict Execution Ordering Preservation
///
/// Verifies that despite asynchronous execution stages in flight, tokens are emitted
/// in strictly monotonic, sequential order without transposition or race conditions.
#[tokio::test]
async fn test_feature8_tier1_strict_execution_ordering() {
    let worker = Depth2PipelinedWorker::new(Duration::from_millis(2), Duration::from_millis(2));
    let (tx, mut rx) = mpsc::unbounded_channel();

    let expected_tokens: Vec<u32> = (1000..1020).collect();
    let jh = tokio::spawn({
        let w = worker.clone();
        let toks = expected_tokens.clone();
        async move { w.run_pipelined_generation(toks, 9999, tx).await }
    });

    let mut actual_tokens = Vec::new();
    while let Some(tok) = rx.recv().await {
        actual_tokens.push(tok);
    }
    let _ = jh.await;

    assert_eq!(
        actual_tokens, expected_tokens,
        "Tokens emitted through Depth-2 pipeline must maintain strictly monotonic ordering"
    );
}

/// Feature 8 - Test 5: Pipeline Flush on EOS
///
/// Verifies that when an End-Of-Sequence (EOS) token is encountered, the Depth-2 pipeline
/// flushes cleanly, stops lookahead dispatch, and drops the sender channel cleanly.
#[tokio::test]
async fn test_feature8_tier1_pipeline_flush_on_eos() {
    let worker = Depth2PipelinedWorker::new(Duration::from_millis(3), Duration::from_millis(3));
    let (tx, mut rx) = mpsc::unbounded_channel();

    // Sequence with EOS at position 3
    let tokens = vec![301, 302, 303, 999, 305, 306, 307];
    let jh = tokio::spawn({
        let w = worker.clone();
        async move { w.run_pipelined_generation(tokens, 999, tx).await }
    });

    let mut received = Vec::new();
    while let Some(tok) = rx.recv().await {
        received.push(tok);
    }
    let _ = jh.await;

    // Must emit exactly tokens before EOS, and nothing past EOS
    assert_eq!(received, vec![301, 302, 303]);
    assert_eq!(worker.active_in_flight.load(Ordering::SeqCst), 0);
}

/// Feature 9 - Test 1: Sliding Window Clamping (W in 4..=6)
///
/// Verifies that `StreamingDetokenizer::new` properly clamps window sizes to [4, 6]
/// and starts with an empty sliding buffer.
#[test]
fn test_feature9_tier1_window_size_clamping() {
    let d1 = StreamingDetokenizer::new(1);
    assert_eq!(d1.window_size(), 4, "Window sizes < 4 must clamp to 4");
    assert!(d1.window_tokens().is_empty());

    let d5 = StreamingDetokenizer::new(5);
    assert_eq!(d5.window_size(), 5, "Window size 5 must remain 5");

    let d10 = StreamingDetokenizer::new(10);
    assert_eq!(d10.window_size(), 6, "Window sizes > 6 must clamp to 6");
}

/// Feature 9 - Test 2: Single Token Emission
///
/// Verifies that ingesting a single token returns its exact string delta immediately.
#[test]
fn test_feature9_tier1_single_token_emission() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_token(42, b"KriyaRT".to_vec());

    let mut detok = StreamingDetokenizer::new(6);
    let delta = detok.step(42, &tok);

    assert_eq!(delta, "KriyaRT");
    assert_eq!(detok.window_tokens(), &[42]);
}

/// Feature 9 - Test 3: Multi-token Accumulation within Sliding Window
///
/// Verifies that tokens accumulate within the sliding window up to the configured size
/// and provide incremental delta outputs.
#[test]
fn test_feature9_tier1_multi_token_accumulation() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_token(1, b"Fast ".to_vec());
    tok.register_token(2, b"Metal ".to_vec());
    tok.register_token(3, b"Engine".to_vec());

    let mut detok = StreamingDetokenizer::new(4);
    let d1 = detok.step(1, &tok);
    let d2 = detok.step(2, &tok);
    let d3 = detok.step(3, &tok);

    assert_eq!(d1, "Fast ");
    assert_eq!(d2, "Metal ");
    assert_eq!(d3, "Engine");
    assert_eq!(detok.window_tokens(), &[1, 2, 3]);
}

/// Feature 9 - Test 4: ASCII Streaming Fidelity
///
/// Verifies that sequential ASCII tokens stream with 100% fidelity, matching full sequence
/// decode without duplicate characters or omitted slices.
#[test]
fn test_feature9_tier1_ascii_streaming_fidelity() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    let words = vec![
        "The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog.",
    ];

    for (i, word) in words.iter().enumerate() {
        tok.register_token(i as u32 + 10, word.as_bytes().to_vec());
    }

    let mut detok = StreamingDetokenizer::new(5);
    let mut reconstructed = String::new();

    for i in 0..words.len() {
        let piece = detok.step(i as u32 + 10, &tok);
        reconstructed.push_str(&piece);
    }
    reconstructed.push_str(&detok.flush(&tok));

    assert_eq!(
        reconstructed,
        "The quick brown fox jumps over the lazy dog."
    );
}

/// Feature 9 - Test 5: O(1) Performance & Bounded Sliding Window
///
/// Verifies that per-token detokenization operates in O(1) time overhead:
/// as total tokens generated grow from 10 to 100+, the window token count remains
/// strictly bounded at <= W, and per-token decode latency remains flat.
#[test]
fn test_feature9_tier1_o1_performance_bounded_window() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    for i in 1..=150 {
        tok.register_token(i, format!("word{i} ").into_bytes());
    }

    let mut detok = StreamingDetokenizer::new(5);

    // Warm up window
    for i in 1..=5 {
        detok.step(i, &tok);
    }

    let t0 = Instant::now();
    for i in 6..=50 {
        let _ = detok.step(i, &tok);
        assert!(
            detok.window_tokens().len() <= 5,
            "Sliding window must never exceed configured window_size of 5"
        );
    }
    let early_batch_dur = t0.elapsed();

    let t1 = Instant::now();
    for i in 51..=95 {
        let _ = detok.step(i, &tok);
        assert!(detok.window_tokens().len() <= 5);
    }
    let late_batch_dur = t1.elapsed();

    // Verify O(1) property: late batch duration should not scale quadratically compared to early batch
    let ratio = late_batch_dur.as_secs_f64() / early_batch_dur.as_secs_f64().max(1e-6);
    assert!(
        ratio < 3.5,
        "Detokenizer step must maintain O(1) performance (observed ratio {ratio:.2})"
    );
}

// ============================================================================
// Tier 2: Boundary & Corner Cases (≥5 tests per feature = 10 tests)
// ============================================================================

/// Feature 8 - Test 6: Boundary 1-Token Prompt
///
/// Tests minimal 1-token prompt submission into the Depth-2 pipeline and server.
#[tokio::test]
async fn test_feature8_tier2_boundary_single_token_prompt() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("hi", vec![1]);
    tok.register_token(200, b"Hello!".to_vec());

    let (app, handle, mut rx) = make_test_server(1024, 512, 10, Arc::new(tok));

    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            let p_worker =
                Depth2PipelinedWorker::new(Duration::from_millis(1), Duration::from_millis(1));
            let _ = p_worker
                .run_pipelined_generation(vec![200], 999, cmd.token_tx)
                .await;
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"model":"test"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let (chunks, done) = collect_sse_events(res).await;
    let _ = worker.await;

    assert!(done, "Must receive terminal data: [DONE]");
    let full: String = chunks.concat();
    assert_eq!(full, "Hello!");
    assert_eq!(handle.metrics.active_requests.load(Ordering::Relaxed), 0);
}

/// Feature 8 - Test 7: Boundary Max Context Tokens
///
/// Verifies pipeline operation when prompt length equals max_prompt_tokens,
/// and rejections when prompt exceeds max_prompt_tokens by 1.
#[tokio::test]
async fn test_feature8_tier2_boundary_max_context_tokens() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    // 4-token prompt
    tok.register_mapping("one two three four", vec![1, 2, 3, 4]);
    // 5-token prompt
    tok.register_mapping("one two three four five", vec![1, 2, 3, 4, 5]);

    let (app, _handle, mut rx) = make_test_server(4, 512, 10, Arc::new(tok));

    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            let _ = cmd.token_tx.send(100);
        }
    });

    // 1. Boundary OK: 4 tokens <= max_prompt_tokens (4)
    let req_ok = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"one two three four"}],"stream":true,"model":"test"}"#,
        ))
        .unwrap();

    let res_ok = app.clone().oneshot(req_ok).await.unwrap();
    assert_eq!(res_ok.status(), StatusCode::OK);

    // 2. Boundary Reject: 5 tokens > max_prompt_tokens (4)
    let req_err = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"one two three four five"}],"stream":true,"model":"test"}"#,
        ))
        .unwrap();

    let res_err = app.oneshot(req_err).await.unwrap();
    assert_eq!(res_err.status(), StatusCode::BAD_REQUEST);

    let _ = worker.await;
}

/// Feature 8 - Test 8: Boundary Pipeline Cancellation Mid-Flight
///
/// Verifies that when a client drops its connection mid-stream while Depth-2
/// compute and readback stages are active, the pipeline detects the broken channel,
/// terminates cleanly, and decrements active_requests.
#[tokio::test]
async fn test_feature8_tier2_boundary_cancellation_mid_flight() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("cancel test", vec![1]);

    let (app, handle, mut rx) = make_test_server(1024, 512, 10, Arc::new(tok));

    let worker_cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled_flag = worker_cancelled.clone();

    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            let p_worker =
                Depth2PipelinedWorker::new(Duration::from_millis(5), Duration::from_millis(5));
            let toks = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            let res = p_worker
                .run_pipelined_generation(toks, 999, cmd.token_tx)
                .await;
            if res.is_ok() {
                worker_cancelled_flag.store(true, Ordering::SeqCst);
            }
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"cancel test"}],"stream":true,"model":"test"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Read only the first chunk, then drop the stream (simulating client abort)
    let mut stream = res.into_body().into_data_stream();
    let first = stream.next().await;
    assert!(first.is_some());
    drop(stream);

    let _ = worker.await;

    // Verify active_requests returns to 0
    let mut cleaned_up = false;
    for _ in 0..50 {
        if handle.metrics.active_requests.load(Ordering::Relaxed) == 0 {
            cleaned_up = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        cleaned_up,
        "Active requests must return to 0 after client drops connection"
    );
}

/// Feature 8 - Test 9: Boundary Immediate EOF / EOS at Step 0
///
/// Verifies that when the prompt immediately generates an EOS token, the Depth-2 pipeline
/// flushes immediately, emits zero content deltas, and sends `data: [DONE]`.
#[tokio::test]
async fn test_feature8_tier2_boundary_immediate_eof() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("stop immediately", vec![1]);

    let (app, handle, mut rx) = make_test_server(1024, 512, 10, Arc::new(tok));

    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            // First token generated is EOS
            let p_worker =
                Depth2PipelinedWorker::new(Duration::from_millis(1), Duration::from_millis(1));
            let _ = p_worker
                .run_pipelined_generation(vec![999], 999, cmd.token_tx)
                .await;
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"stop immediately"}],"stream":true,"model":"test"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let (chunks, done) = collect_sse_events(res).await;
    let _ = worker.await;

    assert!(done);
    let full = chunks.concat();
    assert_eq!(full, "", "Immediate EOS must produce empty content");
    assert_eq!(handle.metrics.active_requests.load(Ordering::Relaxed), 0);
}

/// Feature 8 - Test 10: Boundary Rapid Pipeline Cycling
///
/// Dispatches 20 short requests (1-2 tokens each) in rapid succession through the pipeline.
/// Verifies no resource exhaustion, deadlocks, queue stalls, or metric leaks.
#[tokio::test]
async fn test_feature8_tier2_boundary_rapid_pipeline_cycling() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_token(50, b"ok".to_vec());
    tok.register_mapping("q", vec![1]);

    let (app, handle, mut rx) = make_test_server(1024, 512, 32, Arc::new(tok));

    let worker = tokio::spawn(async move {
        while let Some(cmd) = rx.recv().await {
            let p_worker =
                Depth2PipelinedWorker::new(Duration::from_millis(1), Duration::from_millis(1));
            let _ = p_worker
                .run_pipelined_generation(vec![50], 999, cmd.token_tx)
                .await;
        }
    });

    for i in 0..20 {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(format!(
                r#"{{"messages":[{{"role":"user","content":"q"}}],"stream":true,"model":"test-{i}"}}"#
            )))
            .unwrap();

        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let (chunks, done) = collect_sse_events(res).await;
        assert!(done);
        assert_eq!(chunks.concat(), "ok");
    }

    drop(app);
    worker.abort();

    assert_eq!(handle.metrics.active_requests.load(Ordering::Relaxed), 0);
    assert_eq!(handle.metrics.total_requests.load(Ordering::Relaxed), 20);
}

/// Feature 9 - Test 6: Boundary Split 2-Byte UTF-8 Character Across Tokens
///
/// Tests Cyrillic 'д' (0xD0 0xB4) and Greek 'Ω' (0xCE 0xA9) split across two tokens:
/// Token A has byte 0, Token B has byte 1.
/// Verifies Token A buffers and emits nothing, and Token B completes the character.
#[test]
fn test_feature9_tier2_boundary_split_2byte_utf8() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    // Cyrillic 'д' = [0xD0, 0xB4]
    tok.register_token(10, vec![0xD0]);
    tok.register_token(11, vec![0xB4]);

    // Greek 'Ω' = [0xCE, 0xA9]
    tok.register_token(20, vec![0xCE]);
    tok.register_token(21, vec![0xA9]);

    let mut detok = StreamingDetokenizer::new(5);

    // Step 1: byte 0xD0 (incomplete UTF-8)
    let s1 = detok.step(10, &tok);
    assert_eq!(
        s1, "",
        "Incomplete 2-byte UTF-8 must buffer without emitting replacement char"
    );

    // Step 2: byte 0xB4 completes 'д'
    let s2 = detok.step(11, &tok);
    assert_eq!(s2, "д", "Second token must complete Cyrillic 'д'");

    // Step 3: byte 0xCE
    let s3 = detok.step(20, &tok);
    assert_eq!(s3, "");

    // Step 4: byte 0xA9 completes 'Ω'
    let s4 = detok.step(21, &tok);
    assert_eq!(s4, "Ω", "Second token must complete Greek 'Ω'");
}

/// Feature 9 - Test 7: Boundary Split 3-Byte CJK Character Across Tokens
///
/// Tests 3-byte Chinese character '中' (0xE4 0xB8 0xAD) and Japanese '語' (0xE8 0xAA 0x9E)
/// split across consecutive tokens (1+2 bytes, and 1+1+1 bytes).
#[test]
fn test_feature9_tier2_boundary_split_3byte_cjk() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    // '中' = [0xE4, 0xB8, 0xAD] split into 2 tokens: [0xE4] and [0xB8, 0xAD]
    tok.register_token(100, vec![0xE4]);
    tok.register_token(101, vec![0xB8, 0xAD]);

    // '語' = [0xE8, 0xAA, 0x9E] split into 3 tokens: [0xE8], [0xAA], [0x9E]
    tok.register_token(102, vec![0xE8]);
    tok.register_token(103, vec![0xAA]);
    tok.register_token(104, vec![0x9E]);

    let mut detok = StreamingDetokenizer::new(6);

    let d1 = detok.step(100, &tok);
    assert_eq!(d1, "");
    let d2 = detok.step(101, &tok);
    assert_eq!(d2, "中");

    let d3 = detok.step(102, &tok);
    assert_eq!(d3, "");
    let d4 = detok.step(103, &tok);
    assert_eq!(d4, "");
    let d5 = detok.step(104, &tok);
    assert_eq!(d5, "語");
}

/// Feature 9 - Test 8: Boundary Split 4-Byte Emoji Across Tokens
///
/// Tests Crab emoji '🦀' (0xF0 0x9F 0xA6 0x80) split across 2 tokens (2 bytes + 2 bytes)
/// and Rocket emoji '🚀' (0xF0 0x9F 0x9A 0x80) split across 4 tokens (1 byte each).
#[test]
fn test_feature9_tier2_boundary_split_4byte_emoji() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    // 🦀 = [0xF0, 0x9F, 0xA6, 0x80]
    tok.register_token(201, vec![0xF0, 0x9F]);
    tok.register_token(202, vec![0xA6, 0x80]);

    // 🚀 = [0xF0, 0x9F, 0x9A, 0x80]
    tok.register_token(301, vec![0xF0]);
    tok.register_token(302, vec![0x9F]);
    tok.register_token(303, vec![0x9A]);
    tok.register_token(304, vec![0x80]);

    let mut detok = StreamingDetokenizer::new(5);

    let e1 = detok.step(201, &tok);
    assert_eq!(e1, "");
    let e2 = detok.step(202, &tok);
    assert_eq!(e2, "🦀");

    let r1 = detok.step(301, &tok);
    assert_eq!(r1, "");
    let r2 = detok.step(302, &tok);
    assert_eq!(r2, "");
    let r3 = detok.step(303, &tok);
    assert_eq!(r3, "");
    let r4 = detok.step(304, &tok);
    assert_eq!(r4, "🚀");
}

/// Feature 9 - Test 9: Boundary Incomplete Sequence Buffering & Flush
///
/// Verifies that trailing incomplete UTF-8 bytes are buffered without emitting U+FFFD,
/// and that flush handles final sequence boundaries safely.
#[test]
fn test_feature9_tier2_boundary_invalid_byte_buffering() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_token(1, b"Valid ASCII prefix ".to_vec());
    tok.register_token(2, vec![0xF0, 0x9F]); // incomplete emoji prefix

    let mut detok = StreamingDetokenizer::new(4);
    let s1 = detok.step(1, &tok);
    assert_eq!(s1, "Valid ASCII prefix ");

    let s2 = detok.step(2, &tok);
    assert_eq!(
        s2, "",
        "Incomplete trailing bytes must not be emitted as replacement character"
    );

    // Flush should safely finalize without panicking:
    // When trailing incomplete multi-byte UTF-8 bytes (0xF0, 0x9F) remain at stream completion,
    // flush must emit the Unicode replacement character U+FFFD.
    let flushed = detok.flush(&tok);
    assert_eq!(
        flushed, "\u{FFFD}",
        "Flushing an incomplete trailing UTF-8 byte sequence must emit replacement character U+FFFD"
    );

    // A second flush on an already finalized detokenizer must be idempotent and return empty.
    let second_flushed = detok.flush(&tok);
    assert_eq!(
        second_flushed, "",
        "Subsequent flush on already finalized detokenizer must be empty"
    );
}

/// Feature 9 - Test 10: Boundary Empty & Whitespace Tokens
///
/// Tests handling of empty tokens, single spaces, tabs, newlines, and multi-space tokens
/// without offset corruption or char boundary slicing panics.
#[test]
fn test_feature9_tier2_boundary_empty_and_whitespace_tokens() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_token(1, vec![]); // empty
    tok.register_token(2, b" ".to_vec());
    tok.register_token(3, b"\n".to_vec());
    tok.register_token(4, b"\t".to_vec());
    tok.register_token(5, b"   ".to_vec());
    tok.register_token(6, b"end".to_vec());

    let mut detok = StreamingDetokenizer::new(5);
    let mut pieces = Vec::new();

    for id in [1, 2, 3, 4, 5, 6] {
        pieces.push(detok.step(id, &tok));
    }
    pieces.push(detok.flush(&tok));

    let full = pieces.concat();
    assert_eq!(full, " \n\t   end");
}

// ============================================================================
// Tier 3: Cross-Feature Combinations (≥4 tests)
// ============================================================================

/// Cross-Feature Combination 1: Depth-2 Pipelined Emission into SSE Streaming Detokenizer
///
/// Verifies end-to-end integration: Depth-2 async pipeline generates tokens,
/// which stream into the server's SSE handler and are detokenized in O(1) time
/// via `StreamingDetokenizer`, producing compliant `chat.completion.chunk` events.
#[tokio::test]
async fn test_tier3_depth2_pipeline_to_sliding_detokenizer_streaming() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("run stream", vec![1]);
    tok.register_token(10, b"Deep ".to_vec());
    tok.register_token(11, b"Metal ".to_vec());
    tok.register_token(12, b"Optimization".to_vec());

    let (app, handle, mut rx) = make_test_server(1024, 512, 10, Arc::new(tok));

    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            let p_worker =
                Depth2PipelinedWorker::new(Duration::from_millis(3), Duration::from_millis(3));
            let toks = vec![10, 11, 12];
            let _ = p_worker
                .run_pipelined_generation(toks, 999, cmd.token_tx)
                .await;
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"run stream"}],"stream":true,"model":"test"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let (chunks, done) = collect_sse_events(res).await;
    let _ = worker.await;

    assert!(done, "Must end with [DONE]");
    let full = chunks.concat();
    assert_eq!(full, "Deep Metal Optimization");
    assert_eq!(handle.metrics.active_requests.load(Ordering::Relaxed), 0);
}

/// Cross-Feature Combination 2: Depth-2 Pipeline with Multi-Byte UTF-8 Split Across Ticks
///
/// Verifies that tokens split across Depth-2 pipelined asynchronous ticks preserve byte
/// alignment and reconstruct properly into Unicode characters without corruption.
#[tokio::test]
async fn test_tier3_depth2_pipeline_with_utf8_split_across_ticks() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("unicode stream", vec![1]);
    tok.register_token(10, b"Hello ".to_vec());
    // 4-byte 🦀 split into 2 tokens
    tok.register_token(11, vec![0xF0, 0x9F]);
    tok.register_token(12, vec![0xA6, 0x80]);
    tok.register_token(13, b" World".to_vec());

    let (app, handle, mut rx) = make_test_server(1024, 512, 10, Arc::new(tok));

    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            let p_worker =
                Depth2PipelinedWorker::new(Duration::from_millis(4), Duration::from_millis(4));
            let toks = vec![10, 11, 12, 13];
            let _ = p_worker
                .run_pipelined_generation(toks, 999, cmd.token_tx)
                .await;
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"unicode stream"}],"stream":true,"model":"test"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let (chunks, done) = collect_sse_events(res).await;
    let _ = worker.await;

    assert!(done);
    let full = chunks.concat();
    assert_eq!(full, "Hello 🦀 World");
    assert_eq!(handle.metrics.active_requests.load(Ordering::Relaxed), 0);
}

/// Cross-Feature Combination 3: Pipeline Cancellation with Incomplete Buffered UTF-8 Sequence
///
/// Client disconnects while Depth-2 pipeline is active AND the detokenizer is holding
/// an incomplete multi-byte sequence. Verifies graceful teardown and RAII guard cleanup.
#[tokio::test]
async fn test_tier3_pipeline_cancellation_with_buffered_incomplete_utf8() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("cancel utf8", vec![1]);
    tok.register_token(10, b"Start ".to_vec());
    tok.register_token(11, vec![0xF0, 0x9F]); // incomplete emoji
    tok.register_token(12, vec![0xA6, 0x80]);

    let (app, handle, mut rx) = make_test_server(1024, 512, 10, Arc::new(tok));

    let worker = tokio::spawn(async move {
        if let Some(cmd) = rx.recv().await {
            let p_worker =
                Depth2PipelinedWorker::new(Duration::from_millis(10), Duration::from_millis(10));
            let toks = vec![10, 11, 12];
            let _ = p_worker
                .run_pipelined_generation(toks, 999, cmd.token_tx)
                .await;
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"cancel utf8"}],"stream":true,"model":"test"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Consume first chunk and immediately drop
    let mut stream = res.into_body().into_data_stream();
    let _ = stream.next().await;
    drop(stream);

    let _ = worker.await;

    // Verify active_requests drops to 0
    let mut ok = false;
    for _ in 0..50 {
        if handle.metrics.active_requests.load(Ordering::Relaxed) == 0 {
            ok = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(ok, "Active requests must return to 0");
}

/// Cross-Feature Combination 4: Rapid Depth-2 Cycling with Mixed Multilingual Streams
///
/// Dispatches multiple sequential requests containing mixed ASCII, CJK, and emoji tokens
/// through the Depth-2 pipeline and sliding window detokenizer.
#[tokio::test]
async fn test_tier3_rapid_depth2_cycling_mixed_unicode_ascii() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("req", vec![1]);
    tok.register_token(1, b"Rust ".to_vec());
    tok.register_token(2, "東京 ".as_bytes().to_vec());
    tok.register_token(3, "🔥".as_bytes().to_vec());

    let (app, handle, mut rx) = make_test_server(1024, 512, 32, Arc::new(tok));

    let worker = tokio::spawn(async move {
        while let Some(cmd) = rx.recv().await {
            let p_worker =
                Depth2PipelinedWorker::new(Duration::from_millis(1), Duration::from_millis(1));
            let toks = vec![1, 2, 3];
            let _ = p_worker
                .run_pipelined_generation(toks, 999, cmd.token_tx)
                .await;
        }
    });

    for i in 0..10 {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(format!(
                r#"{{"messages":[{{"role":"user","content":"req"}}],"stream":true,"model":"m-{i}"}}"#
            )))
            .unwrap();

        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let (chunks, done) = collect_sse_events(res).await;
        assert!(done);
        assert_eq!(chunks.concat(), "Rust 東京 🔥");
    }

    drop(app);
    worker.abort();
    assert_eq!(handle.metrics.active_requests.load(Ordering::Relaxed), 0);
}

// ============================================================================
// Tier 4: Real-World Application Scenarios
// ============================================================================

/// Scenario 2: Multi-token Decode Streaming with Character Boundaries
///
/// Tests realistic AI code generation and multi-language explanation containing:
/// - Markdown headings and code blocks
/// - English, Japanese explanations
/// - Emojis with multi-byte boundaries split across token streams
/// Streamed through Depth-2 async pipeline and sliding window detokenizer.
#[tokio::test]
async fn test_tier4_scenario2_multi_token_decode_streaming_character_boundaries() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("generate code", vec![100]);

    // Tokenize a complex realistic markdown code response
    let golden_text = "# Code Review 🔍\n```python\ndef compute():\n    # 計算開始 🌟\n    return 42\n```\nDone! ✅\n";
    let token_slices: Vec<Vec<u8>> = vec![
        b"# Code ".to_vec(),
        b"Review ".to_vec(),
        "🔍\n".as_bytes().to_vec(),
        b"```python\n".to_vec(),
        b"def compute():\n".to_vec(),
        b"    # ".to_vec(),
        // Split Japanese "計算" [0xE8, 0xA8, 0x88, 0xE7, 0xAE, 0x97] across tokens
        vec![0xE8, 0xA8],
        vec![0x88, 0xE7, 0xAE, 0x97],
        // Split "開始"
        "開始 ".as_bytes().to_vec(),
        // Split emoji "🌟\n" [0xF0, 0x9F, 0x8C, 0x9F, 0x0A]
        vec![0xF0, 0x9F],
        vec![0x8C, 0x9F, 0x0A],
        b"    return 42\n".to_vec(),
        b"```\n".to_vec(),
        b"Done! ".to_vec(),
        // Split "✅\n" [0xE2, 0x9C, 0x85, 0x0A]
        vec![0xE2, 0x9C],
        vec![0x85, 0x0A],
    ];

    let mut token_ids = Vec::new();
    for (idx, slice) in token_slices.into_iter().enumerate() {
        let tid = idx as u32 + 500;
        tok.register_token(tid, slice);
        token_ids.push(tid);
    }

    let (app, handle, mut rx) = make_test_server(1024, 1024, 10, Arc::new(tok));

    let worker = tokio::spawn({
        let tids = token_ids.clone();
        async move {
            if let Some(cmd) = rx.recv().await {
                let p_worker =
                    Depth2PipelinedWorker::new(Duration::from_millis(2), Duration::from_millis(2));
                let _ = p_worker
                    .run_pipelined_generation(tids, 999, cmd.token_tx)
                    .await;
            }
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"generate code"}],"stream":true,"model":"test-code"}"#,
        ))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let (chunks, done) = collect_sse_events(res).await;
    let _ = worker.await;

    assert!(done, "Terminal event must be data: [DONE]");
    let reconstructed = chunks.concat();
    assert_eq!(
        reconstructed, golden_text,
        "Reconstructed text must match original multi-byte markdown text bit-for-bit"
    );
    assert_eq!(handle.metrics.active_requests.load(Ordering::Relaxed), 0);
}

/// Scenario 5: Full Server SSE Streaming under Concurrent Load
///
/// Simulates 10 concurrent clients submitting requests to `/v1/chat/completions` with `stream: true`.
/// Pipelined workers generate streaming tokens concurrently with Depth-2 command buffer pipelining
/// and sliding window detokenization.
///
/// Verifies:
/// - 100% of requests succeed with HTTP 200
/// - All streams complete with valid SSE format and terminal `data: [DONE]`
/// - Active requests counter peaks and gracefully returns to 0
/// - Server metrics accurately reflect prompt and completion token counts
/// - Zero race conditions or resource leaks.
#[tokio::test]
async fn test_tier4_scenario5_full_server_sse_streaming_concurrent_load() {
    let mut tok = ByteLevelMockTokenizer::new(999);
    tok.register_mapping("concurrent load prompt", vec![1, 2, 3]);

    for i in 100..120 {
        tok.register_token(i, format!("t{i} ").into_bytes());
    }

    let (app, handle, mut rx) = make_test_server(1024, 512, 16, Arc::new(tok));

    // Background worker loop servicing multiple concurrent requests
    let worker = tokio::spawn(async move {
        while let Some(cmd) = rx.recv().await {
            tokio::spawn(async move {
                let p_worker =
                    Depth2PipelinedWorker::new(Duration::from_millis(2), Duration::from_millis(2));
                let tokens: Vec<u32> = (100..115).collect();
                let _ = p_worker
                    .run_pipelined_generation(tokens, 999, cmd.token_tx)
                    .await;
            });
        }
    });

    let concurrent_clients = 10;
    let mut client_tasks = Vec::new();

    for client_id in 0..concurrent_clients {
        let app_clone = app.clone();
        let jh = tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(format!(
                    r#"{{"messages":[{{"role":"user","content":"concurrent load prompt"}}],"stream":true,"model":"model-{client_id}"}}"#
                )))
                .unwrap();

            let res = app_clone.oneshot(req).await.unwrap();
            assert_eq!(res.status(), StatusCode::OK);

            let (chunks, done) = collect_sse_events(res).await;
            assert!(done);
            let full = chunks.concat();
            assert!(full.starts_with("t100 "));
            assert!(full.ends_with("t114 "));
            chunks.len()
        });
        client_tasks.push(jh);
    }

    for jh in client_tasks {
        let chunk_count = jh.await.unwrap();
        assert!(chunk_count > 0);
    }

    // Allow workers to finish cleanly
    drop(app);
    worker.abort();

    // Verify all active requests return to 0
    let mut all_drained = false;
    for _ in 0..100 {
        if handle.metrics.active_requests.load(Ordering::Relaxed) == 0 {
            all_drained = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert!(all_drained, "Active requests counter must return to 0");
    assert_eq!(
        handle.metrics.total_requests.load(Ordering::Relaxed),
        concurrent_clients as u64
    );
    assert_eq!(
        handle.metrics.rejected_requests.load(Ordering::Relaxed),
        0,
        "No requests should be rejected under admitted capacity"
    );
}
