//! Telemetry, metrics tracking, and timestamps for KriyaRT Server.

use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Returns the current Unix epoch timestamp in microseconds.
pub fn current_timestamp_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_micros() as u64
}

/// Live telemetry & admission metrics for the inference server.
#[derive(Debug)]
pub struct ServerMetrics {
    pub start_time: Instant,
    pub active_requests: AtomicUsize,
    pub total_requests: AtomicU64,
    pub total_tokens_emitted: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    pub rejected_requests: AtomicU64,
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            active_requests: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            total_tokens_emitted: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
        }
    }
}
