//! KriyaRT Server Library.
//!
//! Provides the core runtime, HTTP server routes, admission control, and Hugging Face hub integration.

pub mod hub_fetcher;
pub mod runtime;
pub mod server;

pub use runtime::{BenchmarkReport, BenchmarkSample, EngineRuntime, orca_batch_worker_loop};
pub use server::{
    ActiveStreamState, EnqueueRequestCmd, ServerHandle, ServerMetrics, TokenizerService, create_app,
};
