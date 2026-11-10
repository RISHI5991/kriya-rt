//! Axum HTTP Router definition decoupled from listener binding.

use crate::server::admission::ServerHandle;
use crate::server::handlers::{chat_completions_handler, health_handler, metrics_handler};
use axum::{
    Router,
    routing::{get, post},
};

/// Constructs the complete production Axum Router for KriyaRT.
///
/// This function decouples router definition and middleware configuration from
/// server socket binding, enabling comprehensive in-process testing of the actual
/// production routes and admission controllers.
pub fn create_app(handle: ServerHandle) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(axum::extract::DefaultBodyLimit::max(8 * 1024 * 1024))
        .with_state(handle)
}
