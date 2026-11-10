# ADR 0005: Fail-Closed Descriptor-Driven Checkpoint Verification & Quantization Safety

## Context
Previous iterations of local inference engines silently fell back to unquantized weights, synthetic tensors, or permissive `.ok()` unwrap conversions when optional layers or tensors were missing. This created severe silent failures in production, such as output degradation or hidden NaN propagation.

## Decision
KriyaRT enforces **descriptor-driven fail-closed checkpoint loading**:
- `ModelArchDescriptor` is the sole source of truth for all tensor requirements.
- `from_checkpoint_dir_strict` automatically executes `validate_production_checkpoint` upon entry.
- If any required layer normalization (such as Gemma's `post_feedforward_layernorm` or Qwen's Q/K norms) is specified in the descriptor, the corresponding tensor must exist with the exact shape and data type on disk, or the entire loader aborts with an actionable diagnostic.
- `MetalExecutor::new_production` rejects any fallback to synthetic weights.

## Consequences
- **Positive**: Impossible to run malformed or corrupted checkpoints in production serving.
- **Negative**: Rejects non-standard checkpoint variations that do not match the declared architectural descriptor.
