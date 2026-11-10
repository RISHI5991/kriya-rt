# ADR 0004: EAGLE Medusa Speculative Trees & Fail-Closed Integrity

## Context
Autoregressive decoding is strictly memory-bandwidth bound on single-sequence workloads. Speculative decoding allows generating multiple candidate tokens in parallel and verifying them with a single target model forward pass. However, unverified or malformed draft artifacts risk corrupting generation quality or causing crashes.

## Decision
KriyaRT integrates **EAGLE Medusa speculative decoding** (`crates/kriya-speculate`):
- Speculative acceleration is strictly **optional and fail-closed**. If a draft artifact is absent, corrupted, or fails SHA-256 manifest verification, speculative decoding is cleanly disabled, falling back to standard single-token autoregressive decoding.
- Verification uses a tree topology (canonical 10-token tree for batch sizes $\le 2$, compact tree for batch sizes $> 2$).
- Bonus tokens are committed through the ORCA scheduler only after target model verification passes.

## Consequences
- **Positive**: 1.5x–2.2x speedup on single-stream inference without modifying output token probabilities.
- **Negative**: Adds verification shader passes and requires maintaining compatible trained draft weights.
