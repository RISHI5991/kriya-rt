# E2E Test Infra: kriya-rt Section 4 Deep Metal Backend & Kernel Optimizations

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation design.
- Methodology: Category-Partition + BVA + Pairwise + Workload Testing.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | 32B Aligned Q4 Weights | ORIGINAL_REQUEST R1 | 5 | 5 | ✓ |
| 2 | Fused add_rmsnorm | ORIGINAL_REQUEST R1 | 5 | 5 | ✓ |
| 3 | Consolidated qkv_proj | ORIGINAL_REQUEST R1 | 5 | 5 | ✓ |
| 4 | Function Constants | ORIGINAL_REQUEST R1 | 5 | 5 | ✓ |
| 5 | Type Pruning for Q2/Q3 | ORIGINAL_REQUEST R3 | 5 | 5 | ✓ |
| 6 | HardwareProfile Apple7 FP16 | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ |
| 7 | FlashAttention-2 Prefill | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ |
| 8 | Depth-2 Async Pipelining | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ |
| 9 | Sliding Window Detokenizer | ORIGINAL_REQUEST R3 | 5 | 5 | ✓ |
| 10 | Physical Weight Benchmarks | ORIGINAL_REQUEST R4 | 5 | 5 | ✓ |

## Test Architecture
- Test runner: `cargo test --workspace` and dedicated test harnesses
- Test case format: Rust integration tests in `crates/kriya-backend-metal/tests/` and `crates/kriya-server/tests/`
- Pass/Fail semantics: Exit code 0, 0 clippy warnings, zero regressions on existing 92 tests, all invariant checks clean

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Full Forward & Sample Single Step | F1, F2, F3, F4, F6, F8 | High |
| 2 | Multi-token Decode Streaming with Character Boundaries | F7, F8, F9 | High |
| 3 | Large Batch Prefill with Tiled FlashAttention-2 | F6, F7 | High |
| 4 | Real Qwen2.5-0.5B Inference under 1 GB RAM | F1, F2, F3, F4, F6, F7, F8, F10 | Very High |
| 5 | Full Server SSE Streaming under Concurrent Load | F8, F9, F10 | High |

## Coverage Thresholds
- Tier 1: ≥5 per feature (50 tests minimum)
- Tier 2: ≥5 per feature (50 tests minimum)
- Tier 3: Pairwise coverage of major feature interactions (10 tests minimum)
- Tier 4: ≥5 realistic application scenarios
- Total target: ~115 test cases/assertions
