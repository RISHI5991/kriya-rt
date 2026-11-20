# Project: kriya-rt Section 4 Deep Metal Backend & Kernel Optimizations

## Architecture
- **crates/kriya-core**: Tensor representations, quantization definitions, memory page alignment, and hardware profiling.
- **kernels/metal/**: Metal compute shaders (`q4_gemm_and_moe.metal`, `paged_gqa_and_deltanet.metal`, `transformer_glue.metal`).
- **crates/kriya-backend-metal**: Metal execution engine (`MetalExecutor`), pipeline caching, layer buffer management, forward pass encoding, and checkpoint validation.
- **crates/kriya-kv**: Paged attention and BSR prefill layout planning.
- **crates/kriya-server**: Async inference runtime, Depth-2 command buffer pipelining, HTTP/SSE handlers, and streaming detokenization.
- **crates/kriya-backend-metal/tests & benchmarks/**: Physical weight integration benchmarks and performance measurement harness.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 32B Aligned Q4 Weights | Split `BlockQ4Group64` (36B) into 16-byte aligned 32B `uint4` weights (`ScaledAlignedQ4Block`) and separate `half2` scale/bias buffer (`BlockQ4ScaleBias`) for 100% coalesced 16-byte vector loads | M1 | ORIGINAL_REQUEST R1, media §4.2 |
| 2 | Fused `add_rmsnorm` | Fuses residual addition with RMSNorm in a single Metal kernel pass, eliminating separate dispatches and saving 2 DRAM activation passes per layer | M1 | ORIGINAL_REQUEST R1, media §4.1 |
| 3 | Consolidated `qkv_proj` Buffer | Concatenates Q, K, V weight blocks into unified buffer at load time, reducing 3 dispatches to 1 and dropping per-layer dispatches from 10 to 5 | M1 | ORIGINAL_REQUEST R1, media §4.1 |
| 4 | Metal Function Constants | Replace inner-loop dynamic branching (`has_qkv_bias`, `activation_variant`, `head_dim`, `gqa_ratio`, `has_qk_norm`) with `[[function_constant(n)]]` compiled via `MTLFunctionConstantValues` | M1 | ORIGINAL_REQUEST R1, media §4 |
| 5 | Type Pruning for Q2/Q3 | Prune unwired `BlockQ2Group128` and `BlockQ3Group64` types and dead conversions until ready, maintaining 0 warnings and zero regressions | M1 | ORIGINAL_REQUEST R3, media §5.2 |
| 6 | HardwareProfile Apple7 FP16 | Connect `HardwareProfile::detect` to `MetalExecutor` so `AppleGpuFamily::Apple7` (M1) uses native FP16 (`half`) activations and KV cache instead of emulated BF16 | M2 | ORIGINAL_REQUEST R2, media §4.3 |
| 7 | FlashAttention-2 Tiled Prefill & Deduplication | Implement $B_r=32, B_c=32$ tiled prefill with `simdgroup_matrix` MMA (<28 KB SRAM), and eliminate redundant $M$-row block index duplication in `build_parallel_prefill_layout` | M2 | ORIGINAL_REQUEST R2, media §4.4 |
| 8 | Depth-2 Async Pipelining | Depth-2 async command buffer pipelining in `runtime.rs` to overlap decode token readback with GPU execution, eliminating CPU-GPU synchronization bubbles | M2 | ORIGINAL_REQUEST R2, media §4.5 |
| 9 | Streaming Detokenizer Sliding Window | Replace $O(N^2)$ full sequence re-decoding in `handlers.rs` with 4-6 token sliding window for $O(1)$ decoding while preserving valid UTF-8 character boundary slicing | M3 | ORIGINAL_REQUEST R3, media §3.6 |
| 10 | Physical Weight Benchmarks | Un-ignore and execute 3 physical benchmarks with real `Qwen/Qwen2.5-0.5B-Instruct` weights under 1 GB RAM limit, reporting TTFT, TPOT, and memory bandwidth (GB/s) | M4 | ORIGINAL_REQUEST R4, media §5 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Deep Metal Kernel & Memory Layout Optimizations | F1, F2, F3, F4, F5 (Alignment split, fused add_rmsnorm, consolidated qkv_proj, function constants, Q2/Q3 pruning) | none | IN_PROGRESS (sub_orch: 0570fdac-e2dc-4189-86dd-0d8d8ea0c824) |
| M2 | Hardware Profiling, Attention & Execution Pipelining | F6, F7, F8 (Apple7 FP16 switching, FlashAttention-2 tiled prefill with deduplicated layout, Depth-2 async decode pipelining) | M1 | PLANNED |
| M3 | Streaming Detokenizer Sliding Window | F9 (4-6 token sliding window detokenizer with UTF-8 boundary safety) | none | DONE (StreamingDetokenizer W=6 O(1), 15 api_safety + 26 e2e_streaming tests passed, Invariant 5 verified) |
| M4 | Final Milestone: Benchmarks, E2E Pass & Coverage Hardening | F10 (Real Qwen2.5-0.5B physical weight benchmarks, 100% E2E test pass, Tier 5 adversarial hardening) | M1, M2, M3, TEST_READY.md | PLANNED |
| E2E | Opaque-Box E2E Testing Suite (Tiers 1-4) | Tiers 1-4 comprehensive requirement-driven test suite | none | IN_PROGRESS (sub_orch: 64620db4-4aed-4b85-b811-d721c8559d56) |

## Interface Contracts
### `crates/kriya-core` ↔ `crates/kriya-backend-metal`
- `ScaledAlignedQ4Block`: 32 bytes, `align(16)`. Pure packed nibbles `[u8; 32]`.
- `BlockQ4ScaleBias`: 4 bytes. IEEE `half` or `u16` scale and bias.
- `BlockQ4Group64`: retains helper `split_slice(&[BlockQ4Group64]) -> (Vec<ScaledAlignedQ4Block>, Vec<BlockQ4ScaleBias>)`.
- `HardwareProfile::detect`: Returns `HardwareProfile` containing `gpu_family: AppleGpuFamily` and `supports_native_bf16: bool`.

### `crates/kriya-backend-metal` ↔ `kernels/metal/`
- `q4_gemm_and_moe.metal`:
  - `ScaledAlignedQ4Block`: `uint4 packed_nibbles[2]`.
  - Buffer 0: `device const ScaledAlignedQ4Block* weights [[buffer(0)]]`.
  - Buffer 1: `device const half2* scales_and_biases [[buffer(1)]]`.
  - Consolidated QKV matrix: single buffer indexing `weights + row_idx * num_k_blocks`.
  - Function constants (indices 0..6): `fc_has_qkv_bias`, `fc_activation_variant`, `fc_head_dim`, `fc_gqa_ratio`, `fc_apply_qk_norm`, `fc_has_logit_softcap`, `fc_zero_centered`.
- `transformer_glue.metal`:
  - `fused_add_rmsnorm`: Buffer 0: `residual_inout`, Buffer 1: `proj_in`, Buffer 2: `norm_weight`, Buffer 3: `output_norm`, Buffer 4: `RmsNormParams`.
- `paged_gqa_and_deltanet.metal`:
  - `bsr_paged_gqa_tiled_mma_prefill`: $B_r=32, B_c=32$, `simdgroup_matrix` MMA, deduplicated `page_indices`.

### `crates/kriya-kv` ↔ `crates/kriya-backend-metal`
- `build_parallel_prefill_layout`: `page_indices` contains unique sequence physical block IDs without per-query-token repetition.

### `crates/kriya-server` ↔ Client Stream
- `handlers.rs`: 4-6 token sliding window detokenizer. Output strings preserve exact UTF-8 character boundaries and emit `data: [DONE]` on completion.

## Code Layout
- `crates/kriya-core/src/tensor.rs`: Struct definitions (`ScaledAlignedQ4Block`, `BlockQ4ScaleBias`).
- `crates/kriya-core/src/hardware.rs`: `HardwareProfile`, `AppleGpuFamily`.
- `crates/kriya-backend-metal/src/encoder.rs`: `MetalExecutor`, `LayerGpuBuffers`, `MetalPipelineCache`, forward pass dispatches.
- `kernels/metal/q4_gemm_and_moe.metal`: Split-aligned Q4 GEMV, consolidated QKV, function constants.
- `kernels/metal/transformer_glue.metal`: `fused_add_rmsnorm`.
- `kernels/metal/paged_gqa_and_deltanet.metal`: FlashAttention-2 tiled prefill.
- `crates/kriya-kv/src/bsr_paged.rs`: Deduplicated prefill layout planning.
- `crates/kriya-server/src/runtime.rs`: Depth-2 async decode command buffer pipelining.
- `crates/kriya-server/src/server/handlers.rs`: Streaming detokenizer sliding window.
- `crates/kriya-backend-metal/tests/test_real_qwen.rs`: Physical weight inference benchmarks.
