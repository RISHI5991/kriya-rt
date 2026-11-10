# KriyaRT Engineering Roadmap

This document outlines the architectural milestones and engineering priorities for **KriyaRT**. We welcome community contributions, architectural proposals (RFCs), and collaborative benchmarking across all phases!

---

## 📍 Phase 1: Zero-Conversion Frontier Serving (v0.1.0 — Current)

- [x] **Pure Rust + Metal Shading Language runtime** with zero Python dependencies.
- [x] **Fused MSL 3.1 GEMV & GEMM kernels:**
  - In-register SwiGLU activation via `simd_shuffle_down(acc, 16u)` (eliminates 2 DRAM round-trips per layer).
  - Sub-16 threadgroup dispatch maximizing occupancy on Apple Silicon cores.
  - Direct output projection residual accumulation (`y += x @ W^T`).
- [x] **Zero-Conversion Direct MLX Loading:** Memory-maps native Hugging Face / MLX safetensors triplets (`.weight`, `.scales`, `.biases`) directly at startup.
- [x] **Block-Sparse Row (BSR) Paged KV-Cache:** 16-token pages with radix prefix sharing and copy-on-write page recycling.
- [x] **Vectorized BQ8 Prefill Attention:** 8 query tokens per SIMD threadgroup supporting $D=64, 128, 256$ head dimensions with online softmax in threadgroup SRAM.
- [x] **OpenAI-Compatible HTTP Server:** Streaming Server-Sent Events (SSE) `/v1/chat/completions` API via Axum.
- [x] **Turnkey Release Engineering:** Standalone release tarballs with SHA-256 verification and Homebrew formula (`brew install RISHI5991/tap/kriya`).

---

## 🚀 Phase 2: Closing the Long-Context Prefill Gap (v0.2.0)

**Goal:** Surpass Apple MLX on prompt ingestion ($M \ge 512$) on Apple Silicon.

- [ ] **Metal Performance Shaders (MPSGraph) Tensor Prefill Backend:**
  - Implement a specialized prefill execution path utilizing Apple's `MPSGraph` matrix multiplication routines to access dedicated hardware matrix accelerators.
  - Dynamically switch between MSL ALU kernels (for $M < 64$) and MPSGraph matrix multiplication (for $M \ge 64$).
- [ ] **Double-Buffered DRAM-to-SRAM Tiling:**
  - Implement ping-pong double-buffering in `q4_g64_batched_simdgroup_gemm_m64` (`threadgroup half x_tile[2][64][64]`) to overlap DRAM loads with `simdgroup_multiply_accumulate` loops.
- [ ] **FlashAttention-3 Asynchronous Copy:**
  - Utilize Metal 3.1 `simdgroup_async_copy` to prefetch K and V tiles from Unified Memory directly into threadgroup memory while computing attention scores.

---

## 🌐 Phase 3: Architectural Breadth (v0.3.0)

**Goal:** Expand support to non-canonical, frontier open-weights architectures.

- [ ] **Fused Projection Architectures (Spark X, Falcon):**
  - Add native dequantization and projection splitting for models with fused `q_k_v_proj` matrices.
  - Implement mixed-precision OptiQ dequantization kernels (heterogeneous 4-bit / 8-bit layers).
- [ ] **Multi-Head Latent Attention (MLA) (DeepSeek V2 & V3):**
  - Compressed low-rank KV projection (`c_KV`) with decoupled rotary embeddings.
- [ ] **Sparse Mixture of Experts (MoE) (Mixtral 8x7B, Qwen 2.5 MoE):**
  - GPU-resident top-k softmax routing and fused expert scatter/gather kernels.
- [ ] **Linear Attention & State-Space Models (Mamba, RecurrentGemma):**
  - Associative scan and state-passing kernels in Metal Shading Language.

---

## 🛠️ Phase 4: Ecosystem & Native Integrations (v0.4.0)

**Goal:** Provide first-class bindings for developers building on macOS and iOS.

- [ ] **Swift Package & C-FFI:**
  - Native Swift package (`KriyaKit`) allowing iOS and macOS SwiftUI applications to embed KriyaRT directly with 0 subprocess overhead.
- [ ] **Zero-Overhead Python Bindings:**
  - `pip install kriya-rt` via PyO3, providing a drop-in drop replacement for `mlx_lm.load` and `mlx_lm.generate`.
- [ ] **Speculative Draft Head Zoo:**
  - Pre-trained EAGLE / Medusa draft heads distributed directly on Hugging Face for instant 1.5x–2.0x speculative speedup out of the box.
- [ ] **Hardware Metric Telemetry:**
  - Real-time Metal GPU power (Watts), memory bandwidth utilization (GB/s), and thermal throttling telemetry in the CLI status bar.

---

## How to Get Involved

Want to tackle one of these roadmap items?
1. Check the [Issues](https://github.com/RISHI5991/kriya-rt/issues) tab for items tagged `good first issue` or `help wanted`.
2. Open an RFC in [Discussions](https://github.com/RISHI5991/kriya-rt/discussions) to propose your implementation approach.
3. Read our [Contributing Guide](CONTRIBUTING.md) to set up your environment!
