# ADR 0001: Native Metal Compute Backend for Ultra-Low Latency Inference

## Context
Deploying high-performance LLMs on Apple Silicon requires maximizing memory bandwidth saturation during autoregressive decoding while keeping dispatch latency to a minimum. 

Common alternatives on macOS include:
1. **Apple CoreML**: Optimized for Neural Engine (ANE) and fixed-shape models. Poor support for dynamic sequence lengths, dynamic paged attention, and continuous batching.
2. **MPSGraph / MPS**: High overhead per dispatch graph, proprietary op limitations, lack of low-level SIMDgroup register shuffle control.
3. **Apple MLX**: High-level array framework in C++/Python. Excellent developer experience, but incurs interpreter overhead and does not support kernel-fused multi-tenant continuous scheduling.

## Decision
KriyaRT implements a **pure native Metal compute backend** (`crates/kriya-backend-metal`) utilizing handcrafted MSL (Metal Shading Language) compute pipelines:
- In-register SIMDgroup reduction kernels (`simd_sum`, `simd_shuffle_down`).
- Fused activation and projection kernels (`q4_g64_decode_fused_gate_up_swiglu_sub16`).
- Direct pointer sharing with Apple Silicon Unified Memory without PCIe transfers.

## Consequences
- **Positive**: Sub-microsecond dispatch overhead, zero memory allocation during steady-state decode ticks, complete control over register allocation and threadgroup tiling.
- **Negative**: Requires maintaining architecture-specific MSL kernels across Apple Silicon generations (M1–M4).
