# Contributing to KriyaRT

Thank you for your interest in contributing to **KriyaRT**! 🚀

KriyaRT is an open-source, ultra-high-throughput native Large Language Model (LLM) inference runtime built exclusively for **Apple Silicon (M1–M4, Pro, Max, Ultra)** in **100% pure Rust** and **Metal Shading Language (MSL 3.1)**.

Our mission is to build the fastest, lightest, zero-dependency inference engine on Apple Silicon. We welcome contributors from around the globe—from systems hackers and Metal kernel engineers to Rust developers and documentation writers.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Building from Source](#building-from-source)
   - [Running the Test Suite](#running-the-test-suite)
3. [Architecture Overview](#architecture-overview)
4. [How to Contribute](#how-to-contribute)
   - [Writing Metal Shading Language Kernels](#writing-metal-shading-language-kernels)
   - [Adding New Model Architectures](#adding-new-model-architectures)
   - [Benchmarking & Profiling](#benchmarking--profiling)
5. [Pull Request Process](#pull-request-process)
6. [Coding Guidelines](#coding-guidelines)

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone.

---

## Getting Started

### Prerequisites

* **Hardware:** Any Mac with Apple Silicon (M1, M2, M3, or M4 chip).
* **OS:** macOS 14.0 (Sonoma) or newer.
* **Xcode Command Line Tools:**
  ```bash
  xcode-select --install
  ```
* **Rust Toolchain:** Rust 1.81+ (stable):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### Building from Source

Clone the repository and build the workspace:

```bash
git clone https://github.com/rishisharma/kriya-rt.git
cd kriya-rt

# Build all workspace crates in debug mode
cargo build

# Build optimized release binaries
cargo build --release --target aarch64-apple-darwin -p kriya-server
```

The resulting server binary will be located at `target/aarch64-apple-darwin/release/kriya-server`.

### Running the Test Suite

We maintain a strict 100% test pass rate across the workspace:

```bash
# Run all workspace unit and integration tests
cargo test --workspace

# Run real-world GPU validation tests against CPU reference implementations
cargo test -p kriya-backend-metal --test real_world_validation -- --nocapture
```

---

## Architecture Overview

KriyaRT is organized as a clean Rust workspace:

```
kriya-rt/
├── crates/
│   ├── kriya-core/           # Model descriptors, Safetensors loader, Q4 quantization math
│   ├── kriya-kv/             # Paged Radix KV-Cache, BSR (Block-Sparse Row) schedules
│   ├── kriya-backend-metal/  # Metal 4 compute pipeline, JIT compilation, GPU encoders
│   ├── kriya-scheduler/      # ORCA iteration-level batching & request state machines
│   ├── kriya-speculate/      # Speculative decoding (EAGLE / Medusa) tree verification
│   └── kriya-server/         # Axum HTTP server (OpenAI /v1/chat/completions, SSE) & CLI
├── kernels/
│   └── metal/                # Handcrafted MSL 3.1 compute shaders (.metal)
├── benchmarks/               # Side-by-side reproduction scripts vs Apple MLX
└── packaging/                # Homebrew formula and release packaging scripts
```

---

## How to Contribute

### Writing Metal Shading Language Kernels

All compute shaders are located in `kernels/metal/` and compiled at build time into binary metallib bundles.

When authoring or optimizing MSL kernels:
1. **Leverage SIMDgroup Operations:** Apple Silicon GPUs execute 32 threads per SIMDgroup. Use `simdgroup_matrix`, `simd_shuffle_down`, and `simd_broadcast` to keep operations in registers.
2. **Eliminate DRAM Round-Trips:** Fuse operations wherever possible. For example, our `q4_g64_decode_fused_gate_up_swiglu_sub16` kernel computes Gate and Up projections in the same SIMDgroup and applies SwiGLU non-linear multiplication in registers before writing to memory.
3. **Keep SRAM Utilization High:** Utilize threadgroup shared memory (`threadgroup`) for online softmax and K/V tiling.
4. **Sub-16 Threadgroup Tiling:** Avoid threadgroup sizes that cause register spilling; 128 threads (4 SIMDgroups) is typically optimal for M1/M2/M3 ALU occupancy.

### Adding New Model Architectures

We are actively expanding beyond standard dense Llama/Qwen architectures:
* **Fused Projections (e.g., Spark X):** Models that store combined `q_k_v_proj` rather than separated Q, K, and V tensors need a projection splitter in `crates/kriya-core/src/descriptor.rs` and `encoder.rs`.
* **Multi-Head Latent Attention (MLA) (e.g., DeepSeek V2/V3):** Requires low-rank KV compression and decoupled RoPE handling.
* **Mixture of Experts (MoE) (e.g., Mixtral, Qwen MoE):** Routing and top-k expert selection shaders.

If you plan to add support for a new model, please open an **RFC Issue** first using our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.yml) to align on design.

### Benchmarking & Profiling

Always benchmark changes against Apple MLX:

```bash
# 1. Install MLX in a Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install mlx mlx-lm

# 2. Run the automated comparison runner
python3 benchmarks/run_mlx_vs_kriya.py --model qwen3b
```

Reports will be written to `benchmarks/reports/`. Submit the resulting JSON with your PR!

---

## Pull Request Process

1. **Fork the repository** and create a feature branch (`git checkout -b feat/my-optimization`).
2. **Verify tests and formatting:**
   ```bash
   cargo fmt --all --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```
3. **Commit with clean messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `perf:`, `docs:`, `chore:`).
4. **Open a Pull Request:** Fill out the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
5. **CI Review:** Our GitHub Actions runner will verify compilation, tests, and formatting on macOS Apple Silicon runners.

---

## Coding Guidelines

* **Zero Python Dependencies:** The core runtime must remain 100% pure Rust and Metal. Python is permitted only in `benchmarks/` for comparative MLX validation.
* **Fail-Closed Validation:** Never degrade silently to fallback weights or dummy zeroes. Validate checkpoint shapes and byte counts strictly before execution.
* **Zero Allocations on the Hot Path:** During autoregressive token generation, no heap allocations (`Vec::new`, `Box`, `String`) are allowed in the inner execution loop. Use pre-allocated Unified Memory buffers.
* **Documentation:** Preserve docstrings and comment on non-trivial mathematical or GPU hardware synchronization choices.

Thank you for helping make KriyaRT the gold standard for Apple Silicon inference! 🦀⚡
