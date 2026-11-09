# KriyaRT — High-Performance Apple Silicon LLM Inference Engine

[![CI](https://github.com/rishisharma/kriya-rt/actions/workflows/ci.yml/badge.svg)](https://github.com/rishisharma/kriya-rt/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon%20(M1--M4)-black.svg)](#hardware-requirements)
[![Release](https://img.shields.io/badge/Release-v0.1.0-emerald.svg)](https://github.com/rishisharma/kriya-rt/releases)

**KriyaRT** is an ultra-high-throughput, native Large Language Model (LLM) inference runtime engineered from first principles exclusively for **Apple Silicon (M1–M4, Pro, Max, Ultra)** unified memory architectures. 

Written in **100% pure Rust** and **Metal Shading Language (MSL 3.1)** with zero Python runtime overhead, zero C++ glue, and zero runtime dependencies, KriyaRT directly replaces Apple MLX and llama.cpp for local production serving and frontier edge execution.

---

## ⚡ Empirical Hardware Benchmarks (Apple Silicon M1, 8GB Unified Memory)

Direct side-by-side reproducible benchmarks comparing Apple MLX-LM vs. KriyaRT under identical conditions:
- **Identical Models:** Qwen 2.5 3B, Qwen 2.5 1.5B, Gemma 2 2B, SmolLM2-135M
- **Identical Precision:** 4-bit Group-64 Affine Quantization
- **Identical Sampling:** Greedy deterministic ($T=0.0$)
- **Identical Hardware:** Apple Silicon M1 (8GB unified LPDDR memory, macOS)
- **Zero Conversion:** Running directly on the exact same checkpoint directory

### 1. Autoregressive Decode Throughput (Time Per Output Token — TPOT)

| Model | Context | Apple MLX-LM | KriyaRT | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen 2.5 3B Instruct** | 64 tokens | 27.5 tok/s | **28.9 tok/s** | **+5.1% faster (1.051x)** |
| **Qwen 2.5 3B Instruct** | 255 tokens | 28.1 tok/s | **27.6 tok/s** | *0.982x (Parity)* |
| **Qwen 2.5 3B Instruct** | 512 tokens | 26.7 tok/s | **26.8 tok/s** | **+0.4% faster (1.003x)** |
| **Qwen 2.5 1.5B Instruct** | 64 tokens | 48.8 tok/s | **48.5 tok/s** | *0.993x (Parity)* |
| **SmolLM2 135M Instruct** | 64 tokens | 102.4 tok/s | **106.8 tok/s** | **+4.3% faster (1.043x)** |

### 2. Prefill Latency & Throughput (Time To First Token — TTFT)

| Model | Context Length | KriyaRT TTFT | KriyaRT Prefill Rate | MLX Prefill Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen 2.5 3B Instruct** | 64 tokens | **390.6 ms** | **163.8 tok/s** | 211.8 tok/s |
| **Qwen 2.5 3B Instruct** | 255 tokens | **1553.7 ms** | **164.8 tok/s** | 273.0 tok/s |
| **Qwen 2.5 3B Instruct** | 512 tokens | **3188.5 ms** | **160.6 tok/s** | 273.0 tok/s |
| **Qwen 2.5 1.5B Instruct** | 64 tokens | **219.1 ms** | **292.0 tok/s** | 423.8 tok/s |
| **Qwen 2.5 1.5B Instruct** | 255 tokens | **725.0 ms** | **353.1 tok/s** | 574.1 tok/s |
| **Gemma 2 2B IT** | 65 tokens | **307.5 ms** | **208.1 tok/s** | 282.5 tok/s |

> [!TIP]
> **Peak Memory Footprint:** On 3B-4B parameter models, KriyaRT maintains steady peak RAM usage under **2.4 GB** (well within an 8 GB base Mac), eliminating macOS swap file pressure and memory paging jitter.

---

## 🚀 Key Architectural Innovations

### 1. Zero-Conversion Direct MLX Loading
No manual offline weight conversion required. Point `kriya-server` directly at any MLX 4-bit checkpoint or local Hugging Face hub cache directory. KriyaRT parses `.weight`, `.scales`, and `.biases` triplets, repacks them directly into contiguous GPU unified memory buffers at zero-copy speeds, and converts F16/F32 layernorm weights to native BF16 on the fly.

### 2. Fused GPU Matrix Kernels (MSL 3.1)
- **Fused QKV GEMM & GEMV:** Fuses query, key, and value linear projections into a single GPU compute dispatch with in-register projection bias addition. Saves 2 redundant passes of memory bandwidth across Unified Memory.
- **Fused Gate+Up SwiGLU GEMM:** Evaluates MLP Gate and Up projections simultaneously in threadgroup registers, executing `SiLU(Gate) * Up` with zero global memory intermediate buffers.
- **O-Proj Direct Residual Accumulation:** Accumulates attention output projection directly into the residual buffer (`y += x @ W^T`), eliminating an intermediate tensor write and read cycle.

### 3. Block-Sparse Row (BSR) Paged Radix KV Cache
- 16-token physical KV pages managed by an atomic block allocator with Copy-on-Write (CoW) radix prefix sharing.
- Multi-turn conversational history and shared system prompts reuse existing KV pages with zero recomputation.
- **Vectorized BQ8 Prefill Dispatch:** Dispatches 8 query tokens per SIMD threadgroup with specialized fast paths for head dimensions $D=64$, $D=128$, and $D=256$ with online softmax computed entirely within threadgroup SRAM.

### 4. Continuous Batching & Async Engine Pipelining
- **ORCA Iteration-Level Selective Batching:** Dynamically interleaves chunked prefills with active decode forward passes, ensuring steady decode token latency without prefill head-of-line blocking.
- **Depth-2 Lookahead GPU Pipelining:** CPU prepares and encodes iteration $N+1$ command encoders asynchronously while the Apple Silicon GPU executes iteration $N$, fully saturating the Metal hardware pipeline.

---

## 🎯 Supported Architectures & Scope

KriyaRT is laser-focused on delivering the highest possible throughput and lowest memory footprint for dense 4-bit autoregressive decoder models on Apple Silicon:

- **Supported Models:**
  - **Qwen 2.5** (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)
  - **Llama 3 / 3.1 / 3.2** (1B, 3B, 8B, 70B)
  - **Gemma 2** (2B, 9B, 27B) — including 4 RMSNorms per layer, $\sqrt{d}$ embedding scale, and dual attention/vocab logit softcapping
  - **Mistral / SmolLM2**
- **Quantization:**
  - Native 4-bit Group-64 Affine quantization (`Q4_G64`). Directly loads MLX Community Hugging Face checkpoints without format conversion.
- **Speculative Decoding (EAGLE):**
  - Experimental opt-in feature enabled via `--draft-weights <path>` with tree-structured speculative verification.
- **Current Limitations & Roadmap:**
  - Custom architectures with fused QKV matrices (e.g., Spark X) or non-uniform quantization (OptiQ) require dedicated projection mapping currently in development.
  - Long-context prefill ($M \ge 512$) currently runs on Metal ALU SIMDgroups (~56-65% of MLX); integration of Metal Performance Shaders (MPSGraph) tensor prefill is on the active roadmap.

---

## 📦 Installation & Setup

### Option A: Install via Homebrew (Recommended)

```bash
# Add Kriya tap and install precompiled release binary
brew install rishisharma/tap/kriya
```

### Option B: Download Standalone Release Tarball

Precompiled, codesigned Apple Silicon binaries (`aarch64-apple-darwin`) are available on the [Releases](https://github.com/rishisharma/kriya-rt/releases) page:

```bash
# Download and unpack
curl -L -O https://github.com/rishisharma/kriya-rt/releases/download/v0.1.0/kriya-rt-v0.1.0-aarch64-apple-darwin.tar.gz
tar -xvf kriya-rt-v0.1.0-aarch64-apple-darwin.tar.gz
cd kriya-rt-v0.1.0-aarch64-apple-darwin/bin

# Test execution
./kriya-server --version
```

### Option C: Build from Source

**Prerequisites:** macOS on Apple Silicon (M1/M2/M3/M4), Xcode Command Line Tools (`xcode-select --install`), and Rust 1.81+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`).

```bash
git clone https://github.com/rishisharma/kriya-rt.git
cd kriya-rt

# Build release binaries targeting native aarch64
cargo build --release -p kriya-server
```

The optimized server binary will be located at `target/aarch64-apple-darwin/release/kriya-server`.

---

## 💻 Quick Start & Usage

### 1. Run Interactive CLI Generation

Point directly to any local Hugging Face or MLX community checkpoint:

```bash
./target/aarch64-apple-darwin/release/kriya-server \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit/snapshots/4f83f8f146fdf28b512a06562b671d7af4fab457 \
  --prompt "What are the core advantages of unified memory on Apple Silicon?" \
  --max-tokens 64 \
  --temperature 0.0
```

### 2. Launch Local OpenAI-Compatible Server

```bash
./target/aarch64-apple-darwin/release/kriya-server \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit/snapshots/4f83f8f146fdf28b512a06562b671d7af4fab457 \
  --host 127.0.0.1 \
  --port 8080
```

Query the local server via `curl` with streaming responses:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b-instruct",
    "messages": [
      {"role": "system", "content": "You are a concise AI assistant."},
      {"role": "user", "content": "Explain SIMDgroup matrix operations in Metal."}
    ],
    "stream": true,
    "temperature": 0.0,
    "max_tokens": 128
  }'
```

---

## 🔬 Reproducing Side-by-Side Benchmarks

To independently verify the benchmarks on your own Mac against Apple MLX:

```bash
# 1. Install MLX in a virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm

# 2. Run the automated side-by-side benchmark suite
python3 benchmarks/run_mlx_vs_kriya.py --model qwen3b
python3 benchmarks/run_mlx_vs_kriya.py --model qwen1.5b
python3 benchmarks/run_mlx_vs_kriya.py --model gemma2b
```

Benchmark outputs, TTFT, TPOT, and percentile latencies are automatically validated and written to `benchmarks/reports/`.

---

## 🛡️ Validation & Test Suite

The test suite validates hardware execution, shader compilation, and numerical bit-equivalence:

```bash
# Full workspace test suite (100% pass)
cargo test --workspace

# Real-world Metal GPU kernel validation against CPU reference
cargo test -p kriya-backend-metal --test real_world_validation -- --nocapture
```

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
