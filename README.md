# KriyaRT — Mac-Only LLM Inference Engine

KriyaRT is a high-performance Large Language Model (LLM) and Mixture-of-Experts (MoE)
inference runtime engineered specifically for **Apple Silicon (M1–M4, Pro, Max, Ultra)**. Built entirely
in **pure Rust** and **Metal Shading Language (MSL 3.1)**, it bridges the gap between low-level hardware
kernel research and developer-friendly ecosystems, delivering a zero-overhead local serving engine with request admission control and zero Python GIL.

## Design Goals

| Goal | Implementation |
|------|---------------|
| **Mac-only optimization** | Native Metal 3.1 kernels targeting Apple Silicon unified memory and SIMD-group operations |
| **Performance work** | Q4_G64 kernels, BSR paged KV cache, Depth-2 lookahead pipelining, and ORCA continuous batching |
| **Zero Python GIL** | Pure Rust async server (Axum + Tokio), no Python runtime dependencies |
| **Memory-efficient** | Zero-copy `mmap` weight loading, 4-bit Group-64 quantization, Copy-on-Write KV prefix sharing |
| **Production serving** | OpenAI-compatible `/v1/chat/completions` with SSE streaming, request caps, and admission control |

## Architecture

```
HTTP Client → Axum Server → ORCA Scheduler → Metal Backend → Apple Silicon GPU
                              ↑                                    ↓
                        EAGLE Speculative Engine ←── KV Cache (BSR Paged Radix)
```

**Crates:**
- `kriya-core` — Model descriptors, Q4Group64 quantization, safetensors loader, tokenizer, hardware profiler
- `kriya-backend-metal` — Metal 3.1 command buffer encoder, pipeline cache, GPU executor
- `kriya-kv` — FlashInfer BSR paged KV cache with radix prefix sharing & CoW
- `kriya-scheduler` — ORCA iteration-level selective batching, chunked prefill SLA preemption
- `kriya-speculate` — EAGLE multi-token speculative decoding engine
- `kriya-hybrid` — PowerInfer hot/cold neuron placement solver, FlexGen cost model
- `kriya-server` — Axum HTTP server, HF Hub fetcher, `kriya_convert` quantization CLI

## Supported Model Architectures

| Model | Parameters | Attention | Activation | Special |
|-------|-----------|-----------|------------|---------|
| LLaMA-3.2-3B | 3.21B | GQA (24/8, D=128) | SwiGLU | Dense, 28 layers |
| Qwen3.6-35B-A3B | 35.8B (3.2B active) | Hybrid GQA + DeltaNet | SwiGLU | Descriptor support; production checkpoint execution is not yet implemented |
| Gemma-2-9B-It | 9.24B | GQA (16/8, D=256) | GeGLU | Logit softcap, 42 layers |
| Mistral-7B-v0.3 | 7.25B | GQA (32/8, D=128) | SwiGLU | Sliding window 4096 |
| DeepSeek-R1-Distill-7B | 7.61B | GQA (28/4, D=128) | SwiGLU | 18,944 intermediate |

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Rust 1.81+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

### Build

```bash
cd kriya-rt
cargo build --release --workspace
```

### Run Tests

```bash
cargo test --workspace
```

The suite includes unit tests plus Metal shader compilation and GPU reference tests.

### Convert a Model

```bash
# Convert a Hugging Face checkpoint to KriyaRT's Q4Group64 format
cargo run -p kriya-server --bin kriya_convert --release -- \
  --input /path/to/hf-checkpoint \
  --output /path/to/kriya-q4-checkpoint
```

### Launch the Server

```bash
# Start the OpenAI-compatible server on http://127.0.0.1:8080
cargo run -p kriya-server --release -- \
  --model /path/to/kriya-q4-checkpoint
```

The server auto-downloads from Hugging Face Hub if the model path is not local.

### Send a Request

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kriya-model",
    "messages": [
      {"role": "user", "content": "Write a quicksort implementation in Rust."}
    ],
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

## Key Features

### 1. Q4Group64 Quantization
4-bit affine quantization with 64-element groups (32 bytes weights + 2 bytes BF16 scale + 2 bytes BF16 bias = 36 bytes per block). Effective 4.5 bits/parameter with minimal perplexity degradation. 1D norm tensors preserved in exact BF16.

### 2. BSR Paged Radix KV Cache
16-token physical KV pages with Copy-on-Write radix prefix sharing. Shared system prompts reuse existing pages. Block-Sparse Row layout consumed directly by Metal attention shaders.

### 3. ORCA Continuous Batching & Pipelining
Iteration-level selective batching with dynamic chunked-prefill (512 tokens). Large prompts are sliced and interleaved with decode steps, preventing prompt prefill starvation while maintaining steady decode latency (9.5 ms TPOT on SmolLM2-135M, 18-21 ms TPOT on Qwen2.5-0.5B on base Apple Silicon M1). Depth-2 Lookahead pipelining commits iteration $N+1$ asynchronously while iteration $N$ finishes on the GPU.

### 4. EAGLE Speculative Decoding
Multi-token draft-tree verification is supported with SHA-256 verified `kriya_eagle_draft_v1.bin` checkpoints. When speculative heads are loaded, verification runs on GPU without CPU recomputation; `--no-speculation` is available to enforce direct autoregressive decode.

### 5. Metal 3.1 Kernels
- 128-bit packed vector Q4_G64 dequantization
- Threadgroup-tiled scalar-FMA Q4 prefill GEMM (numerically validated)
- GPU-resident MoE routing (top-K → histogram → prefix scan → scatter)
- Split-K online softmax for long contexts (32K+)
- Gated DeltaNet recurrent state mixer
- GPU-resident token sampler (greedy / temperature / top-p / top-K)

## Hardware Verification

KriyaRT includes hardware shader compilation verification via `xcrun metal` (MSL 3.1, `-O3 -Werror`) and live GPU execution tests on Apple Silicon.

## Development

```bash
# Check compilation (0 warnings)
cargo check --workspace

# Run all tests
cargo test --workspace

# Build release
cargo build --release --workspace

# Format code
cargo fmt --all

# Clippy lints
cargo clippy --workspace -- -D warnings
```

## License

Apache-2.0
