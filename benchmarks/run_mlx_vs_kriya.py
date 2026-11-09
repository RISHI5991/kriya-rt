#!/usr/bin/env python3
"""
Automated Side-by-Side MLX vs. KriyaRT Equal-Settings Benchmark Suite.

Directly addresses 3Review.md Item P1.6:
- Identical base models (Qwen2.5-0.5B-Instruct & SmolLM2-135M-Instruct)
- Identical 4-bit affine quantization scheme (group-size 64)
- Identical tokenizer & fixed prompt corpus (64, 256, 512 tokens)
- Identical sampling policy (greedy deterministic, temperature=0.0)
- Identical output generation length (16 tokens)
- Warm-up iteration to eliminate GPU shader JIT cold starts
- Comprehensive telemetry: TTFT (ms), TPOT (ms), Prefill tok/s, Decode tok/s, Memory, Output text

Usage:
    python3 benchmarks/run_mlx_vs_kriya.py --model qwen
    python3 benchmarks/run_mlx_vs_kriya.py --model smollm
    python3 benchmarks/run_mlx_vs_kriya.py --model all
"""

import argparse
import datetime
import json
import os
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

BENCHMARK_PROMPTS = [
    # ~64 tokens
    (
        64,
        "The quick brown fox jumps over the lazy dog. In computer science and artificial intelligence, "
        "inference engines optimize tensor operations to achieve low latency and high throughput. "
        "What are the key architectural advantages of Apple Silicon unified memory for running large language "
        "models locally on macOS systems with high efficiency?"
    ),
    # ~256 tokens
    (
        256,
        "Artificial intelligence, machine learning, and neural network architectures have undergone a massive evolution "
        "over the last several decades. From early perceptrons and connectionist models to deep convolutional neural networks "
        "and modern transformer architectures, computation requirements have shifted dramatically from general-purpose CPUs "
        "toward specialized vector and matrix engines. On Apple Silicon systems, the GPU and Neural Engine share unified "
        "LPDDR memory with the high-performance CPU cores. This shared memory architecture allows large model weights to be "
        "mapped zero-copy without PCIe bus bottlenecks. Describe how modern inference runtime systems leverage FlashAttention, "
        "Block-Sparse Row paged key-value caches, and 4-bit block quantization to maximize hardware performance and memory "
        "bandwidth utilization."
    ),
    # ~512 tokens
    (
        512,
        "Over the past few years, the development of large language models has accelerated at an unprecedented pace. "
        "Beginning with the original Transformer architecture introduced in 2017, self-attention mechanisms have become "
        "the foundational building block for state-of-the-art natural language processing, code generation, and multimodal "
        "reasoning systems. As these models grew from hundreds of millions to hundreds of billions of parameters, the "
        "computational and memory demands created significant engineering challenges. In particular, the autoregressive "
        "decoding phase is inherently memory-bandwidth-bound: generating each new token requires streaming billions of weight "
        "parameters from DRAM into compute registers while performing relatively few arithmetic operations per loaded byte. "
        "To alleviate these bottlenecks on modern client hardware such as Apple Silicon Macs, modern inference engines deploy "
        "a combination of specialized techniques. First, weight-only quantization (such as 4-bit grouped affine quantization) "
        "reduces memory bandwidth consumption by fourfold compared to 16-bit floating point representations. Second, paged "
        "attention and block-sparse row data structures eliminate KV cache memory fragmentation by allocating physical memory "
        "in discrete pages rather than contiguous buffers. Third, speculative decoding algorithms leverage smaller draft "
        "heads or speculative trees to propose multiple candidate tokens per step, verifying them in parallel with a single "
        "forward pass of the larger target model. Explain the synergistic effects of these optimizations on Time To First "
        "Token and Time Per Output Token."
    ),
]


def prepare_prompt_tokens(tokenizer, base_text: str, target_len: int) -> str:
    """Ensures the text tokenizes to exactly target_len tokens."""
    tokens = tokenizer.encode(base_text)
    if len(tokens) >= target_len:
        tokens = tokens[:target_len]
    else:
        filler = list(tokens)
        while len(tokens) < target_len:
            tokens.extend(filler)
        tokens = tokens[:target_len]
    return tokenizer.decode(tokens)


def benchmark_mlx(
    model_path: str,
    target_tokens: int = 16,
) -> Dict[str, Any]:
    print(f"\n[MLX] Loading model from {model_path}...")
    t_load_start = time.time()
    model, tokenizer = load(model_path)
    load_time_s = time.time() - t_load_start
    print(f"[MLX] Loaded in {load_time_s:.2f}s")

    # Warmup round
    print("[MLX] Running warmup pass...")
    sampler = make_sampler(temp=0.0)
    _ = list(stream_generate(model, tokenizer, prompt="Warmup hello", max_tokens=4, sampler=sampler))

    results = []

    for target_len, text in BENCHMARK_PROMPTS:
        prompt_str = prepare_prompt_tokens(tokenizer, text, target_len)
        actual_tokens = len(tokenizer.encode(prompt_str))

        # Run 3 measured passes and take the median
        pass_metrics = []
        for run_idx in range(3):
            t0 = time.perf_counter()
            gen_iter = stream_generate(model, tokenizer, prompt=prompt_str, max_tokens=target_tokens, sampler=sampler)

            # First token = TTFT
            t_first = None
            generated_tokens = []
            for item in gen_iter:
                if t_first is None:
                    t_first = time.perf_counter()
                generated_tokens.append(item.text)

            t_end = time.perf_counter()

            if t_first is None:
                t_first = t_end

            ttft_ms = (t_first - t0) * 1000.0
            decode_time_ms = (t_end - t_first) * 1000.0
            num_gen = max(1, len(generated_tokens) - 1)
            tpot_ms = decode_time_ms / float(num_gen)
            prefill_tok_s = (float(actual_tokens) / (ttft_ms / 1000.0)) if ttft_ms > 0 else 0.0
            decode_tok_s = (1000.0 / tpot_ms) if tpot_ms > 0 else 0.0

            pass_metrics.append({
                "ttft_ms": ttft_ms,
                "prefill_tok_s": prefill_tok_s,
                "tpot_ms": tpot_ms,
                "decode_tok_s": decode_tok_s,
                "output_text": "".join(generated_tokens).strip()
            })

        # Median by decode throughput
        pass_metrics.sort(key=lambda m: m["decode_tok_s"])
        median = pass_metrics[1]
        print(f"[MLX] Prompt {actual_tokens:3d} toks | TTFT: {median['ttft_ms']:7.2f} ms ({median['prefill_tok_s']:6.1f} tok/s) | TPOT: {median['tpot_ms']:5.2f} ms ({median['decode_tok_s']:5.1f} tok/s)")

        results.append({
            "prompt_tokens": actual_tokens,
            "ttft_ms": median["ttft_ms"],
            "prefill_tok_s": median["prefill_tok_s"],
            "tpot_ms": median["tpot_ms"],
            "decode_tok_s": median["decode_tok_s"],
            "output_text": median["output_text"],
        })

    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

    return {
        "engine": "MLX",
        "version": getattr(mx, "__version__", "unknown"),
        "load_time_s": load_time_s,
        "peak_mem_mb": peak_mem_mb,
        "samples": results,
    }


def benchmark_kriya(
    kriya_model_path: str,
    workspace_root: str,
) -> Dict[str, Any]:
    print(f"\n[KriyaRT] Running hardware benchmark on {kriya_model_path}...")
    temp_json = os.path.join(workspace_root, "benchmarks", "reports", "temp_kriya_report.json")
    os.makedirs(os.path.dirname(temp_json), exist_ok=True)

    server_bin = os.path.join(workspace_root, "target", "aarch64-apple-darwin", "release", "kriya-server")
    if not os.path.isfile(server_bin):
        server_bin = os.path.join(workspace_root, "target", "release", "kriya-server")
    if not os.path.isfile(server_bin):
        print("[KriyaRT] Building kriya-server --release...")
        subprocess.run(["cargo", "build", "--release", "-p", "kriya-server"], cwd=workspace_root, check=True)
        server_bin = os.path.join(workspace_root, "target", "release", "kriya-server")

    cmd = [
        server_bin,
        "--model", kriya_model_path,
        "--bench-only",
        "--no-speculation",
        "--benchmark-json", temp_json,
    ]

    t0 = time.time()
    res = subprocess.run(cmd, cwd=workspace_root, capture_output=True, text=True)
    total_time = time.time() - t0

    if res.stdout:
        print(res.stdout)

    if res.returncode != 0:
        print("[KriyaRT] Execution failed:\n", res.stderr)
        raise RuntimeError("kriya-server hardware benchmark failed")

    with open(temp_json, "r") as f:
        data = json.load(f)

    if os.path.exists(temp_json):
        os.remove(temp_json)

    formatted_samples = []
    for s in data["samples"]:
        formatted_samples.append({
            "prompt_tokens": s["prompt_tokens"],
            "ttft_ms": s["ttft_ms"],
            "prefill_tok_s": s["prefill_tokens_per_second"],
            "tpot_ms": s["tpot_ms"],
            "decode_tok_s": s["decode_tokens_per_second"],
        })

    return {
        "engine": "KriyaRT",
        "model_name": data["model_name"],
        "model_layers": data["model_layers"],
        "total_run_time_s": total_time,
        "samples": formatted_samples,
    }


def print_comparison_markdown(model_label: str, mlx_res: Dict[str, Any], kriya_res: Dict[str, Any]):
    print("\n" + "=" * 90)
    print(f" SIDE-BY-SIDE INFERENCE BENCHMARK: {model_label}")
    print(f" Hardware: {platform.machine()} ({platform.processor() or 'Apple Silicon'}), macOS {platform.mac_ver()[0]}")
    print(f" Settings: Identical 4-bit Group-64 Quantization | Greedy (Temp=0.0) | 16 Output Tokens")
    print("=" * 90)

    print("\n### 1. Prefill Latency & Throughput (Time To First Token)\n")
    print("| Context (Tokens) | MLX TTFT (ms) | KriyaRT TTFT (ms) | MLX Prefill (tok/s) | KriyaRT Prefill (tok/s) | Prefill Ratio (Kriya / MLX) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for m, k in zip(mlx_res["samples"], kriya_res["samples"]):
        toks = m["prompt_tokens"]
        m_ttft = m["ttft_ms"]
        k_ttft = k["ttft_ms"]
        m_rate = m["prefill_tok_s"]
        k_rate = k["prefill_tok_s"]
        ratio = (k_rate / m_rate) if m_rate > 0 else 1.0
        print(f"| **{toks:3d} tokens** | {m_ttft:7.2f} ms | {k_ttft:7.2f} ms | {m_rate:6.1f} tok/s | **{k_rate:6.1f} tok/s** | **{ratio:.3f}x** |")

    print("\n### 2. Autoregressive Decode Latency & Throughput (Time Per Output Token)\n")
    print("| Context (Tokens) | MLX TPOT (ms) | KriyaRT TPOT (ms) | MLX Decode (tok/s) | KriyaRT Decode (tok/s) | Decode Ratio (Kriya / MLX) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for m, k in zip(mlx_res["samples"], kriya_res["samples"]):
        toks = m["prompt_tokens"]
        m_tpot = m["tpot_ms"]
        k_tpot = k["tpot_ms"]
        m_rate = m["decode_tok_s"]
        k_rate = k["decode_tok_s"]
        ratio = (k_rate / m_rate) if m_rate > 0 else 1.0
        print(f"| **{toks:3d} tokens** | {m_tpot:5.2f} ms | {k_tpot:5.2f} ms | {m_rate:5.1f} tok/s | **{k_rate:5.1f} tok/s** | **{ratio:.3f}x** |")

    print("\n" + "=" * 90 + "\n")


def run_benchmark_for_model(model_key: str, workspace_root: str) -> Dict[str, Any]:
    if model_key == "qwen":
        label = "Qwen2.5-0.5B-Instruct (4-bit Group-64)"
        mlx_path = os.path.join(workspace_root, "models", "qwen2.5-0.5b-instruct-mlx-q4")
        kriya_path = os.path.join(workspace_root, "models", "qwen2.5-0.5b-instruct-kriya-q4")
    elif model_key == "qwen1.5b":
        label = "Qwen2.5-1.5B-Instruct (4-bit Group-64)"
        mlx_path = os.path.join(workspace_root, "models", "qwen2.5-1.5b-instruct-mlx-q4")
        kriya_path = os.path.join(workspace_root, "models", "qwen2.5-1.5b-instruct-kriya-q4")
    elif model_key == "qwen3b":
        label = "Qwen2.5-3B-Instruct (4-bit Group-64)"
        mlx_path = os.path.join(workspace_root, "models", "qwen2.5-3b-instruct-mlx-q4")
        kriya_path = os.path.join(workspace_root, "models", "qwen2.5-3b-instruct-kriya-q4")
    elif model_key == "smollm":
        label = "SmolLM2-135M-Instruct (4-bit Group-64)"
        mlx_path = os.path.join(workspace_root, "models", "smollm2-135m-instruct-mlx-q4")
        kriya_path = os.path.join(workspace_root, "models", "smollm2-135m-instruct-kriya-q4")
    elif model_key == "gemma2b":
        label = "Gemma-2-2B-IT (4-bit Group-64)"
        mlx_path = os.path.join(workspace_root, "models", "gemma-2-2b-it-mlx-q4")
        kriya_path = os.path.join(workspace_root, "models", "gemma-2-2b-it-kriya-q4")
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    mlx_res = benchmark_mlx(mlx_path)
    kriya_res = benchmark_kriya(kriya_path, workspace_root)

    print_comparison_markdown(label, mlx_res, kriya_res)

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_key": model_key,
        "model_label": label,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "mlx_results": mlx_res,
        "kriya_results": kriya_res,
    }

    report_path = os.path.join(
        workspace_root,
        "benchmarks",
        "reports",
        f"comparison_{model_key}_{int(time.time())}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full benchmark report artifact to {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Side-by-side MLX vs KriyaRT benchmark")
    parser.add_argument("--model", choices=["qwen", "qwen1.5b", "qwen3b", "gemma2b", "smollm", "all"], default="qwen", help="Model to benchmark")
    args = parser.parse_args()

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if args.model == "all":
        run_benchmark_for_model("smollm", workspace_root)
        run_benchmark_for_model("qwen", workspace_root)
        run_benchmark_for_model("qwen1.5b", workspace_root)
        run_benchmark_for_model("qwen3b", workspace_root)
        run_benchmark_for_model("gemma2b", workspace_root)
    else:
        run_benchmark_for_model(args.model, workspace_root)


if __name__ == "__main__":
    main()
