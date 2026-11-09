## Description

Please include a summary of the change and which issue is fixed or addressed. Include relevant motivation and architectural context.

Fixes #(issue)

---

## Type of Change

- [ ] ⚡ Performance Optimization (MSL kernel, memory bandwidth, latency)
- [ ] 🧩 New Model Architecture / Quantization Support
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] 🛠️ Refactoring / Code Quality Improvement
- [ ] 📖 Documentation update
- [ ] 📦 CI / Packaging / Homebrew update

---

## Validation & Verification

Please describe the tests that you ran to verify your changes:

- [ ] `cargo fmt --all --check` passed cleanly
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passed cleanly
- [ ] `cargo test --workspace` passed 100% across all crates
- [ ] Real-world GPU execution verified with:
  ```bash
  cargo test -p kriya-backend-metal --test real_world_validation -- --nocapture
  ```
- [ ] Tested on physical Apple Silicon hardware (Specify: M1 / M2 / M3 / M4)

---

## Benchmark Evidence (if applicable)

If this PR claims a speedup, please include side-by-side benchmark numbers against Apple MLX or the previous commit:

| Benchmark Metric | Baseline (Before / MLX) | This PR | Speedup / Delta |
| :--- | :--- | :--- | :--- |
| **Decode Throughput (TPOT)** | tok/s | tok/s | x |
| **Prefill Latency (TTFT)** | ms | ms | x |
| **Peak RAM Footprint** | GB | GB | MB |
