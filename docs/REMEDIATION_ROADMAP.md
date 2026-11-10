# KriyaRT — Comprehensive Engineering Remediation Roadmap

## Release Hardening & Architecture Roadmap

**Baseline:** 10 November 2026 · **Commit:** `68cbc71` · **Repository:** RISHI5991/kriya-rt

My recommendation is to treat the next development cycle as a correctness, verification and release-engineering programme, not another feature-development sprint.

KriyaRT already has substantial architectural work. Replacing the architecture or introducing more features right now would be counterproductive.

The objective is to make the existing implementation:

1. Correct under real model weights.  
2. Safe under malformed inputs and missing checkpoint tensors.  
3. Reliable under concurrent inference.  
4. Reproducible across development machines.  
5. Measurably competitive against MLX where the evidence supports that claim.  
6. Installable by an independent developer without manual intervention.

## **1\. The target state**

Before planning individual fixes, establish what a successful KriyaRT v0.1.0 should mean.

## KriyaRT v0.1.0 — Definition of Done

* Build integrity  
  All four CI jobs pass on the same `main` commit.  
* Production checkpoint integrity  
  Missing, malformed or unsupported model tensors cause explicit errors. No synthetic weights can enter the production path.  
* GPU correctness  
  Metal kernels pass CPU-reference comparisons with defined numerical tolerances.  
* Inference correctness  
  At least the explicitly supported real-model checkpoints load and produce valid outputs through the strict production executor.  
* Concurrency  
  Multiple requests can execute with isolated KV state, correct output association and reliable cancellation.  
* API reliability  
  The actual production HTTP routes handle invalid requests, saturation, streaming and disconnects correctly.  
* Benchmark reproducibility  
  Published measurements can be reproduced from committed scripts, exact model artifacts and recorded machine metadata.  
* Distribution  
  A release artifact can be downloaded, verified, installed and executed on a clean Apple Silicon Mac.

These are release acceptance conditions, not features to add to the marketing checklist.

# **2\. Priority structure**

I would divide all remaining work into four engineering priorities.

| Priority | Objective | Release Significance |
| :---: | ----- | ----- |
| **P0** | CI, checkpoint integrity, numerical correctness | Blocking |
| **P1** | Concurrency, API, KV cache, real inference | Blocking |
| **P2** | Benchmark integrity and performance optimization | Required for credible claims |
| **P3** | Packaging, security, repository governance, documentation | Required for public release |

The work should follow dependencies. In particular, don't optimize a kernel before knowing that its output is correct, and don't publish a performance comparison before stabilizing the inference path.

## **Overall execution sequence**

*\[Execution Sequence Flowchart: Baseline (commit 68cbc71) ➔ P0: Establish Green CI & Strict Checkpoint Loading ➔ P1: GPU, Real-Model, Scheduler & API Integration ➔ P2: Reproducible Benchmarks & Optimization ➔ P3: Release Engineering & Governance ➔ Stable v0.1.0 Release Candidate\]*

# **3\. P0 — CI reliability and build integrity**

First blocker

### Problem

The latest CI run on `68cbc71` was cancelled.

Your current workflow has four jobs:

* Code Formatting  
* Clippy Linter  
* Workspace Tests (Apple Silicon)  
* Release Packaging Verification

The workflow configuration is considerably improved, but there is no completed successful run proving that the whole pipeline works.

### Task 1.1 — Investigate workflow cancellation

First establish why the runs are being cancelled.

Do not assume that this is a Rust compilation problem.

Run:

```
gh run list \
  --repo RISHI5991/kriya-rt \
  --workflow ci.yml \
  --limit 10
```

Inspect the latest run:

```
gh run view 42765432856 \
  --repo RISHI5991/kriya-rt
```

Retry the cancelled run:

```
gh run rerun 42765432856 \
  --repo RISHI5991/kriya-rt
```

Then monitor it:

```
gh run watch 42765432856 \
  --repo RISHI5991/kriya-rt \
  --exit-status
```

Check each job individually if the complete workflow fails.

Important: The existing run records do not establish who or what cancelled the workflows. Investigate the actual cancellation source before changing the build configuration to compensate for a guessed cause.

### Task 1.2 — Add manual CI execution

Add `workflow_dispatch` to `.github/workflows/ci.yml`.

```
name: CI

on:
  push:
    branches: [main, master]

  pull_request:
    branches: [main, master]

  workflow_dispatch:
```

This gives you an explicit way to verify a particular commit without needing to create an artificial code change.

Consider concurrency settings as well. If newer pushes are cancelling previous runs, define that policy deliberately. Do not let main-branch verification disappear simply because another commit arrived.

### Task 1.3 — Make CI strict and reproducible

Keep the existing:

```
runs-on: macos-14
```

And retain:

```
KRIYA_REQUIRE_METAL: "1"
```

The latter is important because Metal tests should fail when GPU access is unavailable, not silently count as successful tests.

I recommend these additional checks:

```
cargo fmt --all --check

cargo clippy \
  --workspace \
  --all-targets \
  -- -D warnings

KRIYA_REQUIRE_METAL=1 \
  cargo test --workspace
```

The release package should also be validated with the exact version, target and dependency lockfile used for distribution.

If `Cargo.lock` is committed, use `--locked` in the CI build and packaging commands.

### Task 1.4 — Preserve CI artifacts

Modify the package-check job to upload:

* Release tarball.  
* SHA-256 checksum.  
* Build metadata.  
* Test summary.

Use GitHub Actions artifact storage rather than requiring someone to reconstruct the result manually.

### Acceptance criteria

P0.1 is complete when:

* All four CI jobs return `success`.  
* No Metal test is silently skipped.  
* The build uses the intended Apple Silicon target.  
* Package verification produces a real artifact.  
* Results are associated with a specific Git commit.

Expected deliverable: One reproducible green CI run on `main`.

# **4\. P0 — Strict production checkpoint loading**

Security and correctness blocker

This is the most important source-code remediation.

### Current implementation

Your latest commit introduced:

`ModelGpuWeights::from_checkpoint_dir_strict()`

This is a good architectural decision.

However, the latest code still contains places where errors are converted into absent optional values.

The most obvious example is in:

`crates/kriya-backend-metal/src/encoder.rs`

```
let post_att = bind_bf16(
    &format!("{p}.post_attention_layernorm.weight"), h
).ok();

let post_ff = bind_bf16(
    &format!("{p}.post_feedforward_layernorm.weight"), h
).ok();
```

This is not acceptable for required checkpoint tensors.

### Task 2.1 — Make normalization loading descriptor-driven

The model descriptor should be the source of truth.

The loader should understand whether the model requires:

* Input layer normalization.  
* Pre-feedforward normalization.  
* Post-attention normalization.  
* Post-feedforward normalization.  
* Query/key normalization.

Do not infer optionality solely from whether a tensor happens to exist.

Conceptually:

```
let post_attn_norm = match layer.post_attn_norm {
    Some(_) => Some(bind_bf16(
        &format!("{p}.post_attention_layernorm.weight"),
        hidden_size,
    )?),
    None => None,
};
```

Apply the same logic to `post_ffn_norm`.

The actual implementation must account for each supported architecture's normalization layout.

### Task 2.2 — Tighten Q4 tensor binding

Review the direct Q4 binding branches in `bind_q4`.

For every projection, validate:

| Property | Required Validation |
| ----- | ----- |
| Tensor presence | Mandatory |
| Quantization format | Q4Group64 or explicitly supported MLX format |
| Shape | Exact expected shape |
| Raw byte length | Exact expected packed length |
| Scales | Present and valid for MLX triplets |
| Biases | Present and valid for MLX triplets |
| Packed output | Expected block count |

For the MLX affine path, validate the scale and bias tensor dimensions, data types and block correspondence before constructing the GPU buffer.

Avoid assuming that a successful metadata lookup proves that conversion has succeeded.

### Task 2.3 — Enforce a single production validation boundary

Another improvement:

`from_checkpoint_dir_strict()` is public, and it accepts a descriptor and checkpoint directory.

The production executor validates before calling it, but the strict loader should not rely on every caller remembering to perform that validation.

Choose one of these approaches:

1. Make the strict loading function internal and expose only the validated production constructor.  
2. Have strict loading invoke the complete production checkpoint validator itself.

The second approach offers a stronger independent safety boundary, although duplicate validation should be avoided or organized into a shared internal function.

### Task 2.4 — Build a malformed-checkpoint test matrix

Create a dedicated checkpoint-loading test suite.

| Test Case | Expected Result |
| ----- | ----- |
| Empty checkpoint | Reject |
| Missing embedding tensor | Reject |
| Missing attention projection | Reject |
| Missing MLP projection | Reject |
| Missing required norm | Reject |
| Wrong tensor shape | Reject |
| Wrong quantization dtype | Reject |
| Missing MLX scales | Reject |
| Missing MLX biases | Reject |
| Malformed scale shape | Reject |
| Incorrect packed block count | Reject |
| Incomplete Gemma normalization layout | Reject |
| Complete supported Q4 checkpoint | Accept |
| Complete valid MLX checkpoint | Accept |

The test should verify an error identifies the offending tensor wherever practical.

### Task 2.5 — Test the production constructor, not just the validator

There is a distinction between:

```
validate_production_checkpoint(...)
```

and:

```
MetalExecutor::new_production(...)
```

The validator can accept a checkpoint while the actual GPU binding still fails.

Test both.

### Task 2.6 — Make speculative artifacts fail closed

Your draft-artifact checksum handling is improved. However, decide explicitly what happens when a draft file exists but its manifest does not.

I recommend:

* Base model inference remains available.  
* Speculative decoding is disabled when its integrity cannot be established.  
* An explicitly requested invalid draft artifact generates a clear diagnostic.  
* A present malformed manifest never silently bypasses verification.

Extend the manifest to record:

```
{
  "schema_version": 1,
  "draft_file": "kriya_eagle_draft_v1.bin",
  "sha256": "...",
  "size_bytes": 0,
  "model_identity": "...",
  "tokenizer_identity": "...",
  "architecture": "...",
  "format_version": 1
}
```

Treat SHA-256 as an integrity check, not proof of publisher authenticity. For distribution, add signed release provenance separately.

### Acceptance criteria

* No required tensor is loaded through a silent fallback.  
* Every supported architecture has explicit normalization requirements.  
* Malformed MLX checkpoints fail before execution.  
* Strict production construction rejects incomplete checkpoints.  
* Optional speculative acceleration cannot silently use an unverified draft artifact.

Expected deliverable: A fail-closed checkpoint subsystem with regression tests.

# **5\. P1 — Real-model inference validation**

Release blocker

This is where the project needs to move beyond synthetic test fixtures.

Your repository now has more comprehensive miniature-checkpoint tests. That is useful, but it is not equivalent to proving that actual Hugging Face or MLX checkpoints run correctly.

### Task 5.1 — Fix the real-Qwen integration test

I found an important remaining issue.

The current:

`crates/kriya-backend-metal/tests/test_real_qwen.rs`

has been updated with the GPU requirement helper, but the test still constructs its executor using:

```
MetalExecutor::new(
    &desc,
    &kernels_dir,
    Some(&model_dir),
    512,
    16,
)
```

That uses the permissive constructor.

For production validation, change it to:

```
MetalExecutor::new_production(
    &desc,
    &kernels_dir,
    &model_dir,
    512,
    16,
)
```

This ensures that your real-model integration test exercises the strict loading path rather than the test-oriented fallback path.

### Task 5.2 — Establish a small but representative model suite

Do not attempt to validate every architecture simultaneously.

Start with three checkpoints.

| Model | Purpose | Priority |
| ----- | ----- | :---: |
| Qwen 2.5 0.5B Instruct | Basic production loading and inference | **P0** |
| Qwen 2.5 3B Instruct | Larger model, memory and performance validation | **P0** |
| Gemma 2 2B IT | Architecture-specific normalization and performance validation | **P1** |

All checkpoints must have exact model identities and hashes recorded.

### Task 5.3 — Validate the complete inference lifecycle

For each checkpoint, verify:

Model initialization

* Configuration parsing.  
* Tokenizer loading.  
* Safetensors loading.  
* Quantization format.  
* GPU buffer allocation.  
* Weight binding.

Prompt processing

* Tokenizer output.  
* Prefill execution.  
* KV-cache population.  
* Position encoding.  
* Numerical stability.

Autoregressive generation

* Greedy sampling.  
* Temperature-based sampling.  
* EOS handling.  
* Maximum token enforcement.  
* Stop-condition handling.

Output

* Valid token IDs.  
* Valid decoded text.  
* No NaN or infinity propagation.  
* Correct output termination.  
* Deterministic results under deterministic sampling settings.

### Task 5.4 — Add differential correctness testing

Performance comparison against MLX is only useful if the underlying inference is numerically correct.

For supported architectures, compare selected intermediate values and logits against MLX.

Record:

| Metric | Purpose |
| ----- | ----- |
| Maximum absolute error | Worst numerical difference |
| Maximum relative error | Relative numerical deviation |
| Mean squared error | Aggregate numerical error |
| Top-1 agreement | Sampling decision agreement |
| Top-k agreement | Logit ranking consistency |

Set architecture- and precision-specific tolerances. Do not impose one arbitrary threshold on every quantized operation.

### Task 5.5 — Produce a hardware compatibility matrix

Your README describes M1–M4 support.

That statement needs a clear evidence boundary.

Create something like:

| Hardware | Validation Status | Evidence |
| ----- | ----- | ----- |
| M1 Air, 8 GB | Target validation device | Full test suite |
| M2 | Pending physical verification | To be tested |
| M3 | Pending physical verification | To be tested |
| M4 | Pending physical verification | To be tested |

Do not confuse compilation compatibility with verified runtime compatibility.

### Acceptance criteria

* Real Qwen 0.5B passes through `new_production`.  
* Qwen 3B completes real inference.  
* Gemma passes its architecture-specific checkpoint validation.  
* The tested model outputs are reproducible.  
* Supported hardware claims reflect actual test evidence.

Expected deliverable: A real-model validation suite and an evidence-backed compatibility matrix.

# **6\. P1 — Metal kernel correctness and GPU testing**

High priority

Your current tests cover several important paths:

* Q4 decode GEMV.  
* Prefill GEMM.  
* BSR paged attention.  
* EAGLE verification.  
* Batched speculative features.  
* Memory-budget enforcement.

Keep these. Extend them systematically.

### Task 6.1 — Create a kernel-level validation matrix

Every production-critical GPU kernel should have a corresponding numerical reference test.

| Kernel / Function | Validation Required |
| ----- | ----- |
| Q4 GEMV | CPU reference comparison |
| Q4 GEMM | CPU reference comparison |
| QKV fusion | Compare against separate projections |
| Gate/Up SwiGLU | Compare against unfused reference |
| RMSNorm | Floating-point reference |
| RoPE | Position and dimension edge cases |
| BSR GQA | Attention reference comparison |
| KV append | Layout and boundary validation |
| Token sampler | Deterministic reference tests |
| Tree verification | Target-model logit verification |

### Task 6.2 — Test boundary conditions

For every matrix-based kernel, test:

* Minimum supported dimensions.  
* Multiple of SIMD width.  
* Non-multiple dimensions where supported.  
* Partial threadgroups.  
* Maximum supported shape.  
* Zero-length or invalid input rejection.  
* Alignment errors.  
* Extreme quantized values.  
* NaN and infinity handling.

This matters because many GPU bugs only occur at tile boundaries.

### Task 6.3 — Validate fused kernels

For fused QKV and Gate/Up paths, compare:

*Y\_fused ≈ Y\_unfused*

Do not assume that equivalent mathematical intent guarantees equivalent GPU execution.

### Task 6.4 — Test GPU memory boundaries

Verify:

* Buffer capacities.  
* Index calculations.  
* Byte offsets.  
* Tensor strides.  
* Row indexing.  
* KV-page addresses.  
* Shared-memory assumptions.

Where practical, run the tests using Apple's Metal API validation facilities and Instruments GPU diagnostics.

### Acceptance criteria

* Every critical kernel has an independently testable correctness reference.  
* Supported dimensions have explicit coverage.  
* No out-of-bounds accesses occur in validation runs.  
* Numerical errors are within documented tolerances.

# **7\. P1 — Scheduler, batching and KV-cache validation**

Core runtime blocker

This is where KriyaRT can demonstrate that it is more than a single-request inference implementation.

Your architecture includes ORCA scheduling, paged KV cache, copy-on-write prefix sharing and speculative decoding.

Those capabilities need to work together.

### Task 7.1 — Build a scheduler scenario matrix

Test the following workloads:

| Scenario | Requests | Expected Behaviour |
| ----- | :---: | ----- |
| Single request | 1 | Normal decode |
| Simultaneous requests | 2 | Correct independent outputs |
| Concurrent batch | 4 | Correct row-to-request mapping |
| Larger batch | 8 | Correct batching and resource management |
| Mixed workload | 4 decode \+ 1 prefill | Fair scheduling |
| Uneven sequence lengths | Variable | Correct iteration advancement |
| Client cancellation | 1 of 4 | Immediate cleanup |
| Queue saturation | \> Config limit | Controlled rejection |

The latest executor-level test already covers multiple requests and distinct hidden features. Build on it rather than duplicating its logic.

### Task 7.2 — Test KV-cache invariants

The KV-cache should satisfy several important invariants.

Isolation

One request must never read or overwrite another request's KV state.

Prefix sharing

Identical prefixes should be reusable without corrupting independent branches.

Copy-on-write

When two requests diverge, modifying one branch must not modify the other.

Page lifecycle

Allocated pages should be reclaimed after completed or aborted requests.

Memory pressure

When page capacity is exhausted, the scheduler must reject, defer or otherwise handle the request according to an explicit policy.

Create tests for:

* Prefix sharing.  
* Branch divergence.  
* Copy-on-write.  
* Repeated allocation and release.  
* Page recycling.  
* Capacity exhaustion.  
* Long-context sequences.  
* Cancellation during active decoding.

### Task 7.3 — Test speculative decoding independently

Keep speculative decoding disabled when no valid trained draft head is available.

Then validate:

* Draft-head compatibility.  
* Target verification.  
* Accepted token count.  
* Rejected token rollback.  
* KV-cache consistency.  
* End-of-sequence handling.  
* Speculative fallback to ordinary decoding.

Measure acceptance rates, verification overhead and total throughput.

A higher acceptance rate does not automatically imply a faster runtime.

### Task 7.4 — Test the scheduler under realistic contention

Measure:

* Request queue wait time.  
* Time per iteration.  
* Per-request throughput.  
* Batch size distribution.  
* Request starvation.  
* Cancellation latency.  
* Memory allocated per active sequence.

Introduce a small benchmark suite for 1, 2, 4 and 8 concurrent requests.

### Acceptance criteria

A batch of concurrent requests must complete without cross-request contamination, and cancellation must release its resources.

Expected deliverable: A scheduler/KV-cache correctness and concurrency test suite.

# **8\. P1 — Test the actual HTTP server**

This is a separate issue from testing the inference executor.

I found that `test_api_safety.rs` defines local test handlers.

Those tests demonstrate request-validation ideas, but they do not establish that the actual production HTTP router enforces the same behaviour.

### Task 8.1 — Separate router construction from server startup

Refactor the server so that you can construct the actual router independently.

A conceptual structure:

```
kriya-server/
├── src/
│   ├── main.rs
│   ├── server/
│   │   ├── mod.rs
│   │   ├── router.rs
│   │   ├── handlers.rs
│   │   ├── admission.rs
│   │   └── metrics.rs
│   ├── runtime/
│   └── cli/
└── tests/
    ├── api_integration.rs
    ├── streaming.rs
    └── cancellation.rs
```

Do not duplicate production handler logic inside test files.

### Task 8.2 — Introduce an inference-engine abstraction

Allow the actual HTTP handlers to operate against an injected inference backend.

The test backend can be deterministic and lightweight.

This lets you test production routing, admission control and streaming without requiring a GPU for every API test.

## Task 8.3 — Implement API integration tests

| Test Condition | Expected Outcome |
| ----- | ----- |
| Empty messages | HTTP 400 |
| Invalid request schema | HTTP 400 |
| Excessive completion tokens | HTTP 400 |
| Prompt too long | HTTP 400 |
| Queue saturated | HTTP 429 |
| Valid request | HTTP 200 |
| Streaming request | Valid SSE response |
| Client disconnect | Request cancelled |
| Model initialization failure | Clear server error |
| Shutdown | No orphaned active requests |

Also validate error JSON against a documented API structure.

### Task 8.4 — Add operational metrics

Expose meaningful metrics:

```
kriya_active_requests
kriya_total_requests
kriya_rejected_requests_total
kriya_prompt_tokens_total
kriya_generated_tokens_total
kriya_batch_size
kriya_kv_pages_allocated
kriya_kv_pages_reclaimed
kriya_request_latency_seconds
```

Add these only where they can be measured correctly. Do not expose metrics that are always zero or inferred inaccurately.

### Acceptance criteria

The real production router passes integration tests for validation, queue saturation, streaming, request cancellation and cleanup.

Expected deliverable: An independently testable production API layer.

# **9\. P2 — Benchmark methodology and empirical validation**

Required before performance claims

This is where the public credibility of KriyaRT will be won or lost.

Your current reports establish that KriyaRT can approach MLX decode performance on some Qwen configurations.

They do not establish universal performance superiority.

### Current benchmark problems

The existing harness has:

* Three measured runs.  
* Sixteen generated tokens.  
* A short warm-up.  
* Prompt token-count mismatches in the saved reports.  
* Insufficient comparable memory measurement.

The latest Gemma comparison also shows a substantial regression.

### Task 9.1 — Rebuild the measurement protocol

I recommend the following baseline:

| Parameter | New Protocol |
| ----- | ----- |
| Warm-up | 2 complete warm-up runs |
| Measured runs | At least 10 per configuration |
| Output generation | 128 tokens minimum |
| Extended run | 256 tokens where feasible |
| Prompt lengths | Exactly matched |
| Sampling | Same deterministic greedy policy |
| Quantization | Same quantization representation |
| Model | Same exact checkpoint and tokenizer |
| Run order | Alternate engine order |
| Hardware | Same physical device |
| Results | Raw measurements retained |

The exact benchmark budget can be adjusted for larger models, but don't use extremely short generations to justify general performance claims.

### Task 9.2 — Match tokenizer inputs exactly

This needs explicit correction.

Your reports include examples such as:

* Qwen: 255 versus 256 prompt tokens.  
* Gemma: 65 versus 64 prompt tokens.

The benchmark should record identical token IDs, not merely approximately equal prompt lengths.

A useful addition is a tokenizer parity test that compares token IDs across the two engine paths.

### Task 9.3 — Measure separate performance dimensions

Report these independently.

| Metric | What It Measures |
| ----- | ----- |
| Cold startup | Process start and initialization |
| Model loading | Checkpoint and buffer binding |
| TTFT | Time to first output token |
| Prefill | Prompt processing throughput |
| TPOT | Average time per generated token |
| Decode throughput | Generated tokens per second |
| Concurrent throughput | Aggregate output under multiple requests |
| Memory | Process and unified-memory footprint |

Do not mix model-loading time with steady-state token generation.

### Task 9.4 — Fix memory measurement

The current Python `resource.getrusage` measurement does not give a directly comparable view of the KriyaRT subprocess.

Use a consistent measurement protocol for both engines.

Record:

* Process resident/physical footprint.  
* System memory pressure.  
* Metal buffer allocation totals.  
* Model weight residency.  
* KV-cache residency.  
* Peak memory during generation.

Use macOS tools such as `footprint`, `vmmap`, and Instruments where appropriate, and add explicit Metal allocation counters inside KriyaRT.

### Task 9.5 — Save raw benchmark results

Every run should produce a report containing:

```
{
  "model": "Qwen2.5-3B-Instruct",
  "model_revision": "...",
  "tokenizer_revision": "...",
  "engine": "KriyaRT",
  "commit": "68cbc71",
  "hardware": "Apple M1",
  "memory_gb": 8,
  "macos_version": "...",
  "warmup_runs": 2,
  "measured_runs": 10,
  "output_tokens": 128,
  "results": {
    "ttft_ms": [],
    "prefill_tokens_per_second": [],
    "tpot_ms": [],
    "decode_tokens_per_second": [],
    "peak_memory_mb": []
  }
}
```

Also report medians, p95, interquartile range and confidence intervals where appropriate.

Do not retain only summary numbers.

### Task 9.6 — Profile the two most visible performance gaps

Qwen 3B prefill

Qwen 3B Prefill Performance (160.6 tok/s vs MLX 273.0 tok/s)

Gemma 2B decode

Gemma 2B Decode Performance (17.51 tok/s vs MLX 31.69 tok/s)

These are the existing reported results, not fresh measurements after the latest commit.

Investigate them separately.

For prefill:

* Profile projection kernels.  
* Examine matrix tile utilization.  
* Measure memory access patterns.  
* Evaluate MPSGraph or Metal Performance Shaders.  
* Investigate double buffering.  
* Examine attention's contribution to TTFT.

For Gemma decode:

* Profile Gemma normalization paths.  
* Measure attention and projection costs independently.  
* Check buffer alignment and weight layout.  
* Inspect kernel occupancy.  
* Examine the effects of its normalization architecture.

Use Instruments and Metal GPU profiling rather than guessing at bottlenecks.

### Task 9.7 — Define honest performance targets

Do not set a single target for all models.

Define model-specific performance envelopes:

* Decode throughput.  
* Prefill throughput.  
* Peak memory.  
* Time to first token.  
* Concurrent throughput.

A public statement that KriyaRT is competitive on a particular Qwen configuration should require repeated, reproducible measurements of that exact configuration.

### Acceptance criteria

* Tokenizer inputs match.  
* All measurements have raw repetitions.  
* Memory uses comparable metrics.  
* Qwen and Gemma results are reproduced on the latest commit.  
* Performance claims distinguish model, hardware and context length.

Expected deliverable: A reproducible MLX-versus-KriyaRT benchmark suite.

# **10\. P2 — Performance optimization**

After correctness and benchmark stabilization

Once the measurements are trustworthy, optimize the actual bottlenecks.

Do not introduce speculative optimizations before profiling.

### Task 10.1 — Prioritize the optimization backlog

My suggested order:

| Priority | Area | Reason |
| :---: | ----- | ----- |
| **1** | Prefill GEMM | Largest demonstrated Qwen prefill gap |
| **2** | Gemma normalization and decode | Largest demonstrated cross-model decode regression |
| **3** | KV-cache memory behavior | Important for longer context and concurrent serving |
| **4** | Fused QKV and MLP kernels | Improve decode execution efficiency |
| **5** | Speculative decoding | Additional throughput when acceptance economics justify it |
| **6** | Startup and model loading | Improve interactive usability |

### Task 10.2 — Optimize prefill

Your roadmap already identifies several promising directions:

* MPSGraph-based matrix prefill.  
* Double-buffered DRAM-to-SRAM tiling.  
* Asynchronous attention prefetch.

Retain these as candidate optimizations.

But before implementation, benchmark the individual operators and establish a baseline for each.

For MPSGraph, compare:

* Kernel execution latency.  
* Dispatch overhead.  
* Memory transfers.  
* Small versus large prompt performance.  
* Numerical accuracy.

The crossover point between conventional Metal kernels and MPSGraph must be measured, not assumed.

### Task 10.3 — Optimize Gemma

Create a dedicated Gemma performance investigation.

Do not solve the Gemma regression by lowering precision without measuring numerical consequences.

Examine:

1. Normalization execution.  
2. Intermediate tensor materialization.  
3. Projection layout.  
4. Attention implementation.  
5. Memory traffic.  
6. GPU occupancy.

Create separate microbenchmarks for the expensive operations.

### Task 10.4 — Improve KV-cache efficiency

Profile:

* Allocated versus utilized KV pages.  
* Page fragmentation.  
* Prefix reuse rate.  
* Copy-on-write overhead.  
* Memory reclaimed after cancellation.  
* Cache allocation latency.

Measure actual memory saved by prefix sharing.

A feature should not be marketed as reducing memory requirements merely because its data structure is called a paged cache.

### Task 10.5 — Establish performance regression testing

Once your benchmarks are stable, introduce a dedicated performance comparison against the latest known-good commit.

Use a tolerance band rather than demanding identical measurements.

For example:

* Flag regressions exceeding 5%.  
* Re-run noisy results.  
* Require investigation of sustained regressions.  
* Do not fail ordinary CI because of minor physical-device variability.

Keep GPU correctness tests and performance tests separate.

### Acceptance criteria

Each optimization must demonstrate:

* A measurable improvement.  
* No numerical correctness regression.  
* No increase in memory usage that invalidates its intended benefit.  
* No regressions in other supported model families.

Expected deliverable: A measured performance improvement, backed by reproducible before-and-after reports.

# **11\. P3 — Release engineering and Homebrew distribution**

Stable release blocker

Your packaging infrastructure exists, but the actual public distribution path is not complete.

The latest audit found:

* No published releases.  
* Cancelled release workflow executions.  
* No publicly accessible `RISHI5991/homebrew-tap` repository.  
* An updated Homebrew checksum, but no independently verified published asset.

### Task 11.1 — Make the release package self-validating

Review:

`packaging/build_release_package.sh`

It should:

1. Build both binaries.  
2. Use the exact release target.  
3. Package the README and license.  
4. Generate a SHA-256 checksum.  
5. Validate package contents.  
6. Validate executable architecture.  
7. Run a binary smoke test.

Add explicit verification:

```
shasum -a 256 -c \
  dist/kriya-rt-v0.1.0-aarch64-apple-darwin.tar.gz.sha256

tar -tzf \
  dist/kriya-rt-v0.1.0-aarch64-apple-darwin.tar.gz

file \
  dist/kriya-rt-v0.1.0-aarch64-apple-darwin/bin/kriya-server
```

Also inspect linked libraries using:

```
otool -L path/to/kriya-server
```

This will help establish whether the binary is genuinely standalone in the sense claimed by the documentation.

Be precise with the word static: macOS binaries typically depend on Apple system frameworks.

### Task 11.2 — Strengthen the CI packaging job

The package-check job should:

* Build the release package.  
* Verify its checksum.  
* Inspect its contents.  
* Execute `--version`.  
* Upload the completed artifact.

This should happen before a version tag is published.

### Task 11.3 — Fix release workflow semantics

Your release workflow currently sets:

```
prerelease: false
```

That is unsuitable for a release-candidate tag.

Use separate release-candidate and stable-release behaviour, or derive the prerelease status from the tag.

For example:

```
with:
  prerelease: ${{ contains(github.ref_name, '-') }}
```

I recommend this release progression:

1. v0.1.0-rc.1  
   First public release candidate. Package verification, clean installation and basic inference smoke tests.  
2. v0.1.0-rc.2  
   Fix issues discovered during independent installation and real-model validation.  
3. v0.1.0  
   Stable release only after every release gate passes.

Use new release-candidate tags rather than moving an existing stable tag.

### Task 11.4 — Create the Homebrew tap

Create the GitHub repository:

`RISHI5991/homebrew-tap`

Its structure should be:

```
homebrew-tap/
├── Formula/
│   └── kriya.rb
└── README.md
```

Copy the verified formula into `Formula/kriya.rb`.

Update its:

* Asset download URL.  
* Version.  
* SHA-256 checksum.  
* Installation instructions.

The expected installation command is:

```
brew tap RISHI5991/tap

brew install RISHI5991/tap/kriya
```

Test it on a clean Apple Silicon Mac.

Then run:

```
brew audit --strict --online Formula/kriya.rb
brew test kriya
```

Do not announce working Homebrew distribution until the tap and asset both exist and the installation test passes.

### Task 11.5 — Test the release outside your development workspace

This is essential.

Use a separate clean macOS environment or another Apple Silicon machine.

Verify:

* Download.  
* Checksum.  
* Extraction.  
* Binary execution.  
* CLI help.  
* Version output.  
* Model loading.  
* Basic inference.  
* HTTP API startup.  
* Uninstallation.

A package that runs only because your development environment contains additional files is not a valid release.

### Acceptance criteria

* Release candidate is published.  
* SHA-256 independently verified.  
* Both packaged executables work.  
* Homebrew formula installs and passes its test.  
* At least one clean-machine inference smoke test succeeds.

Expected deliverable: A functional, reproducible v0.1.0 release.

# **12\. P3 — Repository governance and dependency management**

### Task 12.1 — Protect main branch

Your latest branch metadata still shows `main` as unprotected.

Create a branch ruleset targeting `main`.

| Setting | Configuration |
| ----- | ----- |
| Enforcement | Active |
| Restrict deletions | Enabled |
| Block force pushes | Enabled |
| Require pull request | Enabled |
| Required checks | Four CI jobs |
| Conversation resolution | Enabled |
| Approval | One independent reviewer when available |

If you are the sole maintainer, don't impose an approval requirement that makes merging impossible. You can introduce that when there is another reviewer.

The four required checks should correspond to:

* Code Formatting.  
* Clippy Linter.  
* Workspace Tests (Apple Silicon).  
* Release Packaging Verification.

Only make them mandatory after they are reliably passing.

### Task 12.2 — Process open Dependabot PRs

There are currently 11 outstanding PRs.

Do not merge all of them in one operation.

Use three categories:

| Category | Dependencies |
| ----- | ----- |
| Lower-risk updates | `memmap2`, compatible Tokio update |
| Rust API compatibility review | `axum`, `tower-http`, `rand`, `thiserror`, `tokenizers`, `hf-hub` |
| Platform / build tooling | `metal`, `actions/checkout`, `action-gh-release` |

For each PR:

1. Read the changelog.  
2. Examine breaking changes.  
3. Update the relevant code if necessary.  
4. Run formatting, Clippy and tests.  
5. Check the lockfile.  
6. Merge individually.

Pay particular attention to the `metal` upgrade because it changes the foundational GPU bindings.

Do not accept a dependency update solely because Dependabot created it.

### Task 12.3 — Strengthen contribution standards

Add a documented contribution rule:

Every functional PR must contain:

* A problem statement.  
* An implementation explanation.  
* Test evidence.  
* Performance evidence when it changes a performance-critical path.  
* Known limitations.

For GPU work, require a numerical correctness comparison.

### Acceptance criteria

* `main` has an active protection ruleset.  
* Critical CI checks are required.  
* Dependency PRs are resolved individually.  
* Contributors can reproduce the documented development workflow.

# **13\. P3 — Documentation and public technical credibility**

Your README has improved substantially, especially its more conservative decode-performance positioning.

However, several claims still need evidence.

I would revise the following before stable publication.

| Existing Claim | Required Action |
| ----- | ----- |
| Boots in 30ms | Specify how startup is measured |
| Single \~20MB static executable | Verify actual binary size and linked frameworks |
| Under 2.3 GB RAM | Define memory metric, model and machine |
| Production-Ready Serving | Keep only after complete server integration validation |
| Apple Silicon M1–M4 | Distinguish physically tested from expected compatibility |
| Model validated | Require a real checkpoint test for each validated entry |

### Task 13.1 — Publish a tested-model matrix

I would use three statuses.

**Validated:** Real checkpoint loaded and tested through the production executor.

**Experimental:** Architecture implemented, but full model validation is incomplete.

**Planned:** Architecture or functionality not yet implemented.

Attach the following metadata to every validated entry:

* Model repository and revision.  
* Quantization format.  
* Hardware and RAM.  
* macOS version.  
* Inference test date.  
* Benchmark report.

### Task 13.2 — Update the roadmap

Your current roadmap marks turnkey release engineering complete, even though the release has not been published.

Correct that discrepancy.

A useful engineering roadmap should reflect delivered outcomes, not merely implemented infrastructure.

### Task 13.3 — Correct test-count and reliability statements

The contributing guide claims a deterministic 100% test pass rate and 43 self-contained tests.

Do not retain that language until it reflects actual latest CI evidence.

Use the CI test summary to calculate the real number of executed, ignored and failed tests.

Distinguish:

* Unit tests.  
* CPU integration tests.  
* Synthetic-fixture GPU tests.  
* Physical-model integration tests.  
* Benchmarks.

### Task 13.4 — Publish engineering decisions

Introduce a small ADR directory:

```
docs/
└── adr/
    ├── 0001-metal-backend.md
    ├── 0002-kv-cache-design.md
    ├── 0003-orca-scheduling.md
    ├── 0004-speculative-decoding.md
    └── 0005-checkpoint-validation.md
```

Record why the architecture uses each design, what alternatives were considered, and which assumptions remain unverified.

This will make the project significantly easier for an experienced engineer to review and contribute to.

# **14\. Proposed implementation plan: seven focused PRs**

Do not ask a coding agent to fix everything in one enormous PR.

That increases the risk of regressions and makes failures difficult to isolate.

I recommend these seven PRs, in this order.

| PR | Scope | Depends On |
| :---: | ----- | ----- |
| **PR-01** | CI execution and reproducibility | None |
| **PR-02** | Strict checkpoint loading and normalization fixes | PR-01 baseline |
| **PR-03** | Real-model and Metal correctness tests | PR-02 |
| **PR-04** | Scheduler, KV and production API integration | PR-03 |
| **PR-05** | Benchmark rebuild and profiling | PR-03 |
| **PR-06** | Release and Homebrew packaging | PR-01, PR-03, PR-04 |
| **PR-07** | Branch protection, Dependabot and documentation | PR-01 onward |

PR-04 and PR-05 can proceed in parallel once the real-model correctness baseline is established.

## **What each PR must deliver**

PR-01 — CI Stabilization

Files: `.github/workflows/ci.yml`, `.github/workflows/release.yml`

Deliver: Successful complete CI run, manual workflow trigger, clear cancellation policy and retained package artifacts.

Blocking

PR-02 — Production Checkpoint Integrity

Files: `encoder.rs`, `test_shader_compile.rs`

Deliver: Strict norm binding, Q4 validation, malformed-checkpoint tests and explicit production/test loader separation.

Blocking

PR-03 — Real Inference Validation

Files: `test_real_qwen.rs`, `real_world_validation.rs`, model integration tests.

Deliver: Strict constructor use, real checkpoint verification, reference comparisons and hardware compatibility evidence.

Blocking

PR-04 — Runtime Concurrency and API

Files: `main.rs`, scheduler, KV-cache, `test_api_safety.rs` and integration tests.

Deliver: Actual production router testing, multi-request isolation, mixed workloads, streaming and cancellation.

Blocking

PR-05 — Benchmark and Performance Integrity

Files: `benchmarks/run_mlx_vs_kriya.py`, benchmark reports, README.

Deliver: Matched token IDs, stronger repetitions, memory instrumentation and reproducible comparisons.

Required

PR-06 — Release Engineering

Files: release workflow, packaging script, Homebrew formula and tap repository.

Deliver: Release candidate, checksum validation, installation tests and published release assets.

Blocking

PR-07 — Open-source Governance

Files: branch rulesets, dependency workflow, README, CONTRIBUTING, ROADMAP, SECURITY.

Deliver: Protected default branch, dependency hygiene and accurate public documentation.

Final hardening

# **15\. Execution schedule and milestones**

I would organize this into three development sprints and one release-validation stage.

## **KriyaRT remediation programme**

1. Sprint 1 — Establish correctness  
   P0  
   * Complete a green CI run.  
   * Fix strict checkpoint loading.  
   * Eliminate silent required-tensor failures.  
   * Test malformed checkpoint scenarios.  
   * Convert real-Qwen testing to the production constructor.  
2. Milestone: Production model loading is fail-closed and repeatably verified.  
3. Sprint 2 — Prove runtime functionality  
   P1  
   * Validate Metal kernels.  
   * Test Qwen and Gemma real checkpoints.  
   * Test concurrent request isolation.  
   * Validate KV-cache lifecycle.  
   * Integrate actual HTTP router tests.  
   * Verify cancellation and streaming.  
4. Milestone: Complete inference lifecycle works under realistic workloads.  
5. Sprint 3 — Establish performance credibility  
   P2  
   * Rebuild benchmark harness.  
   * Match tokenizer inputs.  
   * Measure memory correctly.  
   * Reproduce MLX comparisons.  
   * Profile prefill and Gemma regressions.  
   * Implement only justified optimizations.  
6. Milestone: Performance characteristics are independently reproducible.  
7. Release stage — Public distribution  
   P3  
   * Publish release candidate.  
   * Verify binaries and checksums.  
   * Create Homebrew tap.  
   * Test clean installation.  
   * Enable branch protection.  
   * Resolve dependency and documentation inconsistencies.  
   * Publish stable v0.1.0.  
8. Milestone: An independent developer can install and use KriyaRT.

## **16\. How to delegate this work to coding agents**

This project has enough separation between subsystems to benefit from multiple engineering agents.

However, I would not let them all modify `main` or the same source files simultaneously.

Use this division:

| Agent | Responsibility | Primary Files |
| :---: | ----- | ----- |
| **Agent A** | CI and build infrastructure | `.github/workflows/` |
| **Agent B** | Strict checkpoint loading | `encoder.rs`, checkpoint tests |
| **Agent C** | GPU correctness | Metal tests and kernels |
| **Agent D** | Server and API integration | `main.rs`, server tests |
| **Agent E** | Benchmark methodology | `benchmarks/` |
| **Agent F** | Packaging and release | `packaging/`, Homebrew, release workflow |

The best sequencing is:

* Agent A establishes the initial CI baseline.  
* Agent B and Agent E can investigate independently.  
* Agent C works after the strict loader changes are agreed.  
* Agent D builds on the validated runtime interfaces.  
* Agent F performs release work after the correctness gates pass.

One important rule: agents should submit focused PRs, not push directly into the same working branch.

### Coding-agent instructions

Use the following as the common engineering contract for every agent:

# KriyaRT Engineering Execution Contract

You are working on KriyaRT, a native Rust \+ Apple Metal inference runtime.

Repository: [https\://github.com/RISHI5991/kriya-rt](https://github.com/RISHI5991/kriya-rt)

Baseline commit: `68cbc71455a20a299fbc804abf8b16e1dbc3e6d7`

## **Engineering principles**

1. Preserve the existing six-crate workspace architecture unless a concrete architectural defect requires a change.  
2. Do not introduce new features outside the assigned remediation scope.  
3. Never suppress compiler warnings globally to obtain a passing build.  
4. Production checkpoint loading must fail closed.  
5. Synthetic weights must never be used in production inference.  
6. Every required model tensor must be validated for existence, shape, dtype and data length.  
7. GPU implementations must have numerical reference tests.  
8. Never claim a test passed unless it was actually executed.  
9. Distinguish skipped, ignored, cancelled and successful tests.  
10. Do not modify benchmark results or claim performance improvements without new measurements.

## **Required delivery**

For your assigned workstream:

* Inspect the existing implementation first.  
* Identify the exact defects and affected files.  
* Propose the smallest safe implementation change.  
* Implement the fix.  
* Add regression tests.  
* Run relevant local checks.  
* Run the complete CI workflow where applicable.  
* Create a focused pull request.

## **Pull request report**

Every PR must contain:

1. Original problem.  
2. Root cause.  
3. Files changed.  
4. Implementation details.  
5. Tests added.  
6. Commands executed.  
7. Actual test results.  
8. Remaining limitations.  
9. Acceptance criteria satisfied or outstanding.

Do not mark the assignment complete when CI is cancelled or when required GPU tests have been skipped.

Maintain backward compatibility for existing supported checkpoints and preserve the production/test execution boundary.

# **17\. Final release checklist**

This should become the actual release gate for KriyaRT.

Release readiness tracker

# **0/13**

Check each item only after the corresponding evidence exists.

CI and compilation

All four main CI jobs pass on the same commit.

Checkpoint integrity

Malformed and incomplete checkpoints are rejected without fallback.

Metal correctness

Critical kernels pass numerical reference tests.

Real-model inference

Qwen 0.5B and 3B pass through the strict production path.

Architecture validation

Gemma normalization and real-model execution are verified.

Concurrency

Multiple requests remain isolated and cancellation reclaims resources.

API

Real production routes pass integration and streaming tests.

Benchmarks

MLX comparisons are reproducible with matched inputs and raw data.

Performance claims

README claims are supported by the actual reports.

Release artifact

Tarball, architecture and SHA-256 are independently verified.

Homebrew

Tap exists and clean installation passes.

Repository governance

Main protection and required checks are configured.

Documentation

Model support, limitations and test counts are accurate.

0% of release gates checked

Reset checklist

# **18\. My strategic recommendation**

There is one decision I would make immediately:

Freeze feature development until the P0 and P1 milestones are complete.

Do not add:

* More model architectures.  
* More speculative decoding algorithms.  
* More quantization formats.  
* More scheduler sophistication.  
* More packaging options.

Those additions will create more implementation surface area before the existing system has been fully verified.

The most valuable work now is reducing the difference between architectural intent and demonstrable runtime behaviour.

I would measure success through five concrete outcomes:

| Outcome | What It Proves |
| ----- | ----- |
| Green CI | The codebase is buildable and testable |
| Strict real-model loading | The runtime uses actual model weights safely |
| Concurrent inference | The architecture works beyond the single-request path |
| Reproducible benchmarks | Performance claims have substance |
| Verified release | Other developers can actually use the project |

One final challenge: Make KriyaRT cloneable, buildable, testable and usable by a Rust/ML systems engineer who has never spoken to you.

When that person can reproduce your results without your intervention, you have crossed a much more meaningful engineering milestone than simply reaching v0.1.0.

Useful starting points: [KriyaRT repository](https://github.com/RISHI5991/kriya-rt?utm_source=chatgpt.com) · [CI workflow](https://github.com/RISHI5991/kriya-rt/blob/main/.github/workflows/ci.yml?utm_source=chatgpt.com) · [Engineering roadmap](https://github.com/RISHI5991/kriya-rt/blob/main/ROADMAP.md?utm_source=chatgpt.com).

