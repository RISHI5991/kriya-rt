# ADR 0003: ORCA Continuous Iteration-Level Batching & Split-K SLA Preemption

## Context
Traditional static request batching requires waiting for all requests in a batch to finish generation, causing high tail latency (P99) and GPU under-utilization when requests have varying prompt lengths and generation budgets.

## Decision
KriyaRT implements an **ORCA continuous iteration-level scheduler** (`crates/kriya-scheduler`):
- New requests enter the batch immediately at the start of any iteration tick.
- Completed requests immediately release their KV cache pages without waiting for slower requests in the batch.
- Decode steps across all active requests are coalesced into a single batched GPU forward pass, streaming model parameters from memory once per tick.
- Dynamic chunked prefill chunks long prompts into SLA-bounded slices (e.g., 512 tokens) to prevent prefill requests from starving active decode streams.

## Consequences
- **Positive**: Low and predictable Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT) under concurrent multi-tenant loads.
- **Negative**: Dynamic scheduling logic introduces minor host-side coordination overhead per tick (~15 microseconds).
