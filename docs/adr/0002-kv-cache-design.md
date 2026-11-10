# ADR 0002: Block-Sparse Row (BSR) Paged KV-Cache & Zero-Copy Prefix Sharing

## Context
Standard continuous memory allocation for key-value (KV) caches leads to severe internal and external memory fragmentation (often exceeding 60% waste in long-context serving). In multi-agent scenarios and parallel candidate generations, identical prompt prefixes must not be duplicated in Unified Memory.

## Decision
KriyaRT adopts an OS-inspired **Block-Sparse Row (BSR) Paged KV Cache Manager** (`crates/kriya-kv`):
- Physical DRAM is partitioned into fixed 16-token pages (`KVBlock`).
- Logical sequence offsets are dynamically translated to physical block IDs through per-request block tables.
- Prompt prefixes and speculative branches use a Radix Tree prefix DAG with Copy-on-Write (CoW) semantics and atomic reference counters.

## Consequences
- **Positive**: Eliminates memory fragmentation, enables instant zero-copy sequence branching for speculative decoding and tree searches, enables sub-millisecond page recycling.
- **Negative**: Requires BSR indirect page addressing inside attention shaders, adding one pointer dereference per 16-token tile.
