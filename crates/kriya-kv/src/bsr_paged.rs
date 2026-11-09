//! FlashInfer Block-Sparse Row (BSR) Unified KV-Cache Manager, vLLM Paged Block
//! Allocator with Copy-on-Write Reference Counting, Radix-Tree Prefix Cache, and
//! Inspector-Executor (`plan` / `run`) Load-Balancing Scheduler.

use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

/// Default number of token positions stored per physical KV block.
pub const DEFAULT_KV_BLOCK_TOKENS: u32 = 16;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum KvCacheError {
    #[error("Out of physical KV blocks on {device}: requested {requested}, available {available}")]
    OutOfBlocks {
        device: &'static str,
        requested: usize,
        available: usize,
    },
    #[error("Invalid sequence ID: {0}")]
    InvalidSequenceId(u64),
    #[error("Invalid physical block ID: {0}")]
    InvalidBlockId(u32),
}

/// Physical KV block metadata tracked by the block allocator.
#[derive(Debug, Clone)]
pub struct PhysicalKvBlock {
    pub block_id: u32,
    /// Reference count supporting prefix sharing, parallel sampling, and beam search.
    pub ref_count: u32,
    /// Number of valid token positions currently populated inside this block (0..=block_size).
    pub num_hashed_tokens: u32,
    /// Optional rolling hash of the prefix token sequence up to and including this block.
    pub prefix_hash: Option<u64>,
}

/// vLLM-style Physical Block Allocator with Reference Counting and Copy-on-Write (CoW).
pub struct BlockAllocator {
    device_name: &'static str,
    block_size_tokens: u32,
    blocks: Vec<PhysicalKvBlock>,
    free_list: VecDeque<u32>,
}

impl BlockAllocator {
    pub fn new(device_name: &'static str, num_blocks: u32, block_size_tokens: u32) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks as usize);
        let mut free_list = VecDeque::with_capacity(num_blocks as usize);

        for id in 0..num_blocks {
            blocks.push(PhysicalKvBlock {
                block_id: id,
                ref_count: 0,
                num_hashed_tokens: 0,
                prefix_hash: None,
            });
            free_list.push_back(id);
        }

        Self {
            device_name,
            block_size_tokens,
            blocks,
            free_list,
        }
    }

    #[inline]
    pub fn num_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    #[inline]
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    pub fn block_size_tokens(&self) -> u32 {
        self.block_size_tokens
    }

    /// Allocates a single physical block and initializes its reference count to 1.
    pub fn allocate_block(&mut self) -> Result<u32, KvCacheError> {
        let block_id = self
            .free_list
            .pop_front()
            .ok_or(KvCacheError::OutOfBlocks {
                device: self.device_name,
                requested: 1,
                available: 0,
            })?;

        let blk = &mut self.blocks[block_id as usize];
        blk.ref_count = 1;
        blk.num_hashed_tokens = 0;
        blk.prefix_hash = None;
        Ok(block_id)
    }

    /// Increments reference count for an existing physical block (for shared prefixes or beams).
    pub fn inc_ref(&mut self, block_id: u32) -> Result<(), KvCacheError> {
        let blk = self
            .blocks
            .get_mut(block_id as usize)
            .ok_or(KvCacheError::InvalidBlockId(block_id))?;
        blk.ref_count += 1;
        Ok(())
    }

    /// Decrements reference count and returns the physical block to the free list when it hits 0.
    pub fn free_block(&mut self, block_id: u32) -> Result<bool, KvCacheError> {
        let blk = self
            .blocks
            .get_mut(block_id as usize)
            .ok_or(KvCacheError::InvalidBlockId(block_id))?;

        if blk.ref_count == 0 {
            return Err(KvCacheError::InvalidBlockId(block_id));
        }

        blk.ref_count -= 1;
        if blk.ref_count == 0 {
            blk.num_hashed_tokens = 0;
            blk.prefix_hash = None;
            self.free_list.push_back(block_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    #[inline]
    pub fn ref_count(&self, block_id: u32) -> Result<u32, KvCacheError> {
        Ok(self
            .blocks
            .get(block_id as usize)
            .ok_or(KvCacheError::InvalidBlockId(block_id))?
            .ref_count)
    }
}

/// Radix-Tree node for automatic prefix KV block sharing across multi-turn prompts.
#[derive(Debug, Default)]
struct RadixNode {
    /// Full block token sequence (length == `block_size_tokens`).
    block_tokens: Vec<u32>,
    /// Mapped physical GPU block ID holding the KV states for `block_tokens`.
    physical_block_id: Option<u32>,
    /// Child nodes keyed by the first token ID of the next block.
    children: HashMap<u32, RadixNode>,
}

/// Radix-Tree Prefix Cache supporting multi-level shared prefix lookup.
#[derive(Debug, Default)]
pub struct RadixPrefixCache {
    root: RadixNode,
    block_size_tokens: u32,
}

impl RadixPrefixCache {
    pub fn new(block_size_tokens: u32) -> Self {
        Self {
            root: RadixNode::default(),
            block_size_tokens,
        }
    }

    /// Matches the longest sequence of full blocks in `prompt_tokens` and increments their ref counts.
    /// Returns `(matched_physical_blocks, matched_token_count)`.
    pub fn match_prefix(
        &self,
        prompt_tokens: &[u32],
        gpu_allocator: &mut BlockAllocator,
    ) -> Result<(Vec<u32>, usize), KvCacheError> {
        let blk_sz = self.block_size_tokens as usize;
        let mut matched_blocks = Vec::new();
        let mut current = &self.root;
        let mut offset = 0;

        while offset + blk_sz <= prompt_tokens.len() {
            let chunk = &prompt_tokens[offset..offset + blk_sz];
            let first_tok = chunk[0];

            if let Some(child) = current.children.get(&first_tok)
                && child.block_tokens == chunk
                && let Some(phys_id) = child.physical_block_id
                && gpu_allocator.ref_count(phys_id).unwrap_or(0) > 0
            {
                gpu_allocator.inc_ref(phys_id)?;
                matched_blocks.push(phys_id);
                current = child;
                offset += blk_sz;
                continue;
            }
            break;
        }

        Ok((matched_blocks, offset))
    }

    /// Inserts full blocks from a sequence into the Radix Tree for future prefix reuse.
    pub fn insert_prefix(&mut self, tokens: &[u32], physical_blocks: &[u32]) {
        let blk_sz = self.block_size_tokens as usize;
        let num_full_blocks = (tokens.len() / blk_sz).min(physical_blocks.len());
        let mut current = &mut self.root;

        for blk_idx in 0..num_full_blocks {
            let chunk = &tokens[blk_idx * blk_sz..(blk_idx + 1) * blk_sz];
            let first_tok = chunk[0];
            let phys_id = physical_blocks[blk_idx];

            let next_node = current
                .children
                .entry(first_tok)
                .or_insert_with(|| RadixNode {
                    block_tokens: chunk.to_vec(),
                    physical_block_id: Some(phys_id),
                    children: HashMap::new(),
                });

            if next_node.block_tokens == chunk {
                next_node.physical_block_id = Some(phys_id);
                current = next_node;
            } else {
                break;
            }
        }
    }
}

/// Logical-to-physical block table for a single active sequence.
#[derive(Debug, Clone)]
pub struct SequenceBlockTable {
    pub seq_id: u64,
    pub logical_blocks: Vec<u32>,
    pub total_tokens: u32,
}

impl SequenceBlockTable {
    /// Number of valid tokens stored in the final page (`1..=block_size_tokens`).
    #[inline]
    pub fn last_page_len(&self, block_size_tokens: u32) -> u32 {
        if self.total_tokens == 0 {
            0
        } else {
            let rem = self.total_tokens % block_size_tokens;
            if rem == 0 { block_size_tokens } else { rem }
        }
    }
}

/// FlashInfer Unified Block-Sparse Row (BSR) KV-Cache Descriptor passed to GPU kernels.
#[derive(Debug, Clone, Default)]
pub struct BsrKvLayout {
    /// Row pointer array of length `batch_size + 1` indexing into `indices`.
    pub page_indptr: Vec<i32>,
    /// Contiguous array of physical KV block IDs across all batched sequences.
    pub page_indices: Vec<i32>,
    /// Number of valid tokens in the final page of each sequence (`batch_size`).
    pub last_page_len: Vec<i32>,
}

/// Single CTA (Cooperative Thread Array / Threadgroup) work tile generated during `plan()`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct CtaWorkTile {
    /// Request index within the current batch (`0..batch_size`).
    pub request_idx: u32,
    /// Starting KV block offset inside the request's logical block table.
    pub kv_block_start: u32,
    /// Number of KV blocks assigned to this CTA tile for partial attention reduction.
    pub kv_block_count: u32,
    /// Flag (1 or 0) indicating whether this request is split across multiple CTAs.
    pub is_split_k: u32,
}

/// Cached Inspector-Executor Scheduling Plan produced once per generation step on CPU
/// and reused across all attention layers on the GPU.
#[derive(Debug, Clone, Default)]
pub struct FlashInferSchedulePlan {
    pub bsr_layout: BsrKvLayout,
    pub cta_work_tiles: Vec<CtaWorkTile>,
    pub max_blocks_per_cta: u32,
}

/// Unified KV-Cache Manager combining vLLM Paged Allocation, Radix Prefix Sharing,
/// and FlashInfer BSR Inspector-Executor Scheduling.
pub struct PagedKvCacheManager {
    pub gpu_allocator: BlockAllocator,
    pub cpu_swap_allocator: BlockAllocator,
    pub radix_cache: RadixPrefixCache,
    sequences: HashMap<u64, SequenceBlockTable>,
}

impl PagedKvCacheManager {
    pub fn new(num_gpu_blocks: u32, num_cpu_swap_blocks: u32, block_size_tokens: u32) -> Self {
        Self {
            gpu_allocator: BlockAllocator::new("GPU", num_gpu_blocks, block_size_tokens),
            cpu_swap_allocator: BlockAllocator::new(
                "CPU_SWAP",
                num_cpu_swap_blocks,
                block_size_tokens,
            ),
            radix_cache: RadixPrefixCache::new(block_size_tokens),
            sequences: HashMap::new(),
        }
    }

    /// Registers a new request prompt, reusing cached prefix blocks from the Radix Tree
    /// and allocating fresh physical blocks only for the uncached suffix.
    pub fn allocate_sequence(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
    ) -> Result<usize, KvCacheError> {
        let blk_sz = self.gpu_allocator.block_size_tokens() as usize;
        let (mut logical_blocks, matched_tokens) = self
            .radix_cache
            .match_prefix(prompt_tokens, &mut self.gpu_allocator)?;

        let total_needed_blocks = prompt_tokens.len().div_ceil(blk_sz);
        let additional_blocks = total_needed_blocks.saturating_sub(logical_blocks.len());

        if self.gpu_allocator.num_free_blocks() < additional_blocks {
            // Roll back incremented prefix ref counts on OOM
            for &blk_id in &logical_blocks {
                let _ = self.gpu_allocator.free_block(blk_id);
            }
            return Err(KvCacheError::OutOfBlocks {
                device: "GPU",
                requested: additional_blocks,
                available: self.gpu_allocator.num_free_blocks(),
            });
        }

        for _ in 0..additional_blocks {
            let blk_id = self.gpu_allocator.allocate_block()?;
            logical_blocks.push(blk_id);
        }

        self.radix_cache
            .insert_prefix(prompt_tokens, &logical_blocks);

        self.sequences.insert(
            seq_id,
            SequenceBlockTable {
                seq_id,
                logical_blocks,
                total_tokens: prompt_tokens.len() as u32,
            },
        );

        Ok(matched_tokens)
    }

    /// Appends one new token slot during autoregressive decoding, performing Copy-on-Write (CoW)
    /// if the tail physical block is shared across multiple sequences (`ref_count > 1`).
    pub fn append_token_slot(
        &mut self,
        seq_id: u64,
    ) -> Result<(u32, Option<(u32, u32)>), KvCacheError> {
        let blk_sz = self.gpu_allocator.block_size_tokens();
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(KvCacheError::InvalidSequenceId(seq_id))?;

        if seq.total_tokens % blk_sz == 0 {
            // Current tail block is completely full; allocate a fresh physical block
            let new_blk = self.gpu_allocator.allocate_block()?;
            seq.logical_blocks.push(new_blk);
            seq.total_tokens += 1;
            Ok((new_blk, None))
        } else {
            // Tail block has free slots; check if Copy-on-Write is required
            let tail_phys_id = *seq
                .logical_blocks
                .last()
                .ok_or(KvCacheError::InvalidSequenceId(seq_id))?;

            if self.gpu_allocator.ref_count(tail_phys_id)? > 1 {
                let fresh_phys_id = self.gpu_allocator.allocate_block()?;
                self.gpu_allocator.free_block(tail_phys_id)?;
                if let Some(last_slot) = seq.logical_blocks.last_mut() {
                    *last_slot = fresh_phys_id;
                }
                let cow_copy = Some((tail_phys_id, fresh_phys_id));
                seq.total_tokens += 1;
                Ok((fresh_phys_id, cow_copy))
            } else {
                seq.total_tokens += 1;
                Ok((tail_phys_id, None))
            }
        }
    }

    /// Forks an existing sequence (for beam search or parallel candidate sampling)
    /// by sharing all physical blocks via reference count increments.
    pub fn fork_sequence(
        &mut self,
        parent_seq_id: u64,
        child_seq_id: u64,
    ) -> Result<(), KvCacheError> {
        let parent = self
            .sequences
            .get(&parent_seq_id)
            .ok_or(KvCacheError::InvalidSequenceId(parent_seq_id))?
            .clone();

        for &blk_id in &parent.logical_blocks {
            self.gpu_allocator.inc_ref(blk_id)?;
        }

        self.sequences.insert(
            child_seq_id,
            SequenceBlockTable {
                seq_id: child_seq_id,
                logical_blocks: parent.logical_blocks,
                total_tokens: parent.total_tokens,
            },
        );
        Ok(())
    }

    /// Frees a completed sequence and decrements reference counts on its physical blocks.
    pub fn free_sequence(&mut self, seq_id: u64) -> Result<(), KvCacheError> {
        let seq = self
            .sequences
            .remove(&seq_id)
            .ok_or(KvCacheError::InvalidSequenceId(seq_id))?;

        for blk_id in seq.logical_blocks {
            self.gpu_allocator.free_block(blk_id)?;
        }
        Ok(())
    }

    /// Whether this manager currently owns a page table for the sequence.
    #[inline]
    pub fn contains_sequence(&self, seq_id: u64) -> bool {
        self.sequences.contains_key(&seq_id)
    }

    /// Returns the (physical_block_id, slot_in_block) for a given sequence position.
    /// Used by speculative decoding to commit accepted bonus tokens' K/V to the cache.
    pub fn block_for_position(
        &self,
        seq_id: u64,
        position: u32,
    ) -> Result<(u32, u32), KvCacheError> {
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(KvCacheError::InvalidSequenceId(seq_id))?;
        let blk_sz = self.gpu_allocator.block_size_tokens() as usize;
        let blk_idx = position as usize / blk_sz;
        let slot_in_blk = position as usize % blk_sz;
        if blk_idx >= seq.logical_blocks.len() {
            return Err(KvCacheError::InvalidSequenceId(seq_id));
        }
        Ok((seq.logical_blocks[blk_idx], slot_in_blk as u32))
    }

    /// Inspector-Executor `plan()` stage: builds the unified BSR page table and partitions
    /// long sequences into balanced `CtaWorkTile` chunks reusable across all attention layers.
    pub fn build_schedule_plan(
        &self,
        active_seq_ids: &[u64],
        max_blocks_per_cta: u32,
    ) -> Result<FlashInferSchedulePlan, KvCacheError> {
        let blk_sz = self.gpu_allocator.block_size_tokens();
        let max_blk_cta = max_blocks_per_cta.max(1);

        let mut page_indptr = Vec::with_capacity(active_seq_ids.len() + 1);
        let mut page_indices = Vec::new();
        let mut last_page_len = Vec::with_capacity(active_seq_ids.len());
        let mut cta_work_tiles = Vec::new();

        page_indptr.push(0);

        for (req_idx, &seq_id) in active_seq_ids.iter().enumerate() {
            let seq = self
                .sequences
                .get(&seq_id)
                .ok_or(KvCacheError::InvalidSequenceId(seq_id))?;

            for &phys_id in &seq.logical_blocks {
                page_indices.push(phys_id as i32);
            }
            page_indptr.push(page_indices.len() as i32);
            last_page_len.push(seq.last_page_len(blk_sz) as i32);

            let num_blocks = seq.logical_blocks.len() as u32;
            let is_split_k = u32::from(num_blocks > max_blk_cta);
            let mut start_blk = 0;

            while start_blk < num_blocks {
                let count = (num_blocks - start_blk).min(max_blk_cta);
                cta_work_tiles.push(CtaWorkTile {
                    request_idx: req_idx as u32,
                    kv_block_start: start_blk,
                    kv_block_count: count,
                    is_split_k,
                });
                start_blk += count;
            }
        }

        Ok(FlashInferSchedulePlan {
            bsr_layout: BsrKvLayout {
                page_indptr,
                page_indices,
                last_page_len,
            },
            cta_work_tiles,
            max_blocks_per_cta: max_blk_cta,
        })
    }

    /// Builds a one-sequence schedule limited to an already-reserved token
    /// prefix.  This is used by correctness-first prefill: each token is
    /// appended and attended against exactly its causal prefix instead of
    /// treating every prompt token as the final query of one request.
    pub fn build_prefix_schedule_plan(
        &self,
        seq_id: u64,
        prefix_tokens: u32,
        max_blocks_per_cta: u32,
    ) -> Result<FlashInferSchedulePlan, KvCacheError> {
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(KvCacheError::InvalidSequenceId(seq_id))?;
        if prefix_tokens == 0 || prefix_tokens > seq.total_tokens {
            return Err(KvCacheError::InvalidSequenceId(seq_id));
        }

        let blk_sz = self.gpu_allocator.block_size_tokens();
        let num_blocks = prefix_tokens.div_ceil(blk_sz) as usize;
        let max_blk_cta = max_blocks_per_cta.max(1);
        let mut cta_work_tiles = Vec::new();
        let mut start_blk = 0u32;
        let block_count = num_blocks as u32;
        let is_split_k = u32::from(block_count > max_blk_cta);
        while start_blk < block_count {
            let count = (block_count - start_blk).min(max_blk_cta);
            cta_work_tiles.push(CtaWorkTile {
                request_idx: 0,
                kv_block_start: start_blk,
                kv_block_count: count,
                is_split_k,
            });
            start_blk += count;
        }

        Ok(FlashInferSchedulePlan {
            bsr_layout: BsrKvLayout {
                page_indptr: vec![0, num_blocks as i32],
                page_indices: seq.logical_blocks[..num_blocks]
                    .iter()
                    .map(|&id| id as i32)
                    .collect(),
                last_page_len: vec![if prefix_tokens.is_multiple_of(blk_sz) {
                    blk_sz as i32
                } else {
                    (prefix_tokens % blk_sz) as i32
                }],
            },
            cta_work_tiles,
            max_blocks_per_cta: max_blk_cta,
        })
    }

    /// Builds a parallel 2D causal prefill layout for a single sequence.
    ///
    /// Unlike `build_prefix_schedule_plan` (which builds a plan for one prefix length),
    /// this builds a layout where every query token position has its own causal prefix.
    ///
    /// - `page_indptr[i]` = cumulative blocks needed for positions 0..i-1 (length M+1)
    /// - `page_indices` = all block IDs for the full sequence
    /// - `last_page_len[i]` = valid tokens in the last page for position i (causal clamp)
    ///
    /// This enables the `bsr_paged_gqa_prefill_stage1` kernel to process all M query
    /// tokens in a single GPU dispatch instead of M sequential dispatches.
    pub fn build_parallel_prefill_layout(
        &self,
        seq_id: u64,
        prefix_tokens: u32,
        num_tokens: u32,
        max_blocks_per_cta: u32,
    ) -> Result<FlashInferSchedulePlan, KvCacheError> {
        let blk_sz = self.gpu_allocator.block_size_tokens();
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(KvCacheError::InvalidSequenceId(seq_id))?;

        if num_tokens == 0 || prefix_tokens.saturating_add(num_tokens) > seq.total_tokens {
            return Err(KvCacheError::InvalidSequenceId(seq_id));
        }

        // The attention kernel treats each query as an independent BSR row.
        // Therefore page_indptr must point into a *concatenated* page_indices
        // array, not directly to logical-block counts. For a 16-token page,
        // query rows 0..2 are [block0], [block0], [block0], yielding [0,1,2,3].
        let mut page_indptr = Vec::with_capacity(num_tokens as usize + 1);
        let mut page_indices = Vec::new();
        page_indptr.push(0);
        for i in 0..num_tokens {
            let visible_tokens = prefix_tokens + i + 1;
            let blocks_needed = visible_tokens.div_ceil(blk_sz) as usize;
            page_indices.extend(
                seq.logical_blocks[..blocks_needed]
                    .iter()
                    .map(|&id| id as i32),
            );
            page_indptr.push(page_indices.len() as i32);
        }

        // last_page_len: valid tokens in last page per position (causal clamp)
        let mut last_page_len = Vec::with_capacity(num_tokens as usize);
        for i in 0..num_tokens {
            let valid = ((prefix_tokens + i) % blk_sz) + 1;
            last_page_len.push(valid as i32);
        }

        // CTA work tiles: one per query token
        let mut cta_work_tiles = Vec::new();
        for i in 0..num_tokens {
            let blocks_for_pos = (prefix_tokens + i + 1).div_ceil(blk_sz);
            let is_split_k = u32::from(blocks_for_pos > max_blocks_per_cta);
            cta_work_tiles.push(CtaWorkTile {
                request_idx: 0,
                kv_block_start: 0,
                kv_block_count: blocks_for_pos,
                is_split_k,
            });
        }

        Ok(FlashInferSchedulePlan {
            bsr_layout: BsrKvLayout {
                page_indptr,
                page_indices,
                last_page_len,
            },
            cta_work_tiles,
            max_blocks_per_cta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator_basics() {
        let mut alloc = BlockAllocator::new("GPU", 8, 16);
        assert_eq!(alloc.num_free_blocks(), 8);

        let b0 = alloc.allocate_block().unwrap();
        assert_eq!(alloc.num_free_blocks(), 7);
        assert_eq!(alloc.ref_count(b0).unwrap(), 1);

        alloc.inc_ref(b0).unwrap();
        assert_eq!(alloc.ref_count(b0).unwrap(), 2);

        // First free decrements but doesn't return to pool
        assert!(!alloc.free_block(b0).unwrap());
        assert_eq!(alloc.num_free_blocks(), 7);

        // Second free returns to pool
        assert!(alloc.free_block(b0).unwrap());
        assert_eq!(alloc.num_free_blocks(), 8);
    }

    #[test]
    fn test_radix_prefix_sharing_and_cow_fork() {
        let mut mgr = PagedKvCacheManager::new(64, 32, 4); // 4 tokens per block
        let prompt1: Vec<u32> = (1..=10).collect(); // 10 tokens -> 3 blocks (2 full + 1 partial with 2 tokens)

        let matched = mgr.allocate_sequence(101, &prompt1).unwrap();
        assert_eq!(matched, 0);
        assert_eq!(mgr.gpu_allocator.num_free_blocks(), 61);

        // Second request shares the first 8 tokens (2 full blocks)
        let prompt2: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 99, 100];
        let matched2 = mgr.allocate_sequence(102, &prompt2).unwrap();
        assert_eq!(matched2, 8);
        // Only 1 new block allocated for the suffix [99, 100]!
        assert_eq!(mgr.gpu_allocator.num_free_blocks(), 60);

        // Fork sequence 101 into beam candidate 103 and trigger Copy-on-Write on append
        mgr.fork_sequence(101, 103).unwrap();
        let (_blk, cow) = mgr.append_token_slot(103).unwrap();
        assert!(cow.is_some());
    }

    #[test]
    fn test_parallel_prefill_layout_has_one_causal_bsr_row_per_query() {
        let mut mgr = PagedKvCacheManager::new(16, 8, 4);
        mgr.allocate_sequence(7, &[10, 11, 12, 13, 14, 15]).unwrap();

        // Queries at absolute positions 2, 3, 4 see 3, 4, and 5 tokens.
        // With four-token pages their BSR rows are [block0], [block0],
        // [block0, block1], represented by concatenated page indices.
        let plan = mgr.build_parallel_prefill_layout(7, 2, 3, 8).unwrap();
        assert_eq!(plan.bsr_layout.page_indptr, vec![0, 1, 2, 4]);
        assert_eq!(plan.bsr_layout.page_indices, vec![0, 0, 0, 1]);
        assert_eq!(plan.bsr_layout.last_page_len, vec![3, 4, 1]);
    }

    #[test]
    fn test_flashinfer_bsr_and_load_balanced_cta_plan() {
        let mut mgr = PagedKvCacheManager::new(128, 32, 16);
        let short_prompt: Vec<u32> = (0..20).collect(); // 2 blocks
        let long_prompt: Vec<u32> = (0..100).collect(); // 7 blocks

        mgr.allocate_sequence(1, &short_prompt).unwrap();
        mgr.allocate_sequence(2, &long_prompt).unwrap();

        // Cap each CTA at 3 KV blocks so seq 2 splits across 3 CTAs
        let plan = mgr.build_schedule_plan(&[1, 2], 3).unwrap();
        assert_eq!(plan.bsr_layout.page_indptr, vec![0, 2, 9]);
        assert_eq!(plan.bsr_layout.last_page_len, vec![4, 4]);
        // Seq 1 = 1 CTA tile (2 blocks); Seq 2 = 3 CTA tiles (3 + 3 + 1 blocks) -> 4 total tiles
        assert_eq!(plan.cta_work_tiles.len(), 4);
        assert_eq!(plan.cta_work_tiles[0].is_split_k, 0);
        assert_eq!(plan.cta_work_tiles[1].is_split_k, 1);
    }

    #[test]
    fn test_sequence_lifecycle() {
        let mut mgr = PagedKvCacheManager::new(32, 16, 8);
        let prompt: Vec<u32> = (0..24).collect(); // 3 blocks

        mgr.allocate_sequence(1, &prompt).unwrap();
        assert_eq!(mgr.gpu_allocator.num_free_blocks(), 29);

        // Append tokens to grow into a 4th block
        for _ in 0..8 {
            mgr.append_token_slot(1).unwrap();
        }

        // Free sequence returns all blocks
        mgr.free_sequence(1).unwrap();
        assert_eq!(mgr.gpu_allocator.num_free_blocks(), 32);
    }

    #[test]
    fn test_prefix_schedule_exposes_only_causal_kv_pages() {
        let mut mgr = PagedKvCacheManager::new(16, 8, 4);
        mgr.allocate_sequence(55, &[1, 2, 3, 4, 5, 6, 7]).unwrap();

        let at_third_token = mgr.build_prefix_schedule_plan(55, 3, 8).unwrap();
        assert_eq!(at_third_token.bsr_layout.page_indptr, vec![0, 1]);
        assert_eq!(at_third_token.bsr_layout.last_page_len, vec![3]);
        assert_eq!(at_third_token.cta_work_tiles.len(), 1);

        let at_fifth_token = mgr.build_prefix_schedule_plan(55, 5, 8).unwrap();
        assert_eq!(at_fifth_token.bsr_layout.page_indptr, vec![0, 2]);
        assert_eq!(at_fifth_token.bsr_layout.last_page_len, vec![1]);
        assert_eq!(at_fifth_token.cta_work_tiles[0].kv_block_count, 2);
    }
}
