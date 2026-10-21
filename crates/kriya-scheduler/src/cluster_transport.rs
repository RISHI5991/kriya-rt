// File: crates/kriya-scheduler/src/cluster_transport.rs
//! Physical Disaggregated KV-Cache Transport (Thunderbolt 5 / 100GbE / TCP).
//!
//! Streams Splitwise `ContiguousKvPutOp` layer-by-layer KV blocks and completion
//! semaphores between disaggregated Prefill and Decode nodes.

use crate::orca_engine::{ContiguousKvPutOp, OrcaMachineScheduler, SchedulerError};
use bytemuck::{Pod, Zeroable};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransportError {
    #[error("Network I/O error during disaggregated KV transfer: {0}")]
    Io(#[from] std::io::Error),
    #[error("Scheduler state error on Decode node: {0}")]
    Scheduler(#[from] SchedulerError),
    #[error("Payload size mismatch: header declared {expected} bytes, received {actual}")]
    PayloadMismatch { expected: usize, actual: usize },
}

/// Compact 32-byte wire header for a one-sided layer-wise KV block transfer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WireKvPutHeader {
    pub req_id: u64,
    pub semaphore_id: u64,
    pub layer_idx: u32,
    pub start_physical_block_id: u32,
    pub num_contiguous_blocks: u32,
    pub payload_byte_len: u32,
}

/// Sends a `ContiguousKvPutOp` and its raw BF16 KV block bytes from a `PrefillOnly` node
/// to a `DecodeOnly` node over a high-speed Thunderbolt / TCP link.
pub fn send_contiguous_kv_put(
    target_addr: SocketAddr,
    op: &ContiguousKvPutOp,
    kv_payload_bytes: &[u8],
) -> Result<(), TransportError> {
    let mut stream = TcpStream::connect(target_addr)?;
    stream.set_nodelay(true)?;

    let header = WireKvPutHeader {
        req_id: op.req_id,
        semaphore_id: op.semaphore_id,
        layer_idx: op.layer_idx,
        start_physical_block_id: op.start_physical_block_id,
        num_contiguous_blocks: op.num_contiguous_blocks,
        payload_byte_len: kv_payload_bytes.len() as u32,
    };

    stream.write_all(bytemuck::bytes_of(&header))?;
    stream.write_all(kv_payload_bytes)?;
    stream.flush()?;
    Ok(())
}

/// Receives a single layer-wise KV transfer on a `DecodeOnly` machine, writes the KV payload
/// into `decode_kv_arena`, and signals `OrcaMachineScheduler::notify_layer_kv_received`.
/// Returns `Ok((req_id, is_all_layers_ready))`.
pub fn receive_one_kv_put_and_notify(
    listener: &TcpListener,
    decode_scheduler: &mut OrcaMachineScheduler,
    decode_kv_arena: &mut [u8],
    block_byte_stride: usize,
) -> Result<(u64, bool), TransportError> {
    let (mut stream, _peer) = listener.accept()?;
    stream.set_nodelay(true)?;

    let mut hdr_buf = [0u8; std::mem::size_of::<WireKvPutHeader>()];
    stream.read_exact(&mut hdr_buf)?;
    let header: WireKvPutHeader = *bytemuck::from_bytes(&hdr_buf);

    let payload_len = header.payload_byte_len as usize;
    let dst_offset = (header.start_physical_block_id as usize) * block_byte_stride;

    if dst_offset + payload_len <= decode_kv_arena.len() {
        stream.read_exact(&mut decode_kv_arena[dst_offset..dst_offset + payload_len])?;
    } else {
        let mut scratch = vec![0u8; payload_len];
        stream.read_exact(&mut scratch)?;
    }

    let ready_for_decode = decode_scheduler.notify_layer_kv_received(header.req_id)?;
    Ok((header.req_id, ready_for_decode))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orca_engine::{InferenceRequest, InstanceRole, ServiceLevelObjective};
    use kriya_kv::bsr_paged::PagedKvCacheManager;

    #[test]
    fn test_live_loopback_disaggregated_kv_put_and_semaphore_activation() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        // Setup Prefill node (2 layers) and Decode node (2 layers)
        let mut prefill_sched = OrcaMachineScheduler::new(
            InstanceRole::PrefillOnly,
            2,
            1024,
            8,
            ServiceLevelObjective::default(),
            PagedKvCacheManager::new(32, 16, 4),
        );
        let mut decode_sched = OrcaMachineScheduler::new(
            InstanceRole::DecodeOnly,
            2,
            1024,
            8,
            ServiceLevelObjective::default(),
            PagedKvCacheManager::new(32, 16, 4),
        );

        let req = InferenceRequest::new(501, vec![10, 20, 30, 40, 50, 60], 8, 999, 100);
        prefill_sched.submit_request(req.clone());
        let _ = prefill_sched.schedule_next_iteration().unwrap().unwrap();

        let layer0_ops = prefill_sched.build_layer_kv_put_ops(501, 0, 1).unwrap();
        let layer1_ops = prefill_sched.build_layer_kv_put_ops(501, 1, 1).unwrap();

        decode_sched
            .register_incoming_disaggregated_decode(req, layer0_ops[0].semaphore_id)
            .unwrap();

        let mut decode_kv_arena = vec![0u8; 4096];
        let payload = vec![0xAAu8; 256]; // 2 contiguous blocks * 128 bytes/block

        // Stream Layer 0 KV blocks over socket
        send_contiguous_kv_put(addr, &layer0_ops[0], &payload).unwrap();
        let (req_id, ready0) =
            receive_one_kv_put_and_notify(&listener, &mut decode_sched, &mut decode_kv_arena, 128)
                .unwrap();
        assert_eq!(req_id, 501);
        assert!(!ready0);
        assert_eq!(decode_kv_arena[0], 0xAA);

        // Stream Layer 1 KV blocks over socket -> request immediately activates for Decode!
        send_contiguous_kv_put(addr, &layer1_ops[0], &payload).unwrap();
        let (_, ready1) =
            receive_one_kv_put_and_notify(&listener, &mut decode_sched, &mut decode_kv_arena, 128)
                .unwrap();
        assert!(ready1);
    }
}
