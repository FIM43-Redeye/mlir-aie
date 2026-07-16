//===- c3_xcol_token.mlir ---------------------------------------*- MLIR -*-===//
//
// Bring-up step C3: the pivot's crux -- CROSS-COLUMN token await. Producer
// core(0,2) in col 0 streams to consumer core(1,3) in col 1 (cross-column data
// flow), and the col-1 consumer signals completion via an issue_token S2MM
// awaited by the host. Token routed via col 1's OWN shim (shim(1,0)) South.
//
// If this completes on HW, a memtile-free core-relay fan-out with cross-column
// completion is viable -- the basis for the SP-3 topology pivot.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_2col) {
    %shim_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %core_0_2 = aie.tile(0, 2)
    %core_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 8>}

    %prod_buf = aie.buffer(%core_0_2) {sym_name = "prod_buf"} : memref<1024xi32>
    %recv_buf = aie.buffer(%core_1_3) {sym_name = "recv_buf"} : memref<1024xi32>

    %c02_empty = aie.lock(%core_0_2, 0) {init = 1 : i32, sym_name = "c02_empty"}
    %c02_full  = aie.lock(%core_0_2, 1) {init = 0 : i32, sym_name = "c02_full"}

    // Cross-column data flow: col 0 producer -> col 1 consumer.
    aie.flow(%core_0_2, DMA : 0, %core_1_3, DMA : 0)

    // Completion-token route for the awaited col-1 consumer: core(1,3)
    // TileControl -> col-1 shim South -> host.
    aie.packet_flow(0x8) {
      aie.packet_source<%core_1_3, "TileControl" : 0>
      aie.packet_dest<%shim_1_0, "South" : 0>
    } {keep_pkt_header = true, priority_route = true}

    %core = aie.core(%core_0_2) {
      aie.use_lock(%c02_empty, AcquireGreaterEqual, 1)
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cN = arith.constant 1024 : index
      scf.for %i = %c0 to %cN step %c1 {
        %iv = arith.index_cast %i : index to i32
        memref.store %iv, %prod_buf[%i] : memref<1024xi32>
      }
      aie.use_lock(%c02_full, Release, 1)
      aie.end
    }

    aie.runtime_sequence(%comp_buf: memref<1024xi32>) {
      %t_prod = aiex.dma_configure_task(%core_0_2, MM2S, 0) {
        aie.use_lock(%c02_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%prod_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c02_empty, Release, 1)
        aie.end
      }
      %t_recv = aiex.dma_configure_task(%core_1_3, S2MM, 0) {
        aie.dma_bd(%recv_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t_recv)
      aiex.dma_start_task(%t_prod)
      aiex.dma_await_task(%t_recv)
    }
  }
}
