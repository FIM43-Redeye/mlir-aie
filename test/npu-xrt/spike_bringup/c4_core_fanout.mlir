//===- c4_core_fanout.mlir --------------------------------------*- MLIR -*-===//
//
// Bring-up step C4: the full SP-3 core-relay dataflow (memtile-free pivot).
// Producer core(0,2) fills a buffer on-chip and fans it out core->core to two
// consumers via its two MM2S channels: core(0,3) (vertical, col 0) and
// core(1,3) (cross-column, col 1). Each consumer receives via an issue_token
// S2MM awaited by the host through its OWN column's shim.
//
// Properties preserved vs the memtile spike: rank-2 fan-out, cross-column,
// born-on-chip (Q=0, no DDR in the data path). Builds only on HW-proven
// primitives: C1 (producer+token await), C3 (cross-column token await).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_2col) {
    %shim_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
    %shim_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %core_0_2 = aie.tile(0, 2)
    %core_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 8>}
    %core_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 9>}

    %prod_buf  = aie.buffer(%core_0_2) {sym_name = "prod_buf"}  : memref<1024xi32>
    %recv_buf0 = aie.buffer(%core_0_3) {sym_name = "recv_buf0"} : memref<1024xi32>
    %recv_buf1 = aie.buffer(%core_1_3) {sym_name = "recv_buf1"} : memref<1024xi32>

    %c02_empty = aie.lock(%core_0_2, 0) {init = 1 : i32, sym_name = "c02_empty"}
    %c02_full  = aie.lock(%core_0_2, 1) {init = 0 : i32, sym_name = "c02_full"}

    // Fan-out data flows: producer's two MM2S channels to the two consumers.
    aie.flow(%core_0_2, DMA : 0, %core_0_3, DMA : 0)   // vertical (col 0)
    aie.flow(%core_0_2, DMA : 1, %core_1_3, DMA : 0)   // cross-column (col 1)

    // Completion-token routes: each consumer's TileControl -> its column shim South.
    aie.packet_flow(0x8) {
      aie.packet_source<%core_0_3, "TileControl" : 0>
      aie.packet_dest<%shim_0_0, "South" : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(0x9) {
      aie.packet_source<%core_1_3, "TileControl" : 0>
      aie.packet_dest<%shim_1_0, "South" : 0>
    } {keep_pkt_header = true, priority_route = true}

    // On-chip producer: fill once, release full by 2 (one credit per fan-out reader).
    %core = aie.core(%core_0_2) {
      aie.use_lock(%c02_empty, AcquireGreaterEqual, 1)
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cN = arith.constant 1024 : index
      scf.for %i = %c0 to %cN step %c1 {
        %iv = arith.index_cast %i : index to i32
        memref.store %iv, %prod_buf[%i] : memref<1024xi32>
      }
      aie.use_lock(%c02_full, Release, 2)
      aie.end
    }

    aie.runtime_sequence(%comp_buf: memref<1024xi32>) {
      // Producer MM2S ch0 -> core(0,3).
      %t_prod0 = aiex.dma_configure_task(%core_0_2, MM2S, 0) {
        aie.use_lock(%c02_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%prod_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c02_empty, Release, 1)
        aie.end
      }
      // Producer MM2S ch1 -> core(1,3) (cross-column).
      %t_prod1 = aiex.dma_configure_task(%core_0_2, MM2S, 1) {
        aie.use_lock(%c02_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%prod_buf : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.use_lock(%c02_empty, Release, 1)
        aie.end
      }
      // Vertical consumer receive (token await).
      %t_c0 = aiex.dma_configure_task(%core_0_3, S2MM, 0) {
        aie.dma_bd(%recv_buf0 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      // Cross-column consumer receive (token await).
      %t_c1 = aiex.dma_configure_task(%core_1_3, S2MM, 0) {
        aie.dma_bd(%recv_buf1 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t_c0)
      aiex.dma_start_task(%t_c1)
      aiex.dma_start_task(%t_prod0)
      aiex.dma_start_task(%t_prod1)
      aiex.dma_await_task(%t_c0)
      aiex.dma_await_task(%t_c1)
    }
  }
}
