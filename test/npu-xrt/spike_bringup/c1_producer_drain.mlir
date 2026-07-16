//===- c1_producer_drain.mlir ----------------------------------*- MLIR -*-===//
//
// Bring-up step C1: morph UP from the known-good core_dmas/dma_configure_task_token
// control (which PASSES on HW through this harness). Replace the control's shim
// DDR source with an ON-CHIP PRODUCER core, and keep only the proven
// "S2MM receiver with issue_token + await" drain.
//
// Topology:  core(0,2) fills prod_buf on-chip -> MM2S -> shim(0,0) S2MM -> comp_buf
// Await:     t_drain (shim S2MM, issue_token) -- token routed shim TileControl->South.
//
// Tests the single new dependency vs the control: an on-chip producer core that
// must actually execute to release the data. No memtile/fan-out/cross-column.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_2col) {
    %shim_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %core_0_2 = aie.tile(0, 2)

    %prod_buf = aie.buffer(%core_0_2) {sym_name = "prod_buf"} : memref<1024xi32>

    %c02_empty = aie.lock(%core_0_2, 0) {init = 1 : i32, sym_name = "c02_empty"}
    %c02_full  = aie.lock(%core_0_2, 1) {init = 0 : i32, sym_name = "c02_full"}

    aie.flow(%core_0_2, DMA : 0, %shim_0_0, DMA : 0)

    // Token route for the awaited shim drain.
    aie.packet_flow(0x7) {
      aie.packet_source<%shim_0_0, "TileControl" : 0>
      aie.packet_dest<%shim_0_0, "South" : 0>
    } {keep_pkt_header = true, priority_route = true}

    // On-chip producer: fill the buffer behind a lock, then release it.
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
      // Producer MM2S: wait for filled buffer, stream to shim, free buffer.
      %t_prod = aiex.dma_configure_task(%core_0_2, MM2S, 0) {
        aie.use_lock(%c02_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%prod_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c02_empty, Release, 1)
        aie.end
      }
      // Shim S2MM drain -> DDR comp_buf, token await.
      %t_drain = aiex.dma_configure_task(%shim_0_0, S2MM, 0) {
        aie.dma_bd(%comp_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t_drain)
      aiex.dma_start_task(%t_prod)
      aiex.dma_await_task(%t_drain)
    }
  }
}
