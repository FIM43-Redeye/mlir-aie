//===- a_notrace.mlir ------------------------------------------*- MLIR -*-===//
//
// SP-3 spike HW bring-up, Experiment A: the FULL spike dataflow + completion
// drain + await, with ALL trace machinery removed (no aie.trace, no
// host_config, no perfcnt, no broadcast sync, no channel-sel write32).
//
// Purpose: bisect trace-machinery vs dataflow as the HW-wedge culprit. If this
// completes on HW, the wedge lives in the trace plumbing. If it still TDRs, the
// wedge is in the dataflow / drain / await chain.
//
// Single arg: comp_buf (the shim-drain DDR sink, also the runner's "trace BO").
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_2col) {
    // Shim hosts the awaited t_drain; its completion token needs a route to the
    // host (controller_id tags the token packet, packet_flow routes TileControl
    // -> South). Both known-good task-token references do exactly this; the
    // redesign that stripped them is what wedged the await on HW.
    %shim_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %mem_0_1  = aie.tile(0, 1)
    %core_0_2 = aie.tile(0, 2)
    %core_0_3 = aie.tile(0, 3)
    %core_1_3 = aie.tile(1, 3)

    %prod_buf  = aie.buffer(%core_0_2) {sym_name = "prod_buf"}  : memref<1024xi32>
    %relay_buf = aie.buffer(%mem_0_1)  {sym_name = "relay_buf"} : memref<1024xi32>
    %recv_buf0 = aie.buffer(%core_0_3) {sym_name = "recv_buf0"} : memref<1024xi32>
    %recv_buf1 = aie.buffer(%core_1_3) {sym_name = "recv_buf1"} : memref<1024xi32>

    %c02_empty = aie.lock(%core_0_2, 0) {init = 1 : i32, sym_name = "c02_empty"}
    %c02_full  = aie.lock(%core_0_2, 1) {init = 0 : i32, sym_name = "c02_full"}
    %mt_empty  = aie.lock(%mem_0_1, 0) {init = 1 : i32, sym_name = "mt_empty"}
    %mt_full   = aie.lock(%mem_0_1, 1) {init = 0 : i32, sym_name = "mt_full"}
    %mt_done0  = aie.lock(%mem_0_1, 2) {init = 0 : i32, sym_name = "mt_done0"}
    %mt_done1  = aie.lock(%mem_0_1, 3) {init = 0 : i32, sym_name = "mt_done1"}
    %mt_comp   = aie.lock(%mem_0_1, 4) {init = 0 : i32, sym_name = "mt_comp"}
    %c03_empty = aie.lock(%core_0_3, 0) {init = 1 : i32, sym_name = "c03_empty"}
    %c03_full  = aie.lock(%core_0_3, 1) {init = 0 : i32, sym_name = "c03_full"}
    %c13_empty = aie.lock(%core_1_3, 0) {init = 1 : i32, sym_name = "c13_empty"}
    %c13_full  = aie.lock(%core_1_3, 1) {init = 0 : i32, sym_name = "c13_full"}

    aie.flow(%core_0_2, DMA : 0, %mem_0_1, DMA : 0)
    aie.flow(%mem_0_1, DMA : 0, %core_0_3, DMA : 0)
    aie.flow(%mem_0_1, DMA : 2, %core_1_3, DMA : 0)
    aie.flow(%mem_0_1, DMA : 4, %shim_0_0, DMA : 0)

    // Completion-token route for the awaited shim drain (t_drain): shim
    // TileControl -> shim South -> host. pkt_id 7 matches the shim controller_id.
    aie.packet_flow(0x7) {
      aie.packet_source<%shim_0_0, "TileControl" : 0>
      aie.packet_dest<%shim_0_0, "South" : 0>
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

    aie.runtime_sequence(%comp_buf: memref<16xi32>) {
      %t_prod = aiex.dma_configure_task(%core_0_2, MM2S, 0) {
        aie.use_lock(%c02_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%prod_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c02_empty, Release, 1)
        aie.end
      }
      %t_fill = aiex.dma_configure_task(%mem_0_1, S2MM, 0) {
        aie.use_lock(%mt_empty, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%mt_full, Release, 2)
        aie.end
      }
      %t_sel0 = aiex.dma_configure_task(%mem_0_1, MM2S, 0) {
        aie.use_lock(%mt_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.use_lock(%mt_done0, Release, 1)
        aie.end
      }
      %t_sel1 = aiex.dma_configure_task(%mem_0_1, MM2S, 2) {
        aie.use_lock(%mt_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 1024) {bd_id = 2 : i32}
        aie.use_lock(%mt_done1, Release, 1)
        aie.end
      }
      %t_c0 = aiex.dma_configure_task(%core_0_3, S2MM, 0) {
        aie.use_lock(%c03_empty, AcquireGreaterEqual, 1)
        aie.dma_bd(%recv_buf0 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c03_full, Release, 1)
        aie.end
      }
      %t_c1 = aiex.dma_configure_task(%core_1_3, S2MM, 0) {
        aie.use_lock(%c13_empty, AcquireGreaterEqual, 1)
        aie.dma_bd(%recv_buf1 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c13_full, Release, 1)
        aie.end
      }
      %t_comp = aiex.dma_configure_task(%mem_0_1, MM2S, 4) {
        aie.use_lock(%mt_done1, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 16) {bd_id = 4 : i32}
        aie.use_lock(%mt_comp, Release, 1)
        aie.end
      }
      %t_drain = aiex.dma_configure_task(%shim_0_0, S2MM, 0) {
        aie.dma_bd(%comp_buf : memref<16xi32>, 0, 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t_drain)
      aiex.dma_start_task(%t_comp)
      aiex.dma_start_task(%t_c0)
      aiex.dma_start_task(%t_c1)
      aiex.dma_start_task(%t_sel0)
      aiex.dma_start_task(%t_sel1)
      aiex.dma_start_task(%t_fill)
      aiex.dma_start_task(%t_prod)

      aiex.dma_await_task(%t_drain)
    }
  }
}
