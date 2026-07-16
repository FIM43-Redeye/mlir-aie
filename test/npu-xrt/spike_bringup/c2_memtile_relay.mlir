//===- c2_memtile_relay.mlir ------------------------------------*- MLIR -*-===//
//
// Bring-up step C2 (builds on C1, which PASSES): insert a MEMTILE RELAY between
// the on-chip producer and the shim drain.
//
// Topology:  core(0,2) fill -> MM2S -> memtile(0,1) S2MM -> MM2S -> shim(0,0) S2MM
// Await:     t_drain (shim S2MM, issue_token) only.
//
// New variable vs C1: the memtile relay (S2MM fill + MM2S drain, even channels).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_2col) {
    %shim_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %mem_0_1  = aie.tile(0, 1)
    %core_0_2 = aie.tile(0, 2)

    %prod_buf  = aie.buffer(%core_0_2) {sym_name = "prod_buf"}  : memref<1024xi32>
    %relay_buf = aie.buffer(%mem_0_1)  {sym_name = "relay_buf"} : memref<1024xi32>

    %c02_empty = aie.lock(%core_0_2, 0) {init = 1 : i32, sym_name = "c02_empty"}
    %c02_full  = aie.lock(%core_0_2, 1) {init = 0 : i32, sym_name = "c02_full"}
    %mt_empty  = aie.lock(%mem_0_1, 0) {init = 1 : i32, sym_name = "mt_empty"}
    %mt_full   = aie.lock(%mem_0_1, 1) {init = 0 : i32, sym_name = "mt_full"}

    aie.flow(%core_0_2, DMA : 0, %mem_0_1, DMA : 0)
    aie.flow(%mem_0_1, DMA : 0, %shim_0_0, DMA : 0)

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

    aie.runtime_sequence(%comp_buf: memref<1024xi32>) {
      %t_prod = aiex.dma_configure_task(%core_0_2, MM2S, 0) {
        aie.use_lock(%c02_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%prod_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c02_empty, Release, 1)
        aie.end
      }
      %t_fill = aiex.dma_configure_task(%mem_0_1, S2MM, 0) {
        aie.use_lock(%mt_empty, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%mt_full, Release, 1)
        aie.end
      }
      %t_relay = aiex.dma_configure_task(%mem_0_1, MM2S, 0) {
        aie.use_lock(%mt_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.use_lock(%mt_empty, Release, 1)
        aie.end
      }
      %t_drain = aiex.dma_configure_task(%shim_0_0, S2MM, 0) {
        aie.dma_bd(%comp_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t_drain)
      aiex.dma_start_task(%t_relay)
      aiex.dma_start_task(%t_fill)
      aiex.dma_start_task(%t_prod)
      aiex.dma_await_task(%t_drain)
    }
  }
}
