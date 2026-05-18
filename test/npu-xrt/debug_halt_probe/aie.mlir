//===- aie.mlir --- debug_halt_probe (Phase A spike) -----------*- MLIR -*-===//
// Probe kernel: minimal straight-line core writes 3 sentinel markers to
// output_buffer, then ends. Runtime sequence issues one control-packet
// register write to a compute-tile debug register to prove the authoring
// mechanism. Marker buffer is DMA'd back to the host.
//===----------------------------------------------------------------------===//
module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_2 = aie.tile(0, 2)

    %output_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "output_lock4"}
    %output_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "output_lock5"}

    %output_buffer = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<8xi32>

    aie.packet_flow(0x3) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
    aie.flow(%tile_0_0, DMA : 1, %tile_0_2, DMA : 1)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %mAA = arith.constant 170 : i32
      %mBB = arith.constant 187 : i32
      %mCC = arith.constant 204 : i32
      aie.use_lock(%output_lock5, AcquireGreaterEqual, 1)
      memref.store %mAA, %output_buffer[%c0] : memref<8xi32>
      memref.store %mBB, %output_buffer[%c1] : memref<8xi32>
      memref.store %mCC, %output_buffer[%c2] : memref<8xi32>
      aie.use_lock(%output_lock4, Release, 1)
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%output_lock4, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_buffer : memref<8xi32>, 0, 8) {packet = #aie.packet_info<pkt_id = 3, pkt_type = 0>}
      aie.use_lock(%output_lock5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 1)

    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c8_i64 = arith.constant 8 : i64
      // SPIKE: prove a control-packet write to a compute-tile debug register
      // is expressible. Debug_Control0 = 0 is innocuous (no-op semantics).
      aiex.npu.write32 {address = 0x32010 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      // DMA the marker buffer back to the host.
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @out0} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
