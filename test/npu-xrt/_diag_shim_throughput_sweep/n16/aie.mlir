//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// xdna-emu shim throughput calibration kernel, BD size N = 16 i32 words.
//
// Data path: ddr -> shim MM2S 0 -> memtile S2MM 0 -> memtile buffer ->
// memtile MM2S 0 -> shim S2MM 0 -> ddr.  No compute core; the memtile
// buffer is a passive loopback so the shim DMA's START/FINISHED_TASK
// events bracket exactly one BD of N words in each direction.
//
// Trace anchors (from the default shim trace event set):
//   shim DMA_MM2S_0_START_TASK    -- push dispatch
//   shim DMA_MM2S_0_FINISHED_TASK -- push complete
//   shim DMA_S2MM_0_START_TASK    -- pull dispatch
//   shim DMA_S2MM_0_FINISHED_TASK -- pull complete
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)

    %loopback_buf = aie.buffer(%tile_0_1) {sym_name = "loopback_buf"} : memref<16xi32>

    %prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "cons_lock"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    aie.shim_dma_allocation @in (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out (%tile_0_0, S2MM, 0)

    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<32xi32>, %arg2: memref<16xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c64_i64 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c64_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @in, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @in}
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c64_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @out, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out}
    }

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // S2MM 0 receive loop
      aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%loopback_buf : memref<16xi32>, 0, 16)
      aie.use_lock(%cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // MM2S 0 send-back loop
      aie.use_lock(%cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%loopback_buf : memref<16xi32>, 0, 16)
      aie.use_lock(%prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}
