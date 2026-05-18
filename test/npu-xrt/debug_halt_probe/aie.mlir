//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// debug_halt_probe (Phase A -- Exp1: separable pre/trap/post/done markers).
// Probe kernel: minimal straight-line core writes 4 sentinel markers
// (pre-trap / trap-bundle / post-trap / done sentinel) to output_buffer, then
// ends. The runtime sequence arms a PC_Event0 breakpoint (+ Debug_Control2
// PC_Event_Halt) at the trap bundle via control-packet register writes to
// compute-tile (0,2), before starting the readback. Marker buffer is DMA'd
// back to the host; the host harness classifies whether the trap halted the
// core before or after the trap-bundle store committed.
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

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %mAA = arith.constant 170 : i32
      %mBB = arith.constant 187 : i32
      %mCC = arith.constant 204 : i32
      %done = arith.constant 1 : i32
      aie.use_lock(%output_lock5, AcquireGreaterEqual, 1)
      memref.store %mAA, %output_buffer[%c0] : memref<8xi32>   // PRE-trap
      memref.store %mBB, %output_buffer[%c1] : memref<8xi32>   // TRAP bundle
      memref.store %mCC, %output_buffer[%c2] : memref<8xi32>   // POST-trap
      memref.store %done, %output_buffer[%c3] : memref<8xi32>  // sentinel: core ran to end
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
      // Arm PC_Event0 at the trap bundle (VALID bit 31 | 14-bit PC).
      // TRAP_PC = 0x114: the VLIW bundle address of the memref.store to
      // output_buffer[%c1] (value 0xBB/187), read from llvm-objdump-aie
      // of the compiled core ELF (main_core_0_2.elf in the chess/ build
      // dir). This is DERIVED, not fixed: if the core kernel changes,
      // re-derive TRAP_PC from a fresh disasm before updating the value
      // below -- a stale PC silently misfires the breakpoint and
      // invalidates the experiment. TRAP_PC14 = TRAP_PC & 0x3FFF = 0x0114
      // (14-bit PC field; TRAP_PC must be < 0x4000).
      // value = 0x80000000 | 0x0114 = 0x80000114.
      // Offset 0x32020 = XAIEMLGBL_CORE_MODULE_PC_EVENT0, per aie-rt
      // driver/src/global/xaiemlgbl_params.h:2536.  VALID @ bit 31
      // (XAIEMLGBL_CORE_MODULE_PC_EVENT0_VALID_LSB=31, line 2539);
      // PC_ADDRESS @ bits 13:0 (XAIEMLGBL_CORE_MODULE_PC_EVENT0_PC_ADDRESS_LSB=0,
      // line 2543, WIDTH=14, line 2544).
      aiex.npu.write32 {address = 0x32020 : ui32, column = 0 : i32, row = 2 : i32, value = 0x80000114 : ui32}
      // Debug_Control2[0] = PC_Event_Halt: halt core when a PC event fires.
      // Offset 0x32018 = XAIEMLGBL_CORE_MODULE_DEBUG_CONTROL2, per aie-rt
      // driver/src/global/xaiemlgbl_params.h:2484.  PC_EVENT_HALT @ bit 0
      // (XAIEMLGBL_CORE_MODULE_DEBUG_CONTROL2_PC_EVENT_HALT_LSB=0, line 2499).
      aiex.npu.write32 {address = 0x32018 : ui32, column = 0 : i32, row = 2 : i32, value = 0x1 : ui32}
      // DMA the marker buffer back to the host.
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @out0} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
