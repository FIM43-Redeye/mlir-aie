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
// compute-tile (0,2), before starting the readback. Observation is via
// control-packet OP_READ of output_buffer slots and Core_Status while the
// core is halted (core-independent -- works even when the core is stopped).
// The verdict is computed from the disassembled store schedule, not source
// order (the compiler reordered the stores; see TRAP_PC comment below).
// The lock-gated @out0 marker DMA is preserved for the EMU no-trap baseline
// (where the core runs to completion); on real HW with a halted core it
// cannot fire because the core never releases output_lock4.
//===----------------------------------------------------------------------===//
module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_2 = aie.tile(0, 2)

    %output_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "output_lock4"}
    %output_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "output_lock5"}

    %output_buffer = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<8xi32>

    // Control-packet flows for tile (0,2): shim DMA:0 -> TileControl:0 (inbound
    // requests from host), TileControl:0 -> shim DMA:0 (outbound responses to
    // host).  Packet IDs match those assembled in test.cpp: 0x1 inbound,
    // 0x2 outbound (response routing per OP_READ stream_id field).
    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_2, TileControl : 0>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
    // Packet flow for marker DMA readback (preserved for EMU no-trap baseline).
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

    // ctrl0: shim DMA S2MM channel 0 -- receives OP_READ responses from tile (0,2)
    // TileControl (packet flow 0x2).  This is the control-packet response BO.
    // out0: shim DMA S2MM channel 1 -- receives marker DMA from tile (0,2) DMA
    // channel 0 (packet flow 0x3).  Preserved for the EMU no-trap baseline.
    aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 1)

    // Shim DMA BD0 template for the ctrl-in MM2S push path.
    // Word layout (shim DMA BD, base 0x1d000 = BD0 in column 0):
    //   [0] 0x1d000: transfer length in 32-bit words (1 -- each OP_READ packet
    //                is a single header word; no data payload for reads)
    //   [1] 0x1d004: buffer address low (patched per packet by address_patch)
    //   [2] 0x1d008: address high / BD config (0x40090000: valid, use-next-bd=0,
    //                enable-packet, out-of-order-id=0)
    //   [3..7]: packet type/id and misc flags (same as add_one_ctrl_packet template)
    // Derived verbatim from add_one_ctrl_packet/aie.mlir blockwrite_data_0,
    // with word[0]=1 because our packets are 1 word (not 2).
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[1, 0, 0x40090000, 0, 0x40000000, 0, 0, 0x2000000]>

    // Runtime sequence args:
    //   %arg0 = ctrl-out BO  (receives OP_READ responses -- ctrl_pkt_resp)
    //   %arg1 = ctrl-in BO   (holds assembled OP_READ request headers)
    //   %arg2 = output BO    (marker DMA readback -- preserved for EMU baseline)
    //
    // group_id mapping (kernel signature, 0-indexed from aiecc):
    //   group_id(1) = instr BO (implicit, always first)
    //   group_id(3) = %arg0 ctrl-out
    //   group_id(4) = %arg1 ctrl-in
    //   group_id(5) = %arg2 output
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: memref<8xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c5_i64 = arith.constant 5 : i64
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

      // --- Control-packet OP_READ readback ---
      // Register ctrl-out (S2MM ch0) and marker-out (S2MM ch1) receivers first.
      // ctrl-out receives exactly 5 words (one response word per OP_READ packet).
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c5_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @ctrl0} : memref<8xi32>
      // marker-out (preserved for EMU no-trap baseline; on HW with halted core
      // this will not fire, but ctrl-packet readback does not depend on it).
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 2 : i64, issue_token = true, metadata = @out0} : memref<8xi32>

      // Set up the shim DMA BD0 (MM2S ch0) template for ctrl-in pushes.
      // address_patch will fill in the buffer pointer per packet.
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 0x1d000 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>

      // Packet 0: OP_READ output_buffer[0] at OUTBUF_ADDR+0 = 0x0400.
      // OUTBUF_ADDR = output_buffer tile-local offset = p0 & 0xFFFF, where p0 is
      // the core's 20-bit pointer to output_buffer[0] read from llvm-objdump-aie
      // (Task 3 disasm: p0 = 0x70400, so OUTBUF_ADDR = 0x0400).  This is DERIVED,
      // not fixed: if the kernel/allocation changes, re-derive from a fresh disasm
      // (or aiecc allocation log) and update all five OUTBUF_ADDR+k occurrences
      // below.  Self-check: the EMU no-trap baseline must return s[0]=170 (0xAA),
      // s[1]=187 (0xBB), s[2]=204 (0xCC), s[3]=1 from the five read packets;
      // if any slot is garbage, OUTBUF_ADDR is wrong.
      // One-time MM2S channel 0 packet-type setup (per add_one_ctrl_packet idiom).
      aiex.npu.maskwrite32 {address = 0x1d210 : ui32, column = 0 : i32, row = 0 : i32, mask = 0x00000F00 : ui32, value = 0x400 : ui32}
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 1: OP_READ output_buffer[1] at OUTBUF_ADDR+4 = 0x0404.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 4 : i32}
      aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 2: OP_READ output_buffer[2] at OUTBUF_ADDR+8 = 0x0408.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 8 : i32}
      aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 3: OP_READ output_buffer[3] at OUTBUF_ADDR+12 = 0x040C.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 12 : i32}
      aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 4: OP_READ Core_Status register at 0x32004.
      // Core_Status: bit 0 = core enabled/running, bit 16 = debug-halted.
      // XAIEMLGBL_CORE_MODULE_CORE_STATUS offset per aie-rt
      // driver/src/global/xaiemlgbl_params.h.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 16 : i32}
      aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Wait for all ctrl-out response words to arrive.
      aiex.npu.dma_wait {symbol = @ctrl0}
      // Wait for marker DMA (EMU no-trap baseline; no-op on HW with halted core).
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
