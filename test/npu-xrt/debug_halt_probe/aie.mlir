//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// debug_halt_probe (Phase A -- Exp2: Debug_Control0 count-step matrix).
// Probe kernel: 8 distinct sequential marker stores (output_buffer[k]=101+k,
// k=0..7) characterize Debug_Control0[5:2] (Single_Step_Count, G2).  The
// runtime sequence writes a single Debug_Control0 value (count/halt config)
// to compute-tile (0,2), then reads back all 8 marker slots + Core_Status via
// control-packet OP_READ.  A post-compile-injected XAIE_IO_MASKPOLL on
// Core_Status[20] (CORE_DONE) ensures the OP_READ happens-after the core
// finishes (or, if the DONE-MASKPOLL wedges, re-run with --witness halt to
// catch a DEBUG_HALT partial-state).
//
// Exp2 inherits all three Exp1 redesigns unchanged:
//   @gate objectfifo: arming-race fix -- the core's first op is a
//     llvm.aie2.acquire hardware-blocking stall on @gate, so the
//     Debug_Control0 write provably lands before the core proceeds.
//     Without @gate a LANDED:8/not-halted result is ambiguous between
//     "count-step inert" and "armed too late" (voiding G2).
//   ctrl-in on shim MM2S ch1: shim-channel-disjointness fix -- @gate occupies
//     ch0 (lowest free, not pinnable per DMAChannelAnalysis); the hand-rolled
//     ctrl-in push is on ch1 so the two host->tile flows use distinct shim
//     slave ports.
//   MASKPOLL happens-after: conditional-witness post-compile injection by
//     inject-maskpoll.py (via emu-bridge-test.sh _inject_maskpoll_if_probe).
//     Default --witness done polls CORE_DONE (inert path); if the DONE-
//     MASKPOLL wedges, re-run with --witness halt.
//
// Exp2 only changes @seq arming: the two Exp1 arming writes (PC_Event0 at
// 0x32020, Debug_Control2 at 0x32018) are replaced by a single
// Debug_Control0 write.  The matrix is swept by editing that single value:
//   count=4 no-halt  0x10   <- start point (initial EMU run)
//   count=4 halt     0x11
//   count=2 no-halt  0x08
//   count=8 no-halt  0x20
//   count=0 control  0x00
//
// Observation path: control-packet OP_READ is the SOLE observation (no marker-
// DMA path -- it would dma_wait forever on a halted core, discovered Task 3).
// The OP_READ reads tile data-memory directly regardless of core state.
//
// OUTBUF_ADDR = output_buffer tile-local offset = p0 & 0xFFFF, where p0 is
// the core's 20-bit pointer to output_buffer[0] from llvm-objdump-aie.
// Task 3 disasm: p0 = 0x70400 => OUTBUF_ADDR = 0x0400.
// DERIVED, not fixed.  Self-check: EMU baseline (count-step inert) returns
// all 8 slots as 101..108; if any is garbage, OUTBUF_ADDR is wrong.
// Re-derive from a fresh disasm or aiecc allocation log if the kernel changes.
//===----------------------------------------------------------------------===//
module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_2 = aie.tile(0, 2)

    %output_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "output_lock4"}
    %output_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "output_lock5"}

    %output_buffer = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<8xi32>

    // Blocking objectfifo gate (Exp1 @gate, retained unchanged for Exp2):
    // The core's first op acquires @gate (llvm.aie2.acquire -- hardware-blocking
    // stall), so the Debug_Control0 write in @seq provably lands before the core
    // can proceed past the gate.  Without this guarantee a LANDED:8/not-halted
    // result would be ambiguous between "count-step inert" and "armed too late".
    aie.objectfifo @gate(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<1xi32>>

    // Control-packet flows for tile (0,2): shim DMA:1 -> TileControl:0 (ctrl-in
    // requests from host, on shim MM2S ch1 to avoid collision with @gate on ch0),
    // TileControl:0 -> shim DMA:0 (ctrl-out responses to host, S2MM ch0).
    // SHIM-CHANNEL-DISJOINTNESS: @gate occupies shim MM2S ch0 (lowest free, not
    // pinnable per DMAChannelAnalysis, grounded in spec §4.2).  The ctrl-in push
    // is on shim MM2S ch1 (the controllable side -- hand-rolled register writes).
    // Regdb confirmation: NOC_MODULE_DMA_MM2S_0_CTRL=0x1d210 ch0,
    // NOC_MODULE_DMA_MM2S_1_CTRL=0x1d218 ch1 (stride 0x8, verified against
    // aie-rt driver/src/global/xaiemlgbl_params.h:18743/18775).
    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_0, DMA : 1>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_2, TileControl : 0>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }

    // Exp2 core: 8 distinct sequential marker stores (output_buffer[k]=101+k,
    // k=0..7).  Markers are NOT assumed to be in source order on-chip -- the
    // Exp2 verdict counts how many *distinct* expected values landed, order-
    // independent (each value is unique in [101..108]).  The happens-after is
    // Core_Status[20] (CORE_DONE), not a memory sentinel -- reorder-immune.
    // The @gate acquire is first, same as Exp1.
    %core_0_2 = aie.core(%tile_0_2) {
      // Gate: block until the host feeds @gate (after writing Debug_Control0).
      %gv = aie.objectfifo.acquire @gate(Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
      %g = aie.objectfifo.subview.access %gv[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
      aie.objectfifo.release @gate(Consume, 1)

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %v1 = arith.constant 101 : i32
      %v2 = arith.constant 102 : i32
      %v3 = arith.constant 103 : i32
      %v4 = arith.constant 104 : i32
      %v5 = arith.constant 105 : i32
      %v6 = arith.constant 106 : i32
      %v7 = arith.constant 107 : i32
      %v8 = arith.constant 108 : i32
      aie.use_lock(%output_lock5, AcquireGreaterEqual, 1)
      memref.store %v1, %output_buffer[%c0] : memref<8xi32>
      memref.store %v2, %output_buffer[%c1] : memref<8xi32>
      memref.store %v3, %output_buffer[%c2] : memref<8xi32>
      memref.store %v4, %output_buffer[%c3] : memref<8xi32>
      memref.store %v5, %output_buffer[%c4] : memref<8xi32>
      memref.store %v6, %output_buffer[%c5] : memref<8xi32>
      memref.store %v7, %output_buffer[%c6] : memref<8xi32>
      memref.store %v8, %output_buffer[%c7] : memref<8xi32>
      aie.use_lock(%output_lock4, Release, 1)
      aie.end
    }

    // ctrl0: shim DMA S2MM channel 0 -- receives OP_READ responses from tile
    // (0,2) TileControl (packet flow 0x2).  This is the control-packet
    // response BO and the sole observation path.
    aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)

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
    //   %arg2 = gate BO      (1xi32 token fed to @gate objectfifo after arming)
    //
    // group_id mapping (kernel signature, 0-indexed from aiecc):
    //   group_id(1) = instr BO (implicit, always first)
    //   group_id(3) = %arg0 ctrl-out
    //   group_id(4) = %arg1 ctrl-in
    //   group_id(5) = %arg2 gate
    aie.runtime_sequence @seq(%arg0: memref<9xi32>, %arg1: memref<9xi32>, %arg2: memref<1xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      // 9 reads: 8 marker slots + Core_Status.
      %c9_i64 = arith.constant 9 : i64

      // Exp2 arming: single Debug_Control0 write replaces the Exp1
      // PC_Event0 (0x32020) + Debug_Control2 (0x32018) writes.
      // Debug_Control0[5:2] = Single_Step_Count (aie-rt
      // XAIEMLGBL_CORE_MODULE_DEBUG_CONTROL0, xaiemlgbl_params.h:2452;
      // SINGLE_STEP_COUNT bits 2-5, DEBUG_HALT_BIT bit 0).
      // Matrix: edit this value per sweep point (see README Exp2 matrix).
      //   count=4 no-halt  (4<<2)|0    = 0x10   <- current (initial run)
      //   count=4 halt     (4<<2)|1    = 0x11
      //   count=2 no-halt  (2<<2)|0    = 0x08
      //   count=8 no-halt  (8<<2)|0    = 0x20
      //   count=0 control  (0<<2)|0    = 0x00
      // The write reaches core_debug via apply_tile_local_effects ->
      // write_debug_control0 (xdna-emu mod.rs:787).  EMU count-step is
      // inert (nothing reads the stored count -- the §5.2 state machine is
      // deferred Phase B work).  So on EMU every matrix point: core runs
      // all 8 stores -> CORE_DONE -> DONE-MASKPOLL satisfies -> LANDED:8.
      aiex.npu.write32 {address = 0x32010 : ui32, column = 0 : i32, row = 2 : i32, value = 0x10 : ui32}

      // Feed the blocking gate objectfifo (@gate) after writing Debug_Control0.
      // The core is stalled at its first instruction (llvm.aie2.acquire on
      // @gate's lock) until this DMA delivers the 1-element token.
      // Runtime-sequence is in-order, so the arming write above is guaranteed
      // to reach the tile before this transfer unblocks the core.
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 2 : i64, metadata = @gate} : memref<1xi32>

      // --- Control-packet OP_READ readback ---
      // Register ctrl-out (S2MM ch0) receiver first: exactly 9 words (one
      // response word per OP_READ packet -- 8 marker slots + Core_Status).
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c9_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @ctrl0} : memref<9xi32>

      // Set up the shim DMA BD0 (MM2S ch1) template for ctrl-in pushes.
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 0x1d000 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>

      // Packet 0: OP_READ output_buffer[0] at OUTBUF_ADDR+0 = 0x0400.
      // 0x1d218 = XAIEMLGBL_NOC_MODULE_DMA_MM2S_1_CTRL (ch1, stride 0x8 from
      // ch0 at 0x1d210; verified against aie-rt xaiemlgbl_params.h:18775).
      aiex.npu.maskwrite32 {address = 0x1d218 : ui32, column = 0 : i32, row = 0 : i32, mask = 0x00000F00 : ui32, value = 0x400 : ui32}
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 0x1d21c : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 1: OP_READ output_buffer[1] at OUTBUF_ADDR+4 = 0x0404.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 4 : i32}
      aiex.npu.write32 {address = 0x1d21c : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 2: OP_READ output_buffer[2] at OUTBUF_ADDR+8 = 0x0408.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 8 : i32}
      aiex.npu.write32 {address = 0x1d21c : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 3: OP_READ output_buffer[3] at OUTBUF_ADDR+12 = 0x040C.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 12 : i32}
      aiex.npu.write32 {address = 0x1d21c : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 4: OP_READ output_buffer[4] at OUTBUF_ADDR+16 = 0x0410.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 16 : i32}
      aiex.npu.write32 {address = 0x1d21c : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 5: OP_READ output_buffer[5] at OUTBUF_ADDR+20 = 0x0414.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 20 : i32}
      aiex.npu.write32 {address = 0x1d21c : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 6: OP_READ output_buffer[6] at OUTBUF_ADDR+24 = 0x0418.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 24 : i32}
      aiex.npu.write32 {address = 0x1d21c : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 7: OP_READ output_buffer[7] at OUTBUF_ADDR+28 = 0x041C.
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 28 : i32}
      aiex.npu.write32 {address = 0x1d21c : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Packet 8: OP_READ Core_Status register at 0x32004 (CORE_STATUS_ADDR).
      // bit 16 = DEBUG_HALT, bit 20 = CORE_DONE (aie-rt xaiemlgbl_params.h:
      // 2363-2365 DEBUG_HALTED, 2347-2349 CORE_DONE).
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 32 : i32}
      aiex.npu.write32 {address = 0x1d21c : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Wait for all ctrl-out response words to arrive.
      aiex.npu.dma_wait {symbol = @ctrl0}
    }
  }
}
