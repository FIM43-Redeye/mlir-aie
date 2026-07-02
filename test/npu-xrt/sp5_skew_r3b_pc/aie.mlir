//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SP-5b R3b-PC broadcast-clock-skew instrument (#140).
//
// Runs two broadcast floods from two corner tiles and, on every measured
// core, times the interval between the two arrivals with a hardware
// Performance Counter (START = flood-1 arrival, STOP = flood-2 arrival),
// then reads each counter back off-chip via a hand-assembled control-packet
// OP_READ stream into a host buffer. The counter buffer feeds the already
// merged {d_h, d_v} observation bridge (tools/calibration/skew/r3b_observe.py).
//
// This kernel produces NO skew number: it is apparatus that must run and
// reproduce. Its silicon verification is SP-5c; here it is validated by
// compile + emulator-smoke-run only (Phoenix is not touched).
//
// TWO FLOODS ON DISTINCT BROADCAST CHANNELS -> DISTINCT ARRIVAL EVENTS:
//   s1 = shim (0,0), flood on broadcast channel 15, received everywhere as
//        BROADCAST_15 (core-module event number 122).
//   s2 = core (2,5), flood on broadcast channel 14, received everywhere as
//        BROADCAST_14 (core-module event number 121).
// Each measured core's Performance_Counter0 arms START=122, STOP=121, so it
// counts from the flood-1 arrival to the flood-2 arrival with no timer
// involvement. The perf counter (0x31500/0x31520) is a SEPARATE hardware unit
// from the timer (0x34000), so NO Timer_Control write is issued on any
// measured tile.
//
// READBACK (write32 is write-only, so the counters cannot be read directly):
// per measured tile, a packet-mode `dma_memcpy_nd` (with a `packet` attribute
// carrying pkt_id = geometry pkt_in) pushes a 1-word OP_READ control header
// into that tile's TileControl port; the tile returns one response word via
// TileControl -> shim S2MM into the readback BO. This is the
// add_one_ctrl_packet_4_cores mechanism, which fans out to multiple tiles by
// routing on the per-push `packet` pkt_id (NOT on an in-band routing word).
// The control headers come from --ctrlpkt (r3b_ctrlpkt.py, Task 2); the
// readback BO is dumped by the host.
//
// DEVIATION FROM THE PLAN'S READBACK DESIGN: the plan specified a 2-word
// request packet [routing header, control header] pushed by a hand-built
// blockwrite BD, with the routing header (word 0) selecting the tile. That is
// wrong for shim control-packet DMA -- the shim BD applies its own stream
// header, so an in-band routing word arrives at the tile as payload and is
// mis-parsed as a control opcode (confirmed on the emulator: every packet
// then routed to one tile and was read as a Write). The correct mechanism
// routes via the `packet` attribute (proven by add_one_ctrl_packet_4_cores).
// The merged Task-2 --ctrlpkt binary is consumed unchanged: only its control
// header (word 2k+1) is used; its routing header (word 2k) is dead. See README.
//
// HOST BINDING (bridge-trace-runner classifier, verified against the
// classifier in xdna-emu/src/npu/classify.rs):
//   runtime_sequence args in order: (%ctrlin, %readback).
//     kernargs = [opcode, instr, ninstr, %ctrlin, %readback]
//     buffer kernargs = [instr, %ctrlin, %readback]
//   %ctrlin  -> packet-mode MM2S BD  -> classified Ctrlpkt -> bound by --ctrlpkt.
//   %readback-> S2MM receive         -> it is the LAST buffer, so the runner
//     treats it as the trace slot and dumps its full contents via --trace-out.
//     (bridge-trace-runner's --output path writes only the XRT-declared
//     pointer size = 8 bytes, so the full 24-byte readback is recovered from
//     the --trace-out sink; see README "Readback mechanism" and the two
//     fallbacks.)
//===----------------------------------------------------------------------===//
module {
  aie.device(npu1_3col) {
    // Flood sources + readback shim.
    // The shim (0,0) both generates flood-1 (s1) and drives the readback DMA.
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    // Core (2,5) generates flood-2 (s2).
    %src2 = aie.tile(2, 5)

    // The 6 measured cores (geometry.json, counter_index order).
    %t0 = aie.tile(0, 3)   // counter_index 0, role h_west
    %t1 = aie.tile(1, 3)   // counter_index 1, role interior
    %t2 = aie.tile(2, 3)   // counter_index 2, role h_east
    %t3 = aie.tile(1, 2)   // counter_index 3, role v_row2
    %t4 = aie.tile(1, 4)   // counter_index 4, role v_row4
    %t5 = aie.tile(1, 5)   // counter_index 5, role v_row5

    // Per-tile control-packet flows (pkt_in / pkt_out from geometry.json).
    // Inbound:  shim MM2S 0 -> tile TileControl 0 (routed by pkt_in).
    // Outbound: tile TileControl 0 -> shim S2MM 0 (routed by pkt_out); all six
    //           responses land on the single @perf_resp S2MM channel.
    // pkt_in / pkt_out must be mask-separable by the packet router
    // (AIECreatePathFindFlows merges flows sharing a hop and masks their common
    // ID bits; a divergent flow must differ from the continuing group in a bit
    // that group holds in common). All six inbound flows merge northward up
    // col 0 and diverge only at (0,3)+, which makes hand-assignment fragile, so
    // these IDs were found by a route-only search (tools scratch, physical
    // routing is ID-independent). They are the geometry.json pkt_in/pkt_out,
    // kept in exact sync -- change both together and re-run the route check.
    aie.packet_flow(8) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%t0, TileControl : 0>
    }
    aie.packet_flow(22) {
      aie.packet_source<%t0, TileControl : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(18) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%t1, TileControl : 0>
    }
    aie.packet_flow(21) {
      aie.packet_source<%t1, TileControl : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(16) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%t2, TileControl : 0>
    }
    aie.packet_flow(9) {
      aie.packet_source<%t2, TileControl : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(20) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%t3, TileControl : 0>
    }
    aie.packet_flow(10) {
      aie.packet_source<%t3, TileControl : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%t4, TileControl : 0>
    }
    aie.packet_flow(31) {
      aie.packet_source<%t4, TileControl : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(27) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%t5, TileControl : 0>
    }
    aie.packet_flow(7) {
      aie.packet_source<%t5, TileControl : 0>
      aie.packet_dest<%shim, DMA : 0>
    }

    // Readback S2MM receiver: all OP_READ responses arrive here in issue order
    // (= counter_index order), little-endian u32, exactly what observe_r3b reads.
    aie.shim_dma_allocation @perf_resp (%shim, S2MM, 0)

    // Shim MM2S allocation for the OP_READ request pushes. Each push is a
    // packet-mode DMA (dma_memcpy_nd with a `packet` attribute) that carries
    // the stream routing pkt_id, so the request reaches the destination tile's
    // TileControl port. This is the add_one_ctrl_packet_4_cores mechanism.
    aie.shim_dma_allocation @ctrlin (%shim, MM2S, 0)

    // Runtime sequence args:
    //   %ctrlin   = OP_READ request stream from --ctrlpkt (Task 2). r3b_ctrlpkt.py
    //               emits 2 words/tile: word[2k]=routing header (UNUSED here -- see
    //               README "Readback mechanism"; routing is carried by the per-push
    //               `packet` attribute, not in-band data) and word[2k+1]=control
    //               header (stream_id=pkt_out, op=READ, addr=0x31520). Each push
    //               reads only the control header at word 2k+1.
    //   %readback = counter response words (6 x u32), dumped via --trace-out.
    aie.runtime_sequence @seq(%ctrlin: memref<12xi32>, %readback: memref<6xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c3_i64 = arith.constant 3 : i64
      %c5_i64 = arith.constant 5 : i64
      %c6_i64 = arith.constant 6 : i64
      %c7_i64 = arith.constant 7 : i64
      %c9_i64 = arith.constant 9 : i64
      %c11_i64 = arith.constant 11 : i64

      // ---- Step 2: Performance-counter config (before any Event_Generate) ----
      // Per measured tile: zero Performance_Counter0, then arm
      // Performance_Control0 = 0x797A = 122 | (121 << 8):
      //   Cnt0_Start_Event (bits 6:0)  = BROADCAST_15 (122)  [flood-1 arrival]
      //   Cnt0_Stop_Event  (bits 14:8) = BROADCAST_14 (121)  [flood-2 arrival]
      // Field layout: regdb Performance_Control0 bits 6:0 / 14:8. Event numbers:
      // events_database.json aie2 core module BROADCAST_15=122, BROADCAST_14=121.
      // Performance_Counter0 @ 0x31520, Performance_Control0 @ 0x31500 (aie-rt
      // xaiemlgbl_params.h:2264 XAIEMLGBL_CORE_MODULE_PERFORMANCE_COUNTER0).
      // No Timer_Control (0x34000) write: the perf counter is a separate unit.
      aiex.npu.write32 {address = 0x31520 : ui32, column = 0 : i32, row = 3 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 0x31500 : ui32, column = 0 : i32, row = 3 : i32, value = 0x797A : ui32}
      aiex.npu.write32 {address = 0x31520 : ui32, column = 1 : i32, row = 3 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 0x31500 : ui32, column = 1 : i32, row = 3 : i32, value = 0x797A : ui32}
      aiex.npu.write32 {address = 0x31520 : ui32, column = 2 : i32, row = 3 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 0x31500 : ui32, column = 2 : i32, row = 3 : i32, value = 0x797A : ui32}
      aiex.npu.write32 {address = 0x31520 : ui32, column = 1 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 0x31500 : ui32, column = 1 : i32, row = 2 : i32, value = 0x797A : ui32}
      aiex.npu.write32 {address = 0x31520 : ui32, column = 1 : i32, row = 4 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 0x31500 : ui32, column = 1 : i32, row = 4 : i32, value = 0x797A : ui32}
      aiex.npu.write32 {address = 0x31520 : ui32, column = 1 : i32, row = 5 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 0x31500 : ui32, column = 1 : i32, row = 5 : i32, value = 0x797A : ui32}

      // ---- Step 3: Map both floods onto their broadcast channels (config) ----
      // s1 shim (0,0): Event_Broadcast15_A @ 0x3404C <- USER_EVENT_0 (126).
      //   0x3404C = 0x34010 + 4*15 (regdb shim Event_Broadcast{N}_A); shim
      //   USER_EVENT_0 = 126 (events_database.json aie2 pl module).
      aiex.npu.write32 {address = 0x3404C : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      // s2 core (2,5): Event_Broadcast14 @ 0x34048 <- USER_EVENT_2 (126).
      //   0x34048 = 0x34010 + 4*14 (regdb core Event_Broadcast{N}); core
      //   USER_EVENT_2 = 126 (events_database.json aie2 core module).
      aiex.npu.write32 {address = 0x34048 : ui32, column = 2 : i32, row = 5 : i32, value = 126 : ui32}

      // ---- Step 4: Arm the readback S2MM receive (must precede the pushes) ----
      // N=6 response words (one per measured tile) into %readback.
      aiex.npu.dma_memcpy_nd(%readback[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c6_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @perf_resp} : memref<6xi32>

      // ---- Step 5: The two floods (load-bearing order: s1 then s2) ----
      // Event_Generate @ 0x34008 with value = event number to generate.
      // The program-order gap between these two writes is T0_2 - T0_1, which
      // cancels in every cross-tile difference r_X - r_Y.
      aiex.npu.write32 {address = 0x34008 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}  // s1: USER_EVENT_0
      aiex.npu.write32 {address = 0x34008 : ui32, column = 2 : i32, row = 5 : i32, value = 126 : ui32}  // s2: USER_EVENT_2

      // ---- Step 6: Readback pushes (counter_index order) then wait ----
      // One packet-mode dma_memcpy_nd per measured tile, in counter_index
      // order. The `packet` attribute carries the stream routing pkt_id
      // (= geometry pkt_in), selecting the destination tile; the transferred
      // word is the OP_READ control header (word 2k+1 of the --ctrlpkt stream:
      // stream_id=pkt_out for the response, op=READ, addr=0x31520). The read
      // TARGET address 0x31520 lives in that control header, not in this MLIR.
      // Responses return via each tile's outbound packet_flow(pkt_out) to the
      // shim S2MM @perf_resp, arriving in issue (= counter_index) order.
      // t0 (0,3) counter_index 0: pkt_in 8, control header @ word 1.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c1_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 8, pkt_type = 1>) {id = 2 : i64, issue_token = true, metadata = @ctrlin} : memref<12xi32>
      // t1 (1,3) counter_index 1: pkt_in 18, control header @ word 3.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c3_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 18, pkt_type = 1>) {id = 3 : i64, issue_token = true, metadata = @ctrlin} : memref<12xi32>
      // t2 (2,3) counter_index 2: pkt_in 16, control header @ word 5.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c5_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 16, pkt_type = 1>) {id = 4 : i64, issue_token = true, metadata = @ctrlin} : memref<12xi32>
      // t3 (1,2) counter_index 3: pkt_in 20, control header @ word 7.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c7_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 20, pkt_type = 1>) {id = 5 : i64, issue_token = true, metadata = @ctrlin} : memref<12xi32>
      // t4 (1,4) counter_index 4: pkt_in 1, control header @ word 9.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c9_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 1, pkt_type = 1>) {id = 6 : i64, issue_token = true, metadata = @ctrlin} : memref<12xi32>
      // t5 (1,5) counter_index 5: pkt_in 27, control header @ word 11.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c11_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 27, pkt_type = 1>) {id = 7 : i64, issue_token = true, metadata = @ctrlin} : memref<12xi32>

      // Drain the MM2S request channel, then wait for the six response words.
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_wait {symbol = @perf_resp}
    }
  }
}
