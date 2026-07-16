//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SP-5c Phase 2 `d_h` shim-row-routed broadcast-clock-skew instrument (#140).
// VARIANT: sp5_skew_r3b_pc_dh_swb -- topology-closure experiment.
//
// This is a COPY of sp5_skew_r3b_pc_dh with ONE delta: it additionally
// unblocks the shim PL tile's SECOND broadcast switch (switch B) East/West,
// not just switch A. The _dh variant (switch-A CLR only) confined both floods
// to their source column on Phoenix silicon; aie-rt shows a single local
// source feeds BOTH shim switches' propagation trees (AMD's own
// `_XAie_ErrorHandlingInitAie` drives A and B in lockstep to cross columns),
// so switch B's still-set E/W block was likely the residual confinement. This
// variant adds the 6 switch-B E/W CLR writes (Step 1, right after the 6
// switch-A CLR writes) and tests whether the flood now crosses columns.
// Keep sp5_skew_r3b_pc_dh intact -- it is the switch-A-only test artifact.
//
// (Inherited from sp5_skew_r3b_pc/aie.mlir -- do NOT mutate that kernel, it is
// the validated free-flood `d_v` / bring-up instrument.) The core d_h
// mechanism, unchanged from _dh: both broadcast floods are shaped with
// `Event_Broadcast_Block_*` mask writes on every tile in cols 0-2, rows 1-5
// (15 tiles) BEFORE either `Event_Generate` fires. This forces both floods to
// route horizontally only along the shim row (row 0) and climb straight up
// each column from there -- the same path the real hardware timer-reset
// broadcast uses (`_XAie_SetupBroadcastConfig`, aie-rt `xaie_timer.c:472-542`)
// -- instead of the unconstrained Manhattan flood the original kernel uses.
//
// Under this routing the per-tile arrival-interval reduces to
//   r_X = const + dn_h * d_h ,    dn_h = |s2.col - X.col| - |s1.col - X.col|
// (`docs/superpowers/plans/2026-07-02-sp5c-phase2-dh-capture.md` Sec.0,
// `reset_routed_coeffs` in `r3b_observe.py`). The vertical terms cancel
// exactly, so THIS CAPTURE MEASURES `d_h` ONLY -- `d_v` is rank-deficient
// here by construction (Phase-0 guard
// `test_block_routed_capture_is_rank_deficient_for_dh_dv`), not a defect.
//
// Everything else (perf-counter arming, channel mapping, readback plumbing,
// packet routing) is verbatim from sp5_skew_r3b_pc -- that path is already
// silicon-validated (SP-5b/SP-5c bring-up, 2026-07-01) and is not touched.
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
// BLOCK-MASK CONFIG (new in this kernel; plan Sec.0, aie-rt-derived):
// On every tile in cols 0-2, rows 1-5 (15 tiles), BEFORE either flood fires:
//   - AIE compute tile (rows 2-5): mem-module Event_Broadcast_Block_East_Set
//     @ 0x14080 <- 0xC000, AND core-module Event_Broadcast_Block_West_Set
//     @ 0x34060 <- 0xC000.
//   - Mem-tile (row 1): switch-A Event_Broadcast_A_Block_East_Set @ 0x94080
//     <- 0xC000, AND switch-A Event_Broadcast_A_Block_West_Set @ 0x94060
//     <- 0xC000.
//   - Shim (row 0): nothing (left at reset default / unblocked).
// 0xC000 = bits 14|15, blocking BOTH flood channels (ch14 = s2, ch15 = s1) --
// the "two-channel trap": our two independent single-channel sources need
// both bits blocked, unlike vanilla `XAie_SyncTimer` which blocks only its
// one reset channel and leaves `BcastId+1` open as a shim-row relay trigger.
// Register names/offsets/values verified against `aie_registers_aie2.json`
// (AM025) -- see README.md for full provenance.
//
// READBACK (write32 is write-only, so the counters cannot be read directly):
// per measured tile, a packet-mode `dma_memcpy_nd` (with a `packet` attribute
// carrying pkt_id = geometry pkt_in) pushes a 1-word OP_READ control header
// into that tile's TileControl port; the tile returns one response word via
// TileControl -> shim S2MM into the readback BO. This is the
// add_one_ctrl_packet_4_cores mechanism, which fans out to multiple tiles by
// routing on the per-push `packet` pkt_id (NOT on an in-band routing word).
// The control headers come from --ctrlpkt (r3b_ctrlpkt.py), the readback BO
// is dumped by the host. See sp5_skew_r3b_pc/README.md for the full
// deviation-from-plan writeup (identical mechanism here, unchanged).
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
    // Core (2,5) generates flood-2 (s2). Also a block-mask target (row 5, col 2).
    %src2 = aie.tile(2, 5)

    // The 6 measured cores (geometry.json, counter_index order).
    %t0 = aie.tile(0, 3)   // counter_index 0, role h_west
    %t1 = aie.tile(1, 3)   // counter_index 1, role interior
    %t2 = aie.tile(2, 3)   // counter_index 2, role h_east
    %t3 = aie.tile(1, 2)   // counter_index 3, role v_row2
    %t4 = aie.tile(1, 4)   // counter_index 4, role v_row4
    %t5 = aie.tile(1, 5)   // counter_index 5, role v_row5

    // Remaining block-mask-only tiles (cols 0-2, rows 1-5), not otherwise
    // used by the kernel -- declared purely so the full 15-tile block-mask
    // footprint is visible in the device region. (aiex.npu.write32's
    // column/row attributes address the physical tile directly; these SSA
    // values are not referenced again, matching how debug_halt_probe /
    // sp5_skew_r3b_pc declare every write32 target tile.)
    %m0 = aie.tile(0, 1)   // mem-tile, col 0
    %m1 = aie.tile(1, 1)   // mem-tile, col 1
    %m2 = aie.tile(2, 1)   // mem-tile, col 2
    %b02 = aie.tile(0, 2)  // AIE compute, col 0
    %b22 = aie.tile(2, 2)  // AIE compute, col 2
    %b04 = aie.tile(0, 4)  // AIE compute, col 0
    %b24 = aie.tile(2, 4)  // AIE compute, col 2
    %b05 = aie.tile(0, 5)  // AIE compute, col 0

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
    //   %ctrlin   = OP_READ request stream from --ctrlpkt (r3b_ctrlpkt.py).
    //               1 word/tile: word[k] = control header (stream_id=pkt_out,
    //               op=READ, addr=0x31520). Routing to the tile is carried by
    //               the per-push `packet` attribute (pkt_id=pkt_in), NOT
    //               in-band data, so push k reads word k.
    //   %readback = counter response words (6 x u32), dumped via --trace-out.
    aie.runtime_sequence @seq(%ctrlin: memref<6xi32>, %readback: memref<6xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %c5_i64 = arith.constant 5 : i64
      %c6_i64 = arith.constant 6 : i64

      // ---- Step 1: Shim-row defensive UNBLOCK (block_shim_row routing) ----
      // MUST be the very first register writes -- before the Step-2 block-SET
      // masks, before counter config, before everything.
      //
      // WHY: on Phoenix silicon the shim row's switch-A East/West broadcast
      // block bits for ch14/ch15 were found already SET at run start -- NOT at
      // the POR-open default the AM025 register DB implies. With them set,
      // neither flood could cross row 0 horizontally, so each flood stayed
      // confined to its source column (ch15 -> col 0 only, ch14 -> col 2
      // only) and the horizontal-spine tiles never saw both floods -- d_h
      // was unmeasurable (Phoenix capture, 2026-07-02). Vanilla
      // XAie_SyncTimer assumes shim is default-open and never clears these;
      // nothing in the toolchain load path clears them either. We therefore
      // defensively CLEAR them at the start so the shim row is the guaranteed
      // horizontal distribution spine this capture's routing derivation
      // (plan Sec.0) depends on.
      //
      // TWELVE writes = BOTH shim broadcast switches (A and B), E/W CLR pair
      // (sticky SET/CLR): writing 0xC000 (bits 14|15) to WEST_CLR/EAST_CLR
      // unblocks channels 14,15 in the West/East directions on that shim tile.
      // PL/shim module.
      //   switch A: Event_Broadcast_A_Block_West_Clr @ 0x34064
      //             Event_Broadcast_A_Block_East_Clr @ 0x34084
      //   switch B: Event_Broadcast_B_Block_West_Clr @ 0x340A4
      //             Event_Broadcast_B_Block_East_Clr @ 0x340C4
      //             (switch-B block bank = switch-A's +0x40)
      //
      // SWITCH-B DELTA (vs. sp5_skew_r3b_pc_dh, which cleared switch A only):
      // the shim PL tile has TWO broadcast switches (A and B); a single local
      // source feeds BOTH switches' propagation trees, each gated by its own
      // block bits. AMD's own Phoenix code (`_XAie_ErrorHandlingInitAie`)
      // drives A and B in LOCKSTEP to make a broadcast cross columns -- so
      // switch-A-only unblock (the _dh variant) was insufficient: switch B's
      // E/W block, still set on silicon, kept confining the flood to its
      // source column. This variant unblocks BOTH switches E/W to test whether
      // the flood now crosses. (aie-rt-derived; see README.)
      //
      // North/South are left alone (the vertical climb already works; only
      // the horizontal shim-row hop was blocked). Only row 0 gets these; the
      // Step-2 row 1-5 block-SET writes are intentional and untouched.
      // Same per-(col,row) addressing as every other write here
      // (col<<25 | row<<20 | local & 0xfffff).
      // -- switch A (6 writes) --
      aiex.npu.write32 {address = 0x34064 : ui32, column = 0 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col0 Event_Broadcast_A_Block_West_Clr
      aiex.npu.write32 {address = 0x34084 : ui32, column = 0 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col0 Event_Broadcast_A_Block_East_Clr
      aiex.npu.write32 {address = 0x34064 : ui32, column = 1 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col1 A West_Clr
      aiex.npu.write32 {address = 0x34084 : ui32, column = 1 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col1 A East_Clr
      aiex.npu.write32 {address = 0x34064 : ui32, column = 2 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col2 A West_Clr
      aiex.npu.write32 {address = 0x34084 : ui32, column = 2 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col2 A East_Clr
      // -- switch B (6 writes, NEW in this variant) --
      aiex.npu.write32 {address = 0x340A4 : ui32, column = 0 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col0 Event_Broadcast_B_Block_West_Clr
      aiex.npu.write32 {address = 0x340C4 : ui32, column = 0 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col0 Event_Broadcast_B_Block_East_Clr
      aiex.npu.write32 {address = 0x340A4 : ui32, column = 1 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col1 B West_Clr
      aiex.npu.write32 {address = 0x340C4 : ui32, column = 1 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col1 B East_Clr
      aiex.npu.write32 {address = 0x340A4 : ui32, column = 2 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col2 B West_Clr
      aiex.npu.write32 {address = 0x340C4 : ui32, column = 2 : i32, row = 0 : i32, value = 0xC000 : ui32}  // shim col2 B East_Clr

      // ---- Step 2: Block-mask configuration (block_shim_row routing) ----
      // MUST precede everything below -- both floods must see the mask
      // already in place before Event_Generate fires. Applied uniformly to
      // all 15 tiles in cols 0-2, rows 1-5 (no origin special-case: blocking
      // gates egress, not local injection, so the source tile (2,5) is
      // blocked too, matching aie-rt's `_XAie_SetupBroadcastConfig`).
      // 0xC000 = bits 14|15 (blocks BOTH ch14=s2 and ch15=s1 -- the
      // "two-channel trap", see header comment and plan Sec.0). Addresses
      // are LOCAL tile offsets; `column`/`row` attributes on
      // aiex.npu.write32 fold into the absolute address as
      // col<<25 | row<<20 | (local_offset & 0xfffff)
      // (AIEXDialect.cpp getAbsoluteAddress, AIE2TargetModel column/row
      // shift = 25/20) -- the same mechanism the counter-config writes below
      // already use.
      //
      // Row 1 (mem-tiles), cols 0-2: switch-A block East + West.
      aiex.npu.write32 {address = 0x94080 : ui32, column = 0 : i32, row = 1 : i32, value = 0xC000 : ui32}  // Event_Broadcast_A_Block_East_Set
      aiex.npu.write32 {address = 0x94060 : ui32, column = 0 : i32, row = 1 : i32, value = 0xC000 : ui32}  // Event_Broadcast_A_Block_West_Set
      aiex.npu.write32 {address = 0x94080 : ui32, column = 1 : i32, row = 1 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x94060 : ui32, column = 1 : i32, row = 1 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x94080 : ui32, column = 2 : i32, row = 1 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x94060 : ui32, column = 2 : i32, row = 1 : i32, value = 0xC000 : ui32}
      // Rows 2-5 (AIE compute tiles), cols 0-2: mem-module block East (real
      // external E port) + core-module block West (real external W port).
      // Do NOT block mem-module West or core-module East -- those address
      // the internal core<->mem wire, which must stay open (AM020 ch2:349-363).
      aiex.npu.write32 {address = 0x14080 : ui32, column = 0 : i32, row = 2 : i32, value = 0xC000 : ui32}  // mem-mod Event_Broadcast_Block_East_Set
      aiex.npu.write32 {address = 0x34060 : ui32, column = 0 : i32, row = 2 : i32, value = 0xC000 : ui32}  // core-mod Event_Broadcast_Block_West_Set
      aiex.npu.write32 {address = 0x14080 : ui32, column = 1 : i32, row = 2 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 1 : i32, row = 2 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 2 : i32, row = 2 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 2 : i32, row = 2 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 0 : i32, row = 3 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 0 : i32, row = 3 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 1 : i32, row = 3 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 1 : i32, row = 3 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 2 : i32, row = 3 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 2 : i32, row = 3 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 0 : i32, row = 4 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 0 : i32, row = 4 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 1 : i32, row = 4 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 1 : i32, row = 4 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 2 : i32, row = 4 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 2 : i32, row = 4 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 0 : i32, row = 5 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 0 : i32, row = 5 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 1 : i32, row = 5 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 1 : i32, row = 5 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x14080 : ui32, column = 2 : i32, row = 5 : i32, value = 0xC000 : ui32}
      aiex.npu.write32 {address = 0x34060 : ui32, column = 2 : i32, row = 5 : i32, value = 0xC000 : ui32}

      // ---- Step 3: Performance-counter config (before any Event_Generate) ----
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

      // ---- Step 4: Map both floods onto their broadcast channels (config) ----
      // s1 shim (0,0): Event_Broadcast15_A @ 0x3404C <- USER_EVENT_0 (126).
      //   0x3404C = 0x34010 + 4*15 (regdb shim Event_Broadcast{N}_A); shim
      //   USER_EVENT_0 = 126 (events_database.json aie2 pl module).
      aiex.npu.write32 {address = 0x3404C : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      // s2 core (2,5): Event_Broadcast14 @ 0x34048 <- USER_EVENT_2 (126).
      //   0x34048 = 0x34010 + 4*14 (regdb core Event_Broadcast{N}); core
      //   USER_EVENT_2 = 126 (events_database.json aie2 core module).
      aiex.npu.write32 {address = 0x34048 : ui32, column = 2 : i32, row = 5 : i32, value = 126 : ui32}

      // ---- Step 5: Arm the readback S2MM receive (must precede the pushes) ----
      // N=6 response words (one per measured tile) into %readback.
      aiex.npu.dma_memcpy_nd(%readback[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c6_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @perf_resp} : memref<6xi32>

      // ---- Step 6: The two floods (load-bearing order: s1 then s2) ----
      // Event_Generate @ 0x34008 with value = event number to generate.
      // The program-order gap between these two writes is T0_2 - T0_1, which
      // cancels in every cross-tile difference r_X - r_Y. The block-mask
      // writes above (Step 2) shape where these floods can propagate.
      aiex.npu.write32 {address = 0x34008 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}  // s1: USER_EVENT_0
      aiex.npu.write32 {address = 0x34008 : ui32, column = 2 : i32, row = 5 : i32, value = 126 : ui32}  // s2: USER_EVENT_2

      // ---- Step 7: Readback pushes (counter_index order) then wait ----
      // One packet-mode dma_memcpy_nd per measured tile, in counter_index
      // order. The `packet` attribute carries the stream routing pkt_id
      // (= geometry pkt_in), selecting the destination tile; the transferred
      // word is the OP_READ control header (word 2k+1 of the --ctrlpkt stream:
      // stream_id=pkt_out for the response, op=READ, addr=0x31520). The read
      // TARGET address 0x31520 lives in that control header, not in this MLIR.
      // Responses return via each tile's outbound packet_flow(pkt_out) to the
      // shim S2MM @perf_resp, arriving in issue (= counter_index) order.
      // t0 (0,3) counter_index 0: pkt_in 8, control header @ word 0.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 8, pkt_type = 1>) {id = 2 : i64, issue_token = true, metadata = @ctrlin} : memref<6xi32>
      // t1 (1,3) counter_index 1: pkt_in 18, control header @ word 1.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c1_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 18, pkt_type = 1>) {id = 3 : i64, issue_token = true, metadata = @ctrlin} : memref<6xi32>
      // t2 (2,3) counter_index 2: pkt_in 16, control header @ word 2.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c2_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 16, pkt_type = 1>) {id = 4 : i64, issue_token = true, metadata = @ctrlin} : memref<6xi32>
      // t3 (1,2) counter_index 3: pkt_in 20, control header @ word 3.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c3_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 20, pkt_type = 1>) {id = 5 : i64, issue_token = true, metadata = @ctrlin} : memref<6xi32>
      // t4 (1,4) counter_index 4: pkt_in 1, control header @ word 4.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c4_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 1, pkt_type = 1>) {id = 6 : i64, issue_token = true, metadata = @ctrlin} : memref<6xi32>
      // t5 (1,5) counter_index 5: pkt_in 27, control header @ word 5.
      aiex.npu.dma_memcpy_nd(%ctrlin[%c0_i64, %c0_i64, %c0_i64, %c5_i64] [%c1_i64, %c1_i64, %c1_i64, %c1_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 27, pkt_type = 1>) {id = 7 : i64, issue_token = true, metadata = @ctrlin} : memref<6xi32>

      // Drain the MM2S request channel, then wait for the six response words.
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_wait {symbol = @perf_resp}
    }
  }
}
