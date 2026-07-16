//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// On-chip 2-hop cross-column spike kernel (SP-3 Q=0 de-risk, issue #140).
//
// Topology (memtile relay, no DDR in the data path):
//
//             col 0              col 1
//   row 3   core(0,3)           core(1,3)    consumers (S2MM receive)
//   row 2   core(0,2)           --           producer: core body fills buf + PERF_CNT_2
//   row 1   memtile(0,1)        --           relay: S2MM fill + 2x MM2S fan-out
//   row 0   shim(0,0)           --           BROADCAST_15 sync + trace drain only
//
// All data is born on-chip: core(0,2)'s core body writes a fixed pattern into
// a local buffer behind a lock; terminating task-API DMAs stream it through the
// memtile to both consumers.  Trace units on the five non-shim modules drain to
// the shim trace buffer (declarative aie.trace API, uniform broadcast=15).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_2col) {
    // ----- Tiles -----------------------------------------------------------
    %shim_0_0 = aie.tile(0, 0)
    %mem_0_1  = aie.tile(0, 1)
    %core_0_2 = aie.tile(0, 2)
    // Consumers receive behind locks only; completion is signalled host-side via
    // the shim drain task (standard await-on-shim path), not core-hosted tokens.
    %core_0_3 = aie.tile(0, 3)
    %core_1_3 = aie.tile(1, 3)

    // ----- Buffers ---------------------------------------------------------
    %prod_buf  = aie.buffer(%core_0_2) {sym_name = "prod_buf"}  : memref<1024xi32>
    %relay_buf = aie.buffer(%mem_0_1)  {sym_name = "relay_buf"} : memref<1024xi32>
    %recv_buf0 = aie.buffer(%core_0_3) {sym_name = "recv_buf0"} : memref<1024xi32>
    %recv_buf1 = aie.buffer(%core_1_3) {sym_name = "recv_buf1"} : memref<1024xi32>

    // ----- Locks -----------------------------------------------------------
    // Producer core(0,2): empty/full ping with the MM2S send task.
    %c02_empty = aie.lock(%core_0_2, 0) {init = 1 : i32, sym_name = "c02_empty"}
    %c02_full  = aie.lock(%core_0_2, 1) {init = 0 : i32, sym_name = "c02_full"}
    // Memtile(0,1): fill lock released by 2 so both fan-out readers can acquire.
    // Each fan-out reader releases its own "done" sink lock to keep every BD
    // block balanced (the lowering requires 0 or 2 lock ops per BD).
    %mt_empty  = aie.lock(%mem_0_1, 0) {init = 1 : i32, sym_name = "mt_empty"}
    %mt_full   = aie.lock(%mem_0_1, 1) {init = 0 : i32, sym_name = "mt_full"}
    %mt_done0  = aie.lock(%mem_0_1, 2) {init = 0 : i32, sym_name = "mt_done0"}
    %mt_done1  = aie.lock(%mem_0_1, 3) {init = 0 : i32, sym_name = "mt_done1"}
    // Completion-drain sink: t_comp releases it once the cross-column fan-out
    // has finished (downstream of every measured on-chip event).
    %mt_comp   = aie.lock(%mem_0_1, 4) {init = 0 : i32, sym_name = "mt_comp"}
    // Consumers.
    %c03_empty = aie.lock(%core_0_3, 0) {init = 1 : i32, sym_name = "c03_empty"}
    %c03_full  = aie.lock(%core_0_3, 1) {init = 0 : i32, sym_name = "c03_full"}
    %c13_empty = aie.lock(%core_1_3, 0) {init = 1 : i32, sym_name = "c13_empty"}
    %c13_full  = aie.lock(%core_1_3, 1) {init = 0 : i32, sym_name = "c13_full"}

    // ----- Stream-switch routing (task-API DMAs need explicit flows) -------
    // hop 1: producer MM2S ch0 -> memtile S2MM ch0
    aie.flow(%core_0_2, DMA : 0, %mem_0_1, DMA : 0)
    // hop 2a (vertical):       memtile MM2S ch0 (SEL0) -> core(0,3) S2MM ch0
    aie.flow(%mem_0_1, DMA : 0, %core_0_3, DMA : 0)
    // hop 2b (cross-column):   memtile MM2S ch2 (SEL1) -> core(1,3) S2MM ch0
    // (even channel: odd memtile MM2S channels get invalid BDs from the task-API
    // allocator -- see t_sel1 below.)
    aie.flow(%mem_0_1, DMA : 2, %core_1_3, DMA : 0)
    // completion drain: memtile MM2S ch4 (even) -> shim S2MM ch0 (trace drain
    // uses shim DMA ch1, so ch0 is free). Strictly downstream of all measured
    // events; the DDR latency only delays host teardown, never the couplings.
    aie.flow(%mem_0_1, DMA : 4, %shim_0_0, DMA : 0)

    // ----- Producer core body: born-on-chip pattern + ACTIVE cycles -------
    // The fill loop supplies the active cycles that PERF_CNT_2 counts.
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

    // ======================================================================
    // TRACE CONFIGURATION (declarative aie.trace, uniform broadcast=15)
    // Five traced modules drain to the shim trace buffer; the shim generates
    // the broadcast-15 sync.  Event slots are overwritten per-batch by the
    // emulator trace engine -- what matters is that each unit exists + routes.
    // ======================================================================

    // (1) core(0,2) core module -- PERF_CNT_2 anchor (1|2|0|PERF_CNT_2).
    aie.trace @trace_core_0_2(%core_0_2) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
      // 8 listed events so the compiled insts EXPOSE all 8 trace slots (write
      // both Trace_Event0 and Trace_Event1); the engine's patch path always
      // overwrites all 8. PERF_CNT_2 is the anchor; NONE pads the unused slots.
      aie.trace.event<"PERF_CNT_2">
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.event<"INSTR_EVENT_1">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // (1b) Explicit PERF_CNT_2 anchor config on core(0,2): counter 2 starts on
    // ACTIVE (event 28), self-resets on PERF_CNT_2 (event 7 << 16), fires every
    // `period` active cycles.  Per tools/perfcnt_defaults.py period = 1024; the
    // 1024-iteration producer loop supplies enough active cycles to fire it.
    // These regs have no named fields in the aie2 regdb, so raw values are used.
    aie.trace.config @perf_core_0_2(%core_0_2) {
      aie.trace.reg register = "Performance_Control1" value = 28
      aie.trace.reg register = "Performance_Control2" value = 458752
      aie.trace.reg register = "Performance_Counter2_Event_Value" value = 1024
      aie.end
    }

    // (2) core(0,2) memory module -- producer MM2S task events.
    aie.trace @trace_mem_0_2(%core_0_2) {
      aie.trace.packet id=2 type=mem
      // Padded to 8 events so all 8 slots are exposed for the engine patcher.
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.event<"DMA_MM2S_0_FINISHED_TASK">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // (3) memtile(0,1) -- S2MM fill + 2x MM2S fan-out task events.
    aie.trace @trace_memtile_0_1(%mem_0_1) {
      aie.trace.packet id=3 type=memtile
      aie.trace.event<"DMA_S2MM_SEL0_START_TASK">
      aie.trace.event<"DMA_S2MM_SEL0_FINISHED_TASK">
      aie.trace.event<"DMA_MM2S_SEL0_START_TASK">
      aie.trace.event<"DMA_MM2S_SEL0_FINISHED_TASK">
      aie.trace.event<"DMA_MM2S_SEL1_START_TASK">
      aie.trace.event<"DMA_MM2S_SEL1_FINISHED_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // (4) core(0,3) memory module -- vertical consumer S2MM task events.
    aie.trace @trace_mem_0_3(%core_0_3) {
      aie.trace.packet id=4 type=mem
      // Padded to 8 events so all 8 slots are exposed for the engine patcher.
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_S2MM_0_FINISHED_TASK">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // (5) core(1,3) memory module -- cross-column consumer S2MM task events.
    aie.trace @trace_mem_1_3(%core_1_3) {
      aie.trace.packet id=5 type=mem
      // Padded to 8 events so all 8 slots are exposed for the engine patcher.
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_S2MM_0_FINISHED_TASK">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // shim(0,0) -- broadcast-15 sync source + trace drain (no data path).
    aie.trace @trace_shim_0_0(%shim_0_0) {
      aie.trace.packet id=6 type=shimtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // ----- Runtime sequence: terminating task-API DMA chain ---------------
    // The bridge trace-runner binds the trace BO to the HIGHEST arg_idx (user
    // buffers come first, trace last). So comp_buf (the tiny DDR completion buffer
    // the shim drain writes into) is arg0, and the trace buffer is arg1 (=highest).
    aie.runtime_sequence(%comp_buf: memref<16xi32>, %trace_buf: memref<8192xi8>) {
      // Trace activation: drain to arg1 (the highest-index BO), then start units.
      aie.trace.host_config buffer_size = 8192 arg_idx = 1
      aie.trace.start_config @trace_core_0_2
      aie.trace.start_config @perf_core_0_2
      aie.trace.start_config @trace_mem_0_2
      aie.trace.start_config @trace_memtile_0_1
      aie.trace.start_config @trace_mem_0_3
      aie.trace.start_config @trace_mem_1_3
      aie.trace.start_config @trace_shim_0_0

      // Map memtile MM2S SEL1 event slot to DMA channel 2 (the cross-column
      // fan-out moved to even channel 2); SEL0 stays on channel 0 (reset default).
      // Register DMA_Event_Channel_Selection: MM2S_SEL1 is the 3-bit field at bit
      // 24, so channel 2 = 2 << 24 = 0x02000000.
      aiex.npu.write32 {column = 0 : i32, row = 1 : i32, address = 657568 : ui32, value = 33554432 : ui32}

      // Producer MM2S: wait for filled buffer, stream to memtile, free buffer.
      %t_prod = aiex.dma_configure_task(%core_0_2, MM2S, 0) {
        aie.use_lock(%c02_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%prod_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c02_empty, Release, 1)
        aie.end
      }

      // Memtile S2MM fill: receive into SRAM, signal both fan-out readers.
      %t_fill = aiex.dma_configure_task(%mem_0_1, S2MM, 0) {
        aie.use_lock(%mt_empty, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%mt_full, Release, 2)
        aie.end
      }

      // Memtile MM2S SEL0 fan-out -> core(0,3).
      %t_sel0 = aiex.dma_configure_task(%mem_0_1, MM2S, 0) {
        aie.use_lock(%mt_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.use_lock(%mt_done0, Release, 1)
        aie.end
      }

      // Memtile MM2S SEL1 fan-out -> core(1,3) (cross-column).
      // AM025 invariant: memtile BDs 0-23 are valid only for EVEN per-direction
      // channels, 24-47 only for ODD. The mlir-aie task-API BD allocator assigns
      // from 0-23 regardless of channel parity, so an ODD MM2S channel gets an
      // invalid BD and never starts. We use EVEN channel 2 (channel 0 is taken by
      // SEL0) so the allocated BD is valid by construction.
      %t_sel1 = aiex.dma_configure_task(%mem_0_1, MM2S, 2) {
        aie.use_lock(%mt_full, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 1024) {bd_id = 2 : i32}
        aie.use_lock(%mt_done1, Release, 1)
        aie.end
      }

      // Consumer core(0,3) S2MM receive (lock-only; no token).
      %t_c0 = aiex.dma_configure_task(%core_0_3, S2MM, 0) {
        aie.use_lock(%c03_empty, AcquireGreaterEqual, 1)
        aie.dma_bd(%recv_buf0 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c03_full, Release, 1)
        aie.end
      }

      // Consumer core(1,3) S2MM receive (lock-only; no token).
      %t_c1 = aiex.dma_configure_task(%core_1_3, S2MM, 0) {
        aie.use_lock(%c13_empty, AcquireGreaterEqual, 1)
        aie.dma_bd(%recv_buf1 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%c13_full, Release, 1)
        aie.end
      }

      // Completion drain (host-teardown signal, strictly downstream of all
      // measured events). t_comp gates on mt_done1 -- released by t_sel1 only
      // after the cross-column fan-out finished -- then streams a few words off
      // the memtile to the shim, which writes them to the DDR comp_buf. EVEN
      // channel 4 (valid memtile BD by construction).
      %t_comp = aiex.dma_configure_task(%mem_0_1, MM2S, 4) {
        aie.use_lock(%mt_done1, AcquireGreaterEqual, 1)
        aie.dma_bd(%relay_buf : memref<1024xi32>, 0, 16) {bd_id = 4 : i32}
        aie.use_lock(%mt_comp, Release, 1)
        aie.end
      }

      // Shim S2MM drain -> DDR comp_buf. Shim-hosted, so the completion token
      // returns to the host via the standard path (no controller_id/packet_flow).
      // This is the single await target.
      %t_drain = aiex.dma_configure_task(%shim_0_0, S2MM, 0) {
        aie.dma_bd(%comp_buf : memref<16xi32>, 0, 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      // Start sinks first, then the relay, then the source, then the drain.
      aiex.dma_start_task(%t_drain)
      aiex.dma_start_task(%t_comp)
      aiex.dma_start_task(%t_c0)
      aiex.dma_start_task(%t_c1)
      aiex.dma_start_task(%t_sel0)
      aiex.dma_start_task(%t_sel1)
      aiex.dma_start_task(%t_fill)
      aiex.dma_start_task(%t_prod)

      // Wait only on the shim drain -- it completes after the whole chain.
      aiex.dma_await_task(%t_drain)
    }
  }
}
