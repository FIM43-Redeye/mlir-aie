//===- spike_traced.mlir ----------------------------------------*- MLIR -*-===//
//
// SP-3 spike: the HW-proven Q=0 objectfifo dataflow (build_q0/aie_frozen.mlir)
// + declarative trace on all six modules incl. the memtile (cross-domain). The
// dataflow is FROZEN from of_q0_fanout.py; only trace machinery is added here.
//
// Topology (all objectfifo, all HW-proven):
//   producer core(0,2) --on-chip fill, Q=0--> memtile(0,1) --distribute-->
//     core(0,3) [col0] + core(1,3) [col1, cross-column] --join--> shim(0,0) --> DDR
//
// Trace: 6 units (broadcast=15 sync / broadcast=14 stop), each exposing 8 slots
// (NONE-padded) so the engine patch path can overwrite all 8 per batch. Events
// listed are the firing objectfifo BD/PORT/PERF events (derived from the
// selfmodel swept menu); the memtile uses SEL0/SEL1 BD events (circular-BD
// objectfifo fires *_FINISHED_BD, not *_TASK).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_2col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    aie.objectfifo @src(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d0(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @d1(%mem_tile_0_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@src] -> [@d0, @d1]([] [0, 8])
    aie.objectfifo @j0(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @j1(%tile_1_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@j0, @j1] -> [@out]([0, 8] [])

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = aie.objectfifo.acquire @src(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %2 = arith.index_cast %arg1 : index to i32
          memref.store %2, %1[%arg1] : memref<16xi32>
        }
        aie.objectfifo.release @src(Produce, 1)
      }
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = aie.objectfifo.acquire @d0(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %2 = aie.objectfifo.acquire @j0(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %c0_0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c8 step %c1_1 {
          %4 = memref.load %1[%arg1] : memref<8xi32>
          %c1_i32 = arith.constant 1 : i32
          %5 = arith.addi %4, %c1_i32 : i32
          memref.store %5, %3[%arg1] : memref<8xi32>
        }
        aie.objectfifo.release @d0(Consume, 1)
        aie.objectfifo.release @j0(Produce, 1)
      }
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = aie.objectfifo.acquire @d1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %2 = aie.objectfifo.acquire @j1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %c0_0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c8 step %c1_1 {
          %4 = memref.load %1[%arg1] : memref<8xi32>
          %c2_i32 = arith.constant 2 : i32
          %5 = arith.addi %4, %c2_i32 : i32
          memref.store %5, %3[%arg1] : memref<8xi32>
        }
        aie.objectfifo.release @d1(Consume, 1)
        aie.objectfifo.release @j1(Produce, 1)
      }
      aie.end
    }

    // ===================== TRACE (declarative, broadcast=15 sync) ============
    // (1) core(0,2) core module -- PERF_CNT_2 anchor.
    aie.trace @t_core02(%tile_0_2) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
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
    aie.trace.config @perf_core02(%tile_0_2) {
      aie.trace.reg register = "Performance_Control1" value = 28
      aie.trace.reg register = "Performance_Control2" value = 458752
      aie.trace.reg register = "Performance_Counter2_Event_Value" value = 1024
      aie.end
    }
    // (2) core(0,2) memory module -- producer MM2S BD events.
    aie.trace @t_mem02(%tile_0_2) {
      aie.trace.packet id=2 type=mem
      aie.trace.event<"DMA_MM2S_0_FINISHED_BD">
      aie.trace.event<"DMA_MM2S_1_FINISHED_BD">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // (3) memtile(0,1) -- relay S2MM fill + distribute MM2S (the memtile DOMAIN).
    aie.trace @t_memtile(%mem_tile_0_1) {
      aie.trace.packet id=3 type=memtile
      aie.trace.event<"DMA_S2MM_SEL0_FINISHED_BD">
      aie.trace.event<"DMA_S2MM_SEL1_FINISHED_BD">
      aie.trace.event<"DMA_MM2S_SEL0_FINISHED_BD">
      aie.trace.event<"DMA_MM2S_SEL1_FINISHED_BD">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // (4) core(0,3) memory module -- vertical consumer S2MM/MM2S BD events.
    aie.trace @t_mem03(%tile_0_3) {
      aie.trace.packet id=4 type=mem
      aie.trace.event<"DMA_S2MM_0_FINISHED_BD">
      aie.trace.event<"DMA_S2MM_1_FINISHED_BD">
      aie.trace.event<"DMA_MM2S_0_FINISHED_BD">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // (5) core(1,3) memory module -- cross-column consumer S2MM/MM2S BD events.
    aie.trace @t_mem13(%tile_1_3) {
      aie.trace.packet id=5 type=mem
      aie.trace.event<"DMA_S2MM_0_FINISHED_BD">
      aie.trace.event<"DMA_S2MM_1_FINISHED_BD">
      aie.trace.event<"DMA_MM2S_0_FINISHED_BD">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // (6) shim(0,0) -- join-output drain (S2MM) + port activity.
    aie.trace @t_shim(%shim_noc_tile_0_0) {
      aie.trace.packet id=6 type=shimtile
      aie.trace.event<"DMA_S2MM_0_FINISHED_BD">
      aie.trace.event<"PORT_RUNNING_0">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.event<"NONE">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // ----- Runtime sequence: trace activation + join-output drain -----------
    // arg0 = join output (DDR completion sink); arg1 = trace buffer (highest
    // arg -> the bridge runner binds it as the trace BO).
    aie.runtime_sequence(%arg0: memref<16xi32>, %trace_buf: memref<8192xi8>) {
      aie.trace.host_config buffer_size = 8192 arg_idx = 1
      aie.trace.start_config @t_core02
      aie.trace.start_config @perf_core02
      aie.trace.start_config @t_mem02
      aie.trace.start_config @t_memtile
      aie.trace.start_config @t_mem03
      aie.trace.start_config @t_mem13
      aie.trace.start_config @t_shim
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
