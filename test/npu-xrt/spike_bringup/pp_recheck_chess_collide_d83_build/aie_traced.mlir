module {
  aie.device(npu1_2col) {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %tile_0_2 = aie.tile(0, 2)
    %march_buf = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "march_buf"} : memref<2048xi32> 
    %dma_buf = aie.buffer(%tile_0_2) {address = 9216 : i32, sym_name = "dma_buf"} : memref<256xi32> 
    %filler_buf = aie.buffer(%tile_0_2) {address = 32768 : i32, sym_name = "filler_buf"} : memref<2048xi32> 
    %lk_empty = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "lk_empty"}
    %lk_full = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "lk_full"}
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @drain(%shim_noc_tile_0_0, S2MM, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c256 = arith.constant 256 : index
      %c1500 = arith.constant 1500 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        aie.use_lock(%lk_empty, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c256 step %c1 {
          %0 = arith.index_cast %arg1 : index to i32
          memref.store %0, %dma_buf[%arg1] : memref<256xi32>
        }
        aie.use_lock(%lk_full, Release, 1)
        %c100 = arith.constant 100 : index
        %c5 = arith.constant 5 : index
        %c1_0 = arith.constant 1 : index
        %c0_1 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        scf.for %arg1 = %c0 to %c100 step %c1 {
          %0 = arith.muli %arg1, %c5 : index
          %1 = arith.muli %arg1, %c1_0 : index
          %2 = arith.addi %1, %c0_1 : index
          %3 = arith.index_cast %2 : index to i32
          memref.store %3, %filler_buf[%2] : memref<2048xi32>
          %4 = arith.addi %0, %c0_1 : index
          %5 = arith.index_cast %4 : index to i32
          memref.store %5, %march_buf[%4] : memref<2048xi32>
          %6 = arith.addi %0, %c1_2 : index
          %7 = arith.index_cast %6 : index to i32
          memref.store %7, %march_buf[%6] : memref<2048xi32>
          %8 = arith.addi %0, %c2 : index
          %9 = arith.index_cast %8 : index to i32
          memref.store %9, %march_buf[%8] : memref<2048xi32>
          %10 = arith.addi %0, %c3 : index
          %11 = arith.index_cast %10 : index to i32
          memref.store %11, %march_buf[%10] : memref<2048xi32>
          %12 = arith.addi %0, %c4 : index
          %13 = arith.index_cast %12 : index to i32
          memref.store %13, %march_buf[%12] : memref<2048xi32>
        }
      }
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lk_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%dma_buf : memref<256xi32>, 0, 256) {bd_id = 0 : i32}
      aie.use_lock(%lk_empty, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<16384xi8>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 256 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 388112647 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 1593114 : ui32}
      aiex.npu.write32 {address = 82132 : ui32, column = 0 : i32, row = 2 : i32, value = 4098 : ui32}
      aiex.npu.write32 {address = 82128 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 82144 : ui32, column = 0 : i32, row = 2 : i32, value = 1313677593 : ui32}
      aiex.npu.write32 {address = 82148 : ui32, column = 0 : i32, row = 2 : i32, value = 623464527 : ui32}
      aiex.npu.write32 {address = 201988 : ui32, column = 0 : i32, row = 2 : i32, value = 28 : ui32}
      aiex.npu.write32 {address = 201992 : ui32, column = 0 : i32, row = 2 : i32, value = 458752 : ui32}
      aiex.npu.write32 {address = 202120 : ui32, column = 0 : i32, row = 2 : i32, value = 1024 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.write32 {address = 81920 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673700 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119304 : ui32, column = 1 : i32, mask = 65280 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119308 : ui32, column = 1 : i32, row = 0 : i32, value = 2147483663 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 1 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 1 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 1 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {id = 0 : i64, metadata = @drain} : memref<4096xi32>
      aiex.npu.dma_wait {symbol = @drain}
      aiex.npu.write32 {address = 213064 : ui32, column = 1 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 1 : i32, row = 0 : i32, value = 126 : ui32}
    }
    aie.trace.config @trace_t0_2_config(%tile_0_2) packet_type = core {
      aie.trace.reg register = "Trace_Control0" value = 2038038528 mask = 2139029507 comment = "trace mode + start event + stop event"
      aie.trace.reg register = "Trace_Control1" value = 1 mask = 28703 comment = "packet ID + packet type"
      aie.trace.reg register = "Stream_Switch_Event_Port_Selection_0" value = 256 mask = 16128 comment = "port 1 ID + port 1 master/slave"
      aie.trace.reg register = "Trace_Event0" value = 388112647 mask = 2139062143 comment = "PERF_CNT_2 + INSTR_EVENT_0 + INSTR_EVENT_1 + MEMORY_STALL"
      aie.trace.reg register = "Trace_Event1" value = 1593114 mask = 8355711 comment = "LOCK_STALL + PORT_RUNNING_1 + STREAM_STALL"
    }
    aie.trace.config @perf_core_0_2(%tile_0_2) {
      aie.trace.reg register = "Performance_Control1" value = 28
      aie.trace.reg register = "Performance_Control2" value = 458752
      aie.trace.reg register = "Performance_Counter2_Event_Value" value = 1024
    }
    aie.trace.config @trace_mem_0_2_config(%tile_0_2) packet_type = mem {
      aie.trace.reg register = "Trace_Control1" value = 4098 mask = 28703 comment = "packet ID + packet type"
      aie.trace.reg register = "Trace_Control0" value = 2038038528 mask = 2139029504 comment = "start event + stop event"
      aie.trace.reg register = "Trace_Event0" value = 1313677593 mask = 2139062143 comment = "DMA_MM2S_0_FINISHED_BD + DMA_MM2S_0_STALLED_LOCK + CONFLICT_DM_BANK_0 + CONFLICT_DM_BANK_1"
      aie.trace.reg register = "Trace_Event1" value = 623464527 mask = 2139062143 comment = "CONFLICT_DM_BANK_2 + CONFLICT_DM_BANK_3 + DMA_MM2S_0_MEMORY_STARVATION + DMA_MM2S_0_STREAM_BACKPRESSURE"
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_1_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_2, Trace : 1>
      aie.packet_dest<%shim_noc_tile_1_0, DMA : 1>
    } {keep_pkt_header = true}
  }
}

