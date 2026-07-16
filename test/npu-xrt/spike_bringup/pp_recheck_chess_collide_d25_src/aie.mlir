//===- producer_probe collide_d25 -------------------------------*- MLIR -*-===//
// march_buf @ 0x0400 (logical bank 0),
// dma_buf @ 0x2400 (logical bank 0),
// march=on.
//===----------------------------------------------------------------------===//
module {
  aie.device(npu1_2col) {
    %shim_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %core_0_2 = aie.tile(0, 2)

    // Ascending address order: the bank-aware allocator rejects a pre-placed
    // buffer that lands below its per-bank cursor.
    %march_buf = aie.buffer(%core_0_2) {sym_name = "march_buf", address = 1024 : i32} : memref<2048xi32>
    %dma_buf   = aie.buffer(%core_0_2) {sym_name = "dma_buf",   address = 9216 : i32} : memref<256xi32>
    %filler_buf = aie.buffer(%core_0_2) {sym_name = "filler_buf", address = 32768 : i32} : memref<2048xi32>

    %lk_empty = aie.lock(%core_0_2, 0) {init = 1 : i32, sym_name = "lk_empty"}
    %lk_full  = aie.lock(%core_0_2, 1) {init = 0 : i32, sym_name = "lk_full"}

    aie.flow(%core_0_2, DMA : 0, %shim_0_0, DMA : 0)
    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)

    %core = aie.core(%core_0_2) {
      %c0     = arith.constant 0 : index
      %c1     = arith.constant 1 : index
      %cREPS  = arith.constant 16 : index
      %cOBJ   = arith.constant 256 : index
      %cMARCH = arith.constant 1500 : index

      scf.for %r = %c0 to %cREPS step %c1 {
        aie.use_lock(%lk_empty, AcquireGreaterEqual, 1)
        scf.for %i = %c0 to %cOBJ step %c1 {
          %iv = arith.index_cast %i : index to i32
          memref.store %iv, %dma_buf[%i] : memref<256xi32>
        }
        // Release: the MM2S may now drain dma_buf.  The march below runs
        // CONCURRENTLY with that drain -- the measurement window.
        aie.use_lock(%lk_full, Release, 1)
        %gcMREPS = arith.constant 150 : index
        %gcS = arith.constant 1 : index
        %gcGmS = arith.constant 3 : index
        %off0 = arith.constant 0 : index
        %off1 = arith.constant 1 : index
        %off2 = arith.constant 2 : index
        scf.for %sup = %c0 to %gcMREPS step %c1 {
          %b0base = arith.muli %sup, %gcS : index
          %b2base = arith.muli %sup, %gcGmS : index
          %o0 = arith.addi %b2base, %off0 : index
          %ov0 = arith.index_cast %o0 : index to i32
          memref.store %ov0, %filler_buf[%o0] : memref<2048xi32>
          %o1 = arith.addi %b2base, %off1 : index
          %ov1 = arith.index_cast %o1 : index to i32
          memref.store %ov1, %filler_buf[%o1] : memref<2048xi32>
          %o2 = arith.addi %b2base, %off2 : index
          %ov2 = arith.index_cast %o2 : index to i32
          memref.store %ov2, %filler_buf[%o2] : memref<2048xi32>
          %o3 = arith.addi %b0base, %off0 : index
          %ov3 = arith.index_cast %o3 : index to i32
          memref.store %ov3, %march_buf[%o3] : memref<2048xi32>
        }
      }
      aie.end
    } {stack_size = 1024 : i32}

    // Self-looping single-BD MM2S: fires once per lk_full release, REPS times.
    %mem = aie.mem(%core_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lk_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%dma_buf : memref<256xi32>, 0, 256) {bd_id = 0 : i32}
      aie.use_lock(%lk_empty, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.runtime_sequence(%comp_buf: memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(%comp_buf[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {id = 0 : i64, metadata = @drain} : memref<4096xi32>
      aiex.npu.dma_wait {symbol = @drain}
    }
  }
}
