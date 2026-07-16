//===- producer_probe collide_sticky -------------------------------*- MLIR -*-===//
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
        %c4s   = arith.constant 4 : index
        %c8s   = arith.constant 8 : index
        %c128s = arith.constant 128 : index
        %c132s = arith.constant 132 : index
        %cMREPS = arith.constant 46 : index
        scf.for %rr = %c0 to %cMREPS step %c1 {
          scf.for %i0 = %c0 to %c128s step %c8s {
            %v0 = arith.index_cast %i0 : index to i32
            memref.store %v0, %march_buf[%i0] : memref<2048xi32>
          }
          scf.for %i1 = %c4s to %c132s step %c8s {
            %v1 = arith.index_cast %i1 : index to i32
            memref.store %v1, %march_buf[%i1] : memref<2048xi32>
          }
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
