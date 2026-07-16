//===- bankdisc apart --------------------------------------*- MLIR -*-===//
// DMA bank-access-width discriminator.  hammer_buf @ 0x0400
// (logical bank 0), dma_buf @ 0x8000
// (logical bank 2), hammer=on.
//===----------------------------------------------------------------------===//
module {
  aie.device(npu1_2col) {
    %shim_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %core_0_2 = aie.tile(0, 2)

    // Declared in ASCENDING address order: the bank-aware allocator walks a
    // per-bank cursor and rejects a pre-allocated buffer that lands below it.
    %hammer_buf = aie.buffer(%core_0_2) {sym_name = "hammer_buf", address = 1024 : i32} : memref<2048xi32>
    %dma_buf    = aie.buffer(%core_0_2) {sym_name = "dma_buf",    address = 32768 : i32} : memref<256xi32>

    %lk_empty = aie.lock(%core_0_2, 0) {init = 1 : i32, sym_name = "lk_empty"}
    %lk_full  = aie.lock(%core_0_2, 1) {init = 0 : i32, sym_name = "lk_full"}

    aie.flow(%core_0_2, DMA : 0, %shim_0_0, DMA : 0)
    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)

    %core = aie.core(%core_0_2) {
      %c0    = arith.constant 0 : index
      %c1    = arith.constant 1 : index
      %cK    = arith.constant 4 : index
      %cREPS = arith.constant 16 : index
      %cOBJ  = arith.constant 256 : index
      %cHAM  = arith.constant 1536 : index

      scf.for %r = %c0 to %cREPS step %c1 {
        aie.use_lock(%lk_empty, AcquireGreaterEqual, 1)
        scf.for %i = %c0 to %cOBJ step %c1 {
          %iv = arith.index_cast %i : index to i32
          memref.store %iv, %dma_buf[%i] : memref<256xi32>
        }
        // Release: the MM2S may now drain dma_buf.  Everything below runs
        // CONCURRENTLY with that drain -- that is the measurement window.
        aie.use_lock(%lk_full, Release, 1)
        // HAMMER: two loads (sibling physical banks, 16 B apart) + one store.
        // Sustains demand on BOTH physical banks of its logical bank every iter.
        scf.for %hi = %c0 to %cHAM step %c1 {
          %hj = arith.addi %hi, %cK : index
          %ha = memref.load %hammer_buf[%hi] : memref<2048xi32>
          %hb = memref.load %hammer_buf[%hj] : memref<2048xi32>
          %hs = arith.addi %ha, %hb : i32
          memref.store %hs, %hammer_buf[%hi] : memref<2048xi32>
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
