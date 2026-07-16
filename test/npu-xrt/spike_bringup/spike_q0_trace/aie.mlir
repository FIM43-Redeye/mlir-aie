module {
  aie.device(npu1_2col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    aie.objectfifo @src(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>> 
    aie.objectfifo @d0(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32xi32>> 
    aie.objectfifo @d1(%mem_tile_0_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<32xi32>> 
    aie.objectfifo.link [@src] -> [@d0, @d1]([] [0, 32])
    aie.objectfifo @j0(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32xi32>> 
    aie.objectfifo @j1(%tile_1_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32xi32>> 
    aie.objectfifo @out(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>> 
    aie.objectfifo.link [@j0, @j1] -> [@out]([0, 32] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %0 = aie.objectfifo.acquire @src(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %2 = arith.index_cast %arg1 : index to i32
          memref.store %2, %1[%arg1] : memref<64xi32>
        }
        aie.objectfifo.release @src(Produce, 1)
      }
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %0 = aie.objectfifo.acquire @d0(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        %2 = aie.objectfifo.acquire @j0(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        %c0_0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c32 step %c1_1 {
          %4 = memref.load %1[%arg1] : memref<32xi32>
          %c1_i32 = arith.constant 1 : i32
          %5 = arith.addi %4, %c1_i32 : i32
          memref.store %5, %3[%arg1] : memref<32xi32>
        }
        aie.objectfifo.release @d0(Consume, 1)
        aie.objectfifo.release @j0(Produce, 1)
      }
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %0 = aie.objectfifo.acquire @d1(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        %2 = aie.objectfifo.acquire @j1(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        %c0_0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c32 step %c1_1 {
          %4 = memref.load %1[%arg1] : memref<32xi32>
          %c2_i32 = arith.constant 2 : i32
          %5 = arith.addi %4, %c2_i32 : i32
          memref.store %5, %3[%arg1] : memref<32xi32>
        }
        aie.objectfifo.release @d1(Consume, 1)
        aie.objectfifo.release @j1(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<1024xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

