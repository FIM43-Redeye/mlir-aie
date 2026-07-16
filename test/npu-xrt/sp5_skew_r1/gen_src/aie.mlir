module {
  aie.device(npu1_2col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @src(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>> 
    aie.objectfifo @d(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<64xi32>> 
    aie.objectfifo.link [@src] -> [@d]([] [0])
    aie.objectfifo @e(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>> 
    aie.objectfifo @f(%mem_tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<64xi32>> 
    aie.objectfifo.link [@e] -> [@f]([] [0])
    aie.objectfifo @j(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>> 
    aie.objectfifo @out(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>> 
    aie.objectfifo.link [@j] -> [@out]([0] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c32 step %c1 {
        %0 = aie.objectfifo.acquire @src(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        aie.objectfifo.release @src(Produce, 1)
      }
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c32 step %c1 {
        %0 = aie.objectfifo.acquire @d(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        %2 = aie.objectfifo.acquire @e(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        aie.objectfifo.release @d(Consume, 1)
        aie.objectfifo.release @e(Produce, 1)
      }
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c32 step %c1 {
        %0 = aie.objectfifo.acquire @f(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        %2 = aie.objectfifo.acquire @j(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        aie.objectfifo.release @f(Consume, 1)
        aie.objectfifo.release @j(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<2048xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<2048xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

