# Full Q=0 objectfifo SP-3 dataflow skeleton: ON-CHIP producer core fills the
# first fifo (no DDR input), memtile distributes to a col-0 and a col-1 (cross-
# column) consumer, then joins back to the shim for host-side completion.
# All-objectfifo: memtile domain + cross-column, both HW-proven primitives.
import sys
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.extras.context import mlir_mod_ctx
import aie.extras.dialects.arith as arith
from aie.helpers.util import np_dtype_to_mlir_type

dev = AIEDevice.npu1_2col


def build():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            t16 = np.ndarray[(16,), np.dtype[np.int32]]
            t8 = np.ndarray[(8,), np.dtype[np.int32]]

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ProdCore = tile(0, 2)  # on-chip producer (col 0)
            ConsA = tile(0, 3)  # consumer col 0 (vertical)
            ConsB = tile(1, 3)  # consumer col 1 (cross-column)

            # Producer core -> memtile -> distribute to the two consumers.
            of_src = object_fifo("src", ProdCore, MemTile, 2, t16)
            of_d0 = object_fifo("d0", MemTile, ConsA, 2, t8)
            of_d1 = object_fifo("d1", MemTile, ConsB, 2, t8)  # cross-column
            object_fifo_link(of_src, [of_d0, of_d1], [], [0, 8])

            # Join back -> shim for completion.
            of_j0 = object_fifo("j0", ConsA, MemTile, 2, t8)
            of_j1 = object_fifo("j1", ConsB, MemTile, 2, t8)  # cross-column
            of_out = object_fifo("out", MemTile, ShimTile, 2, t16)
            object_fifo_link([of_j0, of_j1], of_out, [0, 8], [])

            # On-chip producer: born-on-chip pattern, no DDR input (Q=0).
            @core(ProdCore)
            def prod_body():
                for _ in range_(2):
                    eo = of_src.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(16):
                        eo[i] = arith.index_cast(i, to=np_dtype_to_mlir_type(np.int32))
                    of_src.release(ObjectFifoPort.Produce, 1)

            @core(ConsA)
            def consa_body():
                for _ in range_(2):
                    ei = of_d0.acquire(ObjectFifoPort.Consume, 1)
                    eo = of_j0.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(8):
                        eo[i] = ei[i] + 1
                    of_d0.release(ObjectFifoPort.Consume, 1)
                    of_j0.release(ObjectFifoPort.Produce, 1)

            @core(ConsB)
            def consb_body():
                for _ in range_(2):
                    ei = of_d1.acquire(ObjectFifoPort.Consume, 1)
                    eo = of_j1.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(8):
                        eo[i] = ei[i] + 2
                    of_d1.release(ObjectFifoPort.Consume, 1)
                    of_j1.release(ObjectFifoPort.Produce, 1)

            # Only a DDR OUTPUT (completion sink); no DDR input -> Q=0 data path.
            data_ty = np.ndarray[(16,), np.dtype[np.int32]]

            @runtime_sequence(data_ty)
            def sequence(outTensor):
                npu_dma_memcpy_nd(
                    metadata=of_out, bd_id=0, mem=outTensor, sizes=[1, 1, 1, 16]
                )
                dma_wait(of_out)

    print(ctx.module)


build()
