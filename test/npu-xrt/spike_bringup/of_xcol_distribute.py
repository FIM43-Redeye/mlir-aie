# Probe: can an objectfifo memtile DISTRIBUTE fan out to a CROSS-COLUMN core on HW?
# memtile(0,1) distributes to core(0,2) [col 0] and core(1,2) [col 1, cross-col],
# then joins back to the shim. If this completes on NPU1, objectfifo can route
# memtile->cross-column-core and the SP-3 topology can stay all-objectfifo.
import sys
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.extras.context import mlir_mod_ctx

dev = AIEDevice.npu1_2col


def build():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            t16 = np.ndarray[(16,), np.dtype[np.int32]]
            t8 = np.ndarray[(8,), np.dtype[np.int32]]

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            Core0 = tile(0, 2)  # col 0
            Core1 = tile(1, 2)  # col 1 -- CROSS-COLUMN from the memtile

            of_in = object_fifo("in", ShimTile, MemTile, 2, t16)
            of_in0 = object_fifo("in0", MemTile, Core0, 2, t8)
            of_in1 = object_fifo("in1", MemTile, Core1, 2, t8)  # cross-column
            object_fifo_link(of_in, [of_in0, of_in1], [], [0, 8])

            of_out = object_fifo("out", MemTile, ShimTile, 2, t16)
            of_out0 = object_fifo("out0", Core0, MemTile, 2, t8)
            of_out1 = object_fifo("out1", Core1, MemTile, 2, t8)  # cross-column
            object_fifo_link([of_out0, of_out1], of_out, [0, 8], [])

            @core(Core0)
            def core_body():
                for _ in range_(2):
                    ei = of_in0.acquire(ObjectFifoPort.Consume, 1)
                    eo = of_out0.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(8):
                        eo[i] = ei[i] + 1
                    of_in0.release(ObjectFifoPort.Consume, 1)
                    of_out0.release(ObjectFifoPort.Produce, 1)

            @core(Core1)
            def core_body():
                for _ in range_(2):
                    ei = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    eo = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(8):
                        eo[i] = ei[i] + 2
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)

            data_ty = np.ndarray[(16,), np.dtype[np.int32]]

            @runtime_sequence(data_ty, data_ty)
            def sequence(inTensor, outTensor):
                npu_dma_memcpy_nd(
                    metadata=of_in,
                    bd_id=1,
                    mem=inTensor,
                    sizes=[1, 1, 1, 16],
                    issue_token=True,
                )
                npu_dma_memcpy_nd(
                    metadata=of_out, bd_id=0, mem=outTensor, sizes=[1, 1, 1, 16]
                )
                dma_wait(of_in, of_out)

    print(ctx.module)


build()
