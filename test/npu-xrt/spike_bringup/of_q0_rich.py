# Richer Q=0 objectfifo SP-3 dataflow: identical topology to of_q0_fanout
# (ON-CHIP producer core -> memtile distribute -> col-0 + col-1 consumers ->
# join -> shim), but with REPS iterations over larger per-object buffers so
# the compute cores sustain lock activity across the trace window and fire
# real LOCK_STALL / lock-req / PERF_CNT events (the 2-iteration micro-kernel
# fired ~0 core events -- too short-lived to capture). Still Q=0: the only
# DDR touch is the OUTPUT drain (completion sink); no DDR input.
import sys
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.extras.context import mlir_mod_ctx
import aie.extras.dialects.arith as arith
from aie.helpers.util import np_dtype_to_mlir_type

dev = AIEDevice.npu1_2col

REPS = 16  # pipeline iterations -- sustained activity for the trace window
OBJ = 64  # producer object element count (split HALF/HALF to the consumers)
HALF = OBJ // 2


def build():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tObj = np.ndarray[(OBJ,), np.dtype[np.int32]]
            tHalf = np.ndarray[(HALF,), np.dtype[np.int32]]

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ProdCore = tile(0, 2)  # on-chip producer (col 0)
            ConsA = tile(0, 3)  # consumer col 0 (vertical)
            ConsB = tile(1, 3)  # consumer col 1 (cross-column)

            # Producer core -> memtile -> distribute HALF/HALF to the consumers.
            of_src = object_fifo("src", ProdCore, MemTile, 2, tObj)
            of_d0 = object_fifo("d0", MemTile, ConsA, 2, tHalf)
            of_d1 = object_fifo("d1", MemTile, ConsB, 2, tHalf)  # cross-column
            object_fifo_link(of_src, [of_d0, of_d1], [], [0, HALF])

            # Join back -> shim for completion.
            of_j0 = object_fifo("j0", ConsA, MemTile, 2, tHalf)
            of_j1 = object_fifo("j1", ConsB, MemTile, 2, tHalf)  # cross-column
            of_out = object_fifo("out", MemTile, ShimTile, 2, tObj)
            object_fifo_link([of_j0, of_j1], of_out, [0, HALF], [])

            # On-chip producer: born-on-chip pattern, no DDR input (Q=0).
            @core(ProdCore)
            def prod_body():
                for _ in range_(REPS):
                    eo = of_src.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(OBJ):
                        eo[i] = arith.index_cast(i, to=np_dtype_to_mlir_type(np.int32))
                    of_src.release(ObjectFifoPort.Produce, 1)

            @core(ConsA)
            def consa_body():
                for _ in range_(REPS):
                    ei = of_d0.acquire(ObjectFifoPort.Consume, 1)
                    eo = of_j0.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(HALF):
                        eo[i] = ei[i] + 1
                    of_d0.release(ObjectFifoPort.Consume, 1)
                    of_j0.release(ObjectFifoPort.Produce, 1)

            @core(ConsB)
            def consb_body():
                for _ in range_(REPS):
                    ei = of_d1.acquire(ObjectFifoPort.Consume, 1)
                    eo = of_j1.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(HALF):
                        eo[i] = ei[i] + 2
                    of_d1.release(ObjectFifoPort.Consume, 1)
                    of_j1.release(ObjectFifoPort.Produce, 1)

            # Only a DDR OUTPUT (completion sink); no DDR input -> Q=0 data path.
            # Drains all REPS objects (REPS*OBJ elements) contiguously.
            data_ty = np.ndarray[(REPS * OBJ,), np.dtype[np.int32]]

            @runtime_sequence(data_ty)
            def sequence(outTensor):
                npu_dma_memcpy_nd(
                    metadata=of_out, bd_id=0, mem=outTensor, sizes=[1, 1, 1, REPS * OBJ]
                )
                dma_wait(of_out)

    print(ctx.module)


build()
