# Lean Q=0 objectfifo SP-4a within-domain-exactness substrate (#140).
#
# Same proven Q=0 vertical dataflow spine as of_q0_fanout/rich (ON-CHIP
# producer core -> memtile relay -> vertical consumer -> join -> shim), but:
#   - DROP the cross-column consumer (of_q0_rich's ConsB): cross-column pairs
#     carry irreducible ~30cy jitter (not range-0), so they cannot be byte-
#     matched. SP-4a keeps only the within-column VERTICAL pair prod->consA,
#     which is range-0 (+2) on HW.
#   - Cores do PURE LOCK/DMA HANDSHAKES, no buffer compute. of_q0_rich's
#     consumers ran memory-heavy `eo[i]=ei[i]+1` loops that fire ~220 HW
#     MEMORY_STALL the EMU does not model (cycle-cost gap, known-fidelity-gaps).
#     With no compute, the cores fire only lock events (genuine-blocking
#     LOCK_STALL = the cross-domain anchor) whose cycle cost the EMU DOES model
#     (lock arbitration 737f5505), so EMU within-domain timing is exact and the
#     EMU<->HW cross-domain residual cleanly isolates the broadcast skew.
#
# Still Q=0: on-chip producer, only the OUTPUT drain touches DDR. The trace is
# the deliverable; output bytes are not checked.
import sys
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.extras.context import mlir_mod_ctx

dev = AIEDevice.npu1_2col

REPS = 32  # sustain the lock ping-pong across the trace window
OBJ = 64  # DMA object size (cores never touch it -> no core memory stalls)


def build():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tObj = np.ndarray[(OBJ,), np.dtype[np.int32]]

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ProdCore = tile(0, 2)  # on-chip producer (col 0, row 2)
            ConsA = tile(0, 3)  # vertical consumer (col 0, row 3) -- the pair

            # prod -> memtile -> consA (vertical), then consA -> memtile -> shim.
            of_src = object_fifo("src", ProdCore, MemTile, 2, tObj)
            of_d = object_fifo("d", MemTile, ConsA, 2, tObj)
            object_fifo_link(of_src, [of_d], [], [0])

            of_j = object_fifo("j", ConsA, MemTile, 2, tObj)
            of_out = object_fifo("out", MemTile, ShimTile, 2, tObj)
            object_fifo_link([of_j], of_out, [0], [])

            # On-chip producer: pure lock handshake, no buffer write (Q=0).
            @core(ProdCore)
            def prod_body():
                for _ in range_(REPS):
                    of_src.acquire(ObjectFifoPort.Produce, 1)
                    of_src.release(ObjectFifoPort.Produce, 1)

            # Vertical consumer: pure relay handshake, no buffer read/write.
            @core(ConsA)
            def consa_body():
                for _ in range_(REPS):
                    of_d.acquire(ObjectFifoPort.Consume, 1)
                    of_j.acquire(ObjectFifoPort.Produce, 1)
                    of_d.release(ObjectFifoPort.Consume, 1)
                    of_j.release(ObjectFifoPort.Produce, 1)

            # DDR OUTPUT only (completion sink); no DDR input -> Q=0.
            data_ty = np.ndarray[(REPS * OBJ,), np.dtype[np.int32]]

            @runtime_sequence(data_ty)
            def sequence(outTensor):
                npu_dma_memcpy_nd(
                    metadata=of_out, bd_id=0, mem=outTensor, sizes=[1, 1, 1, REPS * OBJ]
                )
                dma_wait(of_out)

    print(ctx.module)


build()
