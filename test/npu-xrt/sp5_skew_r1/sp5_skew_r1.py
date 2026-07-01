# SP-5b R1 within-column skew-measurement kernel (#140).
#
# Extends the proven lean Q=0 objectfifo spine (of_q0_lean.py: ON-CHIP
# producer -> memtile relay -> vertical consumer -> join -> shim) from 2 to
# 3 COLLINEAR compute cores (rows 2/3/4, single column 0), so R1's
# differencing extractor (tools/calibration/skew/r1_diff_extract.py) has
# >=3 within-column vertical points to solve d_v from (spec Sec.4/4.2,
# docs/superpowers/specs/2026-06-30-sp5b-bring-up-spec.md).
#
# Same design constraints as of_q0_lean.py, now for three tiles:
#   - Q=0: on-chip producer, only the OUTPUT drain touches DDR.
#   - PURE LOCK/DMA HANDSHAKES, no buffer compute in any core -- avoids the
#     MEMORY_STALL cycle-cost gap (of_q0_rich's `eo[i]=ei[i]+1` loops fired
#     ~220 HW MEMORY_STALL the EMU does not model). With no compute, cores
#     fire only lock events (LOCK_STALL = the cross-domain/within-domain
#     anchor), whose cycle cost the EMU DOES model (lock arbitration
#     737f5505).
#   - Vertical (within-column) only: no cross-column consumer. Cross-column
#     pairs carry irreducible ~30cy jitter on HW (not range-0); R1 is
#     specifically the WITHIN-column extractor, so this kernel never
#     crosses columns.
#
# Dataflow: ProdCore(0,2) --src--> MemTile(0,1) --d--> ConsA(0,3) --e-->
#   MemTile(0,1) --f--> ConsB(0,4) --j--> MemTile(0,1) --out--> ShimTile(0,0)
#   --> DDR (completion sink only).
import sys
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.extras.context import mlir_mod_ctx

dev = AIEDevice.npu1_2col  # single-column geometry; col 0 relative frame

REPS = 32  # sustain the lock ping-pong across the trace window
OBJ = 64  # DMA object size (cores never touch it -> no core memory stalls)


def build():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tObj = np.ndarray[(OBJ,), np.dtype[np.int32]]

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ProdCore = tile(0, 2)  # on-chip producer (col 0, row 2), dn_v 2
            ConsA = tile(0, 3)  # vertical consumer 1 (col 0, row 3), dn_v 3
            ConsB = tile(0, 4)  # vertical consumer 2 (col 0, row 4), dn_v 4 (NEW)

            # prod -> memtile -> consA (vertical hop 1).
            of_src = object_fifo("src", ProdCore, MemTile, 2, tObj)
            of_d = object_fifo("d", MemTile, ConsA, 2, tObj)
            object_fifo_link(of_src, [of_d], [], [0])

            # consA -> memtile -> consB (vertical hop 2, NEW third collinear core).
            of_e = object_fifo("e", ConsA, MemTile, 2, tObj)
            of_f = object_fifo("f", MemTile, ConsB, 2, tObj)
            object_fifo_link(of_e, [of_f], [], [0])

            # consB -> memtile -> shim (join + drain).
            of_j = object_fifo("j", ConsB, MemTile, 2, tObj)
            of_out = object_fifo("out", MemTile, ShimTile, 2, tObj)
            object_fifo_link([of_j], of_out, [0], [])

            # On-chip producer: pure lock handshake, no buffer write (Q=0).
            @core(ProdCore)
            def prod_body():
                for _ in range_(REPS):
                    of_src.acquire(ObjectFifoPort.Produce, 1)
                    of_src.release(ObjectFifoPort.Produce, 1)

            # Vertical consumer 1: pure relay handshake, no buffer read/write.
            @core(ConsA)
            def consa_body():
                for _ in range_(REPS):
                    of_d.acquire(ObjectFifoPort.Consume, 1)
                    of_e.acquire(ObjectFifoPort.Produce, 1)
                    of_d.release(ObjectFifoPort.Consume, 1)
                    of_e.release(ObjectFifoPort.Produce, 1)

            # Vertical consumer 2: pure relay handshake, no buffer read/write.
            @core(ConsB)
            def consb_body():
                for _ in range_(REPS):
                    of_f.acquire(ObjectFifoPort.Consume, 1)
                    of_j.acquire(ObjectFifoPort.Produce, 1)
                    of_f.release(ObjectFifoPort.Consume, 1)
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
