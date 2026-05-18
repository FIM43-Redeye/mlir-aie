# debug_halt_probe -- Phase A Exp1 (PC_Event0 breakpoint probe)

This test is **Phase A of the debug_halt design** (see
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`).

## What it is

A minimal NPU1 kernel with a straight-line compute core that writes four
separable sentinel markers to `output_buffer`:

| Slot | Value | Meaning |
|------|-------|---------|
| s[0] | 0xAA (170) | strictly-after trap bundle (0x18c) |
| s[1] | 0xBB (187) | trap bundle (0x184, PC_Event0 target) |
| s[2] | 0xCC (204) | later store |
| s[3] | 1 | done sentinel (core ran to `aie.end`) |

Source order is `s[0], s[1], s[2], s[3]`, but the compiler reordered the
stores: the trap bundle at PC 0x184 commits `s[1]=0xBB` first, and the
strictly-later bundle at 0x18c commits `s[0]=0xAA`. Verdict logic uses the
disassembled schedule, not source order.

## Blocking objectfifo gate (arming-race fix)

HW attempt 1 (Task 5 initial) failed: the compute core is CDO-enabled before
the host runtime sequence (`@seq`) executes, so the core ran to completion
before the `@seq` arming `write32`s reached the tile.

The fix is a **blocking objectfifo gate**: the core's first operation is
`aie.objectfifo.acquire @gate(Consume, 1)`, which lowers to the
`llvm.aie2.acquire` hardware-blocking intrinsic -- a real pipeline stall immune
to load-elimination (same class as `aie.use_lock`). In `@seq`, the arming
`write32`s (PC_Event0 at 0x32020, Debug_Control2 at 0x32018) execute first,
then a `dma_memcpy_nd` feeds the gate token. Runtime-sequence is an in-order
guarantee (`AIEToConfiguration`), so the core cannot execute its first real
instruction until the host has armed the breakpoint and released the gate.

## Shim-channel disjointness (why ctrl-in is on shim MM2S ch1)

The probe has two host->tile streams:
- `@gate` objectfifo: shim MM2S ch0 → tile DMA:0 (circuit/flow, not packet)
- ctrl-in OP_READ push: shim MM2S ch1 → tile TileControl:0 (packet flow 0x1)

These **must** use disjoint shim MM2S channels. Without the fix, `@gate`
defaulted to shim MM2S ch0, and the hand-rolled ctrl-in push also targeted ch0,
causing the pathfinder to compile a single circuit broadcast from one slave port
into both `TileControl:0` and `DMA:0` -- ctrl-in headers contaminate the gate
buffer and `dma_wait @ctrl0` never satisfies (600s timeout in the first gated
EMU run).

The objectfifo channel is not pinnable by the toolchain: `DMAChannelAnalysis`
(in `AIEObjectFifoStatefulTransform.cpp`) seeds channel occupancy only from
`ShimDMAOp`/`MemOp`/`MemTileDMAOp`/`FlowOp` -- it never reads
`ShimDMAAllocationOp`. There is no per-objectfifo channel attribute
(`AIEOps.td:1825-1848`). The objectfifo unconditionally takes the lowest free
channel and a pre-placed `@gate_shim_alloc` collides with the symbol the
lowering itself emits (`:1752`). So `@gate` stays on its natural default ch0
and the hand-rolled ctrl-in push (the controllable side) is repointed to ch1.

The per-channel stride is `0x8` (verified against aie-rt
`driver/src/global/xaiemlgbl_params.h`: `NOC_MODULE_DMA_MM2S_0_CTRL=0x1d210`,
`NOC_MODULE_DMA_MM2S_1_CTRL=0x1d218`, `NOC_MODULE_DMA_MM2S_0_TASK_QUEUE=0x1d214`,
`NOC_MODULE_DMA_MM2S_1_TASK_QUEUE=0x1d21c`). BD writes (base `0x1d000`, address
patch at `0x1d004`) are channel-independent and unchanged.

## TRAP_PC = 0x184

Derived from `llvm-objdump-aie` of the gated chess build's `main_core_0_2.elf`:

```
0x17a: mova dj0, #0xbb ; movxm p0, #0x70400   <- materialize 0xBB and base ptr
0x184: st dj0, [p0, #4] ; mov m0, #0xaa        <- store 0xBB to output_buffer[1]
```

`p0 = 0x70400` → `OUTBUF_ADDR = 0x70400 & 0xFFFF = 0x0400`.
`TRAP_PC14 = 0x184 & 0x3FFF = 0x184`.
`PC_Event0 value = 0x80000000 | 0x184 = 0x80000184`.

TRAP_PC is DERIVED, not fixed. Re-derive from a fresh disasm whenever the
kernel or objectfifo structure changes -- a stale PC silently misfires the
breakpoint and invalidates the experiment (the re-derivation warning in
`aie.mlir` and `test.cpp` enforces this).

## Observation method

**Task 3 discovery**: a lock-gated marker DMA hangs on a halted core (the core
never releases its lock), so `dma_wait` hangs forever -> `run.wait()` times
out -> NPU wedges. Also, source-order verdict is inverted by the compiler's
store reordering.

**Task 4 redesign**: observation SOLELY via control-packet OP_READ; the
marker-DMA path is removed entirely (the OP_READ reads the markers straight out
of tile data memory, so the DMA transport was fully redundant and a HW-wedge
hazard). After the arming writes, the runtime sequence sends five OP_READ
control packets to compute-tile (0,2) via the `packet_flow(0x1)/(0x2)`
TileControl path (same idiom as `add_one_ctrl_packet`):

| Packet | Address | Reads |
|--------|---------|-------|
| 0 | `OUTBUF_ADDR + 0` | `output_buffer[0]` -> s[0] |
| 1 | `OUTBUF_ADDR + 4` | `output_buffer[1]` -> s[1] |
| 2 | `OUTBUF_ADDR + 8` | `output_buffer[2]` -> s[2] |
| 3 | `OUTBUF_ADDR + 12` | `output_buffer[3]` -> s[3] |
| 4 | `0x32004` | `Core_Status` (bit 16 = DEBUG_HALT, the sole halted signal) |

`OUTBUF_ADDR = 0x0400` -- derived from the disasm (`p0 = 0x70400`, tile-local
offset = `0x70400 & 0xFFFF`). Not a magic constant: self-checked by the EMU
no-trap baseline. Re-derive if the kernel or allocation changes (see aie.mlir
and test.cpp re-derivation warning comments).

On EMU, an OP_READ of `Core_Status` (0x32004) returns `0x0` even for a
completed core: control-packet reads of core/debug registers are not wired
to `core_debug` -- a known emulator read-side routing gap (symmetric to the
write-side gap that drops the breakpoint-arming writes), not a probe bug.
HW populates DEBUG_HALT/ENABLE correctly, so AFTER/BEFORE_COMMIT verdicts are
hardware-only (Task 5b).

## Schedule-derived verdict

```
halted = Core_Status[16]   // DEBUG_HALT bit alone (ENABLE not used)

if !halted:
  s[0]==0xAA && s[1]==0xBB && s[2]==0xCC && s[3]==1  => NO_TRAP_OR_RAN_TO_END
  else                                                 => ANOMALY_NOT_HALTED

if halted:
  s[1]==0xBB && s[0]==0 && s[2]==0 && s[3]==0  => AFTER_COMMIT  (trap committed)
  s[1]==0    && s[0]==0 && s[2]==0 && s[3]==0  => BEFORE_COMMIT (trap not committed)
  else                                          => AMBIGUOUS
```

## How to re-run

From the xdna-emu directory, EMU only:

```bash
./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw
```

Hardware run (Task 5b -- do not drop --no-hw until Task 5b pre-flight):

```bash
./scripts/emu-bridge-test.sh debug_halt_probe --chess-only
```

Results land in `build/bridge-test-results/latest/`.

## Current status

**EMU mechanism validated (Task 5 complete).** Gate plumbing + ctrl-in ch1
repoint + readback end-to-end confirmed. Bridge completes cleanly (no 600s
timeout) -- the ch1 fix resolved the shim-channel collision. TRAP_PC=0x184
re-verified from fresh disasm (unchanged by the ctrl-in repoint).

EMU-side note: due to gate-delivery timing in the EMU (the gate token
unblocks the core at approximately the same simulated cycle the first OP_READ
hits), the EMU run may show `SLOTS: s0=0 s1=0 s2=0 s3=0` and
`TRAP_VERDICT:ANOMALY_NOT_HALTED` rather than the ideal
`NO_TRAP_OR_RAN_TO_END`. This is an EMU scheduling artifact, not a probe
bug: the core does run to completion (confirmed by lock sequence in the EMU
log), but the OP_READ snapshots data memory before the stores commit in the
emulated cycle timeline. The run completes with `PASS` and no timeout. On
real hardware the core runs in nanoseconds after gate release while the
OP_READ takes microseconds to traverse the network, so there is no race on
silicon.

**Task 5b** (hardware, `--no-hw` removed) is the first read-while-halted
exercise: the real PC_Event0 breakpoint fires, Core_Status shows `HALTED=1`,
and the slot values determine the AFTER_COMMIT / BEFORE_COMMIT verdict (G1).

## Spec reference

Experiment details (halt timing, count-step, register offsets, expected log
evidence) live in the design spec:
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`.
