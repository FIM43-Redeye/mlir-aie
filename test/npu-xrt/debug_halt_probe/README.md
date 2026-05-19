# debug_halt_probe -- Phase A Exp1 (PC_Event0 breakpoint probe, MASKPOLL halt-sync)

This test is **Phase A of the debug_halt design** (see
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`).

It arms a PC_Event0 breakpoint on a gated compute core, then reads back
`output_buffer` + `Core_Status` via control-packet OP_READ. A firmware
`XAIE_IO_MASKPOLL` post-compile injected into `insts.bin` makes the OP_READ
provably *happen-after* the core halts. EMU and HW run the byte-identical
patched binary; on EMU the poll is unsatisfiable by design (the expected
`MASKPOLL_UNSATISFIED_EMU` baseline), so the G1 answer is hardware-only
(Task 5b).

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

## MASKPOLL halt-synchronization (the happens-after primitive)

**Task 5 P2/P3 discovery**: the static-`@seq` OP_READ had *no happens-after*
against the core halt. Nothing ordered "core halted at the trap PC" before
"OP_READ reads `output_buffer`" -- it was a pure relative-latency assumption.
On EMU the gated core's stores land *after* the run loop terminated at
`dma_wait @ctrl0`; on silicon the on-die halt and the host/NoC OP_READ have
no synchronization primitive between them. (`aie.runtime_sequence` is strictly
static -- no poll/conditional op exists -- and there is no on-device
halt→lock/event actuation on AIE2, so neither the kernel nor `@seq` can
express this ordering.)

The fix is a firmware **`XAIE_IO_MASKPOLL`** (opcode 0x04) **post-compile
injected into `insts.bin`** by `xdna-emu/tools/inject-maskpoll.py`. It blocks
the instruction stream until `Core_Status & (1<<16)` (DEBUG_HALT) is set, so
the OP_READ provably executes *after* the core has halted. No MLIR dialect op
emits this (`NpuMaskPollOp` does not exist), so a post-compile binary patch is
the only route -- consistent with the probe's existing hand-rolled
control-register writes.

Byte-exact streamed form (28 bytes, little-endian; same record shape as
MaskWrite, opcode 0x03): `reg_off = (col0<<25)|(row2<<20)|0x32004`
(= `0x00232004`, Core_Status absolute NPU address); `value = mask =
0x00010000` (DEBUG_HALT bit 16); `size = 28`. The injector splices it
immediately before the first ctrl-in MaskWrite32 at tile-local `0x1d218`
(the ch1 channel-type setup that precedes the first OP_READ push --
channel-robust, not instruction-count-based) and bumps the `insts.bin`
header op-count `+1` and byte-size `+28`. It is idempotent (no-ops if a
MASKPOLL on `0x00232004` is already present). The bridge wires it in
`scripts/emu-bridge-test.sh` post-compile (and on the cache-hit path) so
**EMU and HW run the byte-identical patched binary**.

## `MASKPOLL_UNSATISFIED_EMU` -- the expected EMU baseline

The streamed MASKPOLL has **no timeout**. On hardware the core halts at the
breakpoint, DEBUG_HALT sets, the poll satisfies, and the OP_READ runs. On
**EMU** the breakpoint-arming control-packet writes are dropped (the
`write_core_register` catch-all -- a recorded Phase B input), so DEBUG_HALT
*never* sets and the poll is **unsatisfiable by design**.

The emulator therefore implements a **graceful poll-termination contract**
(`xdna-emu`, `crates/xdna-emu-ffi/src/execution.rs`): when the executor is
parked in `BlockedOnPoll` with no monotonic progress for a bounded window,
the run ends deterministically with a distinct terminal reason
(`XdnaEmuHaltReason::MaskPollUnsatisfied`), a clean `run.wait()` (no hang,
no timeout). It does **not** fake the polled register, pretend the core
halted, or issue the OP_READ -- so `ctrl-out` is never written.

`test.cpp` pre-fills `ctrl-out` with a sentinel (`0xDEADC0DE`); if every
slot still holds it after a clean `run.wait()`, the OP_READ never landed and
the harness emits `TRAP_VERDICT:MASKPOLL_UNSATISFIED_EMU` -> bridge **PASS**.
This is a *recognized expected-EMU outcome*, distinct from any HW verdict --
it is **never** misreported as BEFORE/AFTER/NO_TRAP. EMU thus validates the
readback mechanism, the gate feed, the disjoint route, the injector, and the
poll-termination contract; the read-*while-halted* semantics and the G1
answer itself are hardware-only (Task 5b).

## Schedule-derived verdict (HW path, unchanged)

```
maskpoll_unsatisfied_emu = (all ctrl-out slots == 0xDEADC0DE sentinel)
  => TRAP_VERDICT:MASKPOLL_UNSATISFIED_EMU  (expected EMU baseline, PASS)

otherwise (HW: core halted, MASKPOLL satisfied, OP_READ ran):
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

**EMU mechanism validated (Task 5 complete -- MASKPOLL halt-sync redesign).**
Gate plumbing + ctrl-in ch1 repoint + MASKPOLL injector + readback +
graceful poll-termination confirmed end-to-end. The injected MASKPOLL is
never satisfiable on EMU (DEBUG_HALT never sets), so the emulator terminates
**deterministically** (`halt_reason=maskpoll_unsatisfied`, ~24k cycles, no
600s timeout, clean `run.wait()`), the harness emits
`TRAP_VERDICT:MASKPOLL_UNSATISFIED_EMU`, and the bridge reports **PASS**.
The polled register is never faked, the core is never pretend-halted, and
the OP_READ is never issued (ctrl-out retains its `0xDEADC0DE` sentinel).
TRAP_PC=0x184 re-verified from fresh disasm (the injector patches only
`insts.bin`; it does not touch the core ELF).

This replaces the obsolete static-`@seq` "EMU scheduling artifact"
(`ANOMALY_NOT_HALTED`) behaviour: with the happens-after MASKPOLL there is
no relative-latency race -- the OP_READ is synchronization-ordered after the
core halt on HW, and deterministically not-issued on EMU.

**Task 5b** (hardware, `--no-hw` removed) is the first read-while-halted
exercise: the injected MASKPOLL blocks the stream until the real PC_Event0
breakpoint fires and DEBUG_HALT sets, the OP_READ then runs *after* the
halt, Core_Status shows `HALTED=1`, and the slot values determine the
AFTER_COMMIT / BEFORE_COMMIT verdict (G1). (The no-timeout MASKPOLL means a
non-halting HW core wedges the device -- the §7-bounded path; one
recovery+retry, then §8 forward-commitment.)

## Spec reference

Experiment details (halt timing, count-step, register offsets, expected log
evidence) live in the design spec:
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`.

---

## Experiment 2 sweep matrix (Task 6 -- Debug_Control0 count-step, G2)

Exp2 reshapes the probe for `Debug_Control0[5:2]` (`Single_Step_Count`)
characterization.  The core writes 8 distinct sequential marker stores
(`output_buffer[k]=101+k`, k=0..7) behind the same `@gate` (arming-race
immunity) and the same ctrl-in-on-ch1 OP_READ readback (9 packets: 8 slots
+ Core_Status).  The `@seq` arming is a single `Debug_Control0` write
replacing the Exp1 `PC_Event0`+`Debug_Control2` writes.

### Matrix

| Config | `Debug_Control0` value | Description |
|--------|------------------------|-------------|
| count=4 no-halt | `0x10` | bits [5:2]=4, bit0=0 -- start point |
| count=4 halt | `0x11` | bits [5:2]=4, bit0=1 (halt bit also set) |
| count=2 no-halt | `0x08` | bits [5:2]=2, bit0=0 |
| count=8 no-halt | `0x20` | bits [5:2]=8, bit0=0 |
| count=0 control | `0x00` | bits [5:2]=0, bit0=0 -- control (no count) |

Edit the `aiex.npu.write32` value in `aie.mlir` `@seq` (the `Debug_Control0`
write at offset `0x32010`) to select a matrix point, then rebuild and re-run.

### Conditional-MASKPOLL protocol

The default Exp2 injection is `--witness done` (Core_Status[20] CORE_DONE).
Set via `DEBUG_HALT_PROBE_WITNESS=done ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only`.

- **DONE-MASKPOLL satisfies** (`DONE=1`, `LANDED:8`, `HALTED=0`): count-step
  is inert for that config -- recorded result, move to next config.
- **DONE-MASKPOLL wedges** (core never reached CORE_DONE because count-step
  halted it): recover (staged, per CLAUDE.md), re-run that config with
  `--witness halt` (`DEBUG_HALT_PROBE_WITNESS=halt`).  The DEBUG_HALT MASKPOLL
  satisfies when the core halts; record `LANDED:N` + `HALTED=1`.

```bash
# Default Exp2 run (witness=done -- inert path)
cd /home/triple/npu-work/xdna-emu
DEBUG_HALT_PROBE_WITNESS=done ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw

# Re-run with witness=halt (if DONE-MASKPOLL wedged on HW)
DEBUG_HALT_PROBE_WITNESS=halt ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only
```

### EMU expectation (Exp2)

On EMU, the `Debug_Control0` write reaches `core_debug` via
`apply_tile_local_effects` -> `write_debug_control0` (xdna-emu `mod.rs:787`).
EMU count-step is inert because nothing reads the stored count -- the §5.2
count-step state machine is deferred Phase B work.  So on EMU every matrix
point: core runs all 8 stores -> `CORE_DONE` -> DONE-MASKPOLL satisfies ->
`LANDED:8`, `DONE=1`, `HALTED=0`.

If the DONE-MASKPOLL does NOT satisfy on EMU (EMU `core_debug` does not latch
`CORE_DONE` in computed `Core_Status`), that is a discovered EMU-fidelity gap
of the Unit-1b class -- surface it explicitly, do not force the verdict.

### Switching between Exp1 and Exp2

Task 8 restores the probe to the Exp1 (BEFORE_COMMIT) config by
`git checkout` of the Exp1 SHAs:

```bash
cd /home/triple/npu-work/mlir-aie
git checkout 830972966c -- test/npu-xrt/debug_halt_probe/test.cpp
git checkout 9a12651d99 -- test/npu-xrt/debug_halt_probe/aie.mlir
```

After restore: run `DEBUG_HALT_PROBE_WITNESS=halt` (the default) for Exp1
(DEBUG_HALT witness).  Exp2 re-investigation: re-apply the Exp2 aie.mlir/
test.cpp from their Task 6 commit SHAs.
