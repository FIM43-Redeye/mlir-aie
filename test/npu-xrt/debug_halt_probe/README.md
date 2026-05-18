# debug_halt_probe -- Phase A Exp1 (PC_Event0 breakpoint probe)

This test is **Phase A of the debug_halt design** (see
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`).

## What it is

A minimal NPU1 kernel with a straight-line compute core that writes four
separable sentinel markers to `output_buffer`:

| Slot | Value | Meaning |
|------|-------|---------|
| out0 | 0xAA (170) | pre-trap store |
| out1 | 0xBB (187) | trap-bundle store |
| out2 | 0xCC (204) | post-trap store |
| out3 | 1 | done sentinel (core ran to `aie.end`) |

The runtime sequence arms a PC_Event0 breakpoint at the trap bundle before
starting the DMA readback, via two control-packet `aiex.npu.write32`s to
compute-tile (0,2):

- `PC_Event0` (offset 0x32020) = `VALID(bit31) | TRAP_PC14` = `0x80000114`,
  where `TRAP_PC14 = 0x114` is the low-14-bit PC of the trap-bundle store,
  derived from `llvm-objdump-aie` of the compiled core ELF (see the
  re-derivation warning in `aie.mlir` -- a kernel edit invalidates this PC).
- `Debug_Control2` (offset 0x32018) bit 0 = `PC_Event_Halt` = `0x1`: halt the
  core when a PC event fires.

The host harness reads the marker buffer back, prints the four slot markers,
a `BASELINE` verdict (all four markers present, core reached `aie.end`), and a
`TRAP_VERDICT` classifying whether a trap halted the core before or after the
trap-bundle store committed.

## How to re-run

From the xdna-emu directory, EMU only:

```bash
./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw
```

Results land in `build/bridge-test-results/latest/debug_halt_probe.chess/`.

Expected output from `test.exe`:
```
MARKERS: out0=.. out1=.. out2=.. out3=..
BASELINE:YES|NO
TRAP_VERDICT:AFTER_COMMIT|BEFORE_COMMIT|AMBIGUOUS|NO_TRAP_OR_RAN_TO_END
PASS
```

## Current status

**EMU baseline recorded.** On the emulator the core does not halt: the
PC_Event0/Debug_Control2 control-packet writes reach the NPU executor but
land in the executor's register catch-all (no wired core-debug path yet), so
the core runs to completion -- `MARKERS: out0=170 out1=187 out2=204 out3=1`,
`BASELINE:YES`, `TRAP_VERDICT:NO_TRAP_OR_RAN_TO_END`. This is the recorded
EMU comparison baseline, not the source of truth. The hardware run that
derives the real halt timing is Task 4 (drops `--no-hw`, runs on real NPU1).

Later plan tasks reshape (Task 5 Exp2) and restore (Task 7) this probe; this
README describes the state at the Exp1 breakpoint-armed commit.

## Spec reference

Experiment details (halt timing, count-step, register offsets, expected log
evidence) live in the design spec:
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`.
