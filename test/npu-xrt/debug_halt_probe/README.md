# debug_halt_probe -- Phase A authoring spike

This test is **Phase A of the debug_halt design** (see
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`).

## What it is

A minimal NPU1 kernel with a straight-line compute core that writes three
sentinel markers (0xAA, 0xBB, 0xCC) to an output buffer, then ends. The
runtime sequence issues one `aiex.npu.write32` to compute-tile (0,2)
`Debug_Control0` (offset 0x32010, value 0) before starting the DMA readback.

The Debug_Control0 write is innocuous (value 0 = no-op), but proves that the
control-packet authoring mechanism -- issuing a register write to a
compute-tile debug register from an mlir-aie runtime sequence -- is
expressible and reaches the tile. This is the de-risking step before the G1
(halt timing) and G2 (count-step) experiments are built on top.

## How to re-run

From the xdna-emu directory, EMU only:

```bash
./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw
```

Results land in `build/bridge-test-results/latest/debug_halt_probe.chess/`.

Expected output from `test.exe`:
```
MARKERS: 170 187 204 0 0 0 0 0
SPIKE_PASS
```

The EMU log should show a control-packet/register-write line for tile (0,2)
offset 0x32010.

**Validation status:** EMU-only so far (`--no-hw`). Hardware validation is
pending -- later plan tasks drop the `--no-hw` flag and run the probe on real
NPU1 silicon.

## Spec reference

Experiment details (G1 halt timing, G2 count-step, register offsets, expected
log evidence) live in the design spec:
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`.
