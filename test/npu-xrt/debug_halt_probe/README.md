# debug_halt_probe -- Phase A Exp1 (PC_Event0 breakpoint probe)

This test is **Phase A of the debug_halt design** (see
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`).

## What it is

A minimal NPU1 kernel with a straight-line compute core that writes four
separable sentinel markers to `output_buffer`:

| Slot | Value | Meaning |
|------|-------|---------|
| s[0] | 0xAA (170) | 0x11c bundle store (strictly after trap bundle) |
| s[1] | 0xBB (187) | 0x114 bundle store (trap bundle -- PC_Event0 target) |
| s[2] | 0xCC (204) | later store |
| s[3] | 1 | done sentinel (core ran to `aie.end`) |

Source order is `s[0], s[1], s[2], s[3]`, but the compiler reordered the
stores: the trap bundle at PC 0x114 commits `s[1]=0xBB` first, and the
strictly-later bundle at 0x11c commits `s[0]=0xAA`. Verdict logic uses the
disassembled schedule, not source order.

The runtime sequence arms a PC_Event0 breakpoint at the trap bundle before
starting the readback, via two control-packet `aiex.npu.write32`s to
compute-tile (0,2):

- `PC_Event0` (offset 0x32020) = `VALID(bit31) | TRAP_PC14` = `0x80000114`,
  where `TRAP_PC14 = 0x114` is the low-14-bit PC of the trap-bundle store,
  derived from `llvm-objdump-aie` of the compiled core ELF (see the
  re-derivation warning in `aie.mlir` -- a kernel edit invalidates this PC).
- `Debug_Control2` (offset 0x32018) bit 0 = `PC_Event_Halt` = `0x1`: halt the
  core when a PC event fires.

## Observation method (redesigned in Task 4)

**Task 3 discovery**: the lock-gated `@out0` marker DMA cannot complete when
the core is halted (the core never releases `output_lock4`), so waiting on it
hangs. Also, source-order verdict is inverted by the compiler's store
reordering.

**Task 4 redesign**: observation via control-packet OP_READ. After the
breakpoint-arming writes, the runtime sequence sends five OP_READ control
packets to compute-tile (0,2) via the `packet_flow(0x1)/(0x2)` TileControl
path (same idiom as `add_one_ctrl_packet`):

| Packet | Address | Reads |
|--------|---------|-------|
| 0 | `OUTBUF_ADDR + 0` | `output_buffer[0]` → s[0] |
| 1 | `OUTBUF_ADDR + 4` | `output_buffer[1]` → s[1] |
| 2 | `OUTBUF_ADDR + 8` | `output_buffer[2]` → s[2] |
| 3 | `OUTBUF_ADDR + 12` | `output_buffer[3]` → s[3] |
| 4 | `0x32004` | `Core_Status` (bit 0 = running, bit 16 = debug-halted) |

`OUTBUF_ADDR = 0x0400` — derived: `output_buffer` base pointer from
`llvm-objdump-aie` (Task 3 disasm: `p0 = 0x70400`); tile-local offset =
`0x70400 & 0xFFFF`. Not a magic constant: self-checked by the EMU no-trap
baseline returning `s[0]=170, s[1]=187, s[2]=204, s[3]=1`. Re-derive if
the kernel or allocation changes (see aie.mlir and test.cpp re-derivation
warning comments).

The `@out0` marker DMA path is preserved in the MLIR but is not the primary
verdict source. On EMU (no halt) it fires correctly; on HW with a halted core
it cannot fire (lock never released by the halted core).

## Schedule-derived verdict

```
halted = Core_Status[16] && Core_Status[0]

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

Results land in `build/bridge-test-results/latest/`.

## Current status

**EMU mechanism validated (Task 4).** The control-packet OP_READ readback
returns correct tile-data-memory values. On EMU the core does not halt
(debug-register writes land in a catch-all); the core runs to completion:

```
SLOTS: s0=170 s1=187 s2=204 s3=1 CORE_STATUS=0x0 HALTED=0
TRAP_VERDICT:NO_TRAP_OR_RAN_TO_END
PASS
```

OUTBUF_ADDR=0x0400 self-check passed (all four marker slots returned correct
values). The readback mechanism (control-packet OP_READ of tile data memory
and Core_Status) is confirmed end-to-end on the emulator.

**Task 5** (hardware, `--no-hw` removed) is the first read-while-halted
exercise: the real PC_Event0 breakpoint fires, Core_Status shows `HALTED=1`,
and the slot values determine the AFTER_COMMIT / BEFORE_COMMIT verdict.

## Spec reference

Experiment details (halt timing, count-step, register offsets, expected log
evidence) live in the design spec:
`xdna-emu/docs/superpowers/specs/2026-05-18-debug-halt-design.md`.
