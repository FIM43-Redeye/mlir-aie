# SP-5b R3b-PC broadcast-clock-skew instrument (#140)

REQUIRES: ryzen_ai_npu1

Hand-authored register-level kernel that measures per-hop broadcast
clock-skew. Two broadcast floods run from two corner tiles; on every measured
core a hardware Performance Counter times the interval between the two
arrivals (START = flood-1 arrival event, STOP = flood-2 arrival event); each
counter is read back off-chip via a control-packet OP_READ stream. The counter
buffer feeds the already-merged `{d_h, d_v}` observation bridge
(`xdna-emu/tools/calibration/skew/r3b_observe.py`).

**SP-5b produces NO skew number.** This is apparatus that must *run and
reproduce*. Silicon verification is SP-5c; here it is validated by compile +
emulator-smoke-run only (Phoenix is not touched).

## Design

- Target `aie.device(npu1_3col)` (NOT `npu1_4col`, which does not exist; bare
  `npu1` is 4-col). Virtual frame: rows 0 shim / 1 memtile / 2-5 core.
- `s1 = shim (0,0)`, floods on broadcast channel 15 -> received everywhere as
  `BROADCAST_15` (core event 122).
- `s2 = core (2,5)`, floods on broadcast channel 14 -> `BROADCAST_14` (121).
- Six measured cores (`geometry.json`, counter_index order): `(0,3) (1,3) (2,3)`
  (horizontal, row 3) and `(1,2) (1,3) (1,4) (1,5)` (vertical, col 1; `(1,3)`
  shared). Each arms `Performance_Counter0` START=122 STOP=121.
- No `Timer_Control` (0x34000) write on any measured tile -- the perf counter
  (0x31500/0x31520) is a separate hardware unit from the timer.

## Runtime-sequence order (load-bearing)

1. Per measured tile: zero `Performance_Counter0` (0x31520=0), then arm
   `Performance_Control0` (0x31500 = 0x797A).
2. Map both floods: shim `Event_Broadcast15_A` (0x3404C) <- 126;
   core `Event_Broadcast14` (0x34048) <- 126.
3. Arm the readback S2MM receive (`dma_memcpy_nd(%readback, @perf_resp)`, N=6).
4. The two `Event_Generate` writes (0x34008 <- 126): s1 shim, then s2 core.
   Counter config MUST precede these or the START event is missed.
5. Per tile in counter_index order: push the OP_READ control header
   (packet-mode `dma_memcpy_nd`, routing pkt_id = geometry `pkt_in`), then
   `dma_wait @perf_resp`.

## Readback mechanism (and deviation from the plan)

`aiex.npu.write32` is write-only, so the counters are read back with a
hand-assembled control-packet OP_READ path, mirroring the working templates
`add_one_ctrl_packet_4_cores/aie.mlir` (multi-tile fan-out) and
`debug_halt_probe/aie.mlir` (reads a real core register).

Per measured tile, a **packet-mode `dma_memcpy_nd`** carrying
`packet = <pkt_id = <geometry pkt_in>, pkt_type = 1>` pushes a **1-word OP_READ
control header** into that tile's `TileControl` port. The tile returns one
response word (the counter value) via `TileControl -> shim S2MM (@perf_resp)`
into the readback BO. Responses arrive in issue (= counter_index) order,
little-endian u32 -- exactly what `observe_r3b` parses.

**Deviation from the plan's Readback Design (verified wrong against the
toolchain + emulator).** The plan (`docs/superpowers/plans/2026-07-01-sp5b-r3b-pc-kernel.md`,
Readback Design step 3) specified a **2-word** request packet
`[routing header, control header]` pushed by a hand-built `blockwrite` BD, with
the routing header (word 0) selecting the tile. That is **incorrect for shim
control-packet DMA**: the shim BD applies its *own* stream header, so an
in-band routing word arrives at the tile as payload and is mis-parsed as a
control opcode. Confirmed on the emulator: with the 2-word format every packet
routed to a single tile (pkt_id 1) and word 0 (`0x00000008`) was read as a
`Write` to address `0x8`. The correct mechanism routes on the per-push
`packet` attribute pkt_id (proven by `add_one_ctrl_packet_4_cores`). The merged
Task-2 `--ctrlpkt` binary (`r3b_ctrlpkt.py`) is **consumed unchanged**: only its
control header (word `2k+1`) is pushed; its routing header (word `2k`) is dead.

**Packet-ID assignment.** `pkt_in`/`pkt_out` (in `geometry.json`) were revised
from the plan's placeholders (16-21 / 32-37; 32+ exceeds the 5-bit packet-ID
field, `AIEDialect.cpp:2307`). All six inbound flows merge northward up col 0
and diverge only at `(0,3)+`, and `AIECreatePathFindFlows` masks the common ID
bits of merged flows -- so contiguous IDs false-match. The final IDs were found
by a route-only search (physical routing is ID-independent). Change
`geometry.json` and this kernel's `packet_flow` IDs together, then re-run the
route check (`aie-opt --aie-create-pathfinder-flows=route-packet=true`).

## Host binding (bridge-trace-runner)

`runtime_sequence` args are `(%ctrlin, %readback)`. The FFI classifier
(`xdna-emu/src/npu/classify.rs`) tags `%ctrlin` as `ctrlpkt` (packet-mode MM2S)
and `%readback` as `data_s2mm`. `%readback` is the last buffer, so the runner
treats it as the trace slot and dumps its full contents via `--trace-out`.
Note: the runner's `--output` path writes only the XRT-declared pointer size
(8 bytes), so the 24-byte readback is recovered from `--trace-out` (first
`6*4` bytes). The Task-4 gate should read the counters from the `--trace-out`
sink, not `--output`.

## Derived-fact provenance

| Fact | Value | Source |
|---|---|---|
| target | `npu1_3col` | `AIEAttrs.td` (npu1_4col does not exist) |
| `Performance_Counter0` | `0x31520` | aie-rt `xaiemlgbl_params.h:2264` |
| `Performance_Control0` | `0x31500`; Cnt0 start bits 6:0, stop 14:8 | regdb `aie_registers_aie2.json` |
| `Performance_Control0` value | `0x797A` = `122 \| (121<<8)` | derived from the two event numbers |
| `BROADCAST_15` / `BROADCAST_14` (core) | 122 / 121 | `events_database.json` aie2 core module `number` |
| `USER_EVENT_0` (shim/pl) / `USER_EVENT_2` (core) | 126 / 126 | `events_database.json` aie2 pl / core |
| `Event_Generate` | `0x34008` | regdb (core/mem/shim) |
| `Event_Broadcast15_A` (shim) | `0x3404C` = `0x34010 + 4*15` | regdb shim |
| `Event_Broadcast14` (core) | `0x34048` = `0x34010 + 4*14` | regdb core |
| `Timer_Control` | `0x34000` -- NOT written on measured tiles | regdb; perf counter is a separate unit |
| control header | `stream_id<<24 \| opcode<<22 \| beats<<20 \| addr`, parity bit31 (odd overall) | `AIETargetNPU.cpp:277-344` |
| opcode READ | `1` | `debug_halt_probe/test.cpp` |
| packet-ID field width | 5 bits (<= 31) | `AIEDialect.cpp:2307` |
| per-tile request routing | `packet = <pkt_id, pkt_type=1>` on `dma_memcpy_nd` | `add_one_ctrl_packet_4_cores/aie.mlir` |

## Build recipe

```bash
cd ~/npu-work
source toolchain-build/activate-npu-env.sh
cd mlir-aie/test/npu-xrt/sp5_skew_r3b_pc
env -u XDNA_EMU aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts \
  --no-compile-host --alloc-scheme=basic-sequential \
  --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie.mlir
```

(`--alloc-scheme=basic-sequential` per `add_one_ctrl_packet/run.lit`.) Produces
`aie.xclbin` + `insts.bin`.

## Emulator smoke-run (no HW)

```bash
cd ~/npu-work/xdna-emu
cargo build -p xdna-emu-ffi   # rebuild the FFI .so (NOT bare `cargo build`)
# Generate the --ctrlpkt binary from geometry.json:
python3 -c "import json,sys; sys.path.insert(0,'tools'); \
  from calibration.skew.r3b_ctrlpkt import build_ctrlpkt; \
  open('ctrlpkt.bin','wb').write(build_ctrlpkt(json.load(open('../mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/geometry.json'))))"
BU=../mlir-aie/test/npu-xrt/sp5_skew_r3b_pc
XDNA_EMU=1 XDNA_EMU_RUNTIME=debug \
  bridge-runner/build/bridge-trace-runner \
  --xclbin "$BU/aie.xclbin" --instr "$BU/insts.bin" \
  --ctrlpkt ctrlpkt.bin --trace-out readback.bin --trace-size 256
# First 24 bytes of readback.bin = 6 x u32 counters (counter_index order).
```

### Evidence (2026-07-01, emulator, no HW)

Run completed cleanly: `halt_reason=completed cycles=26846`, no wedge, no panic.
The two `Event_Generate` writes (0x34008 <- 126) fired on the shim and the core
source; all six OP_READ requests routed to the six measured tiles with
`addr=0x31520` and the correct `resp_id` (= `pkt_out`); six response words
returned to the shim S2MM and the readback BO was populated (24 bytes). The
counters read **0** -- the emulator does not drive the perf counter from
broadcast arrivals, which is expected: this run validates the **plumbing**
(config + generate + route + read-back), not the interval value. The counter
value is SP-5c's silicon proof.

## Two fallbacks (build only if the primary is disproven on HW at SP-5c)

- (a) If `bridge-trace-runner`'s `--ctrlpkt` classification mis-binds, embed the
  request stream as an MLIR `memref.global` constant and DMA from it, removing
  the `--ctrlpkt` dependency.
- (b) If a control-packet OP_READ of a *perf-counter* register does not return
  the live value on silicon (the templates read data memory + `Core_Status`;
  `0x31520` is the same core-module class, high confidence but unverified for
  perf registers), fall back to a post-run core `LDA 0x31520 -> store -> DMA
  out`. Record which path was used.

**Genuinely unresolved until HW (SP-5c):** whether a control-packet OP_READ of
`Performance_Counter0` returns the live counter value on Phoenix silicon, and
whether broadcast floods on channels 14/15 arrive as `BROADCAST_14/15` events
that drive the counter. The emulator smoke-run validates the plumbing shape;
the silicon read is the SP-5c gate's first real proof, with fallback (b) ready.
