# SP-5c Phase 2 `d_h` variant `_dh_swb` -- both-shim-switch topology-closure (#140)

REQUIRES: ryzen_ai_npu1

**Topology-closure experiment. A copy of `sp5_skew_r3b_pc_dh` with ONE delta:
it also unblocks the shim PL tile's SECOND broadcast switch (switch B) E/W,
not just switch A.** See "Switch-B delta" below. `sp5_skew_r3b_pc_dh` (the
switch-A-only artifact) is kept intact.

The rest of this README describes the shared `_dh` mechanism; it is a **copy**
of `sp5_skew_r3b_pc` (do NOT mutate that kernel -- it is the
validated free-flood `d_v` / bring-up instrument) with the
substantive change that both broadcast floods are shaped with
`Event_Broadcast_Block_*` mask writes on every tile in cols 0-2, rows 1-5 (15
tiles) BEFORE either `Event_Generate` fires, so the flood is forced onto the
same shim-row-only-then-climb path the real hardware timer-reset broadcast
uses (`_XAie_SetupBroadcastConfig`, aie-rt `xaie_timer.c:472-542`), instead of
the unconstrained Manhattan flood the original kernel uses.

Governing plan: `xdna-emu/docs/superpowers/plans/2026-07-02-sp5c-phase2-dh-capture.md`
(Sec.0 = the block-mask derivation, Sec.1a = this kernel's task spec).

**This capture measures `d_h` ONLY.** Under block routing the per-tile
arrival interval reduces to

```
r_X = const + dn_h * d_h ,    dn_h = |s2.col - X.col| - |s1.col - X.col|
```

(`reset_routed_coeffs`, `r3b_observe.py`). The vertical terms cancel exactly
(`s2`'s climb-down is a constant that dies in reference differencing; each
tile's climb-up is common to both floods and cancels in the interval), so
`d_v` is **rank-deficient by construction** here -- the Phase-0 guard
`test_block_routed_capture_is_rank_deficient_for_dh_dv` encodes this. This is
by design, not a defect; `d_v` is measured elsewhere (Phase 3, the R1
two-sided spine).

Like `sp5_skew_r3b_pc`, **this kernel produces NO skew number.** It is
apparatus validated by compile + emulator-smoke-run only; the HW capture is
Phase 2 Sec.2 of the plan (Phoenix, gated on this + the `observe_r3b`/
`extract_r3b_dh` code deltas landing green).

## What changed vs. `sp5_skew_r3b_pc` (the delta)

Everything is verbatim from the parent kernel -- perf-counter arming, channel
mapping, packet-flow IDs, readback plumbing -- **except**:

0. **12 shim-row UNBLOCK (CLR) writes** (Step 1 of `@seq`), the very first
   register writes, before the block-SET masks and before everything else.
   On the shim row (row 0), cols 0-2, BOTH broadcast switches' E/W CLR pairs
   (6 switch-A writes, then 6 switch-B writes):

   | Switch | Register (local offset) | Value | Module |
   |---|---|---|---|
   | A | `Event_Broadcast_A_Block_West_Clr` `0x34064` | `0xC000` | PL/shim (switch A) |
   | A | `Event_Broadcast_A_Block_East_Clr` `0x34084` | `0xC000` | PL/shim (switch A) |
   | B | `Event_Broadcast_B_Block_West_Clr` `0x340A4` | `0xC000` | PL/shim (switch B) |
   | B | `Event_Broadcast_B_Block_East_Clr` `0x340C4` | `0xC000` | PL/shim (switch B) |

   (The 6 switch-A writes are identical to `sp5_skew_r3b_pc_dh`; the 6
   switch-B writes are the **new delta in this variant** -- see "Switch-B
   delta" below.)

   **Why the shim-row unblock at all (Phoenix silicon finding, 2026-07-02):**
   the first HW run of `sp5_skew_r3b_pc_dh` completed clean but each flood was
   **confined to its source column** -- ch15 (fired at shim (0,0)) reached
   only col 0; ch14 (fired at core (2,5)) reached only col 2 -- so the
   horizontal-spine tiles never saw both floods and `d_h` was unmeasurable.
   Root cause (derived from aie-rt): the Step-2 block-SET masks below worked
   correctly, but the **shim row's** East/West broadcast-block bits for
   ch14/ch15 were already SET on this device at run start -- NOT at the
   POR-open default the AM025 register DB implies. With them set, neither
   flood could cross row 0 horizontally. Vanilla `XAie_SyncTimer` assumes the
   shim is default-open and never clears these; nothing in the toolchain load
   path clears them either. The shim row is precisely the horizontal
   distribution spine the routing derivation (plan Sec.0:
   "shim-row-only-then-climb") depends on, so we defensively CLEAR the block
   there at the very start.

   ### Switch-B delta (the difference vs. `sp5_skew_r3b_pc_dh`)

   `sp5_skew_r3b_pc_dh` cleared only **switch A**'s E/W block and still
   confined the floods on silicon. Further aie-rt derivation: the shim PL tile
   has **TWO** broadcast switches (A and B); a single local broadcast source
   feeds BOTH switches' propagation trees, each gated by its own independent
   block bits. AMD's own Phoenix bring-up code
   (`_XAie_ErrorHandlingInitAie`) drives switch A and switch B **in lockstep**
   when it needs a broadcast to cross columns. So clearing switch A alone was
   insufficient: switch B's E/W block -- also still SET on this device -- kept
   confining the flood to its source column. **This variant additionally
   clears switch B's E/W block** (both switches unblocked E/W) to test whether
   the flood now crosses columns on silicon. The switch-B block bank sits at
   switch A's `+0x40` (`A West_Clr 0x34064 -> B West_Clr 0x340A4`; `A East_Clr
   0x34084 -> B East_Clr 0x340C4`).

   These are all sticky SET/CLR register pairs: writing `0xC000` (bits 14\|15)
   to a `WEST_CLR`/`EAST_CLR` register unblocks channels 14,15 in the
   West/East directions on that shim tile. North/South are left alone (the
   vertical climb already works -- only the horizontal shim-row hop was
   blocked). Only row 0 gets these; the Step-2 row 1-5 block-SET writes below
   are intentional and untouched. All four register names/offsets confirmed
   against `aie_registers_aie2.json` `modules.shim.registers`
   (`Event_Broadcast_A_Block_West_Clr` = `0x34064`,
   `Event_Broadcast_A_Block_East_Clr` = `0x34084`,
   `Event_Broadcast_B_Block_West_Clr` = `0x340A4`,
   `Event_Broadcast_B_Block_East_Clr` = `0x340C4`).

   **Emulator note:** the emulator models the shim row as always-open, so
   these CLR writes are a **no-op on the emulator** -- they exist purely for
   silicon. The emu smoke-run confirms the kernel still runs clean; it does
   not (and cannot) exhibit the routing change.

1. **30 new `aiex.npu.write32` block-mask writes**, emitted as Step 2 of
   `@seq` (right after the CLR writes above), before the (unchanged)
   perf-counter config step. On every tile in cols 0-2, rows 1-5 (15 tiles):

   | Tile type | Register (local offset) | Value | Module |
   |---|---|---|---|
   | AIE compute (rows 2-5) | `Event_Broadcast_Block_East_Set` `0x14080` | `0xC000` | memory (mem-module of the compute tile) |
   | AIE compute (rows 2-5) | `Event_Broadcast_Block_West_Set` `0x34060` | `0xC000` | core |
   | Mem-tile (row 1) | `Event_Broadcast_A_Block_East_Set` `0x94080` | `0xC000` | memory_tile (switch A) |
   | Mem-tile (row 1) | `Event_Broadcast_A_Block_West_Set` `0x94060` | `0xC000` | memory_tile (switch A) |
   | Shim (row 0) | -- none -- | -- | left at reset default (unblocked) |

   `0xC000` = bits 14\|15, blocking **both** flood channels (ch14 = s2, ch15 =
   s1) -- the "two-channel trap": our two independent single-channel sources
   both need the full shaped block, unlike vanilla `XAie_SyncTimer`
   (`xaie_timer.c:823-904`), which blocks only its one reset `BcastId` and
   leaves `BcastId+1` open as a shim-row relay trigger. Copying the
   single-channel form would let the unblocked flood free-route and corrupt
   `d_h`.

   Blocking is uniform across all 15 tiles, including the source tile (2,5)
   itself -- no origin special-case, matching `_XAie_SetupBroadcastConfig`
   (blocking gates egress, not local injection).

   **Core/mem asymmetry (AM020 ch2:349-363, load-bearing, do not "fix"):** the
   core module has only N/S/W *external* ports (its "east" is an internal,
   unblockable wire to the co-located mem module's "west"); the mem module has
   only N/S/E external ports. Blocking mem-East + core-West severs the two
   real external horizontal ports while leaving the internal core<->mem wire
   open (which is what lets one arrival arm both modules). Blocking mem-West
   or core-East would be a no-op on the internal wire -- deliberately not
   done.

2. **`geometry.json`** gains a top-level `"routing": "block_shim_row"` field
   (vs. absent/`"free_flood"` for the parent kernel) and an updated `comment`
   documenting the block-mask delta and the `d_h`-only scope. The `sources`
   and `tiles` arrays (six measured cores, same `counter_index`/`pkt_in`/
   `pkt_out`) are **unchanged** from the parent -- the horizontal spine
   `(0,3),(1,3),(2,3)` carries all the `d_h` leverage (`dn_h = [2,0,-2]`); the
   vertical-spine tiles `(1,2),(1,4),(1,5)` are `dn_h=0` under block routing
   (redundant with `(1,3)`) and are kept only as ballast / to preserve the
   free-flood kernel's counter layout -- they contribute no `d_h` information
   here.

3. Runtime-sequence step numbering: the shim-row CLR writes are Step 1, the
   block-SET masks Step 2; perf-counter config, channel mapping, readback arm,
   floods, and readback pushes are Steps 3-7 respectively -- pure renumbering
   vs. the parent, no reordering of the shared logic.

**Everything else is byte-identical in structure to the parent**: the same
two flood sources/channels (`s1` = shim (0,0) ch15, `s2` = core (2,5) ch14),
the same six measured tiles and packet-flow IDs, the same perf-counter arm
(`START=BROADCAST_15=122`, `STOP=BROADCAST_14=121`, `Performance_Control0 =
0x797A` @ `0x31500`, `Performance_Counter0` @ `0x31520`), and the same
control-packet OP_READ readback mechanism (see the parent's `README.md`
"Readback mechanism" section for the full writeup -- unchanged here).

## Addressing mechanism (how a local offset becomes a full per-tile address)

`aiex.npu.write32` carries a LOCAL (tile-relative) `address` plus explicit
`column`/`row` integer attributes -- exactly the pattern the parent kernel's
counter-config and channel-mapping writes already use (e.g. `address =
0x31520, column = 0, row = 3`). The lowering
(`AIEXDialect.cpp::getAbsoluteAddress`, `lib/Dialect/AIEX/IR/AIEXDialect.cpp:700-747`)
computes the physical address the NPU instruction stream carries as:

```
absolute_address = (column & 0xff) << columnShift
                  | (row    & 0xff) << rowShift
                  | (local_address & 0xfffff)
```

with `columnShift = 25`, `rowShift = 20` for AIE2/Phoenix
(`AIE2TargetModel`, `include/aie/Dialect/AIE/IR/AIETargetModel.h:619-620`; the
`BaseNPU1TargetModel` used by `npu1_3col` inherits this unmodified). This is
the *same* mechanism for every `aiex.npu.write32` in this kernel -- the
block-mask writes use no different addressing path than the pre-existing
counter/event writes. Verified by decoding this kernel's compiled `insts.bin`
(see "Build + verify" below): all 30 block-mask writes appear with the
expected `(column, row, local_offset, value)` tuples, decoded back out of
their absolute addresses via the same shift formula.

No `aie.tile` declaration is structurally required for a plain
column/row-addressed `write32` (the lowering only needs to find the parent
`aie.device`, not a matching `TileOp` -- confirmed by reading the lowering
code directly). This kernel nonetheless declares an `aie.tile` for every
block-mask target tile that isn't already declared for another reason (packet
flows, flood source), purely for self-documentation -- matching the
convention every other `write32`-based kernel in this tree follows
(`debug_halt_probe`, `sp5_skew_r3b_pc`).

## Register offset provenance

All four block-mask register names/offsets were cross-checked directly
against the AM025 register database
(`mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`,
`modules.{core,memory,memory_tile}.registers`) and match exactly:

| Offset | Module | Register name |
|---|---|---|
| `0x34060` | `core` | `Event_Broadcast_Block_West_Set` |
| `0x14080` | `memory` | `Event_Broadcast_Block_East_Set` |
| `0x94060` | `memory_tile` | `Event_Broadcast_A_Block_West_Set` |
| `0x94080` | `memory_tile` | `Event_Broadcast_A_Block_East_Set` |

(Full derivation record, including the aie-rt `_XAie_SetupBroadcastConfig`
cross-reference and the two-channel-trap reasoning: plan Sec.0.)

## Build recipe

```bash
cd ~/npu-work
source toolchain-build/activate-npu-env.sh
cd mlir-aie/test/npu-xrt/sp5_skew_r3b_pc_dh_swb
env -u XDNA_EMU aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts \
  --no-compile-host --alloc-scheme=basic-sequential \
  --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie.mlir
```

Produces `aie.xclbin` + `insts.bin`.

## Emulator smoke-run (no HW)

```bash
cd ~/npu-work/xdna-emu
cargo build -p xdna-emu-ffi   # rebuild the FFI .so (NOT bare `cargo build`)
# Generate the --ctrlpkt binary from geometry.json:
python3 -c "import json,sys; sys.path.insert(0,'tools'); \
  from calibration.skew.r3b_ctrlpkt import build_ctrlpkt; \
  open('ctrlpkt_swb.bin','wb').write(build_ctrlpkt(json.load(open('../mlir-aie/test/npu-xrt/sp5_skew_r3b_pc_dh_swb/geometry.json'))))"
BU=../mlir-aie/test/npu-xrt/sp5_skew_r3b_pc_dh_swb
XDNA_EMU=1 XDNA_EMU_RUNTIME=debug \
  bridge-runner/build/bridge-trace-runner \
  --xclbin "$BU/aie.xclbin" --instr "$BU/insts.bin" \
  --ctrlpkt ctrlpkt_swb.bin --trace-out readback_swb.bin --trace-size 256
# First 24 bytes of readback_swb.bin = 6 x u32 counters (counter_index order).
```

### Evidence (2026-07-02, emulator, no HW; switch-A + switch-B shim CLR)

Run completed cleanly: `halt_reason=completed cycles=31574`, no wedge, no
panic, no WARN/ERROR lines. Both `Event_Generate` writes (`0x34008 <- 126`)
fired (shim `Tile(1,0)` at cycle 6990, core source `Tile(3,5)` at cycle 7094 --
emulator-internal tile addressing runs the kernel shifted one column, per
`feedback_port_cadence_oracle_only_emu_decoded`). The readback BO's first 24
bytes were populated with six little-endian `u32` words, all reading **0** --
expected and correct, since (a) the emulator does not drive the perf counter
from broadcast arrivals and (b) the emulator models the shim row as
always-open, so the shim-row CLR writes (both switches) are a no-op there.
This run validates plumbing only (that the kernel still runs clean after the
switch-B delta); the routing change and the counter values are the silicon
proof, out of scope here.

Additionally, `insts.bin` was decoded directly (parsing the `XAIE_IO_WRITE`
instruction words per `AIETargetNPU.cpp::appendWrite32`) to confirm the
compiled instruction stream contains the writes in the intended order:
- the **12 shim-row CLR writes are the first 12** `write32`s -- 6 switch-A
  `(col, row=0, local=0x34064/0x34084, value=0xC000)` for cols 0,1,2, then 6
  switch-B `(col, row=0, local=0x340A4/0x340C4, value=0xC000)` for cols 0,1,2,
  each in `West_Clr` then `East_Clr` order per column;
- immediately followed by the **30 block-SET writes** (rows 1-5, unchanged,
  starting `(col 0, row 1, 0x94080)`);
- then the perf-counter config, the channel-mapping writes, and both
  `Event_Generate` writes (`0x34008`).

This confirms the load-bearing ordering (shim-row unblock A+B -> block-SET
masks -> counter config -> floods -> readback) holds in the actual compiled
binary, not just the MLIR source.

## Not done here (Phase 2 follow-up, per the plan)

- `observe_r3b` routing-aware branch (`"block_shim_row"` -> `reset_routed_coeffs`)
  and `extract_r3b_dh` (`tools/calibration/skew/r3b_observe.py`,
  `r3b_extract.py`) -- plan Sec.1b/1c, a separate task.
- `r3b_pc_dh_gate.sh` + tally -- plan Sec.1d.
- The Phoenix HW capture protocol (plan Sec.2) -- gated on the above landing
  green and a whole-delta review.
