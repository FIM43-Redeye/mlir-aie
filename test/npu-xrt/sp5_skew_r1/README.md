# SP-5b R1 within-column skew-measurement kernel (#140)

REQUIRES: ryzen_ai_npu1

3-collinear-core extension of the proven lean Q=0 objectfifo spine
(`spike_bringup/of_q0_lean.py`), traced for the R1 differencing extractor
(`xdna-emu/tools/calibration/skew/r1_diff_extract.py` +
`xdna-emu/tools/calibration/skew/r1_observe.py`). Gives R1 >=3 within-column
vertical points (rows 2/3/4, single column) to solve `d_v`, plus one
same-tile core<->mem intra-contrast anchor (row 2) per
`docs/superpowers/specs/2026-06-30-sp5b-bring-up-spec.md` Sec.4/4.2 (in the
xdna-emu repo).

## Dataflow

```
ProdCore(0,2) --src--> MemTile(0,1) --d--> ConsA(0,3) --e--> MemTile(0,1)
  --f--> ConsB(0,4) --j--> MemTile(0,1) --out--> ShimTile(0,0) --> DDR
```

Pure lock/DMA handshakes in every core (no buffer compute) -- Q=0, on-chip
producer, only the output drain touches DDR. REPS=32, OBJ=64 i32 objects.
Same design rationale as `of_q0_lean.py`: no compute avoids the
MEMORY_STALL cycle-cost gap, so within-domain LOCK_STALL timing is the
clean cross-domain/within-domain anchor.

## Files

- `sp5_skew_r1.py` -- the kernel (IRON/aie python, `IRONbuild()`-style like
  `of_q0_lean.py`; run it to emit `aie.mlir` on stdout).
- `trace_config.json` -- the trace-injection config actually produced by
  the build below (checked in as documentation of intended tracing; the
  build regenerates its own copy under
  `mlir-aie/build/test/npu-xrt/sp5_skew_r1/trace_config.json`, which is the
  one Task 5/6 tooling should read).
- `geometry.json` -- pairs + `dn_v`/`kind` for `r1_observe.observe_r1`.

## Deviations from the Task-4 brief (read this before reusing the recipe)

1. **`tools/trace-prepare.py` invocation.** The brief's reconstructed
   `--kernel ... --trace-config ... --out ...` CLI does not exist on
   `trace-prepare.py` (verified against its actual `argparse` block: it
   takes a positional `test_dir` containing `aie.mlir` or `aie2.py`, plus
   `--output/-o`, `--trace-size`, `--device`, `--memtile-sweep-events`,
   `--shim-sweep-events`, `--memmod-sweep-events`, `--core-sweep-events`,
   `--trace-mode`). There is also no `spike_bringup/README.md` on disk (the
   brief's `spike_bringup/README.md:52-82` citation does not resolve to a
   real file) -- the real proven recipe lives in
   `xdna-emu/build/experiments/sp3-spike-trace/README.md` (the `of_q0_rich`
   trace pipeline) and `xdna-emu/docs/superpowers/findings/
   2026-06-29-coldstart-headstart-trace-baseline.md` (cites
   `build_q0_lean_trace/` directly). The recipe below is reconstructed from
   those plus `trace-prepare.py`/`mlir-trace-inject.py` source, not
   transcribed from a nonexistent doc.

2. **Intra-contrast anchor event: `DMA_MM2S_0_FINISHED_BD`, not
   `PORT_RUNNING_0`.** The brief's geometry pair 4 and acceptance criterion
   ask for `PORT_RUNNING_0` on the row-2 **mem** module (pkt_type 1,
   compute tile's own memory-module trace unit -- distinct from the
   standalone memtile at row 1). This event does not exist for that module
   in any of three independent, mutually-confirming sources:
   - `aie-rt/driver/src/events/xaie_events_aieml.h` (AIE2/aieml event
     tables): `PORT_RUNNING_0` is defined for `CORE`, `PL` (shim), and
     `MEM_TILE` -- there is no `MEM`-module (compute tile's own memory
     module) variant.
   - `xdna-emu/tools/mlir-trace-inject.py`'s own `_TRACE_DEFAULT_MEMMOD_EVENTS`
     and its comment ("All eight are GenericEvents -- no port-slot bindings
     to thread through, unlike memtile") -- there is a
     `_core_default_port_config` and `_memtile_default_port_config` helper,
     but no memmod equivalent.
   - `mlir-aie/python/utils/trace/setup.py::_get_default_events_for_tile`'s
     `is_mem_trace` branch: returns `MemEvent.DMA_S2MM_0_START_TASK`,
     `DMA_MM2S_0_START_TASK`, `CONFLICT_DM_BANK_{0..3}`,
     `EDGE_DETECTION_EVENT_{0,1}` -- `MemEvent` has no `PORT_RUNNING_*`
     member at all.

   I additionally confirmed empirically that the `*_START_TASK` variants
   (the memmod defaults) never fire for this kernel's objectfifo-driven
   DMA (0 occurrences across 3 emu runs) -- objectfifo lowers to BD-chained
   transfers, not task-queue-triggered ones, matching the note already
   banked in `build/experiments/sp3-spike-trace/README.md`: "objectfifo
   fires `*_FINISHED_BD`, NOT `*_TASK`". `spike_bringup/build_q0_trace_bd/`
   is prior (untracked, unlanded) evidence of the same finding. I therefore
   set `--memmod-sweep-events` to the `*_FINISHED_BD` set and picked
   `DMA_MM2S_0_FINISHED_BD` for row 2 (ProdCore only ever *produces*, so
   only its MM2S channel finishes a BD -- confirmed empirically: row 2's
   mem module fires exactly that one event and nothing else, while rows
   3/4, which both consume and relay, fire both `DMA_S2MM_0_FINISHED_BD`
   and `DMA_MM2S_0_FINISHED_BD`). `geometry.json`'s pair 4 `b.name` uses
   `DMA_MM2S_0_FINISHED_BD` accordingly -- this is the only change from the
   brief's literal geometry.json; `pkt_type`, `dn_v`, `kind` all match. A
   geometry.json shipping the literal `PORT_RUNNING_0` name would raise
   `KeyError` at first use in `r1_observe.observe_r1` (verified: I ran the
   brief's literal choice against a real decoded trace and it does not
   appear; only `DMA_MM2S_0_FINISHED_BD` does).

3. **Dual-compiler Chess vs Peano sanity check.** Because this kernel's
   cores contain zero vector/scalar compute (pure objectfifo
   acquire/release), a first pass comparing `main_core_0_2.o` between the
   Chess and Peano builds came back byte-identical, which looked like
   `--xchesscc` was silently no-oping. `strace -f -e trace=execve` on the
   `aiecc` invocation proved otherwise: `xchesscc`, `chess-clang`, and
   `chess-backend` are genuinely invoked (confirmed against the known-good
   `add_one_cpp_aiecc_xchesscc` lit-test's own Chess build as a control),
   and the final `.elf` *does* differ between the two builds -- the `.o`
   match is coincidental to an intermediate stage shared by both pipelines
   for this trivial a kernel, not a Chess no-op. Both compiles are real.

## Build recipe (as actually run)

```bash
cd ~/npu-work
source toolchain-build/activate-npu-env.sh
cd xdna-emu

# 1. Generate MLIR from the IRON kernel (mirrors of_q0_lean.py's own
#    generation step -- trace-prepare.py only accepts a directory
#    containing aie.mlir or aie2.py, not an arbitrary-named .py directly).
mkdir -p ../mlir-aie/test/npu-xrt/sp5_skew_r1/gen_src
python3 ../mlir-aie/test/npu-xrt/sp5_skew_r1/sp5_skew_r1.py \
  > ../mlir-aie/test/npu-xrt/sp5_skew_r1/gen_src/aie.mlir

# 2. Inject trace: core defaults carry LOCK_STALL + the lock/stall event
#    set; memmod gets the *_FINISHED_BD set (see deviation 2 above);
#    memtile/shim get their PORT_RUNNING / DMA-task defaults ("all").
SRC=../mlir-aie/test/npu-xrt/sp5_skew_r1/gen_src
BU=../mlir-aie/build/test/npu-xrt/sp5_skew_r1
python3 tools/trace-prepare.py "$SRC" -o "$BU" \
  --trace-size 16384 --device npu1_2col \
  --memtile-sweep-events all --shim-sweep-events all \
  --memmod-sweep-events DMA_S2MM_0_FINISHED_BD,DMA_MM2S_0_FINISHED_BD,DMA_S2MM_1_FINISHED_BD,DMA_MM2S_1_FINISHED_BD \
  --trace-mode event_time

# 3. Compile BOTH compilers from the same traced MLIR (Chess = ground
#    truth, Peano = informational, per CLAUDE.md).
mkdir -p "$BU/chess" "$BU/peano"
cp "$BU/aie_traced.mlir" "$BU/chess/aie_traced.mlir"
cp "$BU/aie_traced.mlir" "$BU/peano/aie_traced.mlir"

( cd "$BU/peano" && env -u XDNA_EMU aiecc.py --no-aiesim \
    --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
    --alloc-scheme=basic-sequential --xclbin-name=aie.xclbin \
    --npu-insts-name=insts.bin ./aie_traced.mlir )

( cd "$BU/chess" && env -u XDNA_EMU aiecc --xchesscc --xbridge \
    --aie-generate-xclbin --xclbin-name=aie.xclbin \
    --aie-generate-npu-insts --npu-insts-name=insts.bin \
    ./aie_traced.mlir )
```

## Run + decode (emu, no HW needed)

```bash
cd ~/npu-work/xdna-emu
cargo build -p xdna-emu-ffi   # rebuild the FFI .so from source (not bare `cargo build`)

BU=../mlir-aie/build/test/npu-xrt/sp5_skew_r1
mkdir -p build/experiments/sp5-skew-r1/run1 && cd build/experiments/sp5-skew-r1/run1

XDNA_EMU=1 XDNA_EMU_RUNTIME=debug \
  ~/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner \
  --xclbin "$BU/chess/aie.xclbin" --instr "$BU/chess/insts.bin" \
  --output ./out.bin --trace-out ./trace.bin --trace-size 16384

python3 ~/npu-work/xdna-emu/tools/parse-trace.py --trace-bin ./trace.bin \
  --xclbin-mlir "$BU/chess/aie_traced.mlir.prj/input_with_addresses.mlir" \
  --trace-mode event_time --out-events ./events.json \
  --out-perfetto ./trace.perfetto.json
```

## Evidence (2026-07-01, emu-only, Chess-compiled xclbin)

3 runs, identical cycle count (119921), identical `halt_reason=completed`,
identical per-tile event-name sets, identical total event count (2326)
each run -- reproducible, no panic, no TDR-equivalent (emu has no TDR
concept; "no TDR" here means clean halt every run).

Per-tile event set (col reported as **1**, not the declared 0 -- the
bridge-trace-runner/FFI path places the kernel at physical col 1 on this
2-column device; this is expected/known behavior of that harness, not a
bug -- see `events.json`'s `"placement": {"origin_col": 1, ...}` field,
which downstream tooling uses to normalize back to `geometry.json`'s
declared col-0 frame):

| tile (col,row,pkt_type) | events |
|---|---|
| (1,2,0) core | `INSTR_LOCK_ACQUIRE_REQ`, `LOCK_STALL`, `MEMORY_STALL`, `PERF_CNT_2` |
| (1,3,0) core | same set |
| (1,4,0) core | same set |
| (1,2,1) mem  | `DMA_MM2S_0_FINISHED_BD` (ProdCore only produces) |
| (1,3,1) mem  | `DMA_MM2S_0_FINISHED_BD`, `DMA_S2MM_0_FINISHED_BD` (ConsA consumes+relays) |
| (1,4,1) mem  | `DMA_MM2S_0_FINISHED_BD`, `DMA_S2MM_0_FINISHED_BD` (ConsB consumes+relays) |
| (1,1,3) memtile | `PORT_RUNNING_{0,1,2,4,5,6}` |
| (1,0,2) shim | `DMA_S2MM_0_{START,FINISHED}_TASK`, `DMA_S2MM_0_STREAM_STARVATION` |

`LOCK_STALL` fires on all three compute cores (satisfies the acceptance
intent); the row-2 mem module fires a real, reproducible, exclusive DMA
event (`DMA_MM2S_0_FINISHED_BD` in place of the nonexistent
`PORT_RUNNING_0` -- see Deviation 2).

I also smoke-tested `geometry.json` end-to-end against
`tools/calibration/skew/r1_observe.py::observe_r1` using this real decoded
trace (col-shifted by the observed `origin_col=1` first, matching the
schema's documented normalization contract) as both the "measured" and
"dwall" side: all 4 pairs resolve with `skew=0.0` (self-consistency, not a
real skew measurement -- proves every named anchor in `geometry.json`
actually exists in a real trace, no `KeyError`).

## Reproduce

```bash
# repeat the "Run + decode" block 3x into run1/run2/run3, then:
python3 - <<'EOF'
import json
from collections import defaultdict
def tileset(p):
    d = json.load(open(p))
    by = defaultdict(set)
    for e in d["events"]:
        by[(e["col"], e["row"], e["pkt_type"])].add(e["name"])
    return {k: frozenset(v) for k, v in by.items()}
sets = [tileset(f"run{i}/events.json") for i in (1, 2, 3)]
assert sets[0] == sets[1] == sets[2], "tile/event set diverged across runs"
print("reproducible:", sets[0].keys())
EOF
```

Real NPU1 hardware no-TDR proof is Task 6 (not attempted here -- CPU/emu
only per the Task-4 acceptance gate).
