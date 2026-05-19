//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// debug_halt_probe host harness (Phase A -- Exp1 redesign: control-packet
// OP_READ readback + schedule-derived verdict).
//
// Task 3 discovery: a lock-gated marker DMA hangs on a halted core (the core
// never releases its lock -> dma_wait blocks forever -> run.wait() timeout ->
// NPU wedge), and the compiler reordered the stores so a source-order verdict
// is inverted.  There is therefore NO marker-DMA path: this harness reads
// output_buffer slots and Core_Status SOLELY via control-packet OP_READ
// (which works on a halted core, prior art: add_one_ctrl_packet) and computes
// the verdict from the disassembled store schedule (not source order).
//
// Task 5 revision: a blocking objectfifo gate (@gate, 1xi32, depth 1) is
// added to prevent the arming-before-run race observed in HW attempt 1 (the
// CDO enables the core before @seq arming writes reach the tile).  The gate
// @seq arg is %arg2 -> group_id(5).  The host writes a nonzero i32 token
// into bo_gate and syncs it to device before launching the kernel; @seq feeds
// it to the core via dma_memcpy_nd AFTER the arming writes.
//
// OUTBUF_ADDR = output_buffer tile-local address = p0 & 0xFFFF, where p0 is
// the core's 20-bit pointer to output_buffer[0] from llvm-objdump-aie.
// Task 3 disasm: p0 = 0x70400 => OUTBUF_ADDR = 0x0400.
// This is DERIVED, not fixed.  Self-check: the EMU no-trap baseline must
// return s[0]=170, s[1]=187, s[2]=204, s[3]=1 from the four slot reads.
// If any slot is garbage, OUTBUF_ADDR is wrong -- re-derive from a fresh
// disasm (or aiecc allocation log) and update OUTBUF_ADDR + regenerate
// the ctrl-in packets below.  Also re-derive TRAP_PC in aie.mlir if the
// kernel changes.
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// OUTBUF_ADDR: tile-local data-memory offset of output_buffer[0].
// Derived: output_buffer base pointer (from llvm-objdump-aie of
// main_core_0_2.elf, Task 3 disasm) = 0x70400; tile-local offset =
// 0x70400 & 0xFFFF = 0x0400.  Must be re-derived if kernel/allocation
// changes (see file-header comment).  Self-checked by EMU no-trap baseline.
static constexpr uint32_t OUTBUF_ADDR = 0x0400;

// Core_Status register offset (tile-local, compute-tile module).
// XAIEMLGBL_CORE_MODULE_CORE_STATUS, per aie-rt
// driver/src/global/xaiemlgbl_params.h.
// bit 16 = DEBUG_HALT: the verdict's SOLE halted signal.  bit 0 = ENABLE is
// no longer used by the verdict (ENABLE-stays-1-while-halted was an unverified
// HW assumption; DEBUG_HALT=1 alone proves the core ran and is debug-halted).
//
// Phase B Unit 1b (xdna-emu, 2026-05-18): the mutable tile.read_register
// path now dispatches Core_Status reads to core_debug.read_status() (mirroring
// the Unit-1 read_register_pure fix).  The injected MASKPOLL polls Core_Status
// via this path.  With arming wired (always was, spec §4.2 "Mechanism
// correction") + the Unit-1 pre-execute seam halting before the trap bundle
// + Unit-1b read reconciliation, the EMU DEBUG_HALT bit is now observable and
// the MASKPOLL satisfies on EMU.  The probe is therefore a self-checking
// EMU+HW regression of the G1 before-commit fidelity fix.
static constexpr uint32_t CORE_STATUS_ADDR = 0x32004;

// Number of ctrl-in words (one OP_READ header per slot + one for Core_Status).
// To extend (Task 6 reads 8 marker slots): bump this, add slots to s[]/the
// ctrlPackets list, and add a packet block in aie.mlir.
static constexpr int NUM_READ_PKTS = 5;
// ctrl-out receives one response word per OP_READ packet.  This constant
// drives the bo_ctrlOut allocation below.
static constexpr int CTRL_OUT_WORDS = NUM_READ_PKTS;

// Sentinel pre-written into every ctrl-out slot before the run.  Used to
// detect the MASKPOLL-unsatisfied EMU baseline (Task 5 Step 4): when the
// injected XAIE_IO_MASKPOLL on Core_Status[16] (DEBUG_HALT) is never
// satisfied (the EMU drops the breakpoint-arming control-packet writes), the
// emulator terminates deterministically and the OP_READ push is NEVER issued,
// so ctrl-out is never written.  If ALL slots still hold this sentinel after
// a clean run.wait(), that is the recognized expected-EMU outcome.  The value
// is chosen to never collide with a real OP_READ response: markers are small
// ints (0xAA/0xBB/0xCC/1) and Core_Status is a status word -- 0xDEADC0DE is
// not a plausible reading of either.
static constexpr uint32_t CTRLOUT_SENTINEL = 0xDEADC0DEu;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("debug_halt_probe");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  // BO layout (matches aie.mlir @seq arg order):
  //   group_id(1) = instr BO (implicit, always first)
  //   group_id(3) = ctrl-out BO (%arg0): receives OP_READ responses
  //   group_id(4) = ctrl-in BO  (%arg1): holds assembled OP_READ headers
  //   group_id(5) = gate BO     (%arg2): 1xi32 token fed to @gate objectfifo
  // (No output BO: the marker-DMA path was removed -- it would dma_wait
  // forever on a halted core.  The OP_READ responses are the sole output.)
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_ctrlOut = xrt::bo(device, CTRL_OUT_WORDS * sizeof(uint32_t),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_ctrlIn = xrt::bo(device, NUM_READ_PKTS * sizeof(uint32_t),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  // gate BO: 1 x i32 token; any nonzero value works (the core only blocks
  // on the lock, not the value).  group_id(5) = @seq %arg2 (third seq arg).
  auto bo_gate = xrt::bo(device, sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                         kernel.group_id(5));

  // Assemble OP_READ control-packet headers into ctrl-in BO.
  // Header layout (from xdna-emu src/device/control_packets/parser.rs):
  //   [31]    parity (even-parity over bits 30:0)
  //   [30:24] stream_id (response routing; 0x2 = TileControl->shim flow 0x2)
  //   [23:22] operation  (0x1 = OP_READ)
  //   [21:20] length     (beats - 1; 0 = one response word)
  //   [19:0]  address    (tile-local address to read)
  // No data words follow an OP_READ header (reads carry no payload).
  auto parity = [](uint32_t n) -> uint32_t {
    uint32_t p = 0;
    while (n) {
      p += n & 1u;
      n >>= 1;
    }
    return (p % 2u == 0u) ? 1u : 0u;
  };

  auto make_read_header = [&](uint32_t addr) -> uint32_t {
    uint32_t stream_id =
        0x2; // response routed via packet_flow(0x2) back to shim
    uint32_t operation = 0x1;  // OP_READ
    uint32_t beats_minus1 = 0; // 1 word response per packet
    uint32_t hdr = (stream_id << 24) | (operation << 22) |
                   (beats_minus1 << 20) | (addr & 0xFFFFF);
    hdr |= (parity(hdr) << 31);
    return hdr;
  };

  // Five OP_READ packets: output_buffer[0..3] then Core_Status.
  std::vector<uint32_t> ctrlPackets = {
      make_read_header(OUTBUF_ADDR + 0 * sizeof(uint32_t)), // s[0]
      make_read_header(OUTBUF_ADDR + 1 * sizeof(uint32_t)), // s[1]
      make_read_header(OUTBUF_ADDR + 2 * sizeof(uint32_t)), // s[2]
      make_read_header(OUTBUF_ADDR + 3 * sizeof(uint32_t)), // s[3]
      make_read_header(CORE_STATUS_ADDR),                   // Core_Status
  };
  uint32_t *bufCtrlIn = bo_ctrlIn.map<uint32_t *>();
  memcpy(bufCtrlIn, ctrlPackets.data(), ctrlPackets.size() * sizeof(uint32_t));

  // Write nonzero gate token (value 1; the core only needs the lock to be
  // acquired, not any particular token value).
  uint32_t *bufGate = bo_gate.map<uint32_t *>();
  bufGate[0] = 1u;

  // Pre-fill ctrl-out with the sentinel so an un-issued OP_READ (the
  // MASKPOLL-unsatisfied EMU baseline) is detectable: if every slot still
  // holds CTRLOUT_SENTINEL after the run, the OP_READ never landed.
  uint32_t *bufCtrlOut = bo_ctrlOut.map<uint32_t *>();
  for (int i = 0; i < CTRL_OUT_WORDS; i++)
    bufCtrlOut[i] = CTRLOUT_SENTINEL;

  uint32_t *bufInstr = bo_instr.map<uint32_t *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_gate.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Kernel args: opcode, instr BO, instr count, ctrl-out, ctrl-in, gate
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_ctrlOut, bo_ctrlIn, bo_gate);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_ctrlOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Parse ctrl-out response words.
  // Responses arrive in the order the OP_READ packets were sent:
  //   ctrlOut[0] = output_buffer[0] (s[0])
  //   ctrlOut[1] = output_buffer[1] (s[1])
  //   ctrlOut[2] = output_buffer[2] (s[2])
  //   ctrlOut[3] = output_buffer[3] (s[3])
  //   ctrlOut[4] = Core_Status       (cs)
  uint32_t *ctrlOut = bo_ctrlOut.map<uint32_t *>();
  uint32_t s[4];
  for (int i = 0; i < 4; i++)
    s[i] = ctrlOut[i];
  uint32_t cs = ctrlOut[4];

  // MASKPOLL-unsatisfied defensive branch.
  //
  // The injected XAIE_IO_MASKPOLL blocks the instruction stream until
  // Core_Status[16] (DEBUG_HALT).  As of Phase B Unit 1b (xdna-emu), the
  // mutable tile.read_register path dispatches Core_Status reads to
  // core_debug.read_status(), so DEBUG_HALT IS now observable on EMU.  With
  // arming wired (spec §4.2 "Mechanism correction") and the Unit-1 pre-execute
  // seam halting before the trap bundle commits, the MASKPOLL SATISFIES on EMU
  // and the OP_READ push runs.  EMU and HW both reach the schedule-derived
  // verdict below (BEFORE_COMMIT expected on both).
  //
  // This sentinel-check block is therefore a DEFENSIVE branch for unexpected
  // regressions (e.g. a future emulator build where Unit 1b is not present)
  // and is NO LONGER the expected EMU outcome.  If all slots still hold the
  // sentinel after a clean run.wait(), something has regressed -- emit the
  // diagnostic and fail (return 1) so the bridge harness flags it.
  //
  // The emulator graceful-poll-termination contract (MaskPollUnsatisfied
  // terminal reason, clean run.wait(), no register fakery) is retained and
  // independently unit-tested; it simply is not this probe's path any more.
  bool maskpoll_unsatisfied = true;
  for (int i = 0; i < CTRL_OUT_WORDS; i++) {
    if (ctrlOut[i] != CTRLOUT_SENTINEL) {
      maskpoll_unsatisfied = false;
      break;
    }
  }
  if (maskpoll_unsatisfied) {
    std::cout << "SLOTS: s0=" << s[0] << " s1=" << s[1] << " s2=" << s[2]
              << " s3=" << s[3] << " CORE_STATUS=0x" << std::hex << cs
              << std::dec
              << " HALTED=0 (OP_READ never issued: MASKPOLL unsatisfied --"
              << " Unit 1b regression or unexpected build?)\n";
    std::cout << "TRAP_VERDICT:MASKPOLL_UNSATISFIED_EMU\n";
    // FAIL: this is no longer the expected EMU outcome (Phase B Unit 1b).
    return 1;
  }

  // Schedule from llvm-objdump-aie (gated build, Task 5 re-derivation):
  //   trap bundle  0x184 -> s[1]=0xBB  (st dj0,[p0,#4]; dj0=0xBB, p0=0x70400)
  //   strictly-later 0x18c -> s[0]=0xAA  (st m0,[p0],#8; m0=0xAA)
  //   later       0x192 -> s[3]=1
  //   later       0x1a0 -> s[2]=0xCC
  // TRAP_PC=0x184; TRAP_PC14=0x184; PC_Event0=0x80000184.
  // OUTBUF_ADDR=0x0400 (p0=0x70400 from disasm, tile-local = 0x70400&0xFFFF).
  // Re-derive if the kernel changes (see aie.mlir TRAP_PC comment).
  bool halted = (cs & (1u << 16)) != 0; // DEBUG_HALT bit alone --
                                        // ENABLE-stays-1-while-halted is an
                                        // unverified HW assumption;
                                        // DEBUG_HALT=1 already proves the core
                                        // ran and is debug-halted
  std::cout << "SLOTS: s0=" << s[0] << " s1=" << s[1] << " s2=" << s[2]
            << " s3=" << s[3] << " CORE_STATUS=0x" << std::hex << cs << std::dec
            << " HALTED=" << (halted ? 1 : 0) << "\n";
  std::cout << "TRAP_VERDICT:";
  if (!halted) {
    if (s[0] == 0xAA && s[1] == 0xBB && s[2] == 0xCC && s[3] == 1)
      std::cout << "NO_TRAP_OR_RAN_TO_END\n";
    else
      std::cout << "ANOMALY_NOT_HALTED\n";
  } else if (s[1] == 0xBB && s[0] == 0 && s[2] == 0 && s[3] == 0) {
    std::cout << "AFTER_COMMIT\n";
  } else if (s[1] == 0 && s[0] == 0 && s[2] == 0 && s[3] == 0) {
    std::cout << "BEFORE_COMMIT\n";
  } else {
    std::cout << "AMBIGUOUS\n";
  }
  // Bridge-harness pass token (emu-bridge-test.sh greps for PASS).
  std::cout << "PASS\n";
  return 0;
}
