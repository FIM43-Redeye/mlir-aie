//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// debug_halt_probe host harness (Phase A -- Exp2: Debug_Control0 count-step
// matrix).
//
// Exp2 uses a NO-POLL DOUBLE OP_READ for the settled-state verdict (spec §4.3).
// @seq issues 18 ctrl-in packets total: pass A (9: 8 marker slots +
// Core_Status) followed immediately by pass B (same 9, verbatim repeat).
// The ctrl-out BO receives 18 response words: A=[0..8], B=[9..17].
// test.cpp requires A==B (every slot + Core_Status identical) => settled
// terminal state.  A!=B => UNSETTLED (core still mid-execution -- clean
// re-run, never a device wedge).
//
// LANDED = number of committed marker stores, order-independent.  Precise
// reason: MLIR `memref.store %v(k), buf[k]` binds value (101+k) to slot k by
// source; the scheduler only reorders when each store commits, never which
// value lands at which address.  So s[k]==101+k iff that store committed.
//
// TRANSIENT Exp2 config.  Task 8 will restore the Exp1 test.cpp
// (git checkout <Exp1 SHA> -- test/npu-xrt/debug_halt_probe/test.cpp).
//
// OUTBUF_ADDR = output_buffer tile-local address = p0 & 0xFFFF, where p0 is
// the core's 20-bit pointer to output_buffer[0] from llvm-objdump-aie.
// Task 3 disasm: p0 = 0x70400 => OUTBUF_ADDR = 0x0400.
// DERIVED, not fixed.  Self-check: the EMU baseline (count-step inert) must
// return s[k]==101+k for k=0..7 from the 8 slot reads.  If any slot is
// garbage, OUTBUF_ADDR is wrong -- re-derive from a fresh disasm (or aiecc
// allocation log) and update OUTBUF_ADDR + regenerate the ctrl-in packets.
// Also re-derive TRAP_PC in aie.mlir if the kernel changes.
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
// bit 16 = DEBUG_HALT (XAIEMLGBL_CORE_MODULE_CORE_STATUS_DEBUG_HALTED_LSB=16,
//   width=1, lines 2363-2365).
// bit 20 = CORE_DONE  (XAIEMLGBL_CORE_MODULE_CORE_STATUS_CORE_DONE_LSB=20,
//   width=1, lines 2347-2349).
static constexpr uint32_t CORE_STATUS_ADDR = 0x32004;

// Number of ctrl-in packets per pass: 8 marker slots + 1 Core_Status = 9.
// ctrl-out BO receives 2 passes: pass A = words [0..8], pass B = words [9..17].
static constexpr int NUM_READ_PKTS = 18; // total ctrl-out words (2 passes)
static constexpr int PKTS_PER_PASS = 9;  // packets per pass
static constexpr int CTRL_OUT_WORDS = NUM_READ_PKTS;

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
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_ctrlOut = xrt::bo(device, CTRL_OUT_WORDS * sizeof(uint32_t),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  // ctrl-in BO holds 9 unique OP_READ headers (one per target address).
  // @seq sends them twice (pass A then pass B) by reusing this same BO;
  // only 9 headers need to be assembled.
  auto bo_ctrlIn = xrt::bo(device, PKTS_PER_PASS * sizeof(uint32_t),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_gate = xrt::bo(device, sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                         kernel.group_id(5));

  // Assemble OP_READ control-packet headers into ctrl-in BO.
  // Header layout (from xdna-emu src/device/control_packets/parser.rs):
  //   [31]    parity (even-parity over bits 30:0)
  //   [30:24] stream_id (response routing; 0x2 = TileControl->shim flow 0x2)
  //   [23:22] operation  (0x1 = OP_READ)
  //   [21:20] length     (beats - 1; 0 = one response word)
  //   [19:0]  address    (tile-local address to read)
  auto parity = [](uint32_t n) -> uint32_t {
    uint32_t p = 0;
    while (n) {
      p += n & 1u;
      n >>= 1;
    }
    return (p % 2u == 0u) ? 1u : 0u;
  };

  auto make_read_header = [&](uint32_t addr) -> uint32_t {
    uint32_t stream_id = 0x2;
    uint32_t operation = 0x1;  // OP_READ
    uint32_t beats_minus1 = 0; // 1 word response per packet
    uint32_t hdr = (stream_id << 24) | (operation << 22) |
                   (beats_minus1 << 20) | (addr & 0xFFFFF);
    hdr |= (parity(hdr) << 31);
    return hdr;
  };

  // Nine OP_READ packet headers: output_buffer[0..7] then Core_Status.
  // @seq reuses this 9-header ctrl-in BO for both pass A and pass B.
  std::vector<uint32_t> ctrlPackets = {
      make_read_header(OUTBUF_ADDR + 0 * sizeof(uint32_t)), // s[0]
      make_read_header(OUTBUF_ADDR + 1 * sizeof(uint32_t)), // s[1]
      make_read_header(OUTBUF_ADDR + 2 * sizeof(uint32_t)), // s[2]
      make_read_header(OUTBUF_ADDR + 3 * sizeof(uint32_t)), // s[3]
      make_read_header(OUTBUF_ADDR + 4 * sizeof(uint32_t)), // s[4]
      make_read_header(OUTBUF_ADDR + 5 * sizeof(uint32_t)), // s[5]
      make_read_header(OUTBUF_ADDR + 6 * sizeof(uint32_t)), // s[6]
      make_read_header(OUTBUF_ADDR + 7 * sizeof(uint32_t)), // s[7]
      make_read_header(CORE_STATUS_ADDR),                   // Core_Status
  };
  uint32_t *bufCtrlIn = bo_ctrlIn.map<uint32_t *>();
  memcpy(bufCtrlIn, ctrlPackets.data(), ctrlPackets.size() * sizeof(uint32_t));

  uint32_t *bufGate = bo_gate.map<uint32_t *>();
  bufGate[0] = 1u;

  uint32_t *bufCtrlOut = bo_ctrlOut.map<uint32_t *>();
  // Zero-init ctrl-out (no sentinel needed: we check agreement between passes,
  // not presence of a specific sentinel value).
  memset(bufCtrlOut, 0, CTRL_OUT_WORDS * sizeof(uint32_t));

  uint32_t *bufInstr = bo_instr.map<uint32_t *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_gate.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_ctrlOut, bo_ctrlIn, bo_gate);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_ctrlOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Parse ctrl-out response words (double-read, spec §4.3).
  // ctrl-out layout:
  //   pass A: [0..7]  = output_buffer[0..7]  (sA[0..7])
  //           [8]     = Core_Status           (csA)
  //   pass B: [9..16] = output_buffer[0..7]  (sB[0..7])
  //           [17]    = Core_Status           (csB)
  uint32_t *ctrlOut = bo_ctrlOut.map<uint32_t *>();
  uint32_t sA[8], sB[8];
  for (int k = 0; k < 8; k++)
    sA[k] = ctrlOut[k];
  uint32_t csA = ctrlOut[8];
  for (int k = 0; k < 8; k++)
    sB[k] = ctrlOut[9 + k];
  uint32_t csB = ctrlOut[17];

  // Settled-state self-validation: A == B (every slot + Core_Status identical).
  // A != B => core was still mid-execution during pass A; clean re-run.
  bool settled = (csA == csB) && (memcmp(sA, sB, sizeof(sA)) == 0);

  // Interpret pass B (the later, more authoritative snapshot).
  auto count_landed = [](const uint32_t *s) {
    int n = 0;
    for (int k = 0; k < 8; k++)
      if (s[k] == (uint32_t)(101 + k))
        n++;
    return n;
  };
  int landedB = count_landed(sB);
  bool halted = (csB & (1u << 16)) != 0; // DEBUG_HALT bit 16 (aie-rt 2363)
  bool done = (csB & (1u << 20)) != 0;   // CORE_DONE  bit 20 (aie-rt 2347)

  // Exp2 LANDED verdict (Transient -- Task 8 restores Exp1 test.cpp).
  //
  // SLOTS: uses pass B (sB) -- later and identical to sA when settled.
  // LANDED: counts slots where s[k]==101+k (value-to-slot bound by source, so
  // count is schedule-reorder-immune -- see file-header comment).
  // SETTLED=1 => A==B => terminal state; SETTLED=0 => re-run clean.
  //
  // Three-way interpretation when settled:
  //   done=1, landed=8, halted=0  -> count-step inert (core ran to end)
  //   halted=1, landed=N (N<8)    -> count-step fired after N committed stores
  //   other                       -> partial / unexpected; record verbatim
  std::cout << "SLOTS:";
  for (int k = 0; k < 8; k++)
    std::cout << " " << sB[k];
  std::cout << " CORE_STATUS=0x" << std::hex << csB << std::dec
            << " HALTED=" << (halted ? 1 : 0) << " DONE=" << (done ? 1 : 0)
            << " SETTLED=" << (settled ? 1 : 0) << "\n";
  std::cout << "LANDED:" << landedB << "\n";

  if (!settled) {
    std::cout << "UNSETTLED\n";
    return 1;
  }

  // Bridge-harness pass token (emu-bridge-test.sh greps for PASS).
  std::cout << "PASS\n";
  return 0;
}
