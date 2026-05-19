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
// Exp2 inherits the Exp1 control-packet OP_READ readback machinery, extended
// from 5 packets (4 marker slots + Core_Status) to 9 (8 marker slots +
// Core_Status).  The verdict counts how many of the 8 expected values
// (101..108) committed, order-independent (source binds value 101+k to slot
// k; the scheduler only reorders commit timing, not value-to-slot -- see the
// verdict-block comment for the precise argument).  Core_Status[16]
// (DEBUG_HALT) and bit 20 (CORE_DONE) are both reported.
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

// Number of ctrl-in/ctrl-out words: 8 marker slots + 1 Core_Status = 9.
static constexpr int NUM_READ_PKTS = 9;
static constexpr int CTRL_OUT_WORDS = NUM_READ_PKTS;

// Sentinel pre-written into every ctrl-out slot before the run.  Used to
// detect un-issued OP_READ (MASKPOLL-unsatisfied path, if any).
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
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_ctrlOut = xrt::bo(device, CTRL_OUT_WORDS * sizeof(uint32_t),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_ctrlIn = xrt::bo(device, NUM_READ_PKTS * sizeof(uint32_t),
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

  // Nine OP_READ packets: output_buffer[0..7] then Core_Status.
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

  // Pre-fill ctrl-out with the sentinel so un-issued OP_READ is detectable.
  uint32_t *bufCtrlOut = bo_ctrlOut.map<uint32_t *>();
  for (int i = 0; i < CTRL_OUT_WORDS; i++)
    bufCtrlOut[i] = CTRLOUT_SENTINEL;

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

  // Parse ctrl-out response words.
  // Responses arrive in the order the OP_READ packets were sent:
  //   ctrlOut[0..7] = output_buffer[0..7] (s[0..7])
  //   ctrlOut[8]    = Core_Status          (cs)
  uint32_t *ctrlOut = bo_ctrlOut.map<uint32_t *>();
  uint32_t s[8];
  for (int i = 0; i < 8; i++)
    s[i] = ctrlOut[i];
  uint32_t cs = ctrlOut[8];

  // Exp2 LANDED verdict (Transient -- Task 8 restores Exp1 test.cpp).
  //
  // landed = number of committed marker stores, order-independent.  Precise
  // reason it is reorder-immune: MLIR `memref.store %v(k), buf[k]` binds the
  // value (101+k) to slot k by *source*; the instruction scheduler only
  // reorders *when* each store commits, never which value lands at which
  // address.  So s[k]==101+k iff that store committed; the positional check
  // counts committed stores regardless of schedule order.  (Caveat for G2
  // interpretation, not this verdict: if the compiler fuses the 8 scalar
  // stores into one vector store they commit atomically, changing count-step
  // granularity -- a Task 7 findings-doc observation, recorded from SLOTS,
  // not a bug in this faithful count.)
  //
  // Report both terminal bits:
  //   HALTED = Core_Status[16] (DEBUG_HALT, aie-rt xaiemlgbl_params.h:2363)
  //   DONE   = Core_Status[20] (CORE_DONE,  aie-rt xaiemlgbl_params.h:2347)
  //
  // Three-way interpretation:
  //   done=1, landed=8, halted=0  -> count-step inert (core ran to end)
  //   halted=1, landed=N (N<8)    -> count-step fired after N committed stores
  //   other                       -> partial / unexpected; record verbatim
  int landed = 0;
  for (int k = 0; k < 8; k++)
    if (s[k] == (uint32_t)(101 + k))
      landed++;
  bool halted = (cs & (1u << 16)) != 0; // DEBUG_HALT bit 16
  bool done = (cs & (1u << 20)) != 0;   // CORE_DONE  bit 20

  std::cout << "SLOTS:";
  for (int k = 0; k < 8; k++)
    std::cout << " " << s[k];
  std::cout << " CORE_STATUS=0x" << std::hex << cs << std::dec
            << " HALTED=" << (halted ? 1 : 0) << " DONE=" << (done ? 1 : 0)
            << "\n";
  std::cout << "LANDED:" << landed << "\n";

  // Bridge-harness pass token (emu-bridge-test.sh greps for PASS).
  std::cout << "PASS\n";
  return 0;
}
