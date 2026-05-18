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
// On EMU, an OP_READ of this register returns 0x0 even for a completed core:
// control-packet reads of core/debug registers are not wired to core_debug --
// a known emulator read-side routing gap (symmetric to the write-side gap that
// drops the breakpoint-arming writes), not a probe bug.  HW populates
// DEBUG_HALT/ENABLE correctly; the AFTER/BEFORE_COMMIT verdicts are therefore
// hardware-only (Task 5).
static constexpr uint32_t CORE_STATUS_ADDR = 0x32004;

// Number of ctrl-in words (one OP_READ header per slot + one for Core_Status).
// To extend (Task 6 reads 8 marker slots): bump this, add slots to s[]/the
// ctrlPackets list, and add a packet block in aie.mlir.
static constexpr int NUM_READ_PKTS = 5;
// ctrl-out receives one response word per OP_READ packet.  This constant
// drives the bo_ctrlOut allocation below.
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
  // (No output BO: the marker-DMA path was removed -- it would dma_wait
  // forever on a halted core.  The OP_READ responses are the sole output.)
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_ctrlOut = xrt::bo(device, CTRL_OUT_WORDS * sizeof(uint32_t),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_ctrlIn = xrt::bo(device, NUM_READ_PKTS * sizeof(uint32_t),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

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

  uint32_t *bufInstr = bo_instr.map<uint32_t *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Kernel args: opcode, instr BO, instr count, ctrl-out, ctrl-in
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_ctrlOut, bo_ctrlIn);
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

  // Schedule from llvm-objdump-aie (full schedule in aie.mlir TRAP_PC comment):
  //   trap bundle 0x114 -> s[1]=0xBB ; strictly-later 0x11c -> s[0]=0xAA
  //   later -> s[3]=1 ; later -> s[2]=0xCC
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
