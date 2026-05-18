//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// debug_halt_probe host harness (Phase A -- Exp1: separable pre/trap/post/done
// markers).
// Reads the marker buffer back from the emulator and prints the four slot
// markers (out0..out3), a BASELINE verdict (all four markers present, core
// reached aie.end), and a TRAP_VERDICT classifying a Task-3 trap as
// before/after-commit. The runtime sequence also issues one aiex.npu.write32
// to compute-tile (0,2) Debug_Control0 to prove the control-packet authoring
// mechanism reaches the tile.
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

  // instr BO at group_id(1); output BO at group_id(3) (first runtime arg)
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_out = xrt::bo(device, 8 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *out = bo_out.map<uint32_t *>();
  std::cout << "MARKERS: out0=" << out[0] << " out1=" << out[1]
            << " out2=" << out[2] << " out3=" << out[3] << "\n";
  // No-trap baseline: all four markers present, core reached aie.end.
  bool baseline =
      (out[0] == 0xAA && out[1] == 0xBB && out[2] == 0xCC && out[3] == 1);
  // Trap armed (Task 3): out0 always lands (pre-trap).
  //  - out1==0xBB && out2==0 -> trap bundle COMMITTED -> HALT AFTER COMMIT
  //  - out1==0    && out2==0 -> trap bundle DID NOT commit -> HALT BEFORE
  //  COMMIT
  // out2 (POST-trap store) must be 0: any trap halts the core before it reaches
  // the post-trap store.
  // out3 (done sentinel) must be 0 when trapped (core halted, never reached
  // end).
  std::cout << "BASELINE:" << (baseline ? "YES" : "NO") << "\n";
  std::cout << "TRAP_VERDICT:";
  if (out[0] == 0xAA && out[3] == 0) {
    if (out[1] == 0xBB && out[2] == 0)
      std::cout << "AFTER_COMMIT\n";
    else if (out[1] == 0 && out[2] == 0)
      std::cout << "BEFORE_COMMIT\n";
    // AMBIGUOUS: out1/out2 in an unexpected combination (e.g. post-trap landed
    // but trap didn't) -- treat as a probe anomaly to investigate, not a clean
    // before/after result.
    else
      std::cout << "AMBIGUOUS\n";
  } else {
    std::cout << "NO_TRAP_OR_RAN_TO_END\n";
  }
  // Signal to the bridge test harness that the run completed without error.
  // Verdict interpretation is in the BASELINE / TRAP_VERDICT lines above.
  std::cout << "PASS\n";
  return 0;
}
