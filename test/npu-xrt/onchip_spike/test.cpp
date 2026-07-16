//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Minimal trace-presence host for the on-chip spike kernel.  The kernel's data
// is born on-chip (memtile relay, no DDR data path), so there is nothing to
// memcmp -- the only host buffer is the trace BO (runtime_sequence arg0).  The
// trace, not a data check, is the deliverable; this host just runs the kernel
// to completion and confirms the trace buffer is non-empty.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Trace buffer size in bytes -- matches aie.trace.host_config buffer_size.
constexpr int TRACE_SIZE = 8192;

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("onchip_spike");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

  // Start the XRT test code
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>()
              << std::endl;
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

  // Kernel arg layout: opcode(0), instr(1), ninstr(2), trace(3).
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_trace =
      xrt::bo(device, TRACE_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Zero the trace buffer so we can detect that the kernel wrote into it.
  uint8_t *bufTrace = bo_trace.map<uint8_t *>();
  memset(bufTrace, 0, TRACE_SIZE);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_trace);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Trace-presence check: a configured + routed trace unit writes packet data
  // into the buffer.  An all-zero buffer means no trace was captured.
  int nonzero = 0;
  for (int i = 0; i < TRACE_SIZE; i++)
    if (bufTrace[i] != 0)
      nonzero++;

  if (verbosity >= 1)
    std::cout << "Trace buffer non-zero bytes: " << nonzero << " / "
              << TRACE_SIZE << std::endl;

  if (nonzero > 0) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << "fail: trace buffer empty (no trace captured)." << std::endl
              << std::endl;
    return 1;
  }
}
