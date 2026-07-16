//===- srs.cc ---------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel: SRS (shift-round-saturate),
// accumulator -> narrower vector. Loads int32 accumulator values, builds an
// acc64 accumulator (the inverse UPS at shift 0 is a plain widen), then applies
// the SRS datapath to int16 with a fixed config: rounding = floor, saturation =
// saturate, user-shift = 4. This is the exact (rnd=0 FLOOR, sat, sym_sat=false,
// bits_o=16, shift=4) slice of the Half-A `srs` golden.
//
// The SRS hardware left-shifts by SRS_SHIFT_BIAS before the user shift to
// provide guard/round precision; `to_vector<int16>(4)` therefore round-floors
// with that bias, exactly matching the emulator's srs_lane. A passing HW==EMU
// bridge is the silicon evidence that the BIAS + floor + saturate path is real.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

// Process the whole batch in full 256-bit-vector chunks (16x int32). Using the
// natural 16-lane width means the compiler loads a full 1024-bit `cm`
// accumulator and the SRS reduces all 16 acc64 lanes to 16 int16 = one 256-bit
// `w` register -- a clean full-register store, no 128-bit q-register splitting.
// N must be a multiple of 16.
#ifndef SRS_N
#define SRS_N 48
#endif
#define SRS_SHIFT 4

extern "C" {

void srs_i32(int32_t *restrict in, int16_t *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::floor);
  ::aie::set_saturation(aie::saturation_mode::saturate);

  for (int i = 0; i < SRS_N; i += 16) {
    aie::vector<int32_t, 16> v = aie::load_v<16>(in + i);
    aie::accum<acc64, 16> acc;
    acc.from_vector(v, 0);
    aie::vector<int16_t, 16> o = acc.to_vector<int16_t>(SRS_SHIFT);
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
