#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#define NUM_TEST_PATCHES 1


// input_scale=9.7952662036e-03  weight_scale=8.3531057462e-03  output_scale=7.6277577318e-03
static elem_t gemm4_input[4] = {47, 29, 12, 118};

static elem_t fc4_weights[2][4] = {
    {61, -21, 85, 35},
    {50, 127, 112, -7},
};
static acc_t fc4_bias[2] = {855, -44};

// input_scale=7.6277577318e-03  weight_scale=6.3607310876e-03  output_scale=4.7581130639e-03
static elem_t output_weights[2][2] = {
    {62, 51},
    {127, -22},
};
static acc_t output_bias[2] = {763, 1580};

static elem_t gemm2_output[2] = {100, 116};

// printf's %f does not work in this baremetal build (same issue braggnn.c
// works around); format manually instead.
static void print_float4(float f) {
  if (f < 0) {
    printf("-");
    f = -f;
  }
  int integer_part = (int)f;
  int frac_part = (int)((f - (float)integer_part) * 10000.0f + 0.5f);
  if (frac_part >= 10000) {
    integer_part++;
    frac_part -= 10000;
  }
  printf("%d.%04d", integer_part, frac_part);
}



void gemmini_gemm2(elem_t *input, float *output) {
  static elem_t prediction[2];

  // Accumulator addressing (bit 31 = is_acc_addr, bit 30 = accumulate; see
  // LocalAddr.scala). Layer 1 uses row 0, layer 2 uses row 1.
  const uint32_t acc_bit = 1u << (ADDR_LEN - 1);
  const uint32_t accum_bit = 1u << (ADDR_LEN - 2);
  const uint32_t D1_ACC = acc_bit | 0;              // layer 1 bias
  const uint32_t C1_ACC = acc_bit | accum_bit | 0;  // layer 1's raw (unactivated) output
  const uint32_t D2_ACC = acc_bit | 1;              // layer 2 bias
  const uint32_t C2_ACC = acc_bit | accum_bit | 1;  // layer 2 (final) output

  // On-chip scratchpad addressing (inputs/weights are tiny -- 1x4, 2x4, 2x2
  // -- so everything fits in a single DIM x DIM tile; no tiling loop
  // needed). FC4_OUT_SP is new: the on-chip landing spot for layer 1's
  // activated+quantized output, written by gemmini_mvout_spad and read
  // straight back by layer 2's compute -- no DRAM involved at any point.
  const uint32_t A1_SP = 0;
  const uint32_t B1_SP = DIM;
  const uint32_t FC4_OUT_SP = 2 * DIM;
  const uint32_t B2_SP = 3 * DIM;

  gemmini_flush(0);

  // dataflow=WS, weights (B) are stored [out_features][in_features] in
  // memory, so B_transpose=1 lets the mesh transpose them on the way in
  // during preload. Activation doesn't matter here: compute's write into
  // the accumulator is always raw/unactivated -- activation is applied
  // later, at read time, via CONFIG_ST (see below).
  gemmini_extended_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0, 1, 0, 1);

  // ---- Layer 1 (fc4): mvin input, weights, bias ----
  gemmini_extended3_config_ld(4 * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0); // A
  gemmini_extended3_config_ld(4 * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1); // B
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);                 // D (acc_t)

  gemmini_extended_mvin(input, A1_SP, 4, 1);
  gemmini_extended_mvin2(fc4_weights, B1_SP, 4, 2);
  gemmini_extended_mvin3(fc4_bias, D1_ACC, 2, 1);

  // ---- Layer 1: preload weights, compute -> accumulator row 0 (raw sum) ----
  gemmini_extended_preload(B1_SP, C1_ACC, /*BD_cols=*/2, /*BD_rows=*/4,
                            /*C_cols=*/2, /*C_rows=*/1);
  gemmini_extended_compute_preloaded(A1_SP, GARBAGE_ADDR, /*A_cols=*/4,
                                      /*A_rows=*/1, DIM, DIM);

  // ---- The fused layer boundary: move layer 1's output from the
  // accumulator to on-chip scratchpad, with RELU + fc4's output scale
  // applied on the way (CONFIG_ST governs mvout_spad's source-side read,
  // exactly like it would for a real mvout to DRAM). No DRAM traffic. ----
  gemmini_extended_config_st(2 * sizeof(elem_t), RELU, 0.0107267);
  gemmini_extended_mvout_spad(FC4_OUT_SP, /*dst_stride=*/1, C1_ACC,
                               /*cols=*/2, /*rows=*/1);

  // ---- Layer 2 (output): mvin weights + bias. Note there's no mvin for
  // "A" here -- layer 2's activation input is FC4_OUT_SP, already sitting
  // in scratchpad from the mvout_spad above. ----
  gemmini_extended3_config_ld(2 * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1); // B
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);                 // D (acc_t)

  gemmini_extended_mvin2(output_weights, B2_SP, 2, 2);
  gemmini_extended_mvin3(output_bias, D2_ACC, 2, 1);

  // ---- Layer 2: preload weights, compute with A = FC4_OUT_SP -- a plain,
  // always-working scratchpad-sourced compute, same as any other matmul. ----
  gemmini_extended_preload(B2_SP, C2_ACC, /*BD_cols=*/2, /*BD_rows=*/2,
                            /*C_cols=*/2, /*C_rows=*/1);
  gemmini_extended_compute_preloaded(FC4_OUT_SP, GARBAGE_ADDR, /*A_cols=*/2,
                                      /*A_rows=*/1, DIM, DIM);

  // ---- Only ONE DRAM write in the whole fused op: the final prediction ----
  gemmini_extended_config_st(2 * sizeof(elem_t), NO_ACTIVATION, 0.0101969);
  gemmini_extended_mvout(prediction, C2_ACC, 2, 1);

  gemmini_fence();

  const float OUTPUT_SCALE = 0.0047581130639f;
  const float PATCH_SIZE = 11.0f;
  output[0] = prediction[0] * OUTPUT_SCALE * PATCH_SIZE;
  output[1] = prediction[1] * OUTPUT_SCALE * PATCH_SIZE;
}


int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  gemmini_flush(0);

  float prediction[2];
  unsigned long long cycle_counts[NUM_TEST_PATCHES];
  unsigned long long total_cycles = 0;

  // Warmup: drop first run to warm caches
  printf("\n--- Warmup ---\n");
  unsigned long long start = read_cycles();
  gemmini_gemm2(gemm4_input, prediction);
  unsigned long long end = read_cycles();
  printf("Warmup done\n");

  unsigned long long elapsed = end - start;

  printf("cycles: %llu\n", elapsed);

  // Measured iterations
  for (int i = 0; i < NUM_TEST_PATCHES; i++) {
    printf("\n--- Inference %d/%d ---\n", i + 1, NUM_TEST_PATCHES);

    unsigned long long start = read_cycles();
    gemmini_gemm2(gemm4_input, prediction);
    unsigned long long end = read_cycles();

    unsigned long long elapsed = end - start;
    cycle_counts[i] = elapsed;
    total_cycles += elapsed;

    printf("cycles: %llu\n", elapsed);
  }

  printf("\n==============================================\n");
  printf("Avg cycles over %d runs: %llu\n", NUM_TEST_PATCHES,
         total_cycles / NUM_TEST_PATCHES);
  printf("==============================================\n");

  printf("\ngemm2 output (patch px): (");
  print_float4(prediction[0]);
  printf(", ");
  print_float4(prediction[1]);
  printf(")\n");

  exit(0);
}
