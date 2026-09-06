#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "ptychonn.h"
#include "include/gemmini_testutils.h"


// Print float with 4 decimal places using only integer printf (bare-metal
// printf does not support %f).
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


// Replicate-pads `in[rows][cols][channels]` by 1 pixel on each side into
// `out[rows+2][cols+2][channels]` (edge-clamped, matching PyTorch bilinear
// upsample's boundary behavior -- NOT Gemmini's own zero-padding).
static void replicate_pad1(const elem_t *in, int rows, int cols, int channels, elem_t *out) {
  int prows = rows + 2, pcols = cols + 2;
  for (int r = 0; r < prows; r++) {
    int sr = r - 1;
    if (sr < 0) sr = 0;
    if (sr >= rows) sr = rows - 1;
    for (int c = 0; c < pcols; c++) {
      int sc = c - 1;
      if (sc < 0) sc = 0;
      if (sc >= cols) sc = cols - 1;
      for (int ch = 0; ch < channels; ch++) {
        out[(r * pcols + c) * channels + ch] = in[(sr * cols + sc) * channels + ch];
      }
    }
  }
}

static void gemmini_upsample2x(const gemmini_conv_layer_t *layer, const elem_t *in,
                               int in_rows, int in_cols, elem_t *padded_scratch, elem_t *out) {
  int channels = layer->out_channels;
  int out_rows = in_rows * 2, out_cols = in_cols * 2;
  tiled_conv_auto(BATCH, in_rows, in_cols, channels, channels, out_rows, out_cols,
                  1, 2, 1, 2, 4, false, false, false, false, false,
                  (elem_t *)in, (elem_t *)layer->upsample_kernel_weight, NULL,
                  out, NO_ACTIVATION, layer->upsample_gemmini_requant_scale, 0, 0, 0, 0, WS);
}

static elem_t enc0[64][64][32];
static elem_t enc1[32][32][32];
static elem_t enc2[32][32][64];
static elem_t enc3[16][16][64];
static elem_t enc4[16][16][128];
static elem_t encoder_out[8][8][128];

static elem_t dec_a[8][8][128];
static elem_t dec_b[8][8][128];
static elem_t dec_b_padded[10][10][128];
static elem_t dec_b_up[16][16][128];
static elem_t dec_c[16][16][64];
static elem_t dec_d[16][16][64];
static elem_t dec_d_padded[18][18][64];
static elem_t dec_d_up[32][32][64];
static elem_t dec_e[32][32][64];
static elem_t dec_f[32][32][64];
static elem_t dec_f_padded[34][34][64];
static elem_t dec_f_up[64][64][64];
//static elem_t dec_final[64][64][1];

static void run_decoder(const gemmini_conv_layer_t *layers, elem_t *output) {
  tiled_conv_auto(BATCH, 8, 8, layers[0].in_channels, layers[0].out_channels, 8, 8,
                  layers[0].stride_h, 1, 1, layers[0].pad_h, layers[0].kernel_h,
                  false, false, false, false, false,
                  (elem_t *)encoder_out, (elem_t *)layers[0].weight, (acc_t *)layers[0].bias,
                  (elem_t *)dec_a, layers[0].use_relu ? RELU : NO_ACTIVATION,
                  layers[0].requant_scale, 0, 0, 0, 0, WS);

  tiled_conv_auto(BATCH, 8, 8, layers[1].in_channels, layers[1].out_channels, 8, 8,
                  layers[1].stride_h, 1, 1, layers[1].pad_h, layers[1].kernel_h,
                  false, false, false, false, false,
                  (elem_t *)dec_a, (elem_t *)layers[1].weight, (acc_t *)layers[1].bias,
                  (elem_t *)dec_b, layers[1].use_relu ? RELU : NO_ACTIVATION,
                  layers[1].requant_scale, 0, 0, 0, 0, WS);
  gemmini_upsample2x(&layers[1], (elem_t *)dec_b, 8, 8, (elem_t *)dec_b_padded, (elem_t *)dec_b_up);

  tiled_conv_auto(BATCH, 16, 16, layers[2].in_channels, layers[2].out_channels, 16, 16,
                  layers[2].stride_h, 1, 1, layers[2].pad_h, layers[2].kernel_h,
                  false, false, false, false, false,
                  (elem_t *)dec_b_up, (elem_t *)layers[2].weight, (acc_t *)layers[2].bias,
                  (elem_t *)dec_c, layers[2].use_relu ? RELU : NO_ACTIVATION,
                  layers[2].requant_scale, 0, 0, 0, 0, WS);

  tiled_conv_auto(BATCH, 16, 16, layers[3].in_channels, layers[3].out_channels, 16, 16,
                  layers[3].stride_h, 1, 1, layers[3].pad_h, layers[3].kernel_h,
                  false, false, false, false, false,
                  (elem_t *)dec_c, (elem_t *)layers[3].weight, (acc_t *)layers[3].bias,
                  (elem_t *)dec_d, layers[3].use_relu ? RELU : NO_ACTIVATION,
                  layers[3].requant_scale, 0, 0, 0, 0, WS);
  gemmini_upsample2x(&layers[3], (elem_t *)dec_d, 16, 16, (elem_t *)dec_d_padded, (elem_t *)dec_d_up);

  tiled_conv_auto(BATCH, 32, 32, layers[4].in_channels, layers[4].out_channels, 32, 32,
                  layers[4].stride_h, 1, 1, layers[4].pad_h, layers[4].kernel_h,
                  false, false, false, false, false,
                  (elem_t *)dec_d_up, (elem_t *)layers[4].weight, (acc_t *)layers[4].bias,
                  (elem_t *)dec_e, layers[4].use_relu ? RELU : NO_ACTIVATION,
                  layers[4].requant_scale, 0, 0, 0, 0, WS);

  tiled_conv_auto(BATCH, 32, 32, layers[5].in_channels, layers[5].out_channels, 32, 32,
                  layers[5].stride_h, 1, 1, layers[5].pad_h, layers[5].kernel_h,
                  false, false, false, false, false,
                  (elem_t *)dec_e, (elem_t *)layers[5].weight, (acc_t *)layers[5].bias,
                  (elem_t *)dec_f, layers[5].use_relu ? RELU : NO_ACTIVATION,
                  layers[5].requant_scale, 0, 0, 0, 0, WS);
  gemmini_upsample2x(&layers[5], (elem_t *)dec_f, 32, 32, (elem_t *)dec_f_padded, (elem_t *)dec_f_up);

  tiled_conv_auto(BATCH, 64, 64, layers[6].in_channels, layers[6].out_channels, 64, 64,
                  layers[6].stride_h, 1, 1, layers[6].pad_h, layers[6].kernel_h,
                  false, false, false, false, false,
                  (elem_t *)dec_f_up, (elem_t *)layers[6].weight, (acc_t *)layers[6].bias,
                  (elem_t *)output, ITANH,
                  layers[6].requant_scale, layers[6].bias_scale, 0, 0, 0, WS);

  //const float out_scale = layers[6].output_scale;
  //for (int i = 0; i < 64 * 64; i++) {
  //  float pre_activation = ((elem_t *)dec_final)[i] * out_scale;
  //  output[i] = is_phase
  //      ? tanhf_custom(pre_activation) * 3.14159265358979323846f
  //      : sigmoidf_custom(pre_activation);
  //}
}

void gemmini_inference(const elem_t *input, elem_t *ph_out) {
  // --- Encoder (gemmini_conv_layers[0..5]) ---
  tiled_conv_auto(BATCH, 64, 64, gemmini_conv_layers[0].in_channels, gemmini_conv_layers[0].out_channels,
                  64, 64, gemmini_conv_layers[0].stride_h, 1, 1, gemmini_conv_layers[0].pad_h,
                  gemmini_conv_layers[0].kernel_h, false, false, false, false, false,
                  (elem_t *)input, (elem_t *)gemmini_conv_layers[0].weight, (acc_t *)gemmini_conv_layers[0].bias,
                  (elem_t *)enc0, gemmini_conv_layers[0].use_relu ? RELU : NO_ACTIVATION,
                  gemmini_conv_layers[0].requant_scale, 0, 0, 0, 0, WS);

  tiled_conv_auto(BATCH, 64, 64, gemmini_conv_layers[1].in_channels, gemmini_conv_layers[1].out_channels,
                  64, 64, gemmini_conv_layers[1].stride_h, 1, 1, gemmini_conv_layers[1].pad_h,
                  gemmini_conv_layers[1].kernel_h, false, false, false, false, false,
                  (elem_t *)enc0, (elem_t *)gemmini_conv_layers[1].weight, (acc_t *)gemmini_conv_layers[1].bias,
                  (elem_t *)enc1, gemmini_conv_layers[1].use_relu ? RELU : NO_ACTIVATION,
                  gemmini_conv_layers[1].requant_scale, 0, 2, 2, 0, WS);

  tiled_conv_auto(BATCH, 32, 32, gemmini_conv_layers[2].in_channels, gemmini_conv_layers[2].out_channels,
                  32, 32, gemmini_conv_layers[2].stride_h, 1, 1, gemmini_conv_layers[2].pad_h,
                  gemmini_conv_layers[2].kernel_h, false, false, false, false, false,
                  (elem_t *)enc1, (elem_t *)gemmini_conv_layers[2].weight, (acc_t *)gemmini_conv_layers[2].bias,
                  (elem_t *)enc2, gemmini_conv_layers[2].use_relu ? RELU : NO_ACTIVATION,
                  gemmini_conv_layers[2].requant_scale, 0, 0, 0, 0, WS);

  tiled_conv_auto(BATCH, 32, 32, gemmini_conv_layers[3].in_channels, gemmini_conv_layers[3].out_channels,
                  32, 32, gemmini_conv_layers[3].stride_h, 1, 1, gemmini_conv_layers[3].pad_h,
                  gemmini_conv_layers[3].kernel_h, false, false, false, false, false,
                  (elem_t *)enc2, (elem_t *)gemmini_conv_layers[3].weight, (acc_t *)gemmini_conv_layers[3].bias,
                  (elem_t *)enc3, gemmini_conv_layers[3].use_relu ? RELU : NO_ACTIVATION,
                  gemmini_conv_layers[3].requant_scale, 0, 2, 2, 0, WS);

  tiled_conv_auto(BATCH, 16, 16, gemmini_conv_layers[4].in_channels, gemmini_conv_layers[4].out_channels,
                  16, 16, gemmini_conv_layers[4].stride_h, 1, 1, gemmini_conv_layers[4].pad_h,
                  gemmini_conv_layers[4].kernel_h, false, false, false, false, false,
                  (elem_t *)enc3, (elem_t *)gemmini_conv_layers[4].weight, (acc_t *)gemmini_conv_layers[4].bias,
                  (elem_t *)enc4, gemmini_conv_layers[4].use_relu ? RELU : NO_ACTIVATION,
                  gemmini_conv_layers[4].requant_scale, 0, 0, 0, 0, WS);

  tiled_conv_auto(BATCH, 16, 16, gemmini_conv_layers[5].in_channels, gemmini_conv_layers[5].out_channels,
                  16, 16, gemmini_conv_layers[5].stride_h, 1, 1, gemmini_conv_layers[5].pad_h,
                  gemmini_conv_layers[5].kernel_h, false, false, false, false, false,
                  (elem_t *)enc4, (elem_t *)gemmini_conv_layers[5].weight, (acc_t *)gemmini_conv_layers[5].bias,
                  (elem_t *)encoder_out, gemmini_conv_layers[5].use_relu ? RELU : NO_ACTIVATION,
                  gemmini_conv_layers[5].requant_scale, 0, 2, 2, 0, WS);

  // --- Decoder of phase ---
  run_decoder(&gemmini_conv_layers[13], ph_out);
}

// ---------------------------------------------------------------------------

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  printf("==============================================\n");
  printf("PtychoNN Gemmini inference (PT2E/GemminiQuantizer weights)\n");
  printf("==============================================\n");

  printf("Flushing Gemmini TLB\n");
  gemmini_flush(0);

  static elem_t ph_out[INPUT_DIM][INPUT_DIM];
  static float ph_out_dq[INPUT_DIM][INPUT_DIM];
  unsigned long long total_cycles = 0;

  printf("\nInput shape: [%d, %d, %d, %d]\n", BATCH, INPUT_CHANNELS, INPUT_DIM, INPUT_DIM);
  printf("Running %d inferences (%d warmup + %d measured)\n",
         NUM_TEST_PATCHES + 1, 1, NUM_TEST_PATCHES);

  // Warmup: drop first run to warm caches, dumping every layer's int8 output
  // once so it can be diffed against a reference run of the same patch.
  printf("\n--- Warmup (sample index %d) ---\n", gemmini_sample_input_indices[0]);
  unsigned long long start = read_cycles();
  gemmini_inference((const elem_t *)gemmini_sample_inputs[0], (elem_t *)ph_out);
  unsigned long long end = read_cycles();

  const float out_scale = gemmini_conv_layers[19].output_scale;
  for (int i = 0; i < 64 * 64; i++) {
    ((float *)ph_out_dq)[i] = ((elem_t *)ph_out)[i] * out_scale;
  }

  printf("Warmup done, cycles: %llu\n", end - start);
  printf(" ph[0][0]="); print_float4(ph_out_dq[0][0]);
  printf(" ph[32][32]="); print_float4(ph_out_dq[32][32]);
  printf("\n");

  // Measured iterations
  for (int i = 0; i < NUM_TEST_PATCHES; i++) {
    printf("\n--- Inference %d/%d (sample index %d) ---\n", i + 1, NUM_TEST_PATCHES,
           gemmini_sample_input_indices[i]);

    unsigned long long istart = read_cycles();
    asm volatile(".word 0x8013");  // TracerV start trigger
    gemmini_inference((const elem_t *)gemmini_sample_inputs[i], (elem_t *)ph_out);
    asm volatile(".word 0x10013"); // TracerV end trigger
    unsigned long long iend = read_cycles();

  	for (int i = 0; i < 64 * 64; i++) {
  	  ((float *)ph_out_dq)[i] = ((elem_t *)ph_out)[i] * out_scale;
  	}

    unsigned long long elapsed = iend - istart;
    total_cycles += elapsed;

    float ph_min = ph_out_dq[0][0], ph_max = ph_out_dq[0][0], ph_sum = 0.0f;
    for (int r = 0; r < INPUT_DIM; r++) {
      for (int c = 0; c < INPUT_DIM; c++) {
        float v = ph_out_dq[r][c];
        if (v < ph_min) ph_min = v;
        if (v > ph_max) ph_max = v;
        ph_sum += v;
      }
    }

    printf("cycles: %llu\n", elapsed);
    printf("\nph:  min="); print_float4(ph_min);
    printf(" max="); print_float4(ph_max);
    printf(" mean="); print_float4(ph_sum / (float)(INPUT_DIM * INPUT_DIM));
    printf("\n");
  }

  printf("\n==============================================\n");
  printf("Avg cycles over %d runs: %llu\n", NUM_TEST_PATCHES, total_cycles / NUM_TEST_PATCHES);
  printf("==============================================\n");
  printf("PtychoNN inference completed successfully\n");
  printf("==============================================\n");

  exit(0);
}
