// Minimal NPU test for BSD V4 model (nc=4, 640x640 RGB planar)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "awnn.h"

#define INPUT_SIZE   (640 * 640 * 3)
#define NUM_CELLS    8400
#define NUM_CLASSES  4

static const char* bsd_names[4] = {"person", "bicycle", "motorcycle", "vehicle"};

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.nb> <input_640_planar.raw>\n", argv[0]);
        return 1;
    }

    const char* nb_path = argv[1];
    const char* raw_path = argv[2];

    // Load input
    unsigned char* input_buf = (unsigned char*)malloc(INPUT_SIZE);
    if (!input_buf) { fprintf(stderr, "malloc input failed\n"); return 1; }

    FILE* f = fopen(raw_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", raw_path); free(input_buf); return 1; }
    size_t n = fread(input_buf, 1, INPUT_SIZE, f);
    fclose(f);
    if (n != INPUT_SIZE) {
        fprintf(stderr, "Short read: %zu / %d bytes\n", n, INPUT_SIZE);
        free(input_buf);
        return 1;
    }
    printf("Input loaded: %zu bytes\n", n);

    // Model info
    awnn_info_t* info = awnn_get_info((char*)nb_path);
    if (info) {
        printf("Model: name=%s, w=%d, h=%d, mem=%u KB\n",
               info->name, info->width, info->height, info->mem_size / 1024);
    }

    // Init and load
    printf("Init NPU...\n");
    awnn_init(16 * 1024 * 1024);

    printf("Load model...\n");
    Awnn_Context_t* ctx = awnn_create((char*)nb_path);
    if (!ctx) { fprintf(stderr, "awnn_create failed\n"); free(input_buf); awnn_uninit(); return 1; }

    // Run
    awnn_set_input_buffers(ctx, &input_buf);
    printf("Run inference...\n");
    awnn_run(ctx);

    // Get outputs
    float** outputs = awnn_get_output_buffers(ctx);
    if (!outputs) { fprintf(stderr, "get_output_buffers failed\n"); awnn_destroy(ctx); awnn_uninit(); free(input_buf); return 1; }

    float* boxes  = outputs[0];  // [4 * 8400]
    float* scores = outputs[1];  // [4 * 8400] raw logits

    printf("\n=== Per-class max ===\n");
    for (int c = 0; c < NUM_CLASSES; c++) {
        float max_logit = -999.0f;
        int best_cell = -1;
        for (int i = 0; i < NUM_CELLS; i++) {
            float logit = scores[c * NUM_CELLS + i];
            if (logit > max_logit) { max_logit = logit; best_cell = i; }
        }
        float sig = sigmoid(max_logit);
        printf("  %d %-12s: logit=%+.4f  sigmoid=%.4f %s\n",
               c, bsd_names[c], max_logit, sig, sig > 0.5 ? "***" : "");
        if (best_cell >= 0 && sig > 0.5) {
            float cx = boxes[0*NUM_CELLS + best_cell];
            float cy = boxes[1*NUM_CELLS + best_cell];
            float w  = boxes[2*NUM_CELLS + best_cell];
            float h  = boxes[3*NUM_CELLS + best_cell];
            printf("         box: cx=%.3f cy=%.3f w=%.3f h=%.3f (cell %d)\n", cx, cy, w, h, best_cell);
        }
    }

    // Box stats
    float min_cx = 999, max_cx = -999, min_cy = 999, max_cy = -999;
    float min_w = 999, max_w = -999, min_h = 999, max_h = -999;
    for (int i = 0; i < NUM_CELLS; i++) {
        float cx = boxes[0*NUM_CELLS + i];
        float cy = boxes[1*NUM_CELLS + i];
        float w  = boxes[2*NUM_CELLS + i];
        float h  = boxes[3*NUM_CELLS + i];
        if (cx < min_cx) min_cx = cx; if (cx > max_cx) max_cx = cx;
        if (cy < min_cy) min_cy = cy; if (cy > max_cy) max_cy = cy;
        if (w < min_w) min_w = w; if (w > max_w) max_w = w;
        if (h < min_h) min_h = h; if (h > max_h) max_h = h;
    }
    printf("\nBox stats: cx[%.3f,%.3f] cy[%.3f,%.3f] w[%.3f,%.3f] h[%.3f,%.3f]\n",
           min_cx, max_cx, min_cy, max_cy, min_w, max_w, min_h, max_h);

    // Cleanup
    awnn_destroy(ctx);
    awnn_uninit();
    free(input_buf);
    printf("\nTest complete!\n");
    return 0;
}
