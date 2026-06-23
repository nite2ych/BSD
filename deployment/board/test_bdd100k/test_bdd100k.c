// Minimal NPU test for BDD100K model (nc=10, 640x640 RGB planar)
// Compile: see Makefile or cross-compile on Ubuntu18
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "awnn.h"

#define INPUT_SIZE   (640 * 640 * 3)   // RGB planar, uint8
#define NUM_CELLS    8400
#define NUM_CLASSES  10

// BDD100K class names
static const char* bdd_names[10] = {
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
};

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

    // 1. Load input image
    printf("Loading input: %s\n", raw_path);
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
    printf("  OK: %zu bytes read\n", n);

    // 2. Get model info
    printf("Model info: %s\n", nb_path);
    awnn_info_t* info = awnn_get_info((char*)nb_path);
    if (info) {
        printf("  name=%s, w=%d, h=%d, mem=%u KB\n",
               info->name, info->width, info->height, info->mem_size / 1024);
    }

    // 3. Init NPU and load model
    printf("Init NPU...\n");
    awnn_init(16 * 1024 * 1024);  // 16 MB heap

    printf("Load model...\n");
    Awnn_Context_t* ctx = awnn_create((char*)nb_path);
    if (!ctx) {
        fprintf(stderr, "awnn_create failed\n");
        free(input_buf);
        awnn_uninit();
        return 1;
    }
    printf("  OK\n");

    // 4. Set input
    printf("Set input...\n");
    awnn_set_input_buffers(ctx, &input_buf);

    // 5. Run inference
    printf("Run inference...\n");
    awnn_run(ctx);
    printf("  Done\n");

    // 6. Get outputs
    float** outputs = awnn_get_output_buffers(ctx);
    if (!outputs) {
        fprintf(stderr, "awnn_get_output_buffers failed\n");
        awnn_destroy(ctx);
        awnn_uninit();
        free(input_buf);
        return 1;
    }

    // outputs[0] = boxes [4 * 8400] (cx, cy, w, h)
    // outputs[1] = scores [10 * 8400] (raw logits)
    float* boxes  = outputs[0];
    float* scores = outputs[1];

    printf("\n=== Results ===\n");
    printf("Boxes range:\n");
    printf("  cx: [%.4f, %.4f]\n", boxes[0], boxes[0]);
    for (int i = 1; i < NUM_CELLS; i++) {
        if (boxes[0*NUM_CELLS + i] < boxes[0]) boxes[0] = boxes[0*NUM_CELLS + i];
        // track min/max across all cells for summary
    }
    // Print per-class max logit and sigmoid
    printf("Per-class max logit/sigmoid:\n");
    for (int c = 0; c < NUM_CLASSES; c++) {
        float max_logit = -999.0f;
        for (int i = 0; i < NUM_CELLS; i++) {
            float logit = scores[c * NUM_CELLS + i];
            if (logit > max_logit) max_logit = logit;
        }
        float sig = sigmoid(max_logit);
        printf("  %2d %-15s: logit=%+.4f  sigmoid=%.4f%s\n",
               c, bdd_names[c], max_logit, sig, sig > 0.5 ? " ***" : "");
    }

    // Box stats
    float min_cx = 999, max_cx = -999, min_cy = 999, max_cy = -999;
    float min_w = 999, max_w = -999, min_h = 999, max_h = -999;
    for (int i = 0; i < NUM_CELLS; i++) {
        float cx = boxes[0*NUM_CELLS + i];
        float cy = boxes[1*NUM_CELLS + i];
        float w  = boxes[2*NUM_CELLS + i];
        float h  = boxes[3*NUM_CELLS + i];
        if (cx < min_cx) min_cx = cx;
        if (cx > max_cx) max_cx = cx;
        if (cy < min_cy) min_cy = cy;
        if (cy > max_cy) max_cy = cy;
        if (w < min_w) min_w = w;
        if (w > max_w) max_w = w;
        if (h < min_h) min_h = h;
        if (h > max_h) max_h = h;
    }
    printf("Box stats:\n");
    printf("  cx: [%.3f, %.3f]\n", min_cx, max_cx);
    printf("  cy: [%.3f, %.3f]\n", min_cy, max_cy);
    printf("  w:  [%.3f, %.3f]\n", min_w, max_w);
    printf("  h:  [%.3f, %.3f]\n", min_h, max_h);

    // 7. Cleanup
    awnn_destroy(ctx);
    awnn_uninit();
    free(input_buf);

    printf("\nTest complete!\n");
    return 0;
}
