// test_npu_direct.c — load BGR file, run NPU inference, print detections
// Usage: ./test_npu_direct <model.nb> <frame.bgr> [width height]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "detect_engine.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.nb> <frame.bgr> [width height]\n", argv[0]);
        return 1;
    }

    int cam_w = 640, cam_h = 360;
    if (argc >= 5) {
        cam_w = atoi(argv[3]);
        cam_h = atoi(argv[4]);
        if (cam_w <= 0 || cam_h <= 0) {
            fprintf(stderr, "Invalid frame size: %s x %s\n", argv[3], argv[4]);
            return 1;
        }
    }
    size_t bgr_sz = (size_t)cam_w * cam_h * 3;

    // Read BGR frame from file
    uint8_t *bgr = (uint8_t*)malloc(bgr_sz);
    if (!bgr) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[2]); free(bgr); return 1; }
    size_t n = fread(bgr, 1, bgr_sz, f);
    fclose(f);
    if (n != bgr_sz) { fprintf(stderr, "Read %zu/%zu bytes\n", n, bgr_sz); free(bgr); return 1; }
    printf("Loaded %s (%zu bytes, %dx%d BGR)\n", argv[2], n, cam_w, cam_h);

    // Init detect engine (low conf threshold for debugging quantization)
    float conf_thr = 0.05f;
    if (detect_init(argv[1], "/bsd_test_shm", cam_w, cam_h, conf_thr, 0.45f) != 0) {
        fprintf(stderr, "detect_init failed\n"); free(bgr); return 1;
    }
    printf("NPU initialized\n");

    // Run inference
    if (detect_process_frame(bgr) != 0) {
        fprintf(stderr, "detect_process_frame failed\n");
        detect_deinit(); free(bgr); return 1;
    }

    // Print results
    const DetResult *res = detect_get_result();
    if (res) {
        printf("frame_id=%u count=%u\n", res->frame_id, res->count);
        const char *names[] = {"person", "bicycle", "motorcycle", "vehicle"};
        for (uint32_t i = 0; i < res->count; i++) {
            const DetObject *d = &res->objects[i];
            printf("  [%u] %s: conf=%.3f box=[%.0f,%.0f,%.0f,%.0f] w=%.0f h=%.0f\n",
                   i,
                   (d->class_id < 4) ? names[d->class_id] : "?",
                   d->conf,
                   d->x1, d->y1, d->x2, d->y2,
                   d->x2 - d->x1, d->y2 - d->y1);
        }
    }

    detect_deinit();
    free(bgr);
    return 0;
}
