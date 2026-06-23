// Continuous NPU benchmark for one BGR frame.
// Usage: ./bench_npu_loop <model.nb> <frame.bgr> [width height model_size loops]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "detect_engine.h"

static long long prof_ms(long long us)
{
    return us / 1000;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.nb> <frame.bgr> [width height model_size loops]\n", argv[0]);
        return 1;
    }

    int cam_w = (argc >= 5) ? atoi(argv[3]) : 1280;
    int cam_h = (argc >= 5) ? atoi(argv[4]) : 720;
    int model_size = (argc >= 6) ? atoi(argv[5]) : 640;
    int loops = (argc >= 7) ? atoi(argv[6]) : 100;
    if (cam_w <= 0 || cam_h <= 0 || loops <= 0) {
        fprintf(stderr, "Invalid width/height/loops\n");
        return 1;
    }

    size_t bgr_sz = (size_t)cam_w * (size_t)cam_h * 3u;
    uint8_t *bgr = (uint8_t*)malloc(bgr_sz);
    if (!bgr) {
        fprintf(stderr, "malloc failed for %zu bytes\n", bgr_sz);
        return 1;
    }

    FILE *f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", argv[2]);
        free(bgr);
        return 1;
    }
    size_t n = fread(bgr, 1, bgr_sz, f);
    fclose(f);
    if (n != bgr_sz) {
        fprintf(stderr, "Read %zu/%zu bytes\n", n, bgr_sz);
        free(bgr);
        return 1;
    }

    if (detect_set_model_size(model_size) != 0) {
        fprintf(stderr, "detect_set_model_size(%d) failed\n", model_size);
        free(bgr);
        return 1;
    }
    printf("[INIT] model=%s frame=%s cam=%dx%d model_size=%d loops=%d\n",
           argv[1], argv[2], cam_w, cam_h, model_size, loops);
    fflush(stdout);
    if (detect_init(argv[1], "/bsd_bench_shm", cam_w, cam_h, 0.5f, 0.45f) != 0) {
        fprintf(stderr, "detect_init failed\n");
        fflush(stderr);
        free(bgr);
        return 1;
    }
    printf("[INIT] detect_init ok\n");
    fflush(stdout);

    long long sum_pre = 0, sum_npu = 0, sum_dec = 0, sum_post = 0, sum_total = 0;
    long long min_npu = 0, max_npu = 0;
    int ok = 0;

    for (int i = 0; i < loops; i++) {
        if (detect_process_frame(bgr) != 0) {
            fprintf(stderr, "detect_process_frame failed at loop %d\n", i);
            break;
        }
        const DetectProfile *p = detect_get_last_profile();
        if (!p || !p->ok) continue;
        if (ok == 0 || p->npu_us < min_npu) min_npu = p->npu_us;
        if (ok == 0 || p->npu_us > max_npu) max_npu = p->npu_us;
        sum_pre += p->preprocess_us;
        sum_npu += p->npu_us;
        sum_dec += p->decode_us;
        sum_post += p->post_us;
        sum_total += p->total_us;
        ok++;
        if ((i + 1) % 10 == 0 || i == 0) {
            printf("[BENCH] loop=%d total=%lldms pre=%lldms npu=%lldms decode=%lldms post=%lldms\n",
                   i + 1,
                   prof_ms(p->total_us),
                   prof_ms(p->preprocess_us),
                   prof_ms(p->npu_us),
                   prof_ms(p->decode_us),
                   prof_ms(p->post_us));
            fflush(stdout);
        }
    }

    if (ok > 0) {
        printf("[SUMMARY] loops=%d avg_total=%.2fms avg_pre=%.2fms avg_npu=%.2fms "
               "avg_decode=%.2fms avg_post=%.2fms min_npu=%.2fms max_npu=%.2fms fps=%.2f\n",
               ok,
               (double)sum_total / ok / 1000.0,
               (double)sum_pre / ok / 1000.0,
               (double)sum_npu / ok / 1000.0,
               (double)sum_dec / ok / 1000.0,
               (double)sum_post / ok / 1000.0,
               (double)min_npu / 1000.0,
               (double)max_npu / 1000.0,
               1000000.0 * ok / (double)sum_total);
    }

    detect_deinit();
    free(bgr);
    return ok > 0 ? 0 : 1;
}
