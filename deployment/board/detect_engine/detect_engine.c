// deployment/board/detect_engine/detect_engine.c
// Main detection engine: NPU init, frame loop, shared memory output
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include "awnn.h"
#include "detect_engine.h"

// Sub-module functions
extern void letterbox_preprocess(const uint8_t* src, int cam_w, int cam_h,
                                 uint8_t* rgb_planar_out,
                                 int* pad_x, int* pad_y, float* scale);
extern void letterbox_preprocess_sized(const uint8_t* src, int cam_w, int cam_h,
                                       int model_size, uint8_t* rgb_planar_out,
                                       int* pad_x, int* pad_y, float* scale);
extern void letterbox_preprocess_nv21m(const uint8_t* y_plane, const uint8_t* vu_plane,
                                       int cam_w, int cam_h, int y_stride, int uv_stride,
                                       uint8_t* rgb_planar_out,
                                       int* pad_x, int* pad_y, float* scale);
extern void letterbox_preprocess_nv21m_sized(const uint8_t* y_plane, const uint8_t* vu_plane,
                                             int cam_w, int cam_h, int y_stride, int uv_stride,
                                             int model_size, uint8_t* rgb_planar_out,
                                             int* pad_x, int* pad_y, float* scale);
extern int  yolo_decode(const float* boxes, const float* scores,
                        DetObject* det_out, int max_det, const float class_thr[4]);
extern int  yolo_set_model_size(int model_size);
extern int  yolo_get_num_cells(void);
extern void yolo_nms(DetObject* dets, int count, float nms_thr);

#define DEFAULT_SHM_NAME "/bsd_detect_shm"

// Engine state
static struct {
    Awnn_Context_t* awnn_ctx;
    int       cam_w, cam_h;
    int       model_size;
    int       num_cells;
    float     conf_thr, nms_thr;
    float     class_conf_thr[4];  // per-class thresholds
    int       input_is_nv21m;
    int       nv21_y_stride;
    int       nv21_uv_stride;

    // Shared memory
    int        shm_fd;
    char       shm_name[64];
    DetResult* shm_result;

    // Preprocess buffer (model_size*model_size*3 bytes RGB planar)
    uint8_t*  rgb_planar;

    DetectProfile last_profile;
} g_detect;

static int g_requested_model_size = 640;

static long long detect_now_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + tv.tv_usec;
}

int detect_set_model_size(int model_size)
{
    if (model_size != 320 && model_size != 416 &&
        model_size != 512 && model_size != 640) {
        return -1;
    }
    if (yolo_set_model_size(model_size) != 0) return -1;
    g_requested_model_size = model_size;
    g_detect.model_size = model_size;
    g_detect.num_cells = yolo_get_num_cells();
    return 0;
}

int detect_init(const char* nb_path, const char* shm_name,
                int cam_width, int cam_height,
                float conf_thr, float nms_thr)
{
    int model_size = g_requested_model_size;
    memset(&g_detect, 0, sizeof(g_detect));

    g_detect.cam_w = cam_width;
    g_detect.cam_h = cam_height;
    detect_set_model_size(model_size);
    g_detect.conf_thr = (conf_thr > 0.0f) ? conf_thr : 0.3f;
    g_detect.nms_thr = (nms_thr > 0.0f) ? nms_thr : 0.45f;
    for (int i = 0; i < 4; i++) g_detect.class_conf_thr[i] = g_detect.conf_thr;
    g_detect.input_is_nv21m = 0;
    g_detect.nv21_y_stride = cam_width;
    g_detect.nv21_uv_stride = cam_width;

    if (shm_name) {
        strncpy(g_detect.shm_name, shm_name, sizeof(g_detect.shm_name) - 1);
    } else {
        strncpy(g_detect.shm_name, DEFAULT_SHM_NAME, sizeof(g_detect.shm_name) - 1);
    }

    // Allocate preprocess buffer (RGB planar model_size x model_size x 3)
    g_detect.rgb_planar = (uint8_t*)malloc((size_t)g_detect.model_size * g_detect.model_size * 3);
    if (!g_detect.rgb_planar) {
        fprintf(stderr, "detect_init: malloc rgb_planar failed\n");
        return -1;
    }

    // Initialize NPU and load model
    awnn_init(16 * 1024 * 1024);
    g_detect.awnn_ctx = awnn_create((char*)nb_path);
    if (!g_detect.awnn_ctx) {
        fprintf(stderr, "detect_init: awnn_create failed\n");
        free(g_detect.rgb_planar);
        awnn_uninit();
        return -1;
    }

    // Create shared memory
    g_detect.shm_fd = shm_open(g_detect.shm_name, O_CREAT | O_RDWR, 0666);
    if (g_detect.shm_fd < 0) {
        fprintf(stderr, "detect_init: shm_open failed\n");
        awnn_destroy(g_detect.awnn_ctx);
        awnn_uninit();
        free(g_detect.rgb_planar);
        return -1;
    }
    if (ftruncate(g_detect.shm_fd, sizeof(DetResult)) != 0) {
        fprintf(stderr, "detect_init: ftruncate failed\n");
        close(g_detect.shm_fd);
        shm_unlink(g_detect.shm_name);
        awnn_destroy(g_detect.awnn_ctx);
        awnn_uninit();
        free(g_detect.rgb_planar);
        return -1;
    }
    g_detect.shm_result = (DetResult*)mmap(NULL, sizeof(DetResult),
                                           PROT_READ | PROT_WRITE,
                                           MAP_SHARED, g_detect.shm_fd, 0);
    if (g_detect.shm_result == MAP_FAILED) {
        fprintf(stderr, "detect_init: mmap failed\n");
        close(g_detect.shm_fd);
        shm_unlink(g_detect.shm_name);
        awnn_destroy(g_detect.awnn_ctx);
        awnn_uninit();
        free(g_detect.rgb_planar);
        return -1;
    }
    memset(g_detect.shm_result, 0, sizeof(DetResult));

    return 0;
}

void detect_set_nv21_stride(int y_stride, int uv_stride)
{
    g_detect.input_is_nv21m = 1;
    g_detect.nv21_y_stride = (y_stride > 0) ? y_stride : g_detect.cam_w;
    g_detect.nv21_uv_stride = (uv_stride > 0) ? uv_stride : g_detect.cam_w;
}

// Process one NV21M frame: preprocess → NPU → decode → NMS → shm
int detect_process_nv21m(const uint8_t* y_plane, const uint8_t* vu_plane)
{
    if (!g_detect.awnn_ctx || !y_plane || !vu_plane) return -1;

    DetectProfile prof;
    memset(&prof, 0, sizeof(prof));
    prof.seq = g_detect.last_profile.seq + 1;
    long long total_t0 = detect_now_us();

    // Step 1: BGR→RGB planar letterbox
    int pad_x, pad_y;
    float scale;
    long long t0 = detect_now_us();
    letterbox_preprocess_nv21m_sized(y_plane, vu_plane, g_detect.cam_w, g_detect.cam_h,
                                     g_detect.nv21_y_stride > 0 ? g_detect.nv21_y_stride : g_detect.cam_w,
                                     g_detect.nv21_uv_stride > 0 ? g_detect.nv21_uv_stride : g_detect.cam_w,
                                     g_detect.model_size,
                                     g_detect.rgb_planar, &pad_x, &pad_y, &scale);
    long long t1 = detect_now_us();
    awnn_set_input_buffers(g_detect.awnn_ctx, &g_detect.rgb_planar);
    awnn_run(g_detect.awnn_ctx);
    float** outputs = awnn_get_output_buffers(g_detect.awnn_ctx);
    long long t2 = detect_now_us();
    if (!outputs) {
        prof.preprocess_us = t1 - t0;
        prof.npu_us = t2 - t1;
        prof.total_us = t2 - total_t0;
        g_detect.last_profile = prof;
        return -1;
    }

    float* boxes  = outputs[0];
    float* scores = outputs[1];

    DetObject dets[MAX_DET];
    int n_det = yolo_decode(boxes, scores, dets, MAX_DET, g_detect.class_conf_thr);
    yolo_nms(dets, n_det, g_detect.nms_thr);
    long long t3 = detect_now_us();

    int valid = 0;
    for (int i = 0; i < n_det; i++) {
        if (dets[i].class_id < 0) break;
        dets[i].x1 = (dets[i].x1 * (float)g_detect.model_size - pad_x) / scale;
        dets[i].y1 = (dets[i].y1 * (float)g_detect.model_size - pad_y) / scale;
        dets[i].x2 = (dets[i].x2 * (float)g_detect.model_size - pad_x) / scale;
        dets[i].y2 = (dets[i].y2 * (float)g_detect.model_size - pad_y) / scale;

        if (dets[i].x1 < 0.0f) dets[i].x1 = 0.0f;
        if (dets[i].x1 >= (float)g_detect.cam_w) dets[i].x1 = (float)(g_detect.cam_w - 1);
        if (dets[i].y1 < 0.0f) dets[i].y1 = 0.0f;
        if (dets[i].y1 >= (float)g_detect.cam_h) dets[i].y1 = (float)(g_detect.cam_h - 1);
        if (dets[i].x2 < 0.0f) dets[i].x2 = 0.0f;
        if (dets[i].x2 >= (float)g_detect.cam_w) dets[i].x2 = (float)(g_detect.cam_w - 1);
        if (dets[i].y2 < 0.0f) dets[i].y2 = 0.0f;
        if (dets[i].y2 >= (float)g_detect.cam_h) dets[i].y2 = (float)(g_detect.cam_h - 1);
        if (dets[i].x2 <= dets[i].x1 || dets[i].y2 <= dets[i].y1) continue;

        float box_w = dets[i].x2 - dets[i].x1;
        float box_h = dets[i].y2 - dets[i].y1;
        if (box_w > (float)g_detect.cam_w * 0.95f &&
            box_h > (float)g_detect.cam_h * 0.95f) continue;

        g_detect.shm_result->objects[valid++] = dets[i];
        if (valid >= MAX_DET) break;
    }

    g_detect.shm_result->frame_id++;
    g_detect.shm_result->count = (uint32_t)valid;
    long long t4 = detect_now_us();

    prof.ok = 1;
    prof.preprocess_us = t1 - t0;
    prof.npu_us = t2 - t1;
    prof.decode_us = t3 - t2;
    prof.post_us = t4 - t3;
    prof.total_us = t4 - total_t0;
    g_detect.last_profile = prof;
    return 0;
}

// Process one camera frame: preprocess → NPU → decode → NMS → shm
// frame_buf: BGR interleaved, cam_w * cam_h * 3 bytes
int detect_process_frame(const uint8_t* frame_buf)
{
    if (!g_detect.awnn_ctx || !frame_buf) return -1;

    DetectProfile prof;
    memset(&prof, 0, sizeof(prof));
    prof.seq = g_detect.last_profile.seq + 1;
    long long total_t0 = detect_now_us();

    // Step 1: BGR→RGB planar letterbox
    int pad_x, pad_y;
    float scale;
    long long t0 = detect_now_us();
    letterbox_preprocess_sized(frame_buf, g_detect.cam_w, g_detect.cam_h,
                               g_detect.model_size,
                               g_detect.rgb_planar, &pad_x, &pad_y, &scale);
    long long t1 = detect_now_us();

    // Step 2: NPU inference
    awnn_set_input_buffers(g_detect.awnn_ctx, &g_detect.rgb_planar);
    awnn_run(g_detect.awnn_ctx);
    float** outputs = awnn_get_output_buffers(g_detect.awnn_ctx);
    long long t2 = detect_now_us();
    if (!outputs) {
        prof.preprocess_us = t1 - t0;
        prof.npu_us = t2 - t1;
        prof.total_us = t2 - total_t0;
        g_detect.last_profile = prof;
        return -1;
    }

    float* boxes  = outputs[0];  // [4, num_cells] layout
    float* scores = outputs[1];  // [4, num_cells] layout

    // Debug: max scores + box stats (per-head analysis)
    {
        static int fcnt = 0;
        fcnt++;
        if (0 && fcnt % 30 == 0) {  // DISABLED per-head debug (slow)
            // Per-head best: [score, cell, class] for each head
            float best_score_h[3] = {-999, -999, -999};
            int   best_cell_h[3]  = {0, 0, 0};
            int   best_cls_h[3]   = {0, 0, 0};
            float best_box_h[3][4];

            int grid0 = g_detect.model_size / 8;
            int grid1 = g_detect.model_size / 16;
            int grid2 = g_detect.model_size / 32;
            int cells0 = grid0 * grid0;
            int cells1 = grid1 * grid1;
            int num_cells = g_detect.num_cells;

            // NCHW layout: scores[c][i] = scores[c * num_cells + i]
            for (int i = 0; i < num_cells; i++) {
                int h = (i < cells0) ? 0 : (i < cells0 + cells1) ? 1 : 2;
                float best_logit = scores[0 * num_cells + i];
                int best_c = 0;
                for (int c = 1; c < 4; c++) {
                    float s = scores[c * num_cells + i];
                    if (s > best_logit) { best_logit = s; best_c = c; }
                }
                if (best_logit > best_score_h[h]) {
                    best_score_h[h] = best_logit;
                    best_cell_h[h] = i;
                    best_cls_h[h] = best_c;
                    best_box_h[h][0] = boxes[0 * num_cells + i];
                    best_box_h[h][1] = boxes[1 * num_cells + i];
                    best_box_h[h][2] = boxes[2 * num_cells + i];
                    best_box_h[h][3] = boxes[3 * num_cells + i];
                }
            }

            const char *names[] = {"person","bicycle","motorcycle","vehicle"};
            printf("[NPU] === Per-head best scores ===\n");
            for (int h = 0; h < 3; h++) {
                float conf = 1.0f / (1.0f + expf(-best_score_h[h]));
                int stride = (h == 0) ? 8 : (h == 1) ? 16 : 32;
                int gx, gy;
                if (h == 0) { gx = best_cell_h[h] % grid0; gy = best_cell_h[h] / grid0; }
                else if (h == 1) { int j = best_cell_h[h] - cells0; gx = j % grid1; gy = j / grid1; }
                else { int j = best_cell_h[h] - cells0 - cells1; gx = j % grid2; gy = j / grid2; }
                printf("  H%d(s=%d): cell=%d g=(%d,%d) cls=%s logit=%.2f conf=%.3f box=[%.1f %.1f %.1f %.1f]\n",
                       h, stride, best_cell_h[h], gx, gy,
                       (best_cls_h[h] < 4) ? names[best_cls_h[h]] : "?",
                       best_score_h[h], conf,
                       best_box_h[h][0], best_box_h[h][1],
                       best_box_h[h][2], best_box_h[h][3]);
            }
        }
    }

    // Step 3: YOLO decode
    DetObject dets[MAX_DET];
    int n_det = yolo_decode(boxes, scores, dets, MAX_DET, g_detect.class_conf_thr);

    // Step 4: NMS
    yolo_nms(dets, n_det, g_detect.nms_thr);
    long long t3 = detect_now_us();

    // Step 5: Coordinate remapping + write shared memory
    int valid = 0;
    for (int i = 0; i < n_det; i++) {
        if (dets[i].class_id < 0) break;

        // Map from letterbox normalized coords → original pixel coords
        dets[i].x1 = (dets[i].x1 * (float)g_detect.model_size - pad_x) / scale;
        dets[i].y1 = (dets[i].y1 * (float)g_detect.model_size - pad_y) / scale;
        dets[i].x2 = (dets[i].x2 * (float)g_detect.model_size - pad_x) / scale;
        dets[i].y2 = (dets[i].y2 * (float)g_detect.model_size - pad_y) / scale;

        // Clip to image bounds [0, cam_size-1]
        if (dets[i].x1 < 0.0f) dets[i].x1 = 0.0f;
        if (dets[i].x1 >= (float)g_detect.cam_w) dets[i].x1 = (float)(g_detect.cam_w - 1);
        if (dets[i].y1 < 0.0f) dets[i].y1 = 0.0f;
        if (dets[i].y1 >= (float)g_detect.cam_h) dets[i].y1 = (float)(g_detect.cam_h - 1);
        if (dets[i].x2 < 0.0f) dets[i].x2 = 0.0f;
        if (dets[i].x2 >= (float)g_detect.cam_w) dets[i].x2 = (float)(g_detect.cam_w - 1);
        if (dets[i].y2 < 0.0f) dets[i].y2 = 0.0f;
        if (dets[i].y2 >= (float)g_detect.cam_h) dets[i].y2 = (float)(g_detect.cam_h - 1);

        // Skip degenerate boxes
        if (dets[i].x2 <= dets[i].x1 || dets[i].y2 <= dets[i].y1) continue;

        // Skip full-frame false positives. Quantized output can produce a box
        // that covers nearly the whole camera frame without touching every edge.
        float box_w = dets[i].x2 - dets[i].x1;
        float box_h = dets[i].y2 - dets[i].y1;
        if (box_w > (float)g_detect.cam_w * 0.95f &&
            box_h > (float)g_detect.cam_h * 0.95f) continue;

        g_detect.shm_result->objects[valid++] = dets[i];
        if (valid >= MAX_DET) break;
    }

    g_detect.shm_result->frame_id++;
    g_detect.shm_result->count = (uint32_t)valid;
    long long t4 = detect_now_us();

    prof.ok = 1;
    prof.preprocess_us = t1 - t0;
    prof.npu_us = t2 - t1;
    prof.decode_us = t3 - t2;
    prof.post_us = t4 - t3;
    prof.total_us = t4 - total_t0;
    g_detect.last_profile = prof;

    return 0;
}

const DetResult* detect_get_result(void)
{
    return g_detect.shm_result;
}

const DetectProfile* detect_get_last_profile(void)
{
    return &g_detect.last_profile;
}

void detect_set_class_threshold(int class_id, float thr)
{
    if (class_id >= 0 && class_id < 4 && thr > 0.0f) {
        g_detect.class_conf_thr[class_id] = thr;
    }
}

void detect_deinit(void)
{
    if (g_detect.shm_result && g_detect.shm_result != MAP_FAILED) {
        munmap(g_detect.shm_result, sizeof(DetResult));
    }
    if (g_detect.shm_fd > 0) {
        close(g_detect.shm_fd);
        shm_unlink(g_detect.shm_name);
    }
    if (g_detect.awnn_ctx) {
        awnn_destroy(g_detect.awnn_ctx);
        awnn_uninit();
    }
    free(g_detect.rgb_planar);
    memset(&g_detect, 0, sizeof(g_detect));
}
