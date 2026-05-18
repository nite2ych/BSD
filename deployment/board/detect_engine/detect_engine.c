// deployment/board/detect_engine/detect_engine.c
// Main detection engine: NPU init, frame loop, shared memory output
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include "detect_engine.h"

// External functions from sub-modules
extern void letterbox_preprocess(const uint8_t* src, int cam_w, int cam_h,
                                 uint8_t* rgb_planar_out,
                                 int* pad_x, int* pad_y, float* scale);
extern void letterbox_to_original(float* x1, float* y1, float* x2, float* y2,
                                   int pad_x, int pad_y, float scale);
extern int  yolo_decode(const float* boxes, const float* scores,
                        DetObject* det_out, int max_det, float conf_thr);
extern void yolo_nms(DetObject* dets, int count, float nms_thr);

// awNN NPU API (from libawnn_full.a)
// Real API reference: V5 test_yolo_input.c in DEPLOYMENT_GUIDE.md
extern void* awnn_init(const char* nb_path);
extern int   awnn_run(void* ctx, const uint8_t* input, float** boxes, float** scores);
extern void  awnn_destroy(void* ctx);
extern void  awnn_uninit(void);

// Default shared memory name
#define DEFAULT_SHM_NAME "/bsd_detect_shm"

// Engine state
static struct {
    void*     awnn_ctx;
    int       cam_w, cam_h;
    float     conf_thr, nms_thr;
    int       running;

    // Shared memory
    int        shm_fd;
    char       shm_name[64];
    DetResult* shm_result;

    // Preprocess buffer (640*640*3 bytes)
    uint8_t*  rgb_planar;
} g_detect;

int detect_init(const char* nb_path, const char* shm_name,
                int cam_width, int cam_height,
                float conf_thr, float nms_thr)
{
    memset(&g_detect, 0, sizeof(g_detect));

    g_detect.cam_w = cam_width;
    g_detect.cam_h = cam_height;
    g_detect.conf_thr = (conf_thr > 0.0f) ? conf_thr : 0.3f;
    g_detect.nms_thr = (nms_thr > 0.0f) ? nms_thr : 0.45f;

    // Store shm name
    if (shm_name) {
        strncpy(g_detect.shm_name, shm_name, sizeof(g_detect.shm_name) - 1);
    } else {
        strncpy(g_detect.shm_name, DEFAULT_SHM_NAME, sizeof(g_detect.shm_name) - 1);
    }

    // Allocate preprocess buffer
    g_detect.rgb_planar = (uint8_t*)malloc(640 * 640 * 3);
    if (!g_detect.rgb_planar) {
        fprintf(stderr, "detect_init: malloc failed\n");
        return -1;
    }

    // Initialize awNN NPU
    g_detect.awnn_ctx = awnn_init(nb_path);
    if (!g_detect.awnn_ctx) {
        fprintf(stderr, "detect_init: awnn_init failed\n");
        free(g_detect.rgb_planar);
        return -1;
    }

    // Create shared memory
    g_detect.shm_fd = shm_open(g_detect.shm_name, O_CREAT | O_RDWR, 0666);
    if (g_detect.shm_fd < 0) {
        fprintf(stderr, "detect_init: shm_open failed\n");
        awnn_destroy(g_detect.awnn_ctx);
        free(g_detect.rgb_planar);
        return -1;
    }
    if (ftruncate(g_detect.shm_fd, sizeof(DetResult)) != 0) {
        fprintf(stderr, "detect_init: ftruncate failed\n");
        close(g_detect.shm_fd);
        shm_unlink(g_detect.shm_name);
        awnn_destroy(g_detect.awnn_ctx);
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
        free(g_detect.rgb_planar);
        return -1;
    }
    memset(g_detect.shm_result, 0, sizeof(DetResult));

    return 0;
}

// Process one frame: capture → preprocess → NPU → decode → write shm
// frame_buf: raw camera frame (BGR interleaved, cam_w * cam_h * 3 bytes)
int detect_process_frame(void)
{
    // Camera capture — platform-specific, caller provides frame
    // For now: caller writes frame to a buffer set elsewhere
    // This function will be called per-frame in a loop

    // Placeholder: frame_buf should come from camera
    // For integration, caller sets frame via detect_set_frame() before calling this
    return 0;
}

// Process a pre-captured frame buffer
int detect_process_frame_buf(const uint8_t* frame_buf)
{
    if (!g_detect.awnn_ctx || !frame_buf) return -1;

    int pad_x, pad_y;
    float scale;
    letterbox_preprocess(frame_buf, g_detect.cam_w, g_detect.cam_h,
                         g_detect.rgb_planar, &pad_x, &pad_y, &scale);

    // NPU inference
    float* boxes;
    float* scores;
    if (awnn_run(g_detect.awnn_ctx, g_detect.rgb_planar, &boxes, &scores) != 0) {
        return -1;
    }

    // YOLO decode
    DetObject dets[MAX_DET];
    int n_det = yolo_decode(boxes, scores, dets, MAX_DET, g_detect.conf_thr);

    // NMS
    yolo_nms(dets, n_det, g_detect.nms_thr);

    // Coordinate remapping + write shared memory
    int valid = 0;
    for (int i = 0; i < n_det; i++) {
        if (dets[i].class_id < 0) break;  // NMS'ed out, rest are invalid

        // Map from letterbox (640x640 normalized) → original pixel coords
        dets[i].x1 = (dets[i].x1 * 640.0f - pad_x) / scale;
        dets[i].y1 = (dets[i].y1 * 640.0f - pad_y) / scale;
        dets[i].x2 = (dets[i].x2 * 640.0f - pad_x) / scale;
        dets[i].y2 = (dets[i].y2 * 640.0f - pad_y) / scale;

        // Clip to image bounds
        if (dets[i].x1 < 0.0f) dets[i].x1 = 0.0f;
        if (dets[i].y1 < 0.0f) dets[i].y1 = 0.0f;
        if (dets[i].x2 >= (float)g_detect.cam_w) dets[i].x2 = (float)(g_detect.cam_w - 1);
        if (dets[i].y2 >= (float)g_detect.cam_h) dets[i].y2 = (float)(g_detect.cam_h - 1);

        g_detect.shm_result->objects[valid++] = dets[i];
        if (valid >= MAX_DET) break;
    }

    g_detect.shm_result->frame_id++;
    g_detect.shm_result->timestamp_ms = 0;  // Set by real clock if available
    g_detect.shm_result->count = (uint32_t)valid;

    return 0;
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
