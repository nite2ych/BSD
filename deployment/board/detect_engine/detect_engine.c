// deployment/board/detect_engine/detect_engine.c
// Main detection engine: NPU init, frame loop, shared memory output
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "awnn.h"
#include "detect_engine.h"

// Sub-module functions
extern void letterbox_preprocess(const uint8_t* src, int cam_w, int cam_h,
                                 uint8_t* rgb_planar_out,
                                 int* pad_x, int* pad_y, float* scale);
extern int  yolo_decode(const float* boxes, const float* scores,
                        DetObject* det_out, int max_det, float conf_thr);
extern void yolo_nms(DetObject* dets, int count, float nms_thr);

#define DEFAULT_SHM_NAME "/bsd_detect_shm"

// Engine state
static struct {
    Awnn_Context_t* awnn_ctx;
    int       cam_w, cam_h;
    float     conf_thr, nms_thr;

    // Shared memory
    int        shm_fd;
    char       shm_name[64];
    DetResult* shm_result;

    // Preprocess buffer (640*640*3 bytes RGB planar)
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

    if (shm_name) {
        strncpy(g_detect.shm_name, shm_name, sizeof(g_detect.shm_name) - 1);
    } else {
        strncpy(g_detect.shm_name, DEFAULT_SHM_NAME, sizeof(g_detect.shm_name) - 1);
    }

    // Allocate preprocess buffer (RGB planar 640x640x3)
    g_detect.rgb_planar = (uint8_t*)malloc(640 * 640 * 3);
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

// Process one camera frame: preprocess → NPU → decode → NMS → shm
// frame_buf: BGR interleaved, cam_w * cam_h * 3 bytes
int detect_process_frame(const uint8_t* frame_buf)
{
    if (!g_detect.awnn_ctx || !frame_buf) return -1;

    // Step 1: BGR→RGB planar letterbox
    int pad_x, pad_y;
    float scale;
    letterbox_preprocess(frame_buf, g_detect.cam_w, g_detect.cam_h,
                         g_detect.rgb_planar, &pad_x, &pad_y, &scale);

    // Step 2: NPU inference
    awnn_set_input_buffers(g_detect.awnn_ctx, &g_detect.rgb_planar);
    awnn_run(g_detect.awnn_ctx);
    float** outputs = awnn_get_output_buffers(g_detect.awnn_ctx);
    if (!outputs) return -1;

    float* boxes  = outputs[0];  // [4 * 8400]
    float* scores = outputs[1];  // [4 * 8400] raw logits

    // Step 3: YOLO decode
    DetObject dets[MAX_DET];
    int n_det = yolo_decode(boxes, scores, dets, MAX_DET, g_detect.conf_thr);

    // Step 4: NMS
    yolo_nms(dets, n_det, g_detect.nms_thr);

    // Step 5: Coordinate remapping + write shared memory
    int valid = 0;
    for (int i = 0; i < n_det; i++) {
        if (dets[i].class_id < 0) break;

        // Map from letterbox normalized coords → original pixel coords
        dets[i].x1 = (dets[i].x1 * 640.0f - pad_x) / scale;
        dets[i].y1 = (dets[i].y1 * 640.0f - pad_y) / scale;
        dets[i].x2 = (dets[i].x2 * 640.0f - pad_x) / scale;
        dets[i].y2 = (dets[i].y2 * 640.0f - pad_y) / scale;

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

        g_detect.shm_result->objects[valid++] = dets[i];
        if (valid >= MAX_DET) break;
    }

    g_detect.shm_result->frame_id++;
    g_detect.shm_result->count = (uint32_t)valid;

    return 0;
}

const DetResult* detect_get_result(void)
{
    return g_detect.shm_result;
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
