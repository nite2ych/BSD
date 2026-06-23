// deployment/board/detect_engine/detect_engine.h
#ifndef DETECT_ENGINE_H
#define DETECT_ENGINE_H

#include "../common/types.h"

typedef struct {
    unsigned int seq;
    int ok;
    long long preprocess_us;
    long long npu_us;
    long long decode_us;
    long long post_us;
    long long total_us;
} DetectProfile;

// Configure model input size before detect_init(). Supported: 320, 416, 512, 640.
int detect_set_model_size(int model_size);

// Init NPU and shared memory. Returns 0 on success.
int  detect_init(const char* nb_path, const char* shm_name,
                 int cam_width, int cam_height,
                 float conf_thr, float nms_thr);

// Configure NV21M strides for the direct NV21 input path.
void detect_set_nv21_stride(int y_stride, int uv_stride);

// Process one NV21M frame (Y plane + VU plane, both using the camera stride).
int  detect_process_nv21m(const uint8_t* y_plane, const uint8_t* vu_plane);

// Process one camera frame (BGR interleaved, cam_w*cam_h*3 bytes).
// Runs: preprocess → NPU → decode → NMS → shm.
int  detect_process_frame(const uint8_t* frame_buf);

// Get pointer to shared memory result (for external readout).
const DetResult* detect_get_result(void);

// Get profiling data for the most recent detect_process_* call.
const DetectProfile* detect_get_last_profile(void);

// Set per-class confidence threshold (0=person, 1=bicycle, 2=motorcycle, 3=vehicle)
void detect_set_class_threshold(int class_id, float thr);

// Cleanup
void detect_deinit(void);

#endif
