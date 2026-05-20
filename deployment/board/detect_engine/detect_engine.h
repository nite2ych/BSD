// deployment/board/detect_engine/detect_engine.h
#ifndef DETECT_ENGINE_H
#define DETECT_ENGINE_H

#include "../common/types.h"

// Init NPU and shared memory. Returns 0 on success.
int  detect_init(const char* nb_path, const char* shm_name,
                 int cam_width, int cam_height,
                 float conf_thr, float nms_thr);

// Process one camera frame (BGR interleaved, cam_w*cam_h*3 bytes).
// Runs: preprocess → NPU → decode → NMS → shm.
int  detect_process_frame(const uint8_t* frame_buf);

// Get pointer to shared memory result (for external readout).
const DetResult* detect_get_result(void);

// Cleanup
void detect_deinit(void);

#endif
