#ifndef DETECT_ENGINE_H
#define DETECT_ENGINE_H

#include "../common/types.h"

int  detect_init(const char* nb_path, const char* shm_name,
                 int cam_width, int cam_height,
                 float conf_thr, float nms_thr);
int  detect_process_frame(void);
void detect_deinit(void);

#endif // DETECT_ENGINE_H
