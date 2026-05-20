// deployment/board/alarm_engine/alarm_engine.h
#ifndef ALARM_ENGINE_H
#define ALARM_ENGINE_H

#include "../common/types.h"

// Initialize alarm engine
//  shm_name: shared memory name (must match detect_init)
int  alarm_init(const char* shm_name);

// Set alarm zone (rectangle, normalized coordinates 0~1)
//  level: 0/1/2 (ZONE_LEVEL_1/2/3)
//  overlap_thr: overlap ratio threshold (default 0.3)
int  alarm_set_zone(int level, float x1, float y1, float x2, float y2,
                    float overlap_thr);

// Set consecutive frame alarm threshold (default 10 frames)
int  alarm_set_frame_threshold(int n);

// Register alarm callback
int  alarm_register_callback(alarm_callback_t cb);

// Start/stop alarm detection loop
int  alarm_start(void);
int  alarm_stop(void);

// Destroy
void alarm_deinit(void);

#endif // ALARM_ENGINE_H
