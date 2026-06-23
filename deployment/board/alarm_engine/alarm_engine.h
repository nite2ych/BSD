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

// Set tracker reliability parameters.
//  max_missed: keep a track for this many missed frames (default 3)
//  cooldown_frames: suppress repeated alarms after a track leaves zone (default 15)
//  reemit_frames: repeat active alarm every N frames, 0 disables repeat (default 15)
//  match_iou_thr: IOU threshold for track association (default 0.30)
//  smooth_alpha: new-box weight for bbox smoothing, 0..1 (default 0.65)
int  alarm_set_tracker_params(int max_missed, int cooldown_frames,
                              int reemit_frames, float match_iou_thr,
                              float smooth_alpha);

// Enable/disable alarm participation per class.
// Defaults preserve the historical behavior: person/bicycle/motorcycle enabled,
// vehicle disabled. live_bsd can explicitly enable vehicle for BSD validation.
int  alarm_set_class_enabled(int class_id, int enabled);

// Register alarm callback
int  alarm_register_callback(alarm_callback_t cb);

// Start/stop alarm detection loop
int  alarm_start(void);
int  alarm_stop(void);

// Destroy
void alarm_deinit(void);

#endif // ALARM_ENGINE_H
