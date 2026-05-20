// deployment/board/alarm_engine/tracker.c
#include <string.h>
#include "../common/types.h"

// Tracked target
typedef struct {
    int32_t track_id;
    int32_t class_id;
    Box     bbox;
    float   conf;
    int32_t alive;           // active this frame
    int32_t zone_hit_level;  // current zone hit level (-1 = none)
    int32_t frame_count;     // consecutive frame hits in zone
    int32_t total_frames;    // total alive frames
} Track;

static Track g_tracks[MAX_TRACKS];
static int   g_next_track_id = 1;

// Calculate IOU between two boxes
static float box_iou(const Box* a, const Box* b)
{
    float ix1 = (a->x1 > b->x1) ? a->x1 : b->x1;
    float iy1 = (a->y1 > b->y1) ? a->y1 : b->y1;
    float ix2 = (a->x2 < b->x2) ? a->x2 : b->x2;
    float iy2 = (a->y2 < b->y2) ? a->y2 : b->y2;

    float iw = ix2 - ix1;
    float ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0.0f;

    float inter = iw * ih;
    float a_area = (a->x2 - a->x1) * (a->y2 - a->y1);
    float b_area = (b->x2 - b->x1) * (b->y2 - b->y1);
    float union_area = a_area + b_area - inter;
    if (union_area <= 0) return 0.0f;

    return inter / union_area;
}

// Initialize tracker
void tracker_init(void)
{
    memset(g_tracks, 0, sizeof(g_tracks));
    g_next_track_id = 1;
}

// Update tracks: associate detections with existing tracks, or create new tracks
//  dets:       current frame detections
//  n_det:      number of detections
//  zone_hits:  zone hit level per detection (-1 = no hit)
//  out_alarms: output alarm list
//  max_alarms: max alarms to output
//  frame_thr:  consecutive frame alarm threshold
// Returns: number of alarms emitted
int tracker_update(const DetObject* dets, int n_det,
                   const int* zone_hits, const float* overlaps,
                   AlarmEvent* out_alarms, int max_alarms,
                   int frame_thr)
{
    int alarm_count = 0;

    // 1. Mark all tracks as unmatched for this frame
    for (int t = 0; t < MAX_TRACKS; t++) {
        if (g_tracks[t].alive) {
            g_tracks[t].alive = 0;
        }
    }

    // 2. For each detection, find the best IOU match among existing tracks
    for (int d = 0; d < n_det; d++) {
        Box bbox = { dets[d].x1, dets[d].y1, dets[d].x2, dets[d].y2 };

        int best_t = -1;
        float best_iou = 0.3f;  // IOU match threshold
        for (int t = 0; t < MAX_TRACKS; t++) {
            if (g_tracks[t].track_id == 0)
                continue;  // empty slot
            if (g_tracks[t].class_id != dets[d].class_id)
                continue;
            float iou = box_iou(&bbox, &g_tracks[t].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_t = t;
            }
        }

        if (best_t >= 0) {
            // 3a. Matched: update existing track
            Track* trk = &g_tracks[best_t];
            trk->bbox = bbox;
            trk->conf = dets[d].conf;
            trk->alive = 1;
            trk->total_frames++;

            if (zone_hits[d] >= 0) {
                if (trk->zone_hit_level == zone_hits[d]) {
                    trk->frame_count++;
                } else {
                    trk->zone_hit_level = zone_hits[d];
                    trk->frame_count = 1;
                }
                // Alarm threshold reached
                if (trk->frame_count >= frame_thr && alarm_count < max_alarms) {
                    out_alarms[alarm_count].zone_level  = zone_hits[d];
                    out_alarms[alarm_count].class_id    = trk->class_id;
                    out_alarms[alarm_count].overlap     = overlaps[d];
                    out_alarms[alarm_count].frame_count = trk->frame_count;
                    out_alarms[alarm_count].x1 = trk->bbox.x1;
                    out_alarms[alarm_count].y1 = trk->bbox.y1;
                    out_alarms[alarm_count].x2 = trk->bbox.x2;
                    out_alarms[alarm_count].y2 = trk->bbox.y2;
                    alarm_count++;
                }
            } else {
                // Left zone, reset
                trk->zone_hit_level = -1;
                trk->frame_count = 0;
            }
        } else {
            // 3b. New target: create track
            for (int t = 0; t < MAX_TRACKS; t++) {
                if (g_tracks[t].track_id == 0) {
                    g_tracks[t].track_id = g_next_track_id++;
                    g_tracks[t].class_id = dets[d].class_id;
                    g_tracks[t].bbox     = bbox;
                    g_tracks[t].conf     = dets[d].conf;
                    g_tracks[t].alive    = 1;
                    g_tracks[t].total_frames = 1;
                    g_tracks[t].zone_hit_level = zone_hits[d];
                    g_tracks[t].frame_count = (zone_hits[d] >= 0) ? 1 : 0;
                    break;
                }
            }
        }
    }

    // 4. Clean up: unmatched tracks are released
    for (int t = 0; t < MAX_TRACKS; t++) {
        if (g_tracks[t].track_id != 0 && g_tracks[t].alive == 0) {
            g_tracks[t].track_id = 0;
        }
    }

    return alarm_count;
}
