// deployment/board/alarm_engine/tracker.c
#include <string.h>
#include "../common/types.h"

// Tracked target
typedef struct {
    int32_t track_id;
    int32_t class_id;
    Box     bbox;
    float   conf;
    int32_t matched;         // matched by a detection this frame
    int32_t miss_count;      // consecutive missed frames
    int32_t zone_hit_level;  // current zone hit level (-1 = none)
    int32_t frame_count;     // consecutive frame hits in zone
    int32_t total_frames;    // total alive frames
    int32_t alarm_active;    // this track has already emitted active alarm
    int32_t last_alarm_frame_count;
    int32_t cooldown_count;  // suppress alarm after leaving zone
    float   last_overlap;
} Track;

static Track g_tracks[MAX_TRACKS];
static int   g_next_track_id = 1;

static int   g_max_missed = 3;
static int   g_cooldown_frames = 15;
static int   g_reemit_frames = 15;
static float g_match_iou_thr = 0.30f;
static float g_smooth_alpha = 0.65f;

static float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

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

int tracker_set_params(int max_missed, int cooldown_frames,
                       int reemit_frames, float match_iou_thr,
                       float smooth_alpha)
{
    if (max_missed < 0 || cooldown_frames < 0 || reemit_frames < 0) return -1;
    if (match_iou_thr <= 0.0f || match_iou_thr > 1.0f) return -1;
    if (smooth_alpha <= 0.0f || smooth_alpha > 1.0f) return -1;
    g_max_missed = max_missed;
    g_cooldown_frames = cooldown_frames;
    g_reemit_frames = reemit_frames;
    g_match_iou_thr = match_iou_thr;
    g_smooth_alpha = smooth_alpha;
    return 0;
}

static void smooth_box(Box* dst, const Box* src)
{
    float a = clampf(g_smooth_alpha, 0.0f, 1.0f);
    dst->x1 = dst->x1 * (1.0f - a) + src->x1 * a;
    dst->y1 = dst->y1 * (1.0f - a) + src->y1 * a;
    dst->x2 = dst->x2 * (1.0f - a) + src->x2 * a;
    dst->y2 = dst->y2 * (1.0f - a) + src->y2 * a;
}

static int should_emit_alarm(const Track* trk, int frame_thr)
{
    if (trk->frame_count < frame_thr) return 0;
    if (trk->cooldown_count > 0) return 0;
    if (!trk->alarm_active) return 1;
    if (g_reemit_frames <= 0) return 0;
    return (trk->frame_count - trk->last_alarm_frame_count) >= g_reemit_frames;
}

static void fill_alarm(const Track* trk, AlarmEvent* evt)
{
    evt->zone_level  = trk->zone_hit_level;
    evt->class_id    = trk->class_id;
    evt->overlap     = trk->last_overlap;
    evt->frame_count = trk->frame_count;
    evt->x1 = trk->bbox.x1;
    evt->y1 = trk->bbox.y1;
    evt->x2 = trk->bbox.x2;
    evt->y2 = trk->bbox.y2;
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

    // 1. Mark all tracks as unmatched for this frame.
    for (int t = 0; t < MAX_TRACKS; t++) {
        if (g_tracks[t].track_id != 0) {
            g_tracks[t].matched = 0;
            if (g_tracks[t].cooldown_count > 0) {
                g_tracks[t].cooldown_count--;
            }
        }
    }

    // 2. For each detection, find the best IOU match among existing tracks
    for (int d = 0; d < n_det; d++) {
        Box bbox = { dets[d].x1, dets[d].y1, dets[d].x2, dets[d].y2 };

        int best_t = -1;
        float best_iou = g_match_iou_thr;
        for (int t = 0; t < MAX_TRACKS; t++) {
            if (g_tracks[t].track_id == 0)
                continue;  // empty slot
            if (g_tracks[t].matched)
                continue;
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
            smooth_box(&trk->bbox, &bbox);
            trk->conf = dets[d].conf;
            trk->matched = 1;
            trk->miss_count = 0;
            trk->total_frames++;

            if (zone_hits[d] >= 0) {
                if (trk->zone_hit_level == zone_hits[d]) {
                    trk->frame_count++;
                } else {
                    trk->zone_hit_level = zone_hits[d];
                    trk->frame_count = 1;
                    trk->alarm_active = 0;
                    trk->last_alarm_frame_count = 0;
                }
                trk->last_overlap = overlaps[d];
                if (should_emit_alarm(trk, frame_thr) && alarm_count < max_alarms) {
                    fill_alarm(trk, &out_alarms[alarm_count]);
                    trk->alarm_active = 1;
                    trk->last_alarm_frame_count = trk->frame_count;
                    alarm_count++;
                }
            } else {
                // Left zone, reset
                trk->zone_hit_level = -1;
                trk->frame_count = 0;
                trk->alarm_active = 0;
                trk->last_alarm_frame_count = 0;
                trk->cooldown_count = g_cooldown_frames;
                trk->last_overlap = 0.0f;
            }
        } else {
            // 3b. New target: create track
            for (int t = 0; t < MAX_TRACKS; t++) {
                if (g_tracks[t].track_id == 0) {
                    g_tracks[t].track_id = g_next_track_id++;
                    g_tracks[t].class_id = dets[d].class_id;
                    g_tracks[t].bbox     = bbox;
                    g_tracks[t].conf     = dets[d].conf;
                    g_tracks[t].matched  = 1;
                    g_tracks[t].miss_count = 0;
                    g_tracks[t].total_frames = 1;
                    g_tracks[t].zone_hit_level = zone_hits[d];
                    g_tracks[t].frame_count = (zone_hits[d] >= 0) ? 1 : 0;
                    g_tracks[t].alarm_active = 0;
                    g_tracks[t].last_alarm_frame_count = 0;
                    g_tracks[t].cooldown_count = 0;
                    g_tracks[t].last_overlap = (zone_hits[d] >= 0) ? overlaps[d] : 0.0f;
                    break;
                }
            }
        }
    }

    // 4. Keep unmatched tracks briefly so one or two dropped detections do not
    // break a pending alarm. Release only after max_missed frames.
    for (int t = 0; t < MAX_TRACKS; t++) {
        Track* trk = &g_tracks[t];
        if (trk->track_id != 0 && !trk->matched) {
            trk->miss_count++;
            if (trk->miss_count > g_max_missed) {
                memset(trk, 0, sizeof(*trk));
            }
        }
    }

    return alarm_count;
}
