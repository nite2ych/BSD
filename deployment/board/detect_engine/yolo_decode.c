// deployment/board/detect_engine/yolo_decode.c
// YOLO output decoder: logit→sigmoid, decode boxes, NMS
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../common/types.h"

#define NUM_CLASSES 4

static int g_model_size = 640;
static int g_grid0 = 80;
static int g_grid1 = 40;
static int g_grid2 = 20;
static int g_cells0 = 6400;
static int g_cells1 = 1600;
static int g_cells2 = 400;
static int g_num_cells = 8400;

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

int yolo_set_model_size(int model_size)
{
    if (model_size != 320 && model_size != 416 &&
        model_size != 512 && model_size != 640) {
        return -1;
    }
    if ((model_size % 32) != 0) return -1;

    g_model_size = model_size;
    g_grid0 = model_size / 8;
    g_grid1 = model_size / 16;
    g_grid2 = model_size / 32;
    g_cells0 = g_grid0 * g_grid0;
    g_cells1 = g_grid1 * g_grid1;
    g_cells2 = g_grid2 * g_grid2;
    g_num_cells = g_cells0 + g_cells1 + g_cells2;
    return 0;
}

int yolo_get_num_cells(void)
{
    return g_num_cells;
}

// YOLO11s reg_max=1 decode. The model outputs raw logits for both boxes
// and scores (no sigmoid in graph). Box values are LTRB distances in GRID
// CELL units (not pixels), directly usable without sigmoid.
//
//   head 0: 80×80 = 6400 cells, stride=8
//   head 1: 40×40 = 1600 cells, stride=16
//   head 2: 20×20 =  400 cells, stride=32
//
// NPU output layout is NCHW [4, NUM_CELLS] (preserving ONNX layout):
//   boxes[c][i]  = boxes[c * NUM_CELLS + i]   (c=0:left, 1:top, 2:right, 3:bottom)
//   scores[c][i] = scores[c * NUM_CELLS + i]  (c=0..3 class logits)
// det_out: output array (caller-allocated, max_det entries)
// class_thr[4]: per-class confidence thresholds (person, bicycle, motorcycle, vehicle)
int yolo_decode(const float* boxes, const float* scores,
                DetObject* det_out, int max_det, const float class_thr[4])
{
    int count = 0;
    for (int i = 0; i < g_num_cells && count < max_det; i++) {
        // NCHW: class c score for cell i is at scores[c * num_cells + i]
        int best_cls = 0;
        float best_logit = scores[0 * g_num_cells + i];
        if (scores[1 * g_num_cells + i] > best_logit) { best_logit = scores[1 * g_num_cells + i]; best_cls = 1; }
        if (scores[2 * g_num_cells + i] > best_logit) { best_logit = scores[2 * g_num_cells + i]; best_cls = 2; }
        if (scores[3 * g_num_cells + i] > best_logit) { best_logit = scores[3 * g_num_cells + i]; best_cls = 3; }

        float conf = sigmoid(best_logit);

        if (conf < class_thr[best_cls]) continue;

        // Determine grid position and stride
        int gx, gy, stride;
        if (i < g_cells0) {
            gx = i % g_grid0;  gy = i / g_grid0;  stride = 8;
        } else if (i < g_cells0 + g_cells1) {
            int j = i - g_cells0;
            gx = j % g_grid1;  gy = j / g_grid1;  stride = 16;
        } else {
            int j = i - g_cells0 - g_cells1;
            gx = j % g_grid2;  gy = j / g_grid2;  stride = 32;
        }

        // NCHW: box coord c for cell i is at boxes[c * num_cells + i]
        float left   = boxes[0 * g_num_cells + i];
        float top    = boxes[1 * g_num_cells + i];
        float right  = boxes[2 * g_num_cells + i];
        float bottom = boxes[3 * g_num_cells + i];

        // Anchor point in grid-cell coordinates
        float anchor_x = (float)gx + 0.5f;
        float anchor_y = (float)gy + 0.5f;

        // dist2bbox: anchor +/- distance -> xyxy in grid units, then normalize.
        float x1 = (anchor_x - left)   * (float)stride / (float)g_model_size;
        float y1 = (anchor_y - top)    * (float)stride / (float)g_model_size;
        float x2 = (anchor_x + right)  * (float)stride / (float)g_model_size;
        float y2 = (anchor_y + bottom) * (float)stride / (float)g_model_size;

        det_out[count].x1 = x1;
        det_out[count].y1 = y1;
        det_out[count].x2 = x2;
        det_out[count].y2 = y2;
        det_out[count].conf = conf;
        det_out[count].class_id = best_cls;

        count++;
    }
    return count;
}

// Simple NMS: sort by conf, greedy suppress by IOU.
// dets modified in-place; suppressed entries get class_id = -1.
// On return, valid dets are compacted to the front.
void yolo_nms(DetObject* dets, int count, float nms_thr)
{
    // Sort by confidence descending (bubble sort — fine for small count)
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (dets[j].conf > dets[i].conf) {
                DetObject tmp = dets[i];
                dets[i] = dets[j];
                dets[j] = tmp;
            }
        }
    }

    // Greedy suppression
    for (int i = 0; i < count; i++) {
        if (dets[i].class_id < 0) continue;
        float a_area = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);

        for (int j = i + 1; j < count; j++) {
            if (dets[j].class_id < 0) continue;
            if (dets[i].class_id != dets[j].class_id) continue;

            float ix1 = fmaxf(dets[i].x1, dets[j].x1);
            float iy1 = fmaxf(dets[i].y1, dets[j].y1);
            float ix2 = fminf(dets[i].x2, dets[j].x2);
            float iy2 = fminf(dets[i].y2, dets[j].y2);
            float iw = ix2 - ix1;
            float ih = iy2 - iy1;
            if (iw <= 0.0f || ih <= 0.0f) continue;

            float inter = iw * ih;
            float b_area = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
            float iou = inter / (a_area + b_area - inter);

            if (iou > nms_thr) {
                dets[j].class_id = -1;
            }
        }
    }

    // Compact valid detections to front
    int out = 0;
    for (int i = 0; i < count; i++) {
        if (dets[i].class_id >= 0) {
            dets[out++] = dets[i];
        }
    }
    for (int i = out; i < count; i++) {
        dets[i].class_id = -1;
    }
}
