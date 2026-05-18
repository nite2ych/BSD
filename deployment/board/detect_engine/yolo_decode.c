// deployment/board/detect_engine/yolo_decode.c
// YOLO output decoder: logit→sigmoid, decode boxes, NMS
#include <math.h>
#include <string.h>
#include "../common/types.h"

#define NUM_CELLS   8400
#define NUM_CLASSES 4

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// Decode YOLO outputs into candidate detections.
// boxes:  [1, 4, 8400]  — cx,cy,w,h (model output, normalized 0~1)
// scores: [1, 4, 8400]  — raw logits
// det_out: output array (caller-allocated, max_det entries)
// conf_thr: confidence threshold
// Returns: number of candidate detections
int yolo_decode(const float* boxes, const float* scores,
                DetObject* det_out, int max_det, float conf_thr)
{
    int count = 0;
    for (int i = 0; i < NUM_CELLS && count < max_det; i++) {
        // Find best class
        int best_cls = 0;
        float best_logit = scores[0 * NUM_CELLS + i];
        for (int c = 1; c < NUM_CLASSES; c++) {
            float logit = scores[c * NUM_CELLS + i];
            if (logit > best_logit) {
                best_logit = logit;
                best_cls = c;
            }
        }
        float conf = sigmoid(best_logit);
        if (conf < conf_thr) continue;

        // Get box: cx, cy, w, h (normalized)
        float cx = boxes[0 * NUM_CELLS + i];
        float cy = boxes[1 * NUM_CELLS + i];
        float w  = boxes[2 * NUM_CELLS + i];
        float h  = boxes[3 * NUM_CELLS + i];

        det_out[count].x1 = cx - w / 2.0f;
        det_out[count].y1 = cy - h / 2.0f;
        det_out[count].x2 = cx + w / 2.0f;
        det_out[count].y2 = cy + h / 2.0f;
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
