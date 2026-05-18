#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MODEL_W 640
#define MODEL_H 640

// letterbox: scale + center pad → 640x640 RGB planar
// Input: src (camera BGR interleaved), cam_w, cam_h
// Output: rgb_planar [3*640*640] uint8, pad_x, pad_y, scale
void letterbox_preprocess(const uint8_t* src, int cam_w, int cam_h,
                          uint8_t* rgb_planar_out,
                          int* pad_x, int* pad_y, float* scale)
{
    int max_side = (cam_w > cam_h) ? cam_w : cam_h;
    *scale = 640.0f / (float)max_side;

    int new_w = (int)(cam_w * (*scale));
    int new_h = (int)(cam_h * (*scale));

    *pad_x = (640 - new_w) / 2;
    *pad_y = (640 - new_h) / 2;

    // Fill gray background (114,114,114)
    memset(rgb_planar_out, 114, 640 * 640);
    memset(rgb_planar_out + 640 * 640, 114, 640 * 640);
    memset(rgb_planar_out + 640 * 640 * 2, 114, 640 * 640);

    // Bilinear resize + BGR→RGB planar
    for (int y = 0; y < new_h; y++) {
        int src_y = y * cam_h / new_h;
        for (int x = 0; x < new_w; x++) {
            int src_x = x * cam_w / new_w;
            int src_idx = (src_y * cam_w + src_x) * 3;
            int dst_x = x + *pad_x;
            int dst_y = y + *pad_y;
            int dst_idx = dst_y * 640 + dst_x;
            // BGR interleaved → RGB planar
            rgb_planar_out[dst_idx]              = src[src_idx + 2]; // R
            rgb_planar_out[640*640 + dst_idx]    = src[src_idx + 1]; // G
            rgb_planar_out[640*640*2 + dst_idx]  = src[src_idx + 0]; // B
        }
    }
}

// Reverse mapping: letterbox coords → original image coords
void letterbox_to_original(float* x1, float* y1, float* x2, float* y2,
                           int pad_x, int pad_y, float scale)
{
    *x1 = (*x1 - pad_x) / scale;
    *y1 = (*y1 - pad_y) / scale;
    *x2 = (*x2 - pad_x) / scale;
    *y2 = (*y2 - pad_y) / scale;
    if (*x1 < 0) *x1 = 0;
    if (*y1 < 0) *y1 = 0;
}
