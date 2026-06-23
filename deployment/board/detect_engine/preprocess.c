#include <stdint.h>
#include <string.h>

static inline uint8_t clamp_byte(int v)
{
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

#define WB_R_GAIN 285
#define WB_G_GAIN 205
#define WB_B_GAIN 307

static inline void fill_gray(uint8_t *dst, int model_size)
{
    int plane = model_size * model_size;
    memset(dst, 114, (size_t)plane);
    memset(dst + plane, 114, (size_t)plane);
    memset(dst + plane * 2, 114, (size_t)plane);
}

void letterbox_preprocess_sized(const uint8_t* src, int cam_w, int cam_h,
                                int model_size, uint8_t* rgb_planar_out,
                                int* pad_x, int* pad_y, float* scale)
{
    int max_side = (cam_w > cam_h) ? cam_w : cam_h;
    *scale = (float)model_size / (float)max_side;

    int new_w = (int)(cam_w * (*scale));
    int new_h = (int)(cam_h * (*scale));

    *pad_x = (model_size - new_w) / 2;
    *pad_y = (model_size - new_h) / 2;

    fill_gray(rgb_planar_out, model_size);

    for (int y = 0; y < new_h; y++) {
        int src_y = y * cam_h / new_h;
        for (int x = 0; x < new_w; x++) {
            int src_x = x * cam_w / new_w;
            int src_idx = (src_y * cam_w + src_x) * 3;
            int dst_x = x + *pad_x;
            int dst_y = y + *pad_y;
            int dst_idx = dst_y * model_size + dst_x;
            int plane = model_size * model_size;

            rgb_planar_out[dst_idx] = src[src_idx + 2];
            rgb_planar_out[plane + dst_idx] = src[src_idx + 1];
            rgb_planar_out[plane * 2 + dst_idx] = src[src_idx + 0];
        }
    }
}

void letterbox_preprocess(const uint8_t* src, int cam_w, int cam_h,
                          uint8_t* rgb_planar_out,
                          int* pad_x, int* pad_y, float* scale)
{
    letterbox_preprocess_sized(src, cam_w, cam_h, 640,
                               rgb_planar_out, pad_x, pad_y, scale);
}

void letterbox_preprocess_nv21m_sized(const uint8_t* y_plane, const uint8_t* vu_plane,
                                      int cam_w, int cam_h, int y_stride, int uv_stride,
                                      int model_size, uint8_t* rgb_planar_out,
                                      int* pad_x, int* pad_y, float* scale)
{
    int max_side = (cam_w > cam_h) ? cam_w : cam_h;
    *scale = (float)model_size / (float)max_side;

    int new_w = (int)(cam_w * (*scale));
    int new_h = (int)(cam_h * (*scale));

    *pad_x = (model_size - new_w) / 2;
    *pad_y = (model_size - new_h) / 2;

    fill_gray(rgb_planar_out, model_size);

    for (int y = 0; y < new_h; y++) {
        int src_y = y * cam_h / new_h;
        const uint8_t *y_row = y_plane + src_y * y_stride;
        const uint8_t *vu_row = vu_plane + (src_y / 2) * uv_stride;
        for (int x = 0; x < new_w; x++) {
            int src_x = x * cam_w / new_w;
            int vu_idx = src_x & ~1;
            int Y = y_row[src_x];
            int V = vu_row[vu_idx];
            int U = vu_row[vu_idx + 1];

            int C = Y, D = U - 128, E = V - 128;
            int b = (256 * C + 454 * D + 128) >> 8;
            int g = (256 * C - 88 * D - 183 * E + 128) >> 8;
            int r = (256 * C + 359 * E + 128) >> 8;

            b = clamp_byte((b * WB_B_GAIN) >> 8);
            g = clamp_byte((g * WB_G_GAIN) >> 8);
            r = clamp_byte((r * WB_R_GAIN) >> 8);

            int dst_x = x + *pad_x;
            int dst_y = y + *pad_y;
            int dst_idx = dst_y * model_size + dst_x;
            int plane = model_size * model_size;

            rgb_planar_out[dst_idx] = (uint8_t)r;
            rgb_planar_out[plane + dst_idx] = (uint8_t)g;
            rgb_planar_out[plane * 2 + dst_idx] = (uint8_t)b;
        }
    }
}

void letterbox_preprocess_nv21m(const uint8_t* y_plane, const uint8_t* vu_plane,
                                int cam_w, int cam_h, int y_stride, int uv_stride,
                                uint8_t* rgb_planar_out,
                                int* pad_x, int* pad_y, float* scale)
{
    letterbox_preprocess_nv21m_sized(y_plane, vu_plane, cam_w, cam_h,
                                     y_stride, uv_stride, 640,
                                     rgb_planar_out, pad_x, pad_y, scale);
}

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
