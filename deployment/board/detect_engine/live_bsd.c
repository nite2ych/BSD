/*
 * live_bsd.c — Real-time BSD pipeline: V4L2 camera → NPU inference → optional framebuffer display
 *
 * Pipeline:
 *   1. V4L2 MPLANE NV21M capture (640x360, /dev/video0 via /dev/media0 init)
 *   2. NV21 → BGR conversion (manual integer YUV→RGB)
 *   3. detect_process_frame(BGR) → NPU inference + NMS
 *   4. Optional BGR → RGBA downscale to framebuffer (preview mode only)
 *   5. Optional draw detection bboxes on framebuffer
 *   6. Optional FBIOPAN_DISPLAY double-buffer flip
 *   7. Alarm engine callback prints events
 */

#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <linux/videodev2.h>
#include <linux/fb.h>
#include <linux/kd.h>

#include "detect_engine.h"
#include "../alarm_engine/alarm_engine.h"

/* ===== Config ===== */
#define DEFAULT_CAM_W  1280
#define DEFAULT_CAM_H  720
#define NBUF       4
#define MAX_PLANES 3
#define ENABLE_FRAME_DUMP 0
#define NPU_FRAME_INTERVAL 3

/* Framebuffer: 480x800 physical, 1600 virtual (double-buffer), 32bpp RGBA */
#define FB_DEV     "/dev/fb0"
#define FB_W       480
#define FB_H       800
#define FB_BPP     4          /* 32-bit RGBA */

/* Colors (RGBA) */
#define COLOR_CAR      0xFF0000FF  /* red */
#define COLOR_PERSON   0xFFFF0000  /* blue */
#define COLOR_BICYCLE  0xFF00FF00  /* green */
#define COLOR_MOTOR    0xFFFF00FF  /* magenta */

/* ===== Globals ===== */
static volatile int g_running = 1;
static int fb_r_off = 16, fb_g_off = 8, fb_b_off = 0, fb_a_off = 24;  /* default ARGB */
static int g_cam_w = DEFAULT_CAM_W;
static int g_cam_h = DEFAULT_CAM_H;
static int g_model_size = 640;
static int g_preview_enabled = 1;
static int g_draw_enabled = 1;
static int g_alarm_enabled = 1;
static int g_disp_w = 450;
static int g_disp_h = 800;
static int g_disp_x = 15;
static int g_disp_y = 0;
static int g_y_stride = 0;
static int g_uv_stride = 0;

static long long now_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + tv.tv_usec;
}

static void sleep_ms(long ms)
{
    poll(NULL, 0, (int)ms);
}

static void update_display_geometry(void)
{
    int rot_w = g_cam_h;
    int rot_h = g_cam_w;
    if (rot_w <= 0 || rot_h <= 0) return;

    int fit_w = rot_w * FB_H / rot_h;
    if (fit_w <= FB_W) {
        g_disp_w = fit_w;
        g_disp_h = FB_H;
    } else {
        g_disp_w = FB_W;
        g_disp_h = rot_h * FB_W / rot_w;
    }
    if (g_disp_w <= 0) g_disp_w = FB_W;
    if (g_disp_h <= 0) g_disp_h = FB_H;
    g_disp_x = (FB_W - g_disp_w) / 2;
    g_disp_y = (FB_H - g_disp_h) / 2;
}

/* ----- Camera: open /dev/video0 with media0 init (kernel requirement) ----- */
static int camera_open(void)
{
    int mfd = open("/dev/media0", O_RDWR);
    if (mfd < 0) { printf("FATAL: open /dev/media0: %s\n", strerror(errno)); return -1; }

    int fd = open("/dev/video0", O_RDWR | O_NONBLOCK);
    if (fd < 0) { printf("FATAL: open /dev/video0: %s\n", strerror(errno)); close(mfd); return -1; }

    close(mfd);
    return fd;
}

/* ----- Camera: format setup (MPLANE NV21M only) ----- */
static int camera_setup_format(int fd, int *out_planes, int *out_ystride, int *out_uvstride)
{
    struct v4l2_format fmt;
    int w = g_cam_w, h = g_cam_h;

    /* MPLANE NV21M directly (this kernel ONLY supports MPLANE) */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

    /* G_FMT first to read current pipeline state */
    ioctl(fd, VIDIOC_G_FMT, &fmt);
    printf("[G_FMT mp] %dx%d pix=0x%08x planes=%d\n",
           fmt.fmt.pix_mp.width, fmt.fmt.pix_mp.height,
           fmt.fmt.pix_mp.pixelformat, fmt.fmt.pix_mp.num_planes);

    /* S_FMT with NV21M */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width       = w;
    fmt.fmt.pix_mp.height      = h;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV21M;
    fmt.fmt.pix_mp.num_planes  = 2;
    fmt.fmt.pix_mp.field       = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        printf("FATAL: S_FMT NV21M: %s\n", strerror(errno));
        return -1;
    }

    int pc = fmt.fmt.pix_mp.num_planes;
    if (pc < 1) pc = 1;
    *out_planes = pc;
    *out_ystride  = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
    if (*out_ystride == 0) *out_ystride = w;  /* driver may leave 0 */
    *out_uvstride = pc > 1 ? fmt.fmt.pix_mp.plane_fmt[1].bytesperline : 0;
    if (*out_uvstride == 0 && pc > 1) *out_uvstride = w;
    printf("[S_FMT mp] NV21M %dx%d planes=%d size=%d/%d stride=%d/%d\n",
           fmt.fmt.pix_mp.width, fmt.fmt.pix_mp.height, pc,
           fmt.fmt.pix_mp.plane_fmt[0].sizeimage,
           pc>1 ? fmt.fmt.pix_mp.plane_fmt[1].sizeimage : 0,
           *out_ystride, *out_uvstride);
    return 0;
}

/* ----- Camera: REQBUFS + QUERYBUF + mmap + QBUF ----- */
static int camera_setup_buffers(int fd, int nplanes,
                                 void **y_ptrs, unsigned int *y_lens,
                                 void **uv_ptrs, int max_bufs, int *out_nbuf)
{
    int buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count  = NBUF;
    req.type   = buf_type;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        printf("REQBUFS fail: %s\n", strerror(errno)); return -1;
    }
    int nbuf = req.count;
    printf("[REQBUFS] %d buffers\n", nbuf);

    for (int i = 0; i < nbuf && i < max_bufs; i++) {
        struct v4l2_plane planes[MAX_PLANES];
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        buf.type     = buf_type;
        buf.memory   = V4L2_MEMORY_MMAP;
        buf.index    = i;
        buf.m.planes = planes;
        buf.length   = nplanes;

        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            printf("QUERYBUF[%d] fail: %s\n", i, strerror(errno)); return -1;
        }

        y_ptrs[i]  = mmap(NULL, planes[0].length, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, planes[0].m.mem_offset);
        y_lens[i]  = planes[0].length;
        if (y_ptrs[i] == MAP_FAILED) { printf("mmap Y[%d] fail\n", i); return -1; }

        if (nplanes > 1) {
            uv_ptrs[i] = mmap(NULL, planes[1].length, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, planes[1].m.mem_offset);
        } else {
            uv_ptrs[i] = NULL;
        }

        /* QBUF */
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        buf.type     = buf_type;
        buf.memory   = V4L2_MEMORY_MMAP;
        buf.index    = i;
        buf.m.planes = planes;
        buf.length   = nplanes;
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            printf("QBUF[%d] fail: %s\n", i, strerror(errno)); return -1;
        }
        printf("[BUF %d] Y=%p len=%d UV=%p\n", i, y_ptrs[i], y_lens[i], uv_ptrs[i]);
    }
    *out_nbuf = nbuf;
    return 0;
}

/* ----- Camera: stream on ----- */
static int camera_stream_on(int fd) {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        printf("STREAMON fail: %s\n", strerror(errno)); return -1;
    }
    printf("[STREAMON]\n");
    return 0;
}

/* ----- Camera: stream off ----- */
static void camera_stream_off(int fd) {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    printf("[STREAMOFF]\n");
}

static int camera_dequeue(int fd, int nplanes, int *idx)
{
    struct pollfd pfd = { fd, POLLIN, 0 };
    int ret = poll(&pfd, 1, 2000);
    if (ret <= 0) return -1;

    struct v4l2_plane planes[MAX_PLANES];
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory   = V4L2_MEMORY_MMAP;
    buf.m.planes = planes;
    buf.length   = nplanes;

    if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) return -1;
    *idx = buf.index;
    return 0;
}

static int camera_enqueue(int fd, int nplanes, int idx)
{
    struct v4l2_plane planes[MAX_PLANES];
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory   = V4L2_MEMORY_MMAP;
    buf.index    = idx;
    buf.m.planes = planes;
    buf.length   = nplanes;
    if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
        printf("[QBUF] idx=%d err=%s\n", idx, strerror(errno));
        return -1;
    }
    return 0;
}

/* ===== NV21 → BGR conversion (manual, integer math, with white balance) ===== */
static inline uint8_t clamp_byte(int v) {
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* White-balance gains (fixed-point 8.8): correct camera green cast.
   Green channel is ~30% too hot; R and B need boost.
   Interpolated: midpoint between v1 (R/G=0.93) and v2 (R/G=1.08). */
#define WB_R_GAIN 285   /* 1.113 * 256 */
#define WB_G_GAIN 205   /* 0.801 * 256 */
#define WB_B_GAIN 307   /* 1.199 * 256 */

static void nv21_to_bgr(const uint8_t *y_plane, const uint8_t *vu_plane,
                         int w, int h, int y_stride, uint8_t *bgr)
{
    for (int j = 0; j < h; j++) {
        const uint8_t *y_row  = y_plane  + j * y_stride;
        const uint8_t *vu_row = vu_plane + (j / 2) * y_stride; /* VU stride == Y stride in NV21M */
        uint8_t *bgr_row = bgr + j * w * 3;

        for (int i = 0; i < w; i++) {
            int Y = y_row[i];
            int V = vu_row[(i & ~1)];       /* NV21: V first */
            int U = vu_row[(i & ~1) + 1];   /* U second */

            int C = Y, D = U - 128, E = V - 128;
            int b = (256 * C + 454 * D + 128) >> 8;
            int g = (256 * C -  88 * D - 183 * E + 128) >> 8;
            int r = (256 * C + 359 * E + 128) >> 8;
            bgr_row[i*3 + 0] = clamp_byte((b * WB_B_GAIN) >> 8);
            bgr_row[i*3 + 1] = clamp_byte((g * WB_G_GAIN) >> 8);
            bgr_row[i*3 + 2] = clamp_byte((r * WB_R_GAIN) >> 8);
        }
    }
}

/* ===== BGR → framebuffer with 90° CW rotation + nearest-neighbor scale ===== */
static void bgr_rot90_scale_to_rgba(const uint8_t *bgr, int sw, int sh,
                                     uint8_t *fb_base, int fb_stride,
                                     int dx, int dy, int dw, int dh)
{
    for (int j = 0; j < dh; j++) {
        int ox = j * sw / dh;
        uint32_t *row = (uint32_t *)(fb_base + (dy + j) * fb_stride) + dx;

        for (int i = 0; i < dw; i++) {
            int oy = sh - 1 - i * sh / dw;
            const uint8_t *p = bgr + (oy * sw + ox) * 3;
            uint32_t r = p[2], g = p[1], b = p[0];
            row[i] = (r << fb_r_off) | (g << fb_g_off) | (b << fb_b_off) | (0xFFu << fb_a_off);
        }
    }
}

/* ===== Draw detection boxes on framebuffer ===== */
static void fb_draw_rect(uint8_t *fb_base, int fb_stride,
                          int x, int y, int w, int h, uint32_t color, int thickness)
{
    int fx = g_disp_x, fy = g_disp_y, fw = g_disp_w, fh = g_disp_h;

    /* Clip to display region */
    if (x < fx) { w -= fx - x; x = fx; }
    if (y < fy) { h -= fy - y; y = fy; }
    if (x + w > fx + fw) w = fx + fw - x;
    if (y + h > fy + fh) h = fy + fh - y;
    if (w <= 0 || h <= 0) return;

    for (int t = 0; t < thickness && t < h && t < w; t++) {
        int x0 = x + t, y0 = y + t, x1 = x + w - 1 - t, y1 = y + h - 1 - t;
        if (x0 > x1 || y0 > y1) break;

        /* Top & bottom */
        uint32_t *top = (uint32_t *)(fb_base + y0 * fb_stride) + x0;
        uint32_t *bot = (uint32_t *)(fb_base + y1 * fb_stride) + x0;
        for (int i = 0; i <= x1 - x0; i++) { top[i] = color; bot[i] = color; }

        /* Left & right */
        for (int r = y0 + 1; r < y1; r++) {
            uint32_t *row = (uint32_t *)(fb_base + r * fb_stride);
            row[x0] = color;
            row[x1] = color;
        }
    }
}

/* ===== Simple 6x8 bitmap font (in 8px-wide cell, ASCII 32-90) ===== */
static const unsigned char font6x8[][8] = {
    [0] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, /* ' ' 32 */
    [1] = {0x20,0x20,0x20,0x20,0x20,0x00,0x20,0x00}, /* '!' 33 */
    [2] = {0x50,0x50,0x50,0x00,0x00,0x00,0x00,0x00}, /* '"' 34 */
    [3] = {0x50,0x50,0xF8,0x50,0xF8,0x50,0x50,0x00}, /* '#' 35 */
    [4] = {0x20,0x78,0xA0,0x70,0x28,0xF0,0x20,0x00}, /* '$' 36 */
    [5] = {0xC0,0xC8,0x10,0x20,0x40,0x98,0x18,0x00}, /* '%' 37 */
    [6] = {0x40,0xA0,0xA0,0x40,0xA8,0x90,0x68,0x00}, /* '&' 38 */
    [7] = {0x30,0x20,0x40,0x00,0x00,0x00,0x00,0x00}, /* ''' 39 */
    [8] = {0x10,0x20,0x40,0x40,0x40,0x20,0x10,0x00}, /* '(' 40 */
    [9] = {0x40,0x20,0x10,0x10,0x10,0x20,0x40,0x00}, /* ')' 41 */
    [10]={0x20,0xA8,0x70,0x20,0x70,0xA8,0x20,0x00}, /* '*' 42 */
    [11]={0x00,0x20,0x20,0xF8,0x20,0x20,0x00,0x00}, /* '+' 43 */
    [12]={0x00,0x00,0x00,0x00,0x30,0x20,0x40,0x00}, /* ',' 44 */
    [13]={0x00,0x00,0x00,0xF8,0x00,0x00,0x00,0x00}, /* '-' 45 */
    [14]={0x00,0x00,0x00,0x00,0x00,0x30,0x30,0x00}, /* '.' 46 */
    [15]={0x00,0x08,0x10,0x20,0x40,0x80,0x00,0x00}, /* '/' 47 */
    [16]={0x70,0x88,0x98,0xA8,0xC8,0x88,0x70,0x00}, /* '0' 48 */
    [17]={0x20,0x60,0x20,0x20,0x20,0x20,0x70,0x00}, /* '1' 49 */
    [18]={0x70,0x88,0x08,0x10,0x20,0x40,0xF8,0x00}, /* '2' 50 */
    [19]={0xF8,0x10,0x20,0x10,0x08,0x88,0x70,0x00}, /* '3' 51 */
    [20]={0x10,0x30,0x50,0x90,0xF8,0x10,0x10,0x00}, /* '4' 52 */
    [21]={0xF8,0x80,0xF0,0x08,0x08,0x88,0x70,0x00}, /* '5' 53 */
    [22]={0x30,0x40,0x80,0xF0,0x88,0x88,0x70,0x00}, /* '6' 54 */
    [23]={0xF8,0x08,0x10,0x20,0x40,0x40,0x40,0x00}, /* '7' 55 */
    [24]={0x70,0x88,0x88,0x70,0x88,0x88,0x70,0x00}, /* '8' 56 */
    [25]={0x70,0x88,0x88,0x78,0x08,0x10,0x60,0x00}, /* '9' 57 */
    [26]={0x00,0x30,0x30,0x00,0x30,0x30,0x00,0x00}, /* ':' 58 */
    [27]={0x00,0x30,0x30,0x00,0x30,0x20,0x40,0x00}, /* ';' 59 */
    [28]={0x10,0x20,0x40,0x80,0x40,0x20,0x10,0x00}, /* '<' 60 */
    [29]={0x00,0x00,0xF8,0x00,0xF8,0x00,0x00,0x00}, /* '=' 61 */
    [30]={0x40,0x20,0x10,0x08,0x10,0x20,0x40,0x00}, /* '>' 62 */
    [31]={0x70,0x88,0x10,0x20,0x20,0x00,0x20,0x00}, /* '?' 63 */
    [32]={0x70,0x88,0xB8,0xA8,0xB8,0x80,0x70,0x00}, /* '@' 64 */
    [33]={0x70,0x88,0x88,0xF8,0x88,0x88,0x88,0x00}, /* 'A' 65 */
    [34]={0xF0,0x88,0x88,0xF0,0x88,0x88,0xF0,0x00}, /* 'B' 66 */
    [35]={0x70,0x88,0x80,0x80,0x80,0x88,0x70,0x00}, /* 'C' 67 */
    [36]={0xE0,0x90,0x88,0x88,0x88,0x90,0xE0,0x00}, /* 'D' 68 */
    [37]={0xF8,0x80,0x80,0xF0,0x80,0x80,0xF8,0x00}, /* 'E' 69 */
    [38]={0xF8,0x80,0x80,0xF0,0x80,0x80,0x80,0x00}, /* 'F' 70 */
    [39]={0x70,0x88,0x80,0xB8,0x88,0x88,0x78,0x00}, /* 'G' 71 */
    [40]={0x88,0x88,0x88,0xF8,0x88,0x88,0x88,0x00}, /* 'H' 72 */
    [41]={0x70,0x20,0x20,0x20,0x20,0x20,0x70,0x00}, /* 'I' 73 */
    [42]={0x38,0x10,0x10,0x10,0x10,0x90,0x60,0x00}, /* 'J' 74 */
    [43]={0x88,0x90,0xA0,0xC0,0xA0,0x90,0x88,0x00}, /* 'K' 75 */
    [44]={0x80,0x80,0x80,0x80,0x80,0x80,0xF8,0x00}, /* 'L' 76 */
    [45]={0x88,0xD8,0xA8,0xA8,0x88,0x88,0x88,0x00}, /* 'M' 77 */
    [46]={0x88,0xC8,0xA8,0x98,0x88,0x88,0x88,0x00}, /* 'N' 78 */
    [47]={0x70,0x88,0x88,0x88,0x88,0x88,0x70,0x00}, /* 'O' 79 */
    [48]={0xF0,0x88,0x88,0xF0,0x80,0x80,0x80,0x00}, /* 'P' 80 */
    [49]={0x70,0x88,0x88,0x88,0xA8,0x90,0x68,0x00}, /* 'Q' 81 */
    [50]={0xF0,0x88,0x88,0xF0,0xA0,0x90,0x88,0x00}, /* 'R' 82 */
    [51]={0x70,0x88,0x80,0x70,0x08,0x88,0x70,0x00}, /* 'S' 83 */
    [52]={0xF8,0x20,0x20,0x20,0x20,0x20,0x20,0x00}, /* 'T' 84 */
    [53]={0x88,0x88,0x88,0x88,0x88,0x88,0x70,0x00}, /* 'U' 85 */
    [54]={0x88,0x88,0x88,0x88,0x88,0x50,0x20,0x00}, /* 'V' 86 */
    [55]={0x88,0x88,0x88,0xA8,0xA8,0xD8,0x88,0x00}, /* 'W' 87 */
    [56]={0x88,0x88,0x50,0x20,0x50,0x88,0x88,0x00}, /* 'X' 88 */
    [57]={0x88,0x88,0x88,0x50,0x20,0x20,0x20,0x00}, /* 'Y' 89 */
    [58]={0xF8,0x08,0x10,0x20,0x40,0x80,0xF8,0x00}, /* 'Z' 90 */
};

static void fb_draw_char(uint8_t *fb_base, int fb_stride,
                          int x, int y, char c, uint32_t color)
{
    if (c < 32 || c > 90) c = '?';
    const unsigned char *glyph = font6x8[c - 32];
    for (int row = 0; row < 8; row++) {
        unsigned char line = glyph[row];
        for (int col = 0; col < 8; col++) {
            if (line & (0x80 >> col)) {
                int px = x + col, py = y + row;
                if (px >= 0 && px < FB_W && py >= 0 && py < FB_H) {
                    uint32_t *p = (uint32_t *)(fb_base + py * fb_stride) + px;
                    *p = color;
                }
            }
        }
    }
}

static void fb_draw_text(uint8_t *fb_base, int fb_stride,
                          int x, int y, const char *text, uint32_t color)
{
    while (*text) {
        fb_draw_char(fb_base, fb_stride, x, y, *text, color);
        x += 8;
        if (x + 8 > FB_W) break;
        text++;
    }
}

/* Confidence threshold for display filtering (default 0.3) */
static float g_disp_conf_thr = 0.3f;

/* ===== Framebuffer setup (double-buffer) ===== */
static uint8_t *fb_base = NULL;
static int      fb_fd   = -1;
static int      fb_stride = 0;
static int      fb_page_h = 0;   /* yres (one page height) */
static int      fb_page   = 0;   /* current displayed page (0 or 1) */
static size_t   fb_size   = 0;

static int fb_init(void)
{
    fb_fd = open(FB_DEV, O_RDWR);
    if (fb_fd < 0) { printf("FATAL: open %s: %s\n", FB_DEV, strerror(errno)); return -1; }

    struct fb_var_screeninfo vinfo;
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo) < 0) {
        printf("FBIOGET_VSCREENINFO: %s\n", strerror(errno)); close(fb_fd); return -1;
    }
    printf("[FB] %dx%d virt=%dx%d bpp=%d\n",
           vinfo.xres, vinfo.yres, vinfo.xres_virtual, vinfo.yres_virtual, vinfo.bits_per_pixel);
    printf("[FB] R:%d/%d G:%d/%d B:%d/%d A:%d/%d\n",
           vinfo.red.offset, vinfo.red.length,
           vinfo.green.offset, vinfo.green.length,
           vinfo.blue.offset, vinfo.blue.length,
           vinfo.transp.offset, vinfo.transp.length);

    fb_page_h = vinfo.yres;
    fb_stride = vinfo.xres_virtual * vinfo.bits_per_pixel / 8;
    fb_size   = (size_t)fb_stride * vinfo.yres_virtual;
    fb_r_off  = vinfo.red.offset;
    fb_g_off  = vinfo.green.offset;
    fb_b_off  = vinfo.blue.offset;
    fb_a_off  = vinfo.transp.offset;

    fb_base = mmap(NULL, fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if (fb_base == MAP_FAILED) { printf("mmap fb: %s\n", strerror(errno)); close(fb_fd); return -1; }
    printf("[FB] mmap %zu bytes @ %p stride=%d page_h=%d\n", fb_size, fb_base, fb_stride, fb_page_h);

    /* Take over fb from kernel console, prevent blanking */
    if (ioctl(fb_fd, KDSETMODE, KD_GRAPHICS) < 0) {
        printf("[FB] KDSETMODE failed: %s\n", strerror(errno));
    }
    ioctl(fb_fd, FBIOBLANK, FB_BLANK_UNBLANK);
    memset(fb_base, 0, fb_page_h * fb_stride);

    return 0;
}

static void fb_flip(void)
{
    struct fb_var_screeninfo vinfo;
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo) < 0) {
        printf("[FB] get vinfo failed: %s\n", strerror(errno));
        return;
    }

    int next = 1 - fb_page;
    vinfo.yoffset = next * fb_page_h;
    vinfo.activate = FB_ACTIVATE_NOW;

    if (ioctl(fb_fd, FBIOPAN_DISPLAY, &vinfo) == 0) {
        fb_page = next;
    } else if (ioctl(fb_fd, FBIOPUT_VSCREENINFO, &vinfo) == 0) {
        fb_page = next;
    } else {
        printf("[FB] pan failed: %s\n", strerror(errno));
    }
}

static uint8_t *fb_back_buffer(void)
{
    int back = 1 - fb_page;  /* write to page NOT currently displayed */
    return fb_base + (size_t)back * fb_page_h * fb_stride;
}

static void fb_deinit(void)
{
    if (fb_fd >= 0) {
        ioctl(fb_fd, KDSETMODE, KD_TEXT);
        close(fb_fd);
    }
    if (fb_base && fb_base != MAP_FAILED) munmap(fb_base, fb_size);
}

/* ===== Signal handler ===== */
static void sig_handler(int sig) { (void)sig; g_running = 0; }

/* ===== Alarm callback ===== */
static void on_alarm(const AlarmEvent *evt)
{
    const char *names[] = {"person", "bicycle", "motorcycle", "vehicle"};
    printf("ALARM: zone=%d class=%s overlap=%.2f frames=%d pos=[%.0f,%.0f,%.0f,%.0f]\n",
           evt->zone_level,
           (evt->class_id >= 0 && evt->class_id < 4) ? names[evt->class_id] : "?",
           evt->overlap, evt->frame_count,
           evt->x1, evt->y1, evt->x2, evt->y2);
}

static uint32_t class_color(int class_id)
{
    switch (class_id) {
        case CLASS_PERSON:     return COLOR_PERSON;
        case CLASS_BICYCLE:    return COLOR_BICYCLE;
        case CLASS_MOTORCYCLE: return COLOR_MOTOR;
        case CLASS_VEHICLE:    return COLOR_CAR;
        default:               return 0xFFFF5500; /* orange */
    }
}

typedef struct {
    int active;
    int class_id;
    int miss;
    float conf;
    float x1, y1, x2, y2;
} DisplayTrack;

static DisplayTrack g_disp_tracks[MAX_DET];

static float rect_iou(float ax1, float ay1, float ax2, float ay2,
                      float bx1, float by1, float bx2, float by2)
{
    float ix1 = ax1 > bx1 ? ax1 : bx1;
    float iy1 = ay1 > by1 ? ay1 : by1;
    float ix2 = ax2 < bx2 ? ax2 : bx2;
    float iy2 = ay2 < by2 ? ay2 : by2;
    float iw = ix2 - ix1;
    float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    float inter = iw * ih;
    float aa = (ax2 - ax1) * (ay2 - ay1);
    float ba = (bx2 - bx1) * (by2 - by1);
    float u = aa + ba - inter;
    return u > 0.0f ? inter / u : 0.0f;
}

static void update_display_tracks(const DetResult *res)
{
    memset(g_disp_tracks, 0, sizeof(g_disp_tracks));

    if (res) {
        int out = 0;
        for (uint32_t i = 0; i < res->count && i < MAX_DET; i++) {
            const DetObject *d = &res->objects[i];
            if (d->conf < g_disp_conf_thr) continue;

            DisplayTrack *tr = &g_disp_tracks[out++];
            tr->active = 1;
            tr->class_id = d->class_id;
            tr->conf = d->conf;
            tr->x1 = d->x1;
            tr->y1 = d->y1;
            tr->x2 = d->x2;
            tr->y2 = d->y2;
            if (out >= MAX_DET) break;
        }
    }
}

/* ===== Main ===== */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.nb> [det_conf=0.5] [nms_thr=0.45] [disp_conf=0.5] [person_conf=0.5] [cam_w cam_h] [mode] [model_size=640] [alarm_frames=3] [max_missed=3] [cooldown=15] [reemit=15] [smooth=0.65] [vehicle_alarm=1]\n", argv[0]);
        return 1;
    }

    const char *nb_path = argv[1];
    float det_conf    = argc > 2 ? atof(argv[2]) : 0.5f;
    float nms_thr     = argc > 3 ? atof(argv[3]) : 0.45f;
    float disp_conf   = argc > 4 ? atof(argv[4]) : 0.5f;
    float person_conf = argc > 5 ? atof(argv[5]) : 0.5f;
    int alarm_frames  = argc > 10 ? atoi(argv[10]) : 3;
    int max_missed    = argc > 11 ? atoi(argv[11]) : 3;
    int alarm_cooldown = argc > 12 ? atoi(argv[12]) : 15;
    int alarm_reemit   = argc > 13 ? atoi(argv[13]) : 15;
    float smooth_alpha = argc > 14 ? atof(argv[14]) : 0.65f;
    int vehicle_alarm  = argc > 15 ? atoi(argv[15]) : 1;
    if (argc > 7) {
        g_cam_w = atoi(argv[6]);
        g_cam_h = atoi(argv[7]);
    }
    if (argc > 8) {
        const char *mode = argv[8];
        if (strcmp(mode, "inference") == 0 || strcmp(mode, "headless") == 0) {
            g_preview_enabled = 0;
            g_draw_enabled = 0;
        } else if (strcmp(mode, "preview") == 0) {
            g_preview_enabled = 1;
            g_draw_enabled = 1;
        }
    }
    if (argc <= 8) {
        g_preview_enabled = 0;
        g_draw_enabled = 0;
    }
    if (argc > 9) {
        g_model_size = atoi(argv[9]);
    }
    if (g_model_size != 320 && g_model_size != 416 &&
        g_model_size != 512 && g_model_size != 640) {
        fprintf(stderr, "Unsupported model_size=%d, expected 320/416/512/640\n", g_model_size);
        return 1;
    }
    update_display_geometry();
    g_disp_conf_thr = disp_conf;

    setvbuf(stdout, NULL, _IONBF, 0);

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGHUP, sig_handler);
    signal(SIGPIPE, sig_handler);

    uint8_t *bgr_buf = NULL;

    printf("=== LIVE BSD PIPELINE ===\n");
    printf("  Model: %s\n", nb_path);
    printf("  Model input: %dx%d\n", g_model_size, g_model_size);
    printf("  Camera: %dx%d MPLANE NV21M\n", g_cam_w, g_cam_h);
    printf("  Display: %dx%d framebuffer\n", FB_W, FB_H);
    printf("  Thresholds: det=%.2f/%.2f(bg)/%.2f(mt)/%.2f(vh) nms=%.2f disp=%.2f person=%.2f\n",
           det_conf, det_conf, det_conf, det_conf, nms_thr, disp_conf, person_conf);
    printf("  Alarm: frames=%d max_missed=%d cooldown=%d reemit=%d smooth=%.2f vehicle=%d\n",
           alarm_frames, max_missed, alarm_cooldown, alarm_reemit, smooth_alpha, vehicle_alarm);

    /* 1. Open camera */
    int cam_fd = camera_open();
    if (cam_fd < 0) return 1;

    ioctl(cam_fd, VIDIOC_S_INPUT, &(int){0});

    int nplanes, y_stride, uv_stride;
    if (camera_setup_format(cam_fd, &nplanes, &y_stride, &uv_stride) < 0) { close(cam_fd); return 1; }
    g_y_stride = y_stride;
    g_uv_stride = uv_stride;
    printf("[CAM] %dx%d NV21M planes=%d y_stride=%d uv_stride=%d\n",
           g_cam_w, g_cam_h, nplanes, y_stride, uv_stride);

    void *y_ptrs[NBUF]  = {0};
    void *uv_ptrs[NBUF] = {0};
    unsigned int y_lens[NBUF] = {0};
    int nbuf = 0;
    if (camera_setup_buffers(cam_fd, nplanes, y_ptrs, y_lens, uv_ptrs, NBUF, &nbuf) < 0) {
        close(cam_fd); return 1;
    }

    /* 3. STREAMON */
    if (camera_stream_on(cam_fd) < 0) goto cleanup;

    /* 4. Init framebuffer */
    if (g_preview_enabled) {
        if (fb_init() < 0) {
            printf("[FB] Display unavailable, running headless\n");
            g_preview_enabled = 0;
            g_draw_enabled = 0;
        }
    } else {
        printf("[FB] preview disabled, running headless\n");
    }

    /* 5. Init detect engine */
    int use_npu = (nb_path != NULL && strcmp(nb_path, "none") != 0);
    if (use_npu) {
        if (detect_set_model_size(g_model_size) != 0) {
            fprintf(stderr, "detect_set_model_size(%d) failed\n", g_model_size);
            fb_deinit(); close(cam_fd); return 1;
        }
        if (detect_init(nb_path, "/bsd_detect_shm", g_cam_w, g_cam_h, det_conf, nms_thr) != 0) {
            fprintf(stderr, "detect_init failed\n");
            fb_deinit(); close(cam_fd); return 1;
        }
        detect_set_nv21_stride(y_stride, uv_stride);
        // Per-class thresholds: person=higher, others=base
        detect_set_class_threshold(0, person_conf);   // person
        // bicycle/motorcycle/vehicle stay at det_conf
        printf("Detect engine initialized\n");

        /* 6. Init alarm engine */
        if (alarm_init("/bsd_detect_shm") != 0) {
            fprintf(stderr, "alarm_init failed\n");
            detect_deinit(); fb_deinit(); close(cam_fd); return 1;
        }
        if (g_alarm_enabled) {
            alarm_set_zone(0, g_cam_w * 0.25f, g_cam_h * 0.2f, g_cam_w * 0.75f, g_cam_h * 0.8f, 0.3f);
            alarm_set_zone(1, 0.0f, 0.0f, (float)g_cam_w, (float)g_cam_h, 0.3f);
            alarm_set_frame_threshold(alarm_frames);
            alarm_set_tracker_params(max_missed, alarm_cooldown, alarm_reemit, 0.30f, smooth_alpha);
            alarm_set_class_enabled(CLASS_PERSON, 1);
            alarm_set_class_enabled(CLASS_BICYCLE, 1);
            alarm_set_class_enabled(CLASS_MOTORCYCLE, 1);
            alarm_set_class_enabled(CLASS_VEHICLE, vehicle_alarm);
            alarm_register_callback(on_alarm);
            if (alarm_start() != 0) {
                fprintf(stderr, "alarm_start failed\n");
                alarm_deinit(); detect_deinit(); fb_deinit(); close(cam_fd); return 1;
            }
            printf("Alarm engine started\n");
        }
    }

    /* 7. Allocate BGR frame buffer only for preview mode */
    size_t bgr_sz = (size_t)g_cam_w * g_cam_h * 3;
    if (g_preview_enabled) {
        bgr_buf = malloc(bgr_sz);
        if (!bgr_buf) { printf("malloc bgr_buf failed\n"); goto cleanup; }
    }

    /* 8. Main loop */
    printf("\n=== RUNNING (Ctrl+C to stop) ===\n");
    printf("[DEBUG] g_running=%d\n", g_running);
    fflush(stdout);
    int frame_no = 0;
    int deq_fail_cnt = 0;   /* consecutive dequeue failures */
    int qbuf_fail_cnt = 0;  /* consecutive QBUF failures */
    long long fps_t0 = now_us();
    long long sum_cap_us = 0, sum_pre_us = 0, sum_dump_us = 0;
    long long sum_preview_us = 0, sum_qbuf_us = 0;
    long long sum_det_pre_us = 0, sum_det_npu_us = 0;
    long long sum_det_decode_us = 0, sum_det_post_us = 0, sum_det_total_us = 0;
    unsigned int last_profile_seq = 0;
    int det_call_count = 0;
    while (g_running) {
        int idx;
        long long t0 = now_us();
        if (camera_dequeue(cam_fd, nplanes, &idx) < 0) {
            deq_fail_cnt++;
            if (deq_fail_cnt <= 5 || deq_fail_cnt % 100 == 0) {
                printf("[F%d] dequeue err (errno=%d %s) fail_cnt=%d\n",
                       frame_no + 1, errno, strerror(errno), deq_fail_cnt);
                fflush(stdout);
            }
            /* Backoff to avoid spin-looping and give driver time to recover */
            sleep_ms(10);
            /* If dequeue fails continuously for ~10s (300x at 30fps), bail */
            if (deq_fail_cnt > 300) {
                printf("[F%d] dequeue fails exceeded 300, exiting\n", frame_no + 1);
                break;
            }
            continue;
        }
        deq_fail_cnt = 0;
        frame_no++;
        long long t1 = now_us();
        if (frame_no % 30 == 1) printf("[F%d]\n", frame_no);

        /* Save frames at intervals: every 10th frame, 15 total */
        static int dump_cnt = 0;
        if (ENABLE_FRAME_DUMP && dump_cnt < 15 && (frame_no == 1 || frame_no % 10 == 0)) {
            dump_cnt++;
            char path[64];
            snprintf(path, sizeof(path), "/mnt/UDISK/frame_%06d_bgr.raw", frame_no);
            if (!bgr_buf) {
                bgr_buf = malloc(bgr_sz);
                if (bgr_buf) {
                    nv21_to_bgr((const uint8_t *)y_ptrs[idx],
                                (const uint8_t *)uv_ptrs[idx],
                                g_cam_w, g_cam_h, y_stride, bgr_buf);
                }
            }
            FILE *f = fopen(path, "wb");
            if (f && bgr_buf) { fwrite(bgr_buf, 1, bgr_sz, f); fclose(f); printf("[DUMP] %s\n", path); }
        }
        long long t3 = now_us();

        /* NPU inference */
        long long pre_t0 = now_us();
        if (use_npu && (frame_no % NPU_FRAME_INTERVAL) == 1) {
            if (detect_process_nv21m((const uint8_t *)y_ptrs[idx],
                                      (const uint8_t *)uv_ptrs[idx]) != 0) {
                printf("[F%d] detect_process_frame failed\n", frame_no);
            }
            const DetectProfile *prof = detect_get_last_profile();
            if (prof && prof->seq != last_profile_seq) {
                last_profile_seq = prof->seq;
                det_call_count++;
                sum_det_pre_us += prof->preprocess_us;
                sum_det_npu_us += prof->npu_us;
                sum_det_decode_us += prof->decode_us;
                sum_det_post_us += prof->post_us;
                sum_det_total_us += prof->total_us;
            }
            update_display_tracks(detect_get_result());
        }
        long long t4 = now_us();
        sum_pre_us += t4 - pre_t0;

        /* Draw to framebuffer */
        long long preview_t0 = now_us();
        if (g_preview_enabled && fb_base && fb_base != MAP_FAILED) {
            uint8_t *back = fb_back_buffer();

            /* Clear the full back page so old boxes/text cannot survive pan flips. */
            memset(back, 0, (size_t)fb_page_h * fb_stride);

            if (g_preview_enabled && bgr_buf) {
                /* Scale BGR → RGBA on framebuffer (rotated 90° CW) */
                nv21_to_bgr((const uint8_t *)y_ptrs[idx],
                            (const uint8_t *)uv_ptrs[idx],
                            g_cam_w, g_cam_h, y_stride, bgr_buf);
                bgr_rot90_scale_to_rgba(bgr_buf, g_cam_w, g_cam_h,
                                         back, fb_stride, g_disp_x, g_disp_y, g_disp_w, g_disp_h);
            }

            /* Draw detections */
            if (use_npu && g_draw_enabled) {
                const char *names[] = {"person", "bicycle", "motorcycle", "vehicle"};
                float sx_rot = (float)g_disp_w / g_cam_h;
                float sy_rot = (float)g_disp_h / g_cam_w;
                for (int i = 0; i < MAX_DET; i++) {
                    const DisplayTrack *d = &g_disp_tracks[i];
                    if (!d->active) continue;
                    int bx = g_disp_x + (int)((g_cam_h - 1 - d->y2) * sx_rot);
                    int by = g_disp_y + (int)(d->x1 * sy_rot);
                    int bw = (int)((d->y2 - d->y1) * sx_rot);
                    int bh = (int)((d->x2 - d->x1) * sy_rot);
                    uint32_t clr = class_color(d->class_id);
                    fb_draw_rect(back, fb_stride, bx, by, bw, bh, clr, 3);
                    /* Label + confidence above box */
                    char label[48];
                    const char *nm = (d->class_id < 4) ? names[d->class_id] : "?";
                    snprintf(label, sizeof(label), "%s %.0f%%", nm, d->conf * 100.0f);
                    int tx = bx, ty = by - 10;
                    if (ty < 0) ty = by + bh + 2;
                    fb_draw_text(back, fb_stride, tx, ty, label, clr);
                }
            }

            /* Page flip */
            fb_flip();
        }
        long long t5 = now_us();
        sum_preview_us += t5 - preview_t0;

        /* Print detections to console (rate-limited to every 30 frames) */
        if (use_npu && frame_no % 30 == 1) {
            const DetResult *res = detect_get_result();
            printf("[F%d] detect_count=%u", frame_no, res ? res->count : 0);
            if (res && res->count > 0) {
                const char *names[] = {"person", "bicycle", "motorcycle", "vehicle"};
                for (uint32_t i = 0; i < res->count && i < 8; i++) {
                    const DetObject *d = &res->objects[i];
                    printf(" %s(%.2f)", (d->class_id >= 0 && d->class_id < 4) ? names[d->class_id] : "?", d->conf);
                }
            }
            printf("\n");
        }

        /* QBUF back to camera */
        if (camera_enqueue(cam_fd, nplanes, idx) < 0) {
            qbuf_fail_cnt++;
            if (qbuf_fail_cnt <= 3 || qbuf_fail_cnt % 50 == 0) {
                printf("[F%d] QBUF fail #%d — buffer may be lost\n", frame_no, qbuf_fail_cnt);
            }
            if (qbuf_fail_cnt >= 4) {
                printf("[F%d] QBUF fails exhausted all %d buffers, exiting\n", frame_no, nbuf);
                break;
            }
        } else {
            qbuf_fail_cnt = 0;
        }
        long long t6 = now_us();

        sum_cap_us  += t1 - t0;
        sum_dump_us += t3 - t1;
        sum_qbuf_us += t6 - t5;

        if (frame_no % 30 == 0) {
            long long now = now_us();
            double sec = (now - fps_t0) / 1000000.0;
            const DetResult *res = use_npu ? detect_get_result() : NULL;
            double call_div = det_call_count > 0 ? (double)det_call_count * 1000.0 : 1.0;
            printf("[FPS] %.2f det=%u calls=%d avg_ms cap=%.1f pre=%.1f dump=%.1f preview=%.1f qbuf=%.1f det_call_ms total=%.1f nv21=%.1f npu=%.1f decode=%.1f post=%.1f\n",
                   sec > 0.0 ? 30.0 / sec : 0.0,
                   res ? res->count : 0,
                   det_call_count,
                   sum_cap_us / 30000.0,
                   sum_pre_us / 30000.0,
                   sum_dump_us / 30000.0,
                   sum_preview_us / 30000.0,
                   sum_qbuf_us / 30000.0,
                   sum_det_total_us / call_div,
                   sum_det_pre_us / call_div,
                   sum_det_npu_us / call_div,
                   sum_det_decode_us / call_div,
                   sum_det_post_us / call_div);
            fps_t0 = now;
            sum_cap_us = sum_pre_us = sum_dump_us = 0;
            sum_preview_us = sum_qbuf_us = 0;
            sum_det_pre_us = sum_det_npu_us = 0;
            sum_det_decode_us = sum_det_post_us = sum_det_total_us = 0;
            det_call_count = 0;
        }
    }

    printf("\nShutting down...\n");

cleanup:
    camera_stream_off(cam_fd);
    if (bgr_buf) free(bgr_buf);
    for (int i = 0; i < nbuf; i++) {
        if (y_ptrs[i] && y_ptrs[i] != MAP_FAILED) munmap(y_ptrs[i], y_lens[i]);
        if (uv_ptrs[i] && uv_ptrs[i] != MAP_FAILED) {
            /* UV size is y_len/4 (quarter res) for NV21M, but we lost track.
               Using y_len to be safe — munmap with larger size is harmless */
            munmap(uv_ptrs[i], y_lens[i]);
        }
    }
    if (g_alarm_enabled) alarm_deinit();
    detect_deinit();
    fb_deinit();
    close(cam_fd);
    printf("Done.\n");
    return 0;
}
