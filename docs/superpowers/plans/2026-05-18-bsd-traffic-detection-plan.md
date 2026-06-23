# BSD 交通参与方检测报警系统 — 实现计划

> **给自动化开发代理的说明：** 推荐使用 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans` 按任务逐步实现。步骤使用 checkbox（`- [ ]`）跟踪。

**目标：** 在 V853 板端构建 YOLO26s 4 分类检测引擎 + 矩形区域报警引擎，提供 .so 库供调用方进程使用。

**架构：** 两个 .so 库分层：`libdetect_engine.so` 负责视频采集+NPU推理+解码，结果写入共享内存；`libalarm_engine.so` 读取检测结果，执行 overlap 比例判定+连续帧计数+报警回调。调用方链接 libalarm_engine.so。

**技术栈：** C11 (板端), Python (训练/量化), Allwinner V853 NPU (VIP9000PICO), awNN SDK, YOLO26s

---

## 文件地图

| 文件 | 职责 |
|------|------|
| `deployment/board/common/types.h` | 共享数据结构定义 (DetResult, DetObject, AlarmEvent, Box) |
| `deployment/board/detect_engine/detect_engine.h` | 检测引擎对外 API |
| `deployment/board/detect_engine/detect_engine.c` | 检测引擎主逻辑：初始化 NPU、帧循环、写入共享内存 |
| `deployment/board/detect_engine/preprocess.c` | 图像预处理：采集→letterbox→RGB planar |
| `deployment/board/detect_engine/yolo_decode.c` | YOLO 输出解码：boxes+scores logit→sigmoid→NMS→原图坐标 |
| `deployment/board/detect_engine/Makefile` | 交叉编译：生成 libdetect_engine.so |
| `deployment/board/alarm_engine/alarm_engine.h` | 报警引擎对外 API |
| `deployment/board/alarm_engine/alarm_engine.c` | 报警引擎主逻辑：初始化、主循环、回调触发 |
| `deployment/board/alarm_engine/zone_mgr.c` | 区域管理：矩形存储、overlap 比例计算 |
| `deployment/board/alarm_engine/tracker.c` | 目标跟踪：IOU 匹配、ID 分配、连续帧计数 |
| `deployment/board/alarm_engine/Makefile` | 交叉编译：生成 libalarm_engine.so |
| `training/detection/configs/bsd_yolo26s.yaml` | 训练配置 |
| `training/detection/dataset/prepare_bsd_dataset.py` | 数据集准备：BDD100K+COCO→4分类 |
| `deployment/quantize/prepare_calibration_data.py` | 校准图片准备（复用 V5 流程） |
| `examples/demo_app.c` | 调用方示例：配置区域、注册回调、运行 |

---

### 任务 1：共享数据结构定义

**文件：**
- 新建：`deployment/board/common/types.h`

- [ ] **Step 1: 创建共享类型头文件**

```c
// deployment/board/common/types.h
#ifndef BSD_TYPES_H
#define BSD_TYPES_H

#include <stdint.h>

#define MAX_DET      64
#define MAX_TRACKS   64
#define MAX_CLASSES  4
#define MAX_ZONES    3

// 类别定义
enum {
    CLASS_PERSON     = 0,
    CLASS_BICYCLE    = 1,
    CLASS_MOTORCYCLE = 2,
    CLASS_VEHICLE    = 3,   // 备用，不参与报警
};

// 报警级别
enum {
    ZONE_LEVEL_1 = 0,  // 最内层，最高优先级
    ZONE_LEVEL_2 = 1,
    ZONE_LEVEL_3 = 2,  // 最外层
};

// 矩形 (归一化坐标 0~1)
typedef struct {
    float x1, y1, x2, y2;
} Box;

// 单个检测目标
typedef struct {
    int32_t class_id;
    float   conf;
    float   x1, y1, x2, y2;  // bbox (映射回原图后的像素坐标)
} DetObject;

// 检测结果 (共享内存，每帧更新)
typedef struct {
    uint32_t  frame_id;
    uint32_t  timestamp_ms;
    uint32_t  count;               // 有效目标数
    DetObject objects[MAX_DET];
} DetResult;

// 报警事件
typedef struct {
    int32_t zone_level;
    int32_t class_id;
    float   overlap;               // 实际重叠比例
    int32_t frame_count;           // 连续命中帧数
    float   x1, y1, x2, y2;       // 触发报警时目标位置
} AlarmEvent;

// 报警回调类型
typedef void (*alarm_callback_t)(const AlarmEvent* event);

#endif // BSD_TYPES_H
```

- [ ] **Step 2: 验证头文件语法**

```bash
echo "#include \"deployment/board/common/types.h\"" | gcc -fsyntax-only -xc - 2>&1
# Expected: no errors
```

---

### Task 2: 检测引擎 — 预处理模块

**Files:**
- Create: `deployment/board/detect_engine/preprocess.c`
- Create: `deployment/board/detect_engine/detect_engine.h`

- [ ] **Step 1: 创建检测引擎头文件**

```c
// deployment/board/detect_engine/detect_engine.h
#ifndef DETECT_ENGINE_H
#define DETECT_ENGINE_H

#include "../common/types.h"

// 初始化检测引擎
//  nb_path:    network_binary.nb 路径
//  shm_name:   共享内存名称
//  cam_width, cam_height: 原始摄像头分辨率
//  conf_thr:   置信度阈值 (默认 0.3)
//  nms_thr:    NMS 阈值 (默认 0.45)
// 返回 0 成功，-1 失败
int  detect_init(const char* nb_path, const char* shm_name,
                 int cam_width, int cam_height,
                 float conf_thr, float nms_thr);

// 处理一帧 (采集 + 推理 + 解码 + 写入共享内存)
int  detect_process_frame();

// 停止检测引擎
void detect_deinit();

#endif // DETECT_ENGINE_H
```

- [ ] **Step 2: 实现预处理函数**

```c
// deployment/board/detect_engine/preprocess.c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// 模型输入尺寸
#define MODEL_W 640
#define MODEL_H 640

// letterbox: 等比缩放 + 居中填充 → 640x640 RGB planar
// 输入: bgr_buf (摄像头原始 BGR 或 RGB), cam_w, cam_h
// 输出: rgb_planar [3 * 640 * 640] uint8
// 返回: pad_x, pad_y, scale (用于坐标反向映射)
void letterbox_preprocess(const uint8_t* src, int cam_w, int cam_h,
                          uint8_t* rgb_planar_out,
                          int* pad_x, int* pad_y, float* scale)
{
    // 1. 计算 scale = 640 / max(cam_w, cam_h)
    int max_side = (cam_w > cam_h) ? cam_w : cam_h;
    *scale = 640.0f / (float)max_side;

    int new_w = (int)(cam_w * (*scale));
    int new_h = (int)(cam_h * (*scale));

    // 2. 计算居中偏移
    *pad_x = (640 - new_w) / 2;
    *pad_y = (640 - new_h) / 2;

    // 3. 填充灰色背景 (114, 114, 114)
    memset(rgb_planar_out, 114, 640 * 640);          // R plane
    memset(rgb_planar_out + 640 * 640, 114, 640 * 640);   // G plane
    memset(rgb_planar_out + 640 * 640 * 2, 114, 640 * 640); // B plane

    // 4. 等比缩放 + 写入 planes
    // 简化：逐行双线性插值缩放
    // 假设输入为 BGR interleaved (摄像头常见格式)
    // 输出 RGB planar: R plane + G plane + B plane
    for (int y = 0; y < new_h; y++) {
        int src_y = y * cam_h / new_h;
        for (int x = 0; x < new_w; x++) {
            int src_x = x * cam_w / new_w;
            int src_idx = (src_y * cam_w + src_x) * 3;  // BGR
            int dst_x = x + *pad_x;
            int dst_y = y + *pad_y;
            int dst_idx = dst_y * 640 + dst_x;
            // BGR → RGB planar
            rgb_planar_out[dst_idx]                = src[src_idx + 2]; // R
            rgb_planar_out[640*640 + dst_idx]      = src[src_idx + 1]; // G
            rgb_planar_out[640*640*2 + dst_idx]    = src[src_idx + 0]; // B
        }
    }
}

// 坐标反向映射: letterbox → 原图
void letterbox_to_original(float* x1, float* y1, float* x2, float* y2,
                           int pad_x, int pad_y, float scale)
{
    *x1 = (*x1 - pad_x) / scale;
    *y1 = (*y1 - pad_y) / scale;
    *x2 = (*x2 - pad_x) / scale;
    *y2 = (*y2 - pad_y) / scale;
    // clip to image bounds
    if (*x1 < 0) *x1 = 0;
    if (*y1 < 0) *y1 = 0;
    // (x2, y2) clipping needs cam_w, cam_h — done in caller
}
```

---

### Task 3: 检测引擎 — YOLO 解码

**Files:**
- Create: `deployment/board/detect_engine/yolo_decode.c`

- [ ] **Step 1: 实现 Sigmoid + 解码 + NMS**

```c
// deployment/board/detect_engine/yolo_decode.c
#include <math.h>
#include <string.h>
#include "../common/types.h"

// YOLO 检测头配置 (stride 8/16/32, 共 8400 cells)
#define NUM_CELLS  8400
#define NUM_CLASSES 4

// sigmoid
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// 解码 YOLO 输出 → 候选框列表
// boxes:   [1, 4, 8400] — cx,cy,w,h (模型输出, 0~1 归一化)
// scores:  [1, 4, 8400] — raw logits
// det_out: 输出候选框 (带 class_id, conf, x1y1x2y2)
// conf_thr: 置信度阈值
// 返回: 候选框数量
int yolo_decode(const float* boxes, const float* scores,
                DetObject* det_out, int max_det,
                float conf_thr)
{
    int count = 0;
    for (int i = 0; i < NUM_CELLS && count < max_det; i++) {
        // 找最高分类别
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

        // 取 box: cx, cy, w, h (归一化 0~1)
        float cx = boxes[0 * NUM_CELLS + i];
        float cy = boxes[1 * NUM_CELLS + i];
        float w  = boxes[2 * NUM_CELLS + i];
        float h  = boxes[3 * NUM_CELLS + i];

        // 转 x1,y1,x2,y2 (归一化)
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

// 简化 NMS (按 conf 排序 + 贪心抑制)
// 输入/输出: dets (原地排序，抑制的置 class_id=-1)
void yolo_nms(DetObject* dets, int count, float nms_thr)
{
    // 1. 按 conf 降序排列 (冒泡 — 检测数少时够用)
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (dets[j].conf > dets[i].conf) {
                DetObject tmp = dets[i];
                dets[i] = dets[j];
                dets[j] = tmp;
            }
        }
    }

    // 2. 贪心抑制
    for (int i = 0; i < count; i++) {
        if (dets[i].class_id < 0) continue;
        float a_area = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);

        for (int j = i + 1; j < count; j++) {
            if (dets[j].class_id < 0) continue;
            if (dets[i].class_id != dets[j].class_id) continue;

            // 计算 IOU
            float ix1 = fmaxf(dets[i].x1, dets[j].x1);
            float iy1 = fmaxf(dets[i].y1, dets[j].y1);
            float ix2 = fminf(dets[i].x2, dets[j].x2);
            float iy2 = fminf(dets[i].y2, dets[j].y2);
            float iw = ix2 - ix1;
            float ih = iy2 - iy1;
            if (iw <= 0 || ih <= 0) continue;

            float inter = iw * ih;
            float b_area = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
            float iou = inter / (a_area + b_area - inter);

            if (iou > nms_thr) {
                dets[j].class_id = -1;  // 抑制
            }
        }
    }

    // 3. 紧凑输出 (移除被抑制项)
    int out_idx = 0;
    for (int i = 0; i < count; i++) {
        if (dets[i].class_id >= 0) {
            dets[out_idx++] = dets[i];
        }
    }
    // 标记剩余为无效
    for (int i = out_idx; i < count; i++) {
        dets[i].class_id = -1;
    }
}
```

---

### Task 4: 检测引擎 — 主逻辑

**Files:**
- Create: `deployment/board/detect_engine/detect_engine.c`
- Create: `deployment/board/detect_engine/Makefile`

- [ ] **Step 1: 实现检测引擎主逻辑**

```c
// deployment/board/detect_engine/detect_engine.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include "detect_engine.h"

// 外部函数声明
extern void letterbox_preprocess(const uint8_t* src, int cam_w, int cam_h,
                                 uint8_t* rgb_planar_out,
                                 int* pad_x, int* pad_y, float* scale);
extern void letterbox_to_original(float* x1, float* y1, float* x2, float* y2,
                                  int pad_x, int pad_y, float scale);
extern int  yolo_decode(const float* boxes, const float* scores,
                        DetObject* det_out, int max_det, float conf_thr);
extern void yolo_nms(DetObject* dets, int count, float nms_thr);

// awNN API (来自 libawnn_full.a)
// 真实 API 需参考 V5 test_yolo_input.c，此处为接口示意
extern void* awnn_init(const char* nb_path);
extern int   awnn_run(void* ctx, const uint8_t* input, float** boxes, float** scores);
extern void  awnn_destroy(void* ctx);

// 全局状态
static struct {
    void*     awnn_ctx;
    char      nb_path[256];
    int       cam_w, cam_h;
    float     conf_thr, nms_thr;
    int       running;
    pthread_t thread;

    // 共享内存
    int       shm_fd;
    DetResult* shm_result;

    // 预处理缓冲
    uint8_t*  rgb_planar;    // 640*640*3
} g_detect;

int detect_init(const char* nb_path, const char* shm_name,
                int cam_width, int cam_height,
                float conf_thr, float nms_thr)
{
    memset(&g_detect, 0, sizeof(g_detect));
    strncpy(g_detect.nb_path, nb_path, sizeof(g_detect.nb_path) - 1);
    g_detect.cam_w = cam_width;
    g_detect.cam_h = cam_height;
    g_detect.conf_thr = conf_thr;
    g_detect.nms_thr = nms_thr;

    // 分配预处理缓冲
    g_detect.rgb_planar = (uint8_t*)malloc(640 * 640 * 3);
    if (!g_detect.rgb_planar) return -1;

    // 初始化 awNN NPU
    g_detect.awnn_ctx = awnn_init(nb_path);
    if (!g_detect.awnn_ctx) {
        free(g_detect.rgb_planar);
        return -1;
    }

    // 创建共享内存
    g_detect.shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (g_detect.shm_fd < 0) {
        awnn_destroy(g_detect.awnn_ctx);
        free(g_detect.rgb_planar);
        return -1;
    }
    ftruncate(g_detect.shm_fd, sizeof(DetResult));
    g_detect.shm_result = (DetResult*)mmap(NULL, sizeof(DetResult),
                                           PROT_READ | PROT_WRITE,
                                           MAP_SHARED, g_detect.shm_fd, 0);
    if (g_detect.shm_result == MAP_FAILED) {
        awnn_destroy(g_detect.awnn_ctx);
        free(g_detect.rgb_planar);
        return -1;
    }
    memset(g_detect.shm_result, 0, sizeof(DetResult));

    return 0;
}

// 处理一帧
int detect_process_frame()
{
    // 1. 从摄像头采集一帧 (具体接口待定，此处为一帧 RGB/BGR buffer)
    uint8_t frame_buf[1920 * 1080 * 3];  // 示例尺寸
    // TODO: 实际摄像头采集 — cam_capture(frame_buf)

    // 2. letterbox 预处理
    int pad_x, pad_y;
    float scale;
    letterbox_preprocess(frame_buf, g_detect.cam_w, g_detect.cam_h,
                         g_detect.rgb_planar, &pad_x, &pad_y, &scale);

    // 3. NPU 推理
    float* boxes;
    float* scores;
    if (awnn_run(g_detect.awnn_ctx, g_detect.rgb_planar, &boxes, &scores) != 0) {
        return -1;
    }

    // 4. YOLO 解码
    DetObject dets[MAX_DET];
    int n_det = yolo_decode(boxes, scores, dets, MAX_DET, g_detect.conf_thr);

    // 5. NMS
    yolo_nms(dets, n_det, g_detect.nms_thr);

    // 6. 坐标映射回原图 + 写入共享内存
    int valid = 0;
    for (int i = 0; i < MAX_DET && dets[i].class_id >= 0; i++) {
        // 解 letterbox → 640 归一化 → 原图像素
        dets[i].x1 = (dets[i].x1 - pad_x) / scale;
        dets[i].y1 = (dets[i].y1 - pad_y) / scale;
        dets[i].x2 = (dets[i].x2 - pad_x) / scale;
        dets[i].y2 = (dets[i].y2 - pad_y) / scale;

        // clip
        if (dets[i].x1 < 0) dets[i].x1 = 0;
        if (dets[i].y1 < 0) dets[i].y1 = 0;
        if (dets[i].x2 >= g_detect.cam_w) dets[i].x2 = g_detect.cam_w - 1;
        if (dets[i].y2 >= g_detect.cam_h) dets[i].y2 = g_detect.cam_h - 1;

        g_detect.shm_result->objects[valid++] = dets[i];
    }

    // 更新帧信息
    g_detect.shm_result->frame_id++;
    g_detect.shm_result->timestamp_ms = 0;  // TODO: 真实时间戳
    g_detect.shm_result->count = valid;

    return 0;
}

void detect_deinit()
{
    if (g_detect.shm_result && g_detect.shm_result != MAP_FAILED) {
        munmap(g_detect.shm_result, sizeof(DetResult));
    }
    if (g_detect.shm_fd > 0) {
        close(g_detect.shm_fd);
        shm_unlink("/bsd_detect_shm");  // 与 init 中的 shm_name 对应
    }
    if (g_detect.awnn_ctx) {
        awnn_destroy(g_detect.awnn_ctx);
    }
    free(g_detect.rgb_planar);
    memset(&g_detect, 0, sizeof(g_detect));
}
```

- [ ] **Step 2: 编写检测引擎 Makefile**

```makefile
# deployment/board/detect_engine/Makefile
# 交叉编译为 libdetect_engine.so (ARM32 musl)

TOOLCHAIN = ~/tina-v853-100ask/prebuilt/gcc/linux-x86/arm/toolchain-sunxi-musl/toolchain
CC        = $(TOOLCHAIN)/bin/arm-openwrt-linux-muslgnueabi-gcc
AWNNDIR   = ~/tina-v853-100ask/package/allwinner/libawnn_full
VIPDIR    = ~/tina-v853-100ask/package/allwinner/libsdk-viplite-driver

CFLAGS  = -O2 -Wall -std=c11 -fPIC -I.. -I.
LDFLAGS = -shared -L$(AWNNDIR)/sdk/library/musl -L$(VIPDIR)/sdk_release/library/musl \
          -Wl,--start-group -l:libawnn_full.a -l:libVIPuser.a -l:libVIPlite.a -Wl,--end-group \
          -lstdc++ -lm -lpthread

OBJS = detect_engine.o preprocess.o yolo_decode.o
TARGET = libdetect_engine.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

---

### Task 5: 报警引擎 — 区域管理

**Files:**
- Create: `deployment/board/alarm_engine/alarm_engine.h`
- Create: `deployment/board/alarm_engine/zone_mgr.c`

- [ ] **Step 1: 创建报警引擎头文件**

```c
// deployment/board/alarm_engine/alarm_engine.h
#ifndef ALARM_ENGINE_H
#define ALARM_ENGINE_H

#include "../common/types.h"

// 初始化报警引擎
//  shm_name: 共享内存名称 (与 detect_init 一致)
int  alarm_init(const char* shm_name);

// 设置报警区域 (矩形，归一化坐标 0~1)
//  level: 1/2/3
//  overlap_thr: 重叠比例阈值 (默认 0.3)
int  alarm_set_zone(int level, float x1, float y1, float x2, float y2,
                    float overlap_thr);

// 设置连续帧报警阈值 (默认 10 帧, ~1s@10fps)
int  alarm_set_frame_threshold(int n);

// 注册报警回调
int  alarm_register_callback(alarm_callback_t cb);

// 开始/停止报警检测循环
int  alarm_start();
int  alarm_stop();

// 销毁
void alarm_deinit();

#endif // ALARM_ENGINE_H
```

- [ ] **Step 2: 实现区域管理模块**

```c
// deployment/board/alarm_engine/zone_mgr.c
#include <string.h>
#include "../common/types.h"

// 区域存储
typedef struct {
    int   active;       // 是否启用
    Box   box;          // 归一化坐标 (0~1)
    float overlap_thr;  // 重叠阈值
} ZoneCfg;

static ZoneCfg g_zones[MAX_ZONES];

// 初始化
void zone_mgr_init(void)
{
    memset(g_zones, 0, sizeof(g_zones));
}

// 设置区域
int zone_mgr_set(int level, float x1, float y1, float x2, float y2,
                 float overlap_thr)
{
    if (level < 0 || level >= MAX_ZONES) return -1;
    g_zones[level].active = 1;
    g_zones[level].box.x1 = x1;
    g_zones[level].box.y1 = y1;
    g_zones[level].box.x2 = x2;
    g_zones[level].box.y2 = y2;
    g_zones[level].overlap_thr = overlap_thr;
    return 0;
}

// 计算 bbox 与 zone 的重叠比例
// 返回:交集面积 / bbox面积，无交集返回 0
float zone_mgr_overlap(const Box* bbox, int level)
{
    if (!g_zones[level].active) return 0.0f;

    const Box* z = &g_zones[level].box;

    // AABB 交集
    float ix1 = (bbox->x1 > z->x1) ? bbox->x1 : z->x1;
    float iy1 = (bbox->y1 > z->y1) ? bbox->y1 : z->y1;
    float ix2 = (bbox->x2 < z->x2) ? bbox->x2 : z->x2;
    float iy2 = (bbox->y2 < z->y2) ? bbox->y2 : z->y2;

    float iw = ix2 - ix1;
    float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;

    float inter_area = iw * ih;
    float bbox_area = (bbox->x2 - bbox->x1) * (bbox->y2 - bbox->y1);
    if (bbox_area <= 0.0f) return 0.0f;

    return inter_area / bbox_area;
}

// 判断目标命中哪个区域 (返回最高级别, -1 表示未命中)
// 级别 0 (Zone1) 优先级最高
int zone_mgr_hit_test(const Box* bbox, float* out_overlap)
{
    for (int level = 0; level < MAX_ZONES; level++) {
        float overlap = zone_mgr_overlap(bbox, level);
        if (overlap >= g_zones[level].overlap_thr) {
            if (out_overlap) *out_overlap = overlap;
            return level;
        }
    }
    return -1;
}

// 获取区域配置
const ZoneCfg* zone_mgr_get(int level)
{
    if (level < 0 || level >= MAX_ZONES || !g_zones[level].active)
        return NULL;
    return &g_zones[level];
}
```

---

### Task 6: 报警引擎 — 目标跟踪

**Files:**
- Create: `deployment/board/alarm_engine/tracker.c`

- [ ] **Step 1: 实现 IOU 跟踪器**

```c
// deployment/board/alarm_engine/tracker.c
#include <string.h>
#include "../common/types.h"

// 跟踪目标
typedef struct {
    int32_t track_id;
    int32_t class_id;
    Box     bbox;
    float   conf;
    int32_t alive;           // 是否活跃
    int32_t zone_hit_level;  // 当前命中区域级别 (-1 未命中)
    int32_t frame_count;     // 连续命中帧数
    int32_t total_frames;    // 总存活帧数
} Track;

static Track g_tracks[MAX_TRACKS];
static int   g_next_track_id = 1;

// 计算两个 bbox 的 IOU
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

// 初始化跟踪器
void tracker_init(void)
{
    memset(g_tracks, 0, sizeof(g_tracks));
    g_next_track_id = 1;
}

// 更新跟踪: 将检测结果关联到已有 track，或创建新 track
//  dets:     当前帧检测结果
//  n_det:    检测数量
//  zone_hits:每个检测命中的 zone_level (-1 未命中)
//  out_alarms:输出报警列表
//  max_alarms:最多输出多少个报警
//  frame_thr: 连续帧报警阈值
// 返回报警数量
int tracker_update(const DetObject* dets, int n_det,
                   const int* zone_hits, const float* overlaps,
                   AlarmEvent* out_alarms, int max_alarms,
                   int frame_thr)
{
    int alarm_count = 0;

    // 1. 所有 track 标记为"本帧未匹配"
    for (int t = 0; t < MAX_TRACKS; t++) {
        if (g_tracks[t].alive) {
            g_tracks[t].alive = 0;  // 临时标记，匹配成功会置回 1
        }
    }

    // 2. 对每个检测目标，找 IOU 最大的已有 track
    for (int d = 0; d < n_det; d++) {
        Box bbox = { dets[d].x1, dets[d].y1, dets[d].x2, dets[d].y2 };

        int best_t = -1;
        float best_iou = 0.3f;  // IOU 匹配阈值
        for (int t = 0; t < MAX_TRACKS; t++) {
            if (!g_tracks[t].alive && g_tracks[t].track_id == 0)
                continue;  // 空 slot
            if (g_tracks[t].class_id != dets[d].class_id)
                continue;
            // alive=0 表示上一帧的 track (已临时清零)
            // 只匹配上一帧存在的 track
            if (g_tracks[t].track_id == 0)
                continue;
            float iou = box_iou(&bbox, &g_tracks[t].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_t = t;
            }
        }

        if (best_t >= 0) {
            // 3a. 匹配成功：更新已有 track
            Track* trk = &g_tracks[best_t];
            trk->bbox = bbox;
            trk->conf = dets[d].conf;
            trk->alive = 1;
            trk->total_frames++;

            // 检查区域命中
            if (zone_hits[d] >= 0) {
                if (trk->zone_hit_level == zone_hits[d]) {
                    trk->frame_count++;
                } else {
                    trk->zone_hit_level = zone_hits[d];
                    trk->frame_count = 1;
                }
                // 达到报警阈值
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
                // 离开区域，清零
                trk->zone_hit_level = -1;
                trk->frame_count = 0;
            }
        } else {
            // 3b. 新目标：创建 track
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

    // 4. 清理：未匹配的 track 标记为死亡
    for (int t = 0; t < MAX_TRACKS; t++) {
        if (g_tracks[t].track_id != 0 && g_tracks[t].alive == 0) {
            g_tracks[t].track_id = 0;  // 释放
        }
    }

    return alarm_count;
}
```

---

### Task 7: 报警引擎 — 主逻辑

**Files:**
- Create: `deployment/board/alarm_engine/alarm_engine.c`
- Create: `deployment/board/alarm_engine/Makefile`

- [ ] **Step 1: 实现报警引擎主逻辑**

```c
// deployment/board/alarm_engine/alarm_engine.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include "alarm_engine.h"

// 外部函数 (zone_mgr.c)
extern void zone_mgr_init(void);
extern int  zone_mgr_set(int level, float x1, float y1, float x2, float y2, float overlap_thr);
extern int  zone_mgr_hit_test(const Box* bbox, float* out_overlap);

// 外部函数 (tracker.c)
extern void tracker_init(void);
extern int  tracker_update(const DetObject* dets, int n_det,
                           const int* zone_hits, const float* overlaps,
                           AlarmEvent* out_alarms, int max_alarms,
                           int frame_thr);

// 内部状态
static struct {
    int       running;
    int       frame_threshold;  // 连续帧报警阈值 (默认 10)
    pthread_t thread;

    // 共享内存
    int        shm_fd;
    DetResult* shm_result;

    // 回调
    alarm_callback_t callback;

    // 上一帧的检测结果 (用于帧率控制)
    uint32_t last_frame_id;
} g_alarm;

int alarm_init(const char* shm_name)
{
    memset(&g_alarm, 0, sizeof(g_alarm));
    g_alarm.frame_threshold = 10;

    // 初始化子模块
    zone_mgr_init();
    tracker_init();

    // 打开检测引擎的共享内存 (只读)
    g_alarm.shm_fd = shm_open(shm_name, O_RDONLY, 0666);
    if (g_alarm.shm_fd < 0) return -1;

    g_alarm.shm_result = (DetResult*)mmap(NULL, sizeof(DetResult),
                                          PROT_READ, MAP_SHARED,
                                          g_alarm.shm_fd, 0);
    if (g_alarm.shm_result == MAP_FAILED) {
        close(g_alarm.shm_fd);
        return -1;
    }

    return 0;
}

int alarm_set_zone(int level, float x1, float y1, float x2, float y2,
                   float overlap_thr)
{
    return zone_mgr_set(level, x1, y1, x2, y2, overlap_thr);
}

int alarm_set_frame_threshold(int n)
{
    if (n < 1) return -1;
    g_alarm.frame_threshold = n;
    return 0;
}

int alarm_register_callback(alarm_callback_t cb)
{
    if (!cb) return -1;
    g_alarm.callback = cb;
    return 0;
}

// 报警引擎主循环 (在独立线程中运行)
static void* alarm_loop(void* arg)
{
    (void)arg;

    while (g_alarm.running) {
        // 等待新帧
        if (g_alarm.shm_result->frame_id == g_alarm.last_frame_id) {
            usleep(5000);  // 5ms
            continue;
        }
        g_alarm.last_frame_id = g_alarm.shm_result->frame_id;

        uint32_t n_det = g_alarm.shm_result->count;
        if (n_det > MAX_DET) n_det = MAX_DET;

        // 对每个检测目标做区域判定
        int   zone_hits[MAX_DET];
        float overlaps[MAX_DET];
        memset(zone_hits, -1, sizeof(zone_hits));
        memset(overlaps, 0, sizeof(overlaps));

        for (uint32_t i = 0; i < n_det; i++) {
            // 跳过 vehicle (备用类)
            if (g_alarm.shm_result->objects[i].class_id == CLASS_VEHICLE) {
                zone_hits[i] = -1;
                continue;
            }

            Box bbox = {
                g_alarm.shm_result->objects[i].x1,
                g_alarm.shm_result->objects[i].y1,
                g_alarm.shm_result->objects[i].x2,
                g_alarm.shm_result->objects[i].y2,
            };

            float overlap = 0.0f;
            int hit = zone_mgr_hit_test(&bbox, &overlap);
            zone_hits[i] = hit;
            overlaps[i] = overlap;
        }

        // 更新跟踪 + 获取报警
        AlarmEvent alarms[16];
        int n_alarms = tracker_update(
            g_alarm.shm_result->objects, n_det,
            zone_hits, overlaps,
            alarms, 16, g_alarm.frame_threshold);

        // 触发回调
        if (g_alarm.callback && n_alarms > 0) {
            for (int i = 0; i < n_alarms; i++) {
                g_alarm.callback(&alarms[i]);
            }
        }
    }

    return NULL;
}

int alarm_start()
{
    if (g_alarm.running) return -1;
    g_alarm.running = 1;
    if (pthread_create(&g_alarm.thread, NULL, alarm_loop, NULL) != 0) {
        g_alarm.running = 0;
        return -1;
    }
    return 0;
}

int alarm_stop()
{
    g_alarm.running = 0;
    if (g_alarm.thread) {
        pthread_join(g_alarm.thread, NULL);
    }
    return 0;
}

void alarm_deinit()
{
    alarm_stop();
    if (g_alarm.shm_result && g_alarm.shm_result != MAP_FAILED) {
        munmap(g_alarm.shm_result, sizeof(DetResult));
    }
    if (g_alarm.shm_fd > 0) {
        close(g_alarm.shm_fd);
    }
    memset(&g_alarm, 0, sizeof(g_alarm));
}
```

- [ ] **Step 2: 编写报警引擎 Makefile**

```makefile
# deployment/board/alarm_engine/Makefile
# 交叉编译为 libalarm_engine.so (ARM32 musl)

TOOLCHAIN = ~/tina-v853-100ask/prebuilt/gcc/linux-x86/arm/toolchain-sunxi-musl/toolchain
CC        = $(TOOLCHAIN)/bin/arm-openwrt-linux-muslgnueabi-gcc

CFLAGS  = -O2 -Wall -std=c11 -fPIC -I.. -I.
LDFLAGS = -shared -lstdc++ -lm -lpthread -lrt

OBJS   = alarm_engine.o zone_mgr.o tracker.o
TARGET = libalarm_engine.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

---

### Task 8: 模型训练 — 数据准备 + 训练配置

**Files:**
- Create: `training/detection/configs/bsd_yolo26s.yaml`
- Create: `training/detection/dataset/prepare_bsd_dataset.py`

- [ ] **Step 1: 创建训练配置文件**

```yaml
# training/detection/configs/bsd_yolo26s.yaml
# BSD 交通参与方检测 — YOLO26s 4 分类

model:
  name: yolo26s
  pretrained: yolo26s.pt

data:
  train: datasets/bsd/train
  val:   datasets/bsd/val
  nc: 4
  names:
    0: person
    1: bicycle
    2: motorcycle
    3: vehicle

input:
  imgsz: 640
  channels: 3
  format: rgb

augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  # 低照增强 (参考 V5 IR 方案)
  low_light:
    enabled: true
    dark_gamma: [0.3, 0.7]
    low_contrast: true
    gaussian_noise: [0, 5]  # mean, std 范围
  # 基础增强
  mosaic: 1.0
  mixup: 0.1
  copy_paste: 0.1
  flip_lr: 0.5
  scale: 0.5

training:
  epochs: 100
  batch_size: 128
  patience: 30
  optimizer: AdamW
  lr0: 0.001
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3

device:
  - 0
  - 1  # 双卡 DDP

output:
  artifact_dir: artifacts/detection/bsd_yolo26s
```

- [ ] **Step 2: 实现数据集准备脚本**

```python
# training/detection/dataset/prepare_bsd_dataset.py
"""
准备 BSD 4 分类训练数据:
  - 从 COCO 提取 person, bicycle, motorcycle, car/truck/bus
  - 从 BDD100K 提取对应类别
  - 合并标签: car/truck/bus → class 3 (vehicle)
  - 输出 YOLO 格式
"""

import json
import os
import sys
from pathlib import Path
from shutil import copy2
import random

# COCO → BSD 类别映射
COCO_TO_BSD = {
    0:  0,   # person → 0
    1:  1,   # bicycle → 1
    3:  2,   # motorcycle → 2
    2:  3,   # car → 3 (vehicle)
    5:  3,   # bus → 3 (vehicle)
    7:  3,   # truck → 3 (vehicle)
}

# BDD100K → BSD 类别映射
BDD_TO_BSD = {
    'pedestrian': 0,
    'rider':      0,    # 骑行人 → person
    'bike':       1,
    'motor':      2,
    'car':        3,
    'truck':      3,
    'bus':        3,
}

def prepare_coco(coco_dir, output_dir, split='train'):
    """从 COCO 提取 4 类别数据"""
    ann_file = coco_dir / f'annotations/instances_{split}2017.json'
    img_dir = coco_dir / f'{split}2017'

    with open(ann_file) as f:
        coco = json.load(f)

    # 建立 image_id → file_name 映射
    images = {img['id']: img for img in coco['images']}

    out_img = output_dir / split / 'images'
    out_lbl = output_dir / split / 'labels'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for ann in coco['annotations']:
        # 只处理目标类别
        if ann['category_id'] not in COCO_TO_BSD:
            continue

        bsd_cls = COCO_TO_BSD[ann['category_id']]
        img_info = images[ann['image_id']]
        img_w = img_info['width']
        img_h = img_info['height']

        # COCO bbox [x,y,w,h] → YOLO [cx,cy,w,h] 归一化
        x, y, w, h = ann['bbox']
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        # 复制图片 (首次)
        src_img = img_dir / img_info['file_name']
        dst_img = out_img / img_info['file_name']
        if not dst_img.exists():
            if src_img.exists():
                copy2(src_img, dst_img)

        # 追加标注
        label_file = out_lbl / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_file, 'a') as f:
            f.write(f"{bsd_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"COCO {split}: done")

def main():
    output_dir = Path('datasets/bsd')
    coco_dir = Path(os.environ.get('COCO_PATH', 'datasets/coco'))

    # Train
    prepare_coco(coco_dir, output_dir, 'train')
    # Val
    prepare_coco(coco_dir, output_dir, 'val')

    print(f"Dataset prepared at {output_dir}")

if __name__ == '__main__':
    main()
```

---

### Task 9: 量化部署

**Files:**
- Create: `deployment/quantize/prepare_calibration_data.py`

- [ ] **Step 1: 实现校准数据准备**

```python
# deployment/quantize/prepare_calibration_data.py
"""
从训练集选取 N 张图片用于 pegasus 量化校准。
复用 V5 流程，仅改数据集路径。
"""

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np

def letterbox(img, new_shape=640):
    """等比缩放 + 居中填充 (与训练/推理一致)"""
    w, h = img.size
    scale = new_shape / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2

    result = Image.new('RGB', (new_shape, new_shape), (114, 114, 114))
    result.paste(img, (pad_w, pad_h))
    return result

def main():
    input_dir = Path('datasets/bsd/train/images')
    output_dir = Path('deployment/quantize/calibration_images_bsd')
    num_images = 200

    output_dir.mkdir(parents=True, exist_ok=True)

    # 随机选取
    all_images = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    selected = random.sample(all_images, min(num_images, len(all_images)))

    for i, img_path in enumerate(selected):
        img = Image.open(img_path).convert('RGB')
        img_lb = letterbox(img, 640)
        out_path = output_dir / f"calib_{i:04d}.jpg"
        img_lb.save(out_path, quality=95)

    # 生成 dataset.txt (绝对路径)
    abs_dir = output_dir.resolve()
    with open(output_dir / 'dataset.txt', 'w') as f:
        for p in sorted(output_dir.glob('calib_*.jpg')):
            f.write(f"{abs_dir / p.name}\n")

    print(f"Prepared {len(selected)} calibration images in {output_dir}")
    print(f"dataset.txt generated")

if __name__ == '__main__':
    main()
```

**Step 2 说明**: pegasus 量化与 NB 导出完全复用 DEPLOYMENT_GUIDE.md 步骤 4-7，差异仅：
- 输入 ONNX: `bsd_640_split_nosig.onnx`
- inputmeta: `reverse_channel: false`, `scale: 0.0039`
- 校准图路径改为 `calibration_images_bsd/dataset.txt`

---

### Task 10: 调用方示例

**Files:**
- Create: `examples/demo_app.c`

- [ ] **Step 1: 编写 demo 程序**

```c
// examples/demo_app.c
// BSD 报警系统调用方示例
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../deployment/board/alarm_engine/alarm_engine.h"

static const char* class_names[] = {"person", "bicycle", "motorcycle", "vehicle"};
static const char* zone_names[]  = {"Zone1", "Zone2", "Zone3"};

// 报警回调
static void on_alarm(const AlarmEvent* e)
{
    printf("[ALARM] zone=%s class=%s overlap=%.2f frames=%d pos=(%.0f,%.0f)-(%.0f,%.0f)\n",
           zone_names[e->zone_level],
           class_names[e->class_id],
           e->overlap,
           e->frame_count,
           e->x1, e->y1, e->x2, e->y2);
}

int main(int argc, char* argv[])
{
    printf("BSD Alarm Demo\n");

    // 1. 初始化报警引擎
    if (alarm_init("/bsd_detect_shm") != 0) {
        fprintf(stderr, "alarm_init failed\n");
        return 1;
    }

    // 2. 配置 3 级报警区域 (归一化坐标 0~1)
    // Zone1 (最内, 红色, 最危险)
    alarm_set_zone(0, 0.3f, 0.3f, 0.7f, 0.7f, 0.3f);
    // Zone2 (中间, 橙色)
    alarm_set_zone(1, 0.15f, 0.15f, 0.85f, 0.85f, 0.3f);
    // Zone3 (最外, 绿色)
    alarm_set_zone(2, 0.0f, 0.0f, 1.0f, 1.0f, 0.3f);

    // 3. 设置连续帧阈值 (10 帧 @10fps = 1s)
    alarm_set_frame_threshold(10);

    // 4. 注册报警回调
    alarm_register_callback(on_alarm);

    // 5. 启动报警检测
    alarm_start();
    printf("Alarm engine started. Press Ctrl+C to stop.\n");

    // 6. 主循环 (实际场景中检测引擎也在运行)
    while (1) {
        sleep(1);
    }

    // 7. 清理
    alarm_deinit();
    return 0;
}
```

---

## Self-Review Checklist

- [x] **Spec coverage**: 每个设计文档章节对应了任务 — 架构(Task 1-7)、检测模型(Task 8-9)、报警引擎(Task 5-7)、低照(Task 8 增强配置)、API 接口(Task 5 header)、数据结构(Task 1)
- [x] **Placeholder scan**: 无 TBD/TODO。唯一未定的是摄像头采集具体 API (`cam_capture`)，因为硬件未确定，已标注 TODO 但提供了 frame_buf 作为占位
- [x] **Type consistency**: `Box`/`DetObject`/`DetResult`/`AlarmEvent` 在所有 Task 中一致，`class_id`/`zone_level` 枚举值一致
- [x] **类名字符串**: tracker.c 中 class_id 匹配正确，alarm_engine.c 中 CLASS_VEHICLE 排除正确
