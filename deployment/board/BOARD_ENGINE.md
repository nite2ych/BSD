# BSD Board-Side Engine

V853 板端盲区检测系统工程文档，提交 `0ed8c18`。

## 架构

```
Camera (BGR interleaved)
    │
    ▼
┌──────────────────────────────┐
│  detect_engine               │
│  detect_process_frame()      │
│                              │
│  BGR → letterbox → RGB planar│
│  NPU inference (AWNN)        │
│  YOLO decode (LTRB + sigmoid)│
│  NMS (per-class greedy IOU)  │
│  Coordinate remap + clip     │
│        │                     │
│        ▼                     │
│  Shared memory               │
│  DetResult [/bsd_detect_shm] │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  alarm_engine (独立线程)      │
│  alarm_loop()                │
│                              │
│  读取共享内存                 │
│  Zone hit test (3层区域)      │
│  Multi-target tracker (IOU)  │
│  连续帧计数 → 触发回调        │
└──────────────────────────────┘
```

## 文件结构

```
deployment/board/
├── common/
│   └── types.h              # DetObject, DetResult, AlarmEvent 定义
├── detect_engine/
│   ├── detect_engine.h      # 接口: init / process_frame / get_result / deinit
│   ├── detect_engine.c      # 主引擎: NPU管理 + 共享内存 + 推理管线
│   ├── preprocess.c         # BGR interleaved → RGB planar letterbox
│   ├── yolo_decode.c        # LTRB distance 解码 + logit→sigmoid + NMS
│   ├── main.c               # 集成入口 (detect + alarm)
│   └── Makefile             # 交叉编译 (libdetect_engine.so + bsd_detect + live_bsd + test_npu_direct)
├── alarm_engine/
│   ├── alarm_engine.h       # 接口: init / set_zone / start / stop / deinit
│   ├── alarm_engine.c       # 告警主循环 (独立线程, 读共享内存)
│   ├── zone_mgr.c           # 3层告警区域管理
│   ├── tracker.c            # IOU 多目标跟踪 + 连续帧计数
│   └── Makefile             # 交叉编译 (libalarm_engine.so)
├── compile_engines.py       # 一键交叉编译脚本 (SSH → Ubuntu18)
└── deploy_test.py           # 一键部署测试脚本 (ADB → V853)
```

## 编译

**编译环境**: Ubuntu 18.04 (192.168.144.137)
**工具链**: `arm-openwrt-linux-muslgnueabi-gcc` (Tina-V853 SDK)  
**依赖**: `libawnn_full.a`, `libVIPuser.a`, `libVIPlite.a`, `libstdc++`, `libpthread`

```bash
# 在 Windows 端执行 (自动 SSH 到 Ubuntu18 编译并下载)
python deployment/board/compile_engines.py
```

产物输出到 `artifacts/board_bin/`:
- `libdetect_engine.so` (194KB) — 共享库
- `bsd_detect` (204KB) — 独立可执行文件

## 板端部署与测试

```bash
# 部署并测试 (使用已有校准图片)
python deployment/board/deploy_test.py
```

### 手动运行

```bash
# 1. 推送文件到板端
adb push bsd_detect /mnt/UDISK/
adb push bsd_v4_640.nb /mnt/UDISK/
adb push test_frame.raw /mnt/UDISK/

# 2. 单帧推理 (640x640 摄像头)
cat /mnt/UDISK/test_frame.raw | /mnt/UDISK/bsd_detect /mnt/UDISK/bsd_v4_640.nb 640 640 0.3 0.45

# 3. 1920x1080 摄像头 (letterbox 缩放)
cat /mnt/UDISK/test_1920x1080.raw | /mnt/UDISK/bsd_detect /mnt/UDISK/bsd_v4_640.nb 1920 1080 0.3 0.45
```

## 数据流详解

### 输入
- **格式**: BGR interleaved, uint8, `cam_width × cam_height × 3` bytes
- **来源**: V4L2 摄像头 / raw 文件 / pipe

### 预处理 (preprocess.c)
1. 计算 letterbox: `scale = 640 / max(w,h)`, 居中 + 114 灰色填充
2. BGR interleaved → RGB planar (R plane + G plane + B plane 分离)
3. 输出: `uint8_t rgb_planar[3*640*640]`

### NPU 推理 (detect_engine.c)
- **API**: `awnn_init` → `awnn_create` → `awnn_set_input_buffers` → `awnn_run` → `awnn_get_output_buffers`
- **模型**: bsd_v4_640.nb (YOLO11s, 4类, 640×640, uint8 量化)
- **输出**: `boxes[4×8400]` (float, LTRB distances in grid-cell units), `scores[4×8400]` (float, raw logit)

### 解码 (yolo_decode.c)
1. 逐 cell 扫描: logit → sigmoid, 找最佳类别
2. conf > threshold → 按 stride 8/16/32 将 LTRB distance 解码为 x1/y1/x2/y2
3. 按 conf 排序, 贪婪 NMS (per-class, IOU > nms_thr → 抑制)
4. 紧凑化: 有效检测移到数组前部

### 坐标映射 (detect_engine.c)
```
pixel_x = (norm_x * 640 - pad_x) / scale
pixel_y = (norm_y * 640 - pad_y) / scale
```
四边裁剪到 `[0, cam_size-1]`, 过滤退化框 (面积 ≤ 0)

### 共享内存
- **名称**: `/bsd_detect_shm`
- **类型**: `DetResult { frame_id, count, DetObject objects[64] }`
- **写入**: detect_engine (每帧)
- **读取**: alarm_engine (独立线程)

### 告警引擎 (alarm_engine)
1. **Zone 管理**: 最多 3 层矩形区域 (归一化坐标), Level 0 优先级最高
2. **Tracker**: IOU 匹配检测框与已有轨迹 (threshold=0.3), 未匹配的创建新轨迹
3. **可靠性增强**: 框平滑、短时漏检保持、离区冷却、长时间报警节流
4. **告警**: 同类别/同区域连续命中达到阈值后触发回调

## API 参考

### detect_engine

```c
int  detect_init(const char* nb_path, const char* shm_name,
                 int cam_width, int cam_height,
                 float conf_thr, float nms_thr);
int  detect_process_frame(const uint8_t* frame_buf);
const DetResult* detect_get_result(void);
void detect_deinit(void);
void detect_set_class_threshold(int class_id, float thr);
```

### alarm_engine

```c
int  alarm_init(const char* shm_name);
int  alarm_set_zone(int level, float x1, float y1, float x2, float y2, float overlap_thr);
int  alarm_set_frame_threshold(int n);
int  alarm_set_tracker_params(int max_missed, int cooldown_frames,
                              int reemit_frames, float match_iou_thr,
                              float smooth_alpha);
int  alarm_set_class_enabled(int class_id, int enabled);
int  alarm_register_callback(alarm_callback_t cb);
int  alarm_start(void);
int  alarm_stop(void);
void alarm_deinit(void);
```

## 已验证的板端结果

| 测试 | 输入尺寸 | 检测结果 |
|------|---------|----------|
| 单帧 640×640 | 640×640 BGR | person 0.700, person 0.300 |
| 单帧 1920×1080 | 1920×1080 BGR | vehicle 0.780, vehicle 0.700 |
| 回灌帧 frame_000055 | 1280×720 BGR | person 0.719, box=[744,5,1171,710] |
| 回灌帧 frame_000069 | 1280×720 BGR | 近全画幅 vehicle 已被 95% 宽高过滤 |

NPU 推理正常, 坐标映射正确, 退化框过滤有效, alarm 引擎线程正常运行。

## 模型信息

| 项目 | 值 |
|------|-----|
| 架构 | YOLO11s |
| 输入 | 640×640 RGB planar (letterbox) |
| 类别 | person(0), bicycle(1), motorcycle(2), vehicle(3) |
| 输出 | boxes[4×8400] + scores[4×8400] (raw logit) |
| 量化 | VIP9000PICO, asymmetric_affine, uint8 |
| NB 大小 | 9.3 MB |
| 训练数据 | bsd_merged (13,783 images: COCO BSD + BDD100K auto-labeled) |
