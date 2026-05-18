# BSD 交通参与方检测报警系统 — 设计文档

**日期**: 2026-05-18
**状态**: Draft
**硬件**: Allwinner V853 (VIP9000PICO NPU)
**参考**: DEPLOYMENT_GUIDE.md (V5 YOLO 部署流程)

---

## 1. 概述

在 V853 板端实现实时视频分析报警系统：
- **检测**：行人、非机动车（自行车/电瓶车、摩托车/三轮车）、机动车（备用，不参与报警）
- **报警**：基于矩形区域 + overlap 比例判定 + 连续帧计数
- **接口**：通过 `.so` 库对外提供 API，供板端其他进程调用

---

## 2. 架构

### 2.1 模块划分

```
调用方进程
    │
    ├─ libalarm_engine.so     ← 报警引擎（本系统输出）
    │    · 区域管理 (矩形, 3级)
    │    · 目标跟踪 (IOU 关联)
    │    · 报警判定 (overlap比例 + 连续帧)
    │
    └─ libdetect_engine.so    ← 检测引擎（本系统输出）
         · 视频采集 + letterbox 预处理
         · NPU 推理 (YOLO26s, VIP9000PICO)
         · 后处理解码 → 检测结果
```

两层通过**共享内存**传递检测结果，调用方直接链接 `libalarm_engine.so`。

### 2.2 数据流

```
摄像头帧 → letterbox 640×640 → RGB planar → NPU推理
  → boxes [1,4,8400] + scores [1,4,8400]
  → 坐标映射回原图 → 写入共享内存 (DetResult)
  → 报警引擎读取 → AABB overlap判定 → 连续帧计数 → 报警回调
```

---

## 3. 检测模型

### 3.1 模型规格

| 参数 | 值 |
|------|-----|
| 架构 | YOLO26s |
| 输入 | 640×640 RGB |
| 检测头 | 3头 (stride 8/16/32, 共 8400 cells) |
| 类别 | 0=person, 1=bicycle(含电瓶车), 2=motorcycle(含三轮车), 3=vehicle(备用) |
| 输出 | boxes [1,4,8400] + scores [1,4,8400] (raw logit) |
| 预训练 | yolo26s.pt (COCO) |

### 3.2 训练计划

- **主力数据集**: BDD100K + COCO，取 person/bicycle/motorcycle/car 四类（car 含 bus/truck 等机动车）
- **数据增强**: HSV 颜色抖动 + 低照增强 (dark/gamma, low_contrast, gaussian_noise)
- **Train/Val**: 80/20 随机切分
- **训练环境**: AutoDL 双卡, batch=128, 100 epochs, patience=30

### 3.3 部署流程

复用 DEPLOYMENT_GUIDE.md V5 链路：

```
训练 (AutoDL) → export ONNX → split YOLO head
  → remove Sigmoid → pegasus 量化 (uint8 asymmetric_affine)
  → network_binary.nb → 板端部署
```

差异点：类别数 4→4（class 3 vehicle 为备用，报警引擎忽略此类）。

---

## 4. 报警引擎

### 4.1 区域模型

- 矩形区域，归一化坐标 (0~1)
- 最多 3 级，Zone1 最内/最高优先级
- `alarm_set_zone(level, x1, y1, x2, y2, overlap_thr)` 动态配置

### 4.2 判定规则

```
对每帧每个检测目标:
  1. 计算 bbox 矩形 与 Zone 矩形的交集面积
  2. overlap_ratio = 交集面积 / bbox面积
  3. overlap_ratio >= threshold (默认 0.3) → 命中该 Zone
  4. 取命中最高级别 Zone
  5. 连续帧计数: 命中 frame_count++, 未命中清零
  6. frame_count >= N → 触发报警回调
```

### 4.3 目标跟踪

帧间 IOU 匹配，无 Kalman/DeepSORT。V853 ~10fps 帧率下相邻帧位移小，IOU 足够。

### 4.4 API

```c
int  alarm_init(const char* shm_name);
int  alarm_set_zone(int level, float x1, float y1, float x2, float y2, float overlap_thr);
int  alarm_set_frame_threshold(int n);       // 连续帧报警阈值
int  alarm_start();
int  alarm_stop();
int  alarm_register_callback(alarm_callback_t cb);
void alarm_deinit();
```

### 4.5 数据结构

```c
// 检测结果 (共享内存，每帧更新)
typedef struct {
    uint32_t  frame_id;
    uint32_t  timestamp_ms;
    uint32_t  count;
    DetObject  objects[MAX_DET];    // MAX_DET = 64
} DetResult;

typedef struct {
    int32_t  class_id;              // 0=person, 1=bicycle, 2=motorcycle, 3=vehicle(备用,不报警)
    float    conf;
    float    x1, y1, x2, y2;       // bbox 原图坐标
} DetObject;

// 报警事件
typedef struct {
    int    zone_level;
    int    class_id;
    float  overlap;
    int    frame_count;
} AlarmEvent;
```

---

## 5. 低照支持

- **训练侧**: Dark/LowContrast/Noise 数据增强 + BDD100K 自带夜间数据
- **推理侧**: 无额外处理，模型通过增强已适应暗光
- **硬件**: 如需要，摄像头选 IR-CUT + 红外补光灯

---

## 6. 项目文件结构

```
BSD/
├── DEPLOYMENT_GUIDE.md              # 现有 V5 部署参考
├── docs/superpowers/specs/          # 设计文档
├── training/detection/              # 模型训练
│   ├── configs/bsd_yolo26s.yaml
│   ├── train.py
│   └── export_onnx.py
├── deployment/
│   ├── quantize/                    # ONNX 拆分 + pegasus 量化
│   │   ├── split_yolo_head.py
│   │   ├── remove_sigmoid.py
│   │   └── prepare_calibration_data.py
│   └── board/
│       ├── detect_engine/           # libdetect_engine.so
│       │   ├── detect_engine.h
│       │   ├── detect_engine.c
│       │   ├── yolo_decode.c
│       │   ├── preprocess.c
│       │   └── Makefile
│       └── alarm_engine/            # libalarm_engine.so
│           ├── alarm_engine.h
│           ├── alarm_engine.c
│           ├── zone_mgr.c
│           ├── tracker.c
│           └── Makefile
└── examples/                        # 调用方示例
    └── demo_app.c
```

---

## 7. 待定 / 未来迭代

- 报警区域 4 点标定的**可视化标定工具**（现在先手动设矩形）
- 更复杂的报警规则（方向、速度、目标类型组合）
- 多摄像头支持
- 报警事件持久化 / 远程上报
- 电瓶车/三轮车的细粒度区分（需要标注数据补充）
