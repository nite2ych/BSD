# 板端代码

V853 板端 BSD 检测和报警链路源码。

| 路径 | 用途 |
|---|---|
| `common/` | 检测和报警引擎共用的 C ABI 类型 |
| `detect_engine/` | AWNN 模型加载、预处理、YOLO 解码、NMS、共享内存输出 |
| `alarm_engine/` | 区域命中、IOU 跟踪、短暂保持、平滑、冷却、连续帧报警逻辑 |
| `test_v4/` | 板端模型测试代码 |
| `test_bdd100k/` | BDD100K 板端测试代码，当前默认不跟踪 |
| `camera/` | 摄像头 bring-up 和调试脚本，当前仍属实验目录 |

主要入口：

| 文件 | 用途 |
|---|---|
| `detect_engine/main.c` | stdin BGR raw 测试程序，编译为 `bsd_detect` |
| `detect_engine/live_bsd.c` | 实时 V4L2 camera -> NPU -> framebuffer 程序 |
| `detect_engine/test_npu_direct.c` | raw BGR 文件 -> NB/NPU 直连对比测试程序 |
| `compile_engines.py` | 旧版远程编译辅助脚本 |
| `compile_live.py` | `live_bsd` 和 `test_npu_direct` 远程编译辅助脚本，当前默认不跟踪 |

生成的 object、动态库、板端可执行文件和复制出来的 SDK 库默认被 Git 忽略。

## 视频可靠性层

模型输出是单帧检测结果。BSD 运行时在输出报警前，通过 `alarm_engine` 增加一层轻量视频可靠性处理：

| 步骤 | 用途 | 默认值 |
|---|---|---|
| 区域命中 | 只在配置的盲区 ROI 内报警 | zone 0 中心区域，zone 1 全画面 |
| IoU 关联 | 跨帧保持目标身份 | `match_iou=0.30` |
| BBox 平滑 | 在区域/报警输出前降低框抖动 | `smooth_alpha=0.65` |
| 连续确认 | 避免单帧误报 | `alarm_frames=3` |
| 短暂保持 | 承受检测器短暂漏检 | `max_missed=3` |
| 冷却 | 目标离开 ROI 后抑制重复报警 | `cooldown=15` |
| 重复上报间隔 | 长时间活跃报警按固定节奏重复上报 | `reemit=15` |
| 类别开关 | 控制哪些类别允许报警 | `live_bsd` 默认 person/bicycle/motorcycle/vehicle 全开；vehicle 可关闭 |

`DetResult` 和 `AlarmEvent` ABI 不变。可靠性层通过 `alarm_set_tracker_params()` 和 `alarm_set_class_enabled()` 配置。

`live_bsd` 保持旧参数兼容，并在末尾追加可选报警参数：

```sh
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v9_relu_from_v8_640.nb \
  0.5 0.45 0.5 0.5 1280 720 headless 640 \
  3 3 15 15 0.65 1
```

最后 6 个值含义：

```text
alarm_frames max_missed cooldown reemit smooth_alpha vehicle_alarm
```

早期板端测试建议保持 `alarm_frames=3` 和 `max_missed=3`。只有在 ROI 调整后仍有误报时，再提高 `alarm_frames`。
