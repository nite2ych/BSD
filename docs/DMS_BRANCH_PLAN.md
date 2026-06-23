# DMS/Cockpit 检测分支计划

本文档定义如何复用 BSD 检测器工作，启动新的 cockpit/DMS 分支。目标是保留已经验证过的 V853 部署链路，只替换产品相关的数据、标签、后处理和报警逻辑。

## 目标

基于 BSD 链路创建 cockpit/DMS 检测分支：

```text
dataset -> YOLO26 candidate -> ONNX export -> raw-logit split ONNX
  -> Pegasus NB -> V853 live video -> temporal post-processing -> event output
```

DMS 第一阶段目标应是同一板端路径上跑通单摄座舱检测器，而不是一次性做完整驾驶员监控产品。检测链路稳定后，再叠加更复杂的行为分析。

## 当前 BSD 基线快照

截至 2026-06-17，BSD 板端路线已从早期 v7/v9 对比阶段，推进到已选择 v9.1 base 候选：

| 项目 | 当前值 |
|---|---|
| BSD 当前优先候选 | `bsd_v9p1_yolo26n_noattn_relu_640_pv_priority` |
| 架构 | `YOLO26n_noattn + ReLU + 640` |
| PC 侧选择原因 | BSD 关键 person/vehicle recall 优于 v9 |
| 本地 PT | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/weights/best.pt` |
| 本地拆分 ONNX | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/best_640_split_nosig.onnx` |
| 本地 NB | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/bsd_v9p1_pv_base_640.nb` |
| 板端 NB | `/mnt/UDISK/bsd_v9p1_pv_base_640.nb` |
| NB 大小 | `2,111,168 bytes`，低于硬性 `<5 MB` 门槛 |
| 板端运行 | `/mnt/UDISK/live_bsd` 加载 v9.1 NB，支持 `preview` 和 `headless` |

`bdd_proxy_val` 上 v9 到 v9.1 的关键对比：

| 候选 | P | R | mAP50 | PV Recall | PV mAP50 |
|---|---:|---:|---:|---:|---:|
| v9 ReLU from v8 | `0.753` | `0.577` | `0.673` | `0.741` | `0.798` |
| v9.1 base | `0.720` | `0.602` | `0.677` | `0.760` | `0.797` |

v9.1 base 牺牲部分 precision 换取更好的 recall，符合 BSD 少漏检 person/vehicle 的优先级。v7 仍是稳定回退和对比锚点，但 v9.1 是当前可部署 BSD 候选。

## 建议分支

使用独立分支，避免影响 BSD 模型历史和板端默认配置：

```bash
git switch -c codex/dms-cockpit-detector
```

在 DMS 分支有自己的部署文件名前，不要动 BSD 板端模型和启动脚本。

## 可复用 BSD 资产

| BSD 资产 | DMS 是否复用 | 备注 |
|---|---|---|
| `training/detection/run_bsd_candidate.py` | 是，后续可重命名/泛化 | 已支持训练/评估/导出和激活函数覆盖 |
| `training/detection/models/yolo26n_noattn.yaml` | 是 | V853 上已验证的最快可部署形态 |
| `deployment/quantize/split_yolo_head.py` | 是 | 保持 raw-logit `boxes/scores` 输出 |
| Pegasus 量化流程 | 是 | 同样受 `<5 MB` NB 约束 |
| `deployment/board/detect_engine` | 大部分复用 | DMS ABI 定义后再重命名 |
| `live_bsd` camera/NPU loop | 可作为模板 | 将产品 overlay/alarm 从 camera/NPU loop 中拆出来 |
| headless/preview 模式 | 是 | 生产仍用 headless；preview 只调试 |
| profiling 字段 | 是 | 保留 `capture/preprocess/npu/decode/postprocess/display` 耗时 |
| display-frame 清理修复 | 是 | preview overlay 不能保留旧框或留下 framebuffer 残影 |

## 产品差异

BSD 是道路场景目标检测。DMS/cockpit 是座舱内乘员/行为检测。不要直接复用 BSD 类别或报警规则。

候选 DMS 类别集：

| 版本 | 类别 | 用途 |
|---|---|---|
| dms_v1_min | `driver`, `face`, `phone`, `smoking`, `seatbelt` | 实用第一版检测器 |
| dms_v1_occ | `person`, `face`, `hand`, `phone`, `seatbelt` | 如果下游行为逻辑依赖身体部位，更适合 |
| dms_v2_behavior | `eyes_closed`, `yawn`, `head_down`, `phone`, `smoking`, `seatbelt` | 需要更强标签和时序逻辑 |

建议第一版类别：

```text
driver/person, face, phone, smoking, seatbelt
```

原因：

- `driver/person` 提供占位和 ROI 锚点。
- `face` 支持后续视线/疲劳模型。
- `phone/smoking/seatbelt` 是常见 DMS 产品事件。
- 除非已有高质量数据，否则不要一开始就做细粒度 eye/yawn 标签。

## 模型建议

从 BSD 已验证最快结构开始：

```text
YOLO26n_noattn + ReLU + 640
```

理由：

- NB 大小应能稳定低于 `5 MB`。
- 板端 NPU 行为已有实测：v9/v9.1 路线约 `31-32 ms` NPU，单路 1280x720 detect call 总耗时约 `48-49 ms`。
- 避开了 VIPLite 上较慢的 attention 和 SiLU/Swish 算子。
- DMS 通常类别更少、ROI 更集中，该模型适合第一版部署候选。
- 最新 BSD v9.1 NB 约 `2.1 MB`，在硬性 `<5 MB` 限制下仍有余量。

除非没有更合适的预训练权重，否则 DMS 不建议从 BSD 交通权重开始。优先使用通用 YOLO26n 预训练权重或 DMS/座舱 checkpoint。BSD 权重有明显道路域偏置。

## 数据计划

DMS 必须重新构建座舱场景数据。公开道路数据不能直接迁移。

可评估数据源：

| 数据类型 | 例子 | 价值 |
|---|---|---|
| 驾驶员监控 | 公开 DMS 分心/疲劳数据集 | phone、smoking、gaze、fatigue 标签 |
| 座舱占位 | 车内 person/seat 数据集 | 乘员和座位 ROI 鲁棒性 |
| 通用检测 | COCO/OpenImages 子集 | 仅用于 phone/person/face 启动 |
| 内部采集 | 目标摄像头和红外条件 | 生产模型选择前必须有 |

数据规则：

- 按 video/session/person 划分 train/val/test，不按帧随机切。
- 避免相邻近重复帧主导训练。
- 在 manifest 中区分白天/夜晚/IR 曝光变体。
- 记录摄像头位置、镜头、分辨率、IR/可见光模式。
- 选择最终模型前必须构建 `dms_board_val`。

期望数据布局：

```text
datasets/dms_v1/
  train/images
  train/labels
  val/images
  val/labels
  board_val/images
  board_val/labels
  dataset.yaml
  board_val.yaml
  manifest.json
```

## 训练路线

新建 DMS 专用配置，不要直接改 BSD 配置：

```text
training/detection/configs/dms_v1_yolo26n_noattn_relu_640.yaml
training/detection/configs/dms_v1_yolo26n_noattn_relu_512.yaml  # 仅兜底
```

初始训练默认值：

| 项目 | 值 |
|---|---|
| architecture | `training/detection/models/yolo26n_noattn.yaml` |
| activation | `ReLU` |
| imgsz | `640` |
| batch | RTX 4090 上先从 `48` 开始 |
| epochs | `80-120`，开启 early stop |
| conf/nms eval | 固定 `0.5 / 0.45` 用于部署对比 |
| 主指标 | 安全事件逐类 recall，而不是只看 overall mAP |

DMS 模型选择沿用 BSD v9.1 经验：先看产品关键事件 recall，再看 precision 和 mAP。若 phone、smoking、driver/face 或 seatbelt 的 recall 有明显提升，precision 略低的模型也可能是更合适的部署模型。

## 板端运行路线

尽量让 camera/NPU loop 保持产品无关：

```text
live_video_core
  -> detect_engine
  -> product_postprocess
  -> product_event_sink
```

DMS 第一版可以先复制 `live_bsd.c` 为 `live_dms.c`，但后续应抽出共享 camera/NPU 代码，避免 BSD 和 DMS 分叉漂移。

部署文件名必须 DMS 专用：

```text
/mnt/UDISK/live_dms
/mnt/UDISK/dms_v1_yolo26n_noattn_relu_640.nb
/mnt/UDISK/dms_live_autostart.log
/etc/init.d/dms_live
```

不要覆盖：

```text
/mnt/UDISK/live_bsd
/mnt/UDISK/bsd_v7_yolo26n_640.nb
/mnt/UDISK/bsd_v9_relu_from_v8_640.nb
/mnt/UDISK/bsd_v9p1_pv_base_640.nb
```

当前 BSD preview 模式实测：

| 分段 | 典型耗时 |
|---|---:|
| NV21 preprocess | `15.8-16.0 ms` |
| NPU | `31.7-31.8 ms` |
| decode | `1.1-1.2 ms` |
| detect total | `48.7-49.0 ms` |
| framebuffer preview | 约 `89 ms` |
| preview FPS | 约 `9.4-9.5 FPS` |

不要把 preview FPS 当作生产 FPS。生产运行必须在 `headless` 模式下测。

## 运行时架构方向

当前 `live_bsd` 主要用于打通 camera、NPU、显示和报警集成，整体仍偏单线程：

```text
DQBUF -> preprocess -> NPU -> decode -> preview draw -> QBUF
```

这足够用于单摄调试，但不是四路 BSD 或 DMS 产品运行时的目标形态。下一步共享运行时应改成生产者/消费者流水线：

```text
camera capture thread(s)
  -> latest-frame slots
  -> preprocess worker
  -> single NPU worker
  -> postprocess/alarm worker
  -> optional low-frequency preview worker
```

共享运行时规则：

- 每路只保留最新帧；丢弃过期帧，不堆积旧图队列。
- 尽量让 CPU 预处理与 NPU 推理重叠。
- 除非 SDK 证明并发提交安全且更快，否则 NPU 访问保持串行。
- preview 不能在检测关键路径上。调试 preview 可以低 FPS 绘制最新结果。
- 产品事件状态和显示状态分离。显示只画最近一次推理；报警逻辑仍可使用短暂 missed-frame hold 和时序确认。

## 视频后处理

DMS 必须依赖视频级逻辑。单帧检测不足以支撑可靠报警。

建议第一版后处理：

| 层 | DMS 用途 |
|---|---|
| ROI gating | 驾驶席 / face 区域 / hand-phone 区域 |
| temporal confirm | 事件持续 N 帧才报警 |
| short hold | 目标漏检 1-3 帧仍保持活跃 |
| lightweight tracking | 初期 IoU + center distance 即可 |
| event state machine | `pending -> active -> cooldown` |
| class-specific rules | Phone/smoking 需要持续确认；seatbelt 可用更长确认 |

最新 BSD preview 问题是一个重要提醒：显示平滑和报警跟踪是两件事。BSD 曾因 preview 渲染保留自己的 held/smoothed boxes 且 framebuffer page 未完全清理，导致旧框残留。DMS preview 应只绘制当前/最新推理结果，而时序保持和平滑应放在事件逻辑中。

不要把这些规则写进检测模型。它们应作为产品逻辑保留，保证同一检测引擎可以支持 BSD、DMS 和其他产品。

## 建议代码布局

短期：

```text
deployment/board/dms_engine/
  live_dms.c
  dms_postprocess.c
  dms_postprocess.h
  Makefile
```

长期：

```text
deployment/board/video_core/
  v4l2_capture.c
  awnn_runner.c
  framebuffer_preview.c
  profiler.c
  pipeline_scheduler.c

deployment/board/products/
  bsd/
  dms/
```

先跑通 DMS，再抽共享代码。不要让大型重构阻塞第一版分支。

## 第一阶段检查清单

1. 创建 `codex/dms-cockpit-detector`。
2. 定义 DMS 类别列表和 `dataset.yaml`。
3. 训练前准备小规模 DMS 验证集。
4. 训练 `dms_v1_yolo26n_noattn_relu_640`。
5. 导出 ONNX，并拆分 raw-logit `boxes/scores`。
6. Pegasus 量化并确认 NB `<5 MB`。
7. 在板端运行等价 `test_npu_direct`。
8. 跑通 `live_dms headless`。
9. 增加 ROI + temporal confirmation 后处理。
10. 在声明最终模型前构建 `dms_board_val`。

## 从 BSD 继承的经验

- 当图结构 NPU 友好时，固定 `640x640` 仍能满足大小和速度。
- VIPLite 对算子映射差时，移除 attention 比缩小输入更重要。
- 在当前板端路径上，ReLU 明显快于 SiLU/Swish。
- preview 模式会显著拉低表观 FPS；生产性能必须测 headless。
- 当前板端 preview 绘制可达到约 `89 ms/frame`，即使 NPU 只有约 `32 ms`。
- CPU 预处理和 NPU 推理应流水并行；否则有效 detect call 是 `preprocess + NPU + decode`，当前约 `49 ms`。
- 四路运行时应使用每路 latest-frame slot，避免旧帧队列堆积。
- 不要只按 overall mAP 选模型；要看产品关键类别 recall 和事件级验证。
- v9.1 取代 v9 的原因是 `bdd_proxy_val` 上 BSD 关键 person/vehicle recall 从 `0.741` 提升到 `0.760`，尽管 precision 降低。
- 板端稳定回退和候选文件名必须分开。
- 没有真实板端域验证数据，不能定最终模型。
