# BSD 交通参与者检测

V853 板端 BSD 盲区检测项目，当前检测 4 类交通参与者：

- `person`
- `bicycle`
- `motorcycle`
- `vehicle`

优先阅读：

| 文档 | 用途 |
|---|---|
| `BSD_DEV_GUIDE.md` | 主开发手册：数据、ONNX、量化、板端编译、部署 |
| `docs/BSD_PUBLIC_DATASET_PLAN.md` | 公开数据集路线和 BSD 精度目标 |
| `docs/candidates/bsd_v9p1_pv_priority_plan.md` | 当前优先候选 `YOLO26n_noattn + ReLU + 640` 的训练、精度和部署记录 |
| `docs/candidates/bsd_v7_yolo26n_640_public_kitti.md` | 稳定回退候选 `YOLO26n@640` 的指标、量化和板端命令 |
| `deployment/board/BOARD_ENGINE.md` | 板端检测/报警引擎说明 |
| `DEPLOYMENT_GUIDE.md` | 从 DMS 项目沉淀下来的 V853 部署历史记录 |

核心目录：

| 路径 | 用途 |
|---|---|
| `deployment/dataset/` | COCO/BDD100K 转 BSD 数据集脚本 |
| `deployment/quantize/` | ONNX 检测头拆分和校准数据工具 |
| `deployment/board/common/` | 检测和报警共用的 C ABI 类型 |
| `deployment/board/detect_engine/` | NPU 检测引擎和 `live_bsd` 实时链路 |
| `deployment/board/alarm_engine/` | 区域命中、跟踪和报警回调 |
| `training/detection/configs/` | BSD 训练配置 |
| `tools/` | PC、SSH、量化、验证辅助脚本，本仓库默认忽略 |

生成模型、板端二进制、SDK 和本地工具链默认不进 Git。统一放在 `artifacts/` 或 `_toolchain/`。
