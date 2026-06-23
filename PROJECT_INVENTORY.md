# BSD 工程清单

本文档把当前工程拆成可维护源码、本地生成产物和实验脚本，方便后续接管开发和提交管理。

## 可维护核心

这些文件和目录是主线代码与文档，通常应保留在版本控制中：

| 路径 | 作用 |
|---|---|
| `README.md` | 工程入口 |
| `BSD_DEV_GUIDE.md` | 主开发手册 |
| `DEPLOYMENT_GUIDE.md` | V853/DMS 历史部署参考 |
| `deployment/board/BOARD_ENGINE.md` | 板端引擎参考 |
| `deployment/board/common/` | 共用 ABI 类型 |
| `deployment/board/detect_engine/` | 检测引擎和实时 `live_bsd` 链路 |
| `deployment/board/alarm_engine/` | 区域、跟踪和报警引擎 |
| `deployment/board/test_v4/` | 板端模型测试代码 |
| `deployment/dataset/` | 数据集转换/准备脚本 |
| `deployment/quantize/` | ONNX 拆分和校准脚本 |
| `examples/` | 集成示例 |
| `training/detection/configs/` | 训练配置 |

## 待确认源码

这些内容看起来是源码或有用工具，但提交前需要逐个确认，避免把临时脚本、主机信息或密码带进仓库：

| 路径 | 说明 |
|---|---|
| `deployment/board/detect_engine/live_bsd.c` | 实时 camera/NPU/framebuffer 链路 |
| `deployment/board/detect_engine/awnn.h` | 板端编译需要的本地 AWNN 声明 |
| `deployment/board/detect_engine/test_npu_direct.c` | NPU 直连冒烟测试 |
| `deployment/board/detect_engine/*.sh` | 板端编译辅助脚本 |
| `deployment/board/compile_live.py` | 远程上传并编译 `live_bsd` |
| `deployment/board/check_camera.py` | 板端摄像头诊断辅助脚本 |
| `deployment/board/test_bdd100k/` | BDD100K 板端测试 |
| `deployment/quantize/prepare_calibration_data.py` | 校准图生成器 |
| `deployment/quantize/split_yolo_head_sig.py` | sigmoid 输出拆分实验 |
| `tools/` | PC、SSH、验证和量化辅助工具 |
| `training/detection/configs/bsd_yolo26s*.yaml` | BSD 训练配置 |

建议优先纳入版本控制的集合：

```text
README.md
PROJECT_INVENTORY.md
deployment/board/detect_engine/awnn.h
deployment/board/detect_engine/live_bsd.c
deployment/board/detect_engine/test_npu_direct.c
deployment/board/detect_engine/*.sh
deployment/board/compile_live.py
deployment/board/check_camera.py
deployment/board/test_bdd100k/
deployment/quantize/prepare_calibration_data.py
deployment/quantize/split_yolo_head_sig.py
training/detection/configs/
```

提交前继续人工检查：

```text
tools/
deployment/board/camera/
root-level *.py scripts
DEBUG_RECORD.md
```

待检查集合里有有用的调试历史，但不少文件包含硬编码机器路径、主机名或凭据。

## 本地产物

这些内容刻意被忽略，不应提交：

| 路径/模式 | 内容 |
|---|---|
| `artifacts/` | 模型、NB 文件、板端二进制、测试帧、校准图 |
| `_toolchain/` | SDK/工具链缓存 |
| `*.pt`, `*.onnx`, `*.nb`, `*.raw` | 模型和验证产物 |
| `*.o`, `*.so`, board executables | 编译输出 |
| `__pycache__/` | Python 缓存 |

## 实验脚本和清理候选

根目录有很多一次性服务器/模型脚本，例如 `check_server.py`、`download_v4_model.py`、`quick_val.py`、`test_pt_model.py`。它们可能仍有参考价值，但与 `tools/` 下脚本重复，并且经常包含硬编码主机或凭据。

建议清理路径：

1. 可复用脚本统一放到 `tools/`。
2. 一次性脚本移到 `tools/experiments/`，或确认废弃后删除。
3. 硬编码凭据改成环境变量。
4. 生成输出统一放到 `artifacts/`。
