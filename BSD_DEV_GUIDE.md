# BSD 交通参与方检测报警系统 — 开发流程手册

从训练数据准备到板端部署验证的完整操作手册。

## 机器分工

| 机器 | IP | 用途 |
|------|-----|------|
| AutoDL 训练服务器 | autodl.com/console | GPU 训练，导出 ONNX |
| Ubuntu18 编译机 | 192.168.144.136 | ARM 交叉编译（TINA SDK） |
| Ubuntu20 | 192.168.144.133 | pegasus 量化、runtime 编译验证 |
| V853 板端 | COM5 ADB | 部署测试 |

## 模型规格

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 640×640 |
| 输入格式 | RGB planar (R plane + G plane + B plane, uint8) |
| 类别 | person=0, bicycle(含电瓶车)=1, motorcycle(含三轮车)=2, vehicle(备用)=3 |
| 检测头 | 3 头 (stride 8/16/32) |
| 总 cell 数 | 8400 (80×80 + 40×40 + 20×20) |
| 输出 | boxes [1,4,8400] + scores [1,4,8400] (logit, 无 Sigmoid) |

---

## 步骤 1：准备训练数据

**机器**: 本地 Windows

从 COCO + BDD100K 提取 4 分类数据，合并 car/truck/bus → vehicle：

```bash
export COCO_PATH=/path/to/coco
python training/detection/dataset/prepare_bsd_dataset.py
```

产物：`datasets/bsd/train/` 和 `datasets/bsd/val/`（YOLO 格式）

**重要**：COCO car/bus/truck 三者合并为 class 3 (vehicle)

---

## 步骤 2：训练与导出 ONNX

**机器**: AutoDL 训练服务器

```bash
# 训练
python training/detection/train.py \
  --config training/detection/configs/bsd_yolo26s.yaml

# 导出 ONNX (640×640, NMS 内置)
python training/detection/export_onnx.py \
  --weights artifacts/detection/bsd_yolo26s/best.pt \
  --imgsz 640 \
  --output artifacts/detection/bsd_yolo26s/best.onnx
```

导出产物：`best.onnx`（输入 [1,3,640,640], 输出 [1,300,6] 含 NMS）

### 训练要点

- **数据集**：COCO + BDD100K，person/bicycle/motorcycle/car(truck+bus) 4 类
- **数据增强**：低照增强（dark/infrared/low_contrast）+ HSV 颜色抖动
- **Train/Val 切分**：80/20 随机切分
- **预训练权重**：yolo26s.pt（继承 COCO 预训练）
- **训练设置**：双 RTX 4090 DDP, batch=128, 100 epochs, patience=30
- **Per-class 验证**：检查 person/bicycle/motorcycle/vehicle 各类 AP

---

## 步骤 3：拆分 YOLO 头 + 移除 Sigmoid

**机器**: 本地 Windows

```bash
# 3a. 拆分：保留 boxes [1,4,N] + scores [1,4,N]，移除 NMS 后处理
python deployment/quantize/split_yolo_head.py \
  --input artifacts/detection/bsd_yolo26s/best.onnx \
  --output artifacts/detection/bsd_yolo26s/best_640_split.onnx

# 3b. 移除 Sigmoid：scores 直接输出 logit
python deployment/quantize/remove_sigmoid.py \
  artifacts/detection/bsd_yolo26s/best_640_split.onnx \
  artifacts/detection/bsd_yolo26s/best_640_split_nosig.onnx
```

产物：`best_640_split_nosig.onnx`（输出 boxes [1,4,8400] + scores [1,4,8400] raw logit）

---

## 步骤 4：准备校准数据

**机器**: 本地 Windows

```bash
python deployment/quantize/prepare_calibration_data.py
```

将校准图片和 ONNX 文件上传到 Ubuntu20：

```bash
scp artifacts/detection/bsd_yolo26s/best_640_split_nosig.onnx \
    ubuntu@192.168.144.133:/home/ubuntu/bsd_quantize/
scp -r deployment/quantize/calibration_images_bsd/* \
    ubuntu@192.168.144.133:/home/ubuntu/bsd_quantize/calib_images/
```

---

## 步骤 5：生成校准数据集文件

**机器**: Ubuntu20

```bash
cd /home/ubuntu/bsd_quantize
find $(pwd)/calib_images -name '*.jpg' > dataset.txt
```

建议 200-300 张校准图片，覆盖所有类别。

---

## 步骤 6：pegasus 量化

**机器**: Ubuntu20

```bash
PEGASUS=/home/ubuntu/VeriSilicon/acuity-toolkit-binary-6.9.3/bin/acuitylib/pegasus
VIV_SDK=$HOME/VeriSilicon/VivanteIDE5.7.1/cmdtools/vsimulator

NAME=bsd_640
OUT_DIR=/home/ubuntu/bsd_quantize/out_640
mkdir -p $OUT_DIR && cd $OUT_DIR

# 6a. 导入 ONNX
cd $(dirname $PEGASUS) && ./pegasus import onnx \
  --model /home/ubuntu/bsd_quantize/best_640_split_nosig.onnx \
  --output-data ${NAME}.data \
  --output-model ${NAME}.json

# 6b. 生成 inputmeta 模板
./pegasus generate inputmeta \
  --model ${NAME}.json \
  --input-meta-output ${NAME}_inputmeta.yml

# 6c. 手动编辑 inputmeta.yml
#   - 修改 path: 指向 dataset.txt
#   - scale: 0.0039
#   - reverse_channel: false (RGB 输入)
#   - mean: [0, 0, 0]

# 6d. 量化（asymmetric_affine, uint8）
./pegasus quantize \
  --model ${NAME}.json \
  --model-data ${NAME}.data \
  --batch-size 1 \
  --device CPU \
  --with-input-meta /home/ubuntu/bsd_quantize/${NAME}_inputmeta.yml \
  --rebuild \
  --model-quantize ${NAME}.quantize \
  --quantizer asymmetric_affine \
  --qtype uint8

# 6e. 导出 NB
./pegasus export ovxlib \
  --model ${NAME}.json \
  --model-data ${NAME}.data \
  --dtype quantized \
  --model-quantize ${NAME}.quantize \
  --batch-size 1 \
  --save-fused-graph \
  --target-ide-project linux64 \
  --with-input-meta /home/ubuntu/bsd_quantize/${NAME}_inputmeta.yml \
  --output-path ovxlib/${NAME}/${NAME}prj \
  --pack-nbg-unify \
  --optimize VIP9000PICO_PID0XEE \
  --viv-sdk ${VIV_SDK}
```

产物：`ovxlib/${NAME}/${NAME}_nbg_unify/network_binary.nb`

**关键踩坑提醒**（参考 DEPLOYMENT_GUIDE.md 步骤 6e）：
1. pegasus 必须在 bin 目录下运行
2. 先用 `generate inputmeta` 生成模板再编辑，确认 lid 名
3. 需要 Vivante 模拟器库 (`ldconfig`)
4. `reverse_channel: false`（模型训练用 RGB）
5. 校准数据路径必须绝对路径
6. NB 文件固定名为 `network_binary.nb`，不要用 `export.data`

---

## 步骤 7：交叉编译板端程序

**机器**: Ubuntu18 (192.168.144.136)

### 7a. 编译 libdetect_engine.so

```bash
cd deployment/board/detect_engine
make clean && make
```

### 7b. 编译 libalarm_engine.so

```bash
cd deployment/board/alarm_engine
make clean && make
```

### 7c. 编译 demo_app

```bash
cd examples
# 链接两个 .so
make
```

---

## 步骤 8：部署到板端

**机器**: 本地 Windows PowerShell

```powershell
$adb = "D:\WORK\TOOL\AllwinnertechPhoeniSuit (1)\AllwinnertechPhoeniSuitRelease20201225\adb.exe"

# 检查连接
& $adb devices

# 推送文件
& $adb push artifacts/detection/bsd_yolo26s/nb/network_binary.nb /mnt/UDISK/bsd_640.nb
& $adb push deployment/board/detect_engine/libdetect_engine.so /mnt/UDISK/
& $adb push deployment/board/alarm_engine/libalarm_engine.so /mnt/UDISK/
& $adb push examples/demo_app /mnt/UDISK/bsd_demo

# 运行
& $adb shell "chmod +x /mnt/UDISK/bsd_demo"
& $adb shell "export LD_LIBRARY_PATH=/mnt/UDISK && /mnt/UDISK/bsd_demo"
```

---

## 步骤 9：PC 端验证（板端部署前）

在 PC 端用 ONNX Runtime 验证模型精度，确保检测效果 OK 再走量化+板端流程：

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

def letterbox(img, new_shape=640):
    w, h = img.size
    scale = new_shape / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2
    result = Image.new('RGB', (new_shape, new_shape), (114, 114, 114))
    result.paste(img, (pad_w, pad_h))
    return result, pad_w, pad_h, scale

session = ort.InferenceSession("best_640_split_nosig.onnx")
img = Image.open("test.jpg").convert("RGB")
img_lb, pad_x, pad_y, scale = letterbox(img, 640)
arr = np.array(img_lb, dtype=np.float32) / 255.0
arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

boxes, scores = session.run(None, {"images": arr})

for c, name in enumerate(["person", "bicycle", "motorcycle", "vehicle"]):
    logit = float(scores[0, c, :].max())
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    print(f"  {name}: logit={logit:.4f} sigmoid={sigmoid:.4f}")
```

---

## 关键脚本与文件索引

| 文件 | 用途 | 运行机器 |
|------|------|---------|
| `training/detection/train.py` | 训练 YOLO 检测模型 | AutoDL |
| `training/detection/export_onnx.py` | 导出 PyTorch → ONNX | AutoDL/本地 |
| `training/detection/dataset/prepare_bsd_dataset.py` | 准备 BDD100K+COCO 4分类数据 | 本地 |
| `deployment/quantize/split_yolo_head.py` | 裁掉 NMS 后处理 | 本地 |
| `deployment/quantize/remove_sigmoid.py` | 移除 Sigmoid | 本地 |
| `deployment/quantize/prepare_calibration_data.py` | 准备校准图片集 | 本地 |
| `deployment/board/detect_engine/` | 检测引擎源码 (libdetect_engine.so) | Ubuntu18 编译 |
| `deployment/board/alarm_engine/` | 报警引擎源码 (libalarm_engine.so) | Ubuntu18 编译 |
| `examples/demo_app.c` | 调用方示例 | Ubuntu18 编译 |

---

## 更新记录

- 2026-05-18: 初版，基于 V5 YOLO 部署流程，BSD 4 分类模型 + 报警引擎
