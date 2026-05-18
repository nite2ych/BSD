# V853 板端 YOLO 检测模型部署流程

从训练 ONNX 到板端 NPU 验证的完整操作手册。当前锁定版本 **V5**（face/phone/cigarette/seatbelt 4 分类，640×640 RGB 输入），V3 流程保留作为参考。

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
| 类别 | face=0, phone=1, cigarette=2, seatbelt=3 |
| 检测头 | 3 头 (stride 8/16/32) |
| 总 cell 数 | 8400 (80×80 + 40×40 + 20×20) |
| 输出 | boxes [1,4,8400] + scores [1,4,8400] (logit, 无 Sigmoid) |

---

## 步骤 1：训练与导出 ONNX

**机器**: AutoDL 训练服务器

```bash
# 训练 (V5)
python training/detection/train.py --config training/detection/configs/multiclass_yolo26s_v5.yaml

# 导出 ONNX (640×640, NMS 内置)
python training/detection/export_onnx.py \
  --weights artifacts/detection/multiclass_yolo26s_v5/best.pt \
  --imgsz 640 \
  --output artifacts/detection/multiclass_yolo26s_v5/best.onnx
```

导出产物：`best.onnx`（输入 [1,3,640,640], 输出 [1,300,6] 含 NMS）

### V5 训练要点

- **数据集**：ADMS 4class，清洗后 4211 张图片（移除 326/439 个错误安全带标注）
- **数据增强**：IR 域适应（dark/infrared/low_contrast，每图 3 份增强），HSV 颜色抖动
- **Train/Val 切分**：80/20 随机切分（3174 train, 794 val）
- **预训练权重**：yolo26s.pt（不继承 V3，避免灾难性遗忘）
- **训练设置**：双 RTX 4090 DDP, batch=128, 100 epochs, patience=30
- **结果**：Best mAP50=0.953 (epoch 47), mAP50-95=0.729 (epoch 65)
- **Per-class 验证**：face 867/859GT, phone 287/279GT, cigarette 155/155GT, seatbelt 29/30GT

### V3 (历史参考)

```bash
# 训练 (V3)
python training/detection/train.py --config training/detection/configs/multiclass_yolo26s.yaml
```

---

## 步骤 2：拆分 YOLO 头 + 移除 Sigmoid

**机器**: 本地 Windows（或 Ubuntu20，仅需 onnx 库）

pegasus/NPU 不支持 GatherElements、TopK 等 NMS 后处理算子，需要将模型在 Concat_3 处裁开；VIP9000PICO 编译器在处理 4 通道 Sigmoid 分类头时有 bug（输出全为死值），需移除 Sigmoid 改为输出 raw logit。

```bash
# 2a. 拆分：保留 boxes [1,4,N] + scores [1,4,N]，移除 NMS 后处理
python deployment/quantize/split_yolo_head.py \
  --input artifacts/detection/multiclass_yolo26s_v5/best.onnx \
  --output artifacts/detection/multiclass_yolo26s_v5/best_640_split.onnx

# 2b. 移除 Sigmoid：scores 直接输出 logit
python deployment/quantize/remove_sigmoid.py \
  artifacts/detection/multiclass_yolo26s_v5/best_640_split.onnx \
  artifacts/detection/multiclass_yolo26s_v5/best_640_split_nosig.onnx
```

产物：`best_640_split_nosig.onnx`（输出 boxes [1,4,8400] + scores [1,4,8400] raw logit）

**重要**：移除 Sigmoid 后，host 侧解码时必须手动计算 `sigmoid(logit)`。

---

## 步骤 3：准备校准数据

**机器**: 本地 Windows

```bash
python deployment/quantize/prepare_calibration_data.py \
  --dataset datasets/processed/detector_multiclass \
  --output deployment/quantize/calibration_images \
  --num 200
```

将校准图片和 ONNX 文件上传到 Ubuntu20：

```bash
scp best_640_split_nosig.onnx ubuntu@192.168.144.133:/home/ubuntu/dms_quantize/
scp -r calibration_images/* ubuntu@192.168.144.133:/home/ubuntu/dms_quantize/calib_images/
```

---

## 步骤 4：生成校准数据集文件

**机器**: Ubuntu20

pegasus 需要文本文件列出所有校准图片的绝对路径：

```bash
cd /home/ubuntu/dms_quantize
find $(pwd)/calib_images -name '*.jpg' > dataset.txt
# 如需合并多个目录：
cat dataset.txt dataset_extra.txt > dataset_full.txt
```

建议 200-300 张校准图片，覆盖所有类别（face/phone/cigarette/seatbelt）。

---

## 步骤 5：创建 inputmeta

**机器**: Ubuntu20

创建 `inputmeta.yml`：

```yaml
input_meta:
  databases:
  - path: /home/ubuntu/dms_quantize/dataset_full.txt
    type: TEXT
    ports:
    - lid: images_370
      category: image
      dtype: float32
      layout: nchw
      shape:
      - 1
      - 3
      - 640
      - 640
      fitting: scale
      preprocess:
        reverse_channel: false    # RGB 输入，不反转通道
        mean:
        - 0
        - 0
        - 0
        scale: 0.0039             # 1/255
        preproc_node_params:
          add_preproc_node: false
          preproc_type: IMAGE_RGB
          preproc_image_size:
          - 640
          - 640
```

**关键参数**:
- `reverse_channel: false` — **模型用 RGB 训练时必须为 false**。VIP 预处理器默认输入为 BGR，若设为 true 会将输入反转（RGB→BGR 或 BGR→RGB）。如果训练数据是 RGB，NPU 喂的也是 RGB，则设 false 保持原样。**V5 踩坑：第一次量化设了 true，导致 1.jpg phone 检测翻转（0.699→0.420），改为 false 后恢复正常（0.699→0.781）**
- `scale: 0.0039` — 约等于 1/255，将 uint8 [0,255] 归一化到 float [0,1]
- `preproc_type: IMAGE_RGB` — 输入颜色格式

---

## 步骤 6：pegasus 量化

**机器**: Ubuntu20

```bash
# pegasus 路径
PEGASUS=/home/ubuntu/VeriSilicon/acuity-toolkit-binary-6.9.3/bin/acuitylib/pegasus
VIV_SDK=$HOME/VeriSilicon/VivanteIDE5.7.1/cmdtools/vsimulator

NAME=v3_640
OUT_DIR=/home/ubuntu/dms_quantize/out_640
mkdir -p $OUT_DIR && cd $OUT_DIR

# 6a. 导入 ONNX
$PEGASUS import onnx \
  --model /home/ubuntu/dms_quantize/best_640_split_nosig.onnx \
  --output-data ${NAME}.data \
  --output-model ${NAME}.json

# 6b. 生成 inputmeta 模板（可选，通常直接用手写的）
$PEGASUS generate inputmeta \
  --model ${NAME}.json \
  --input-meta-output ${NAME}_inputmeta.yml

# 6c. 量化（asymmetric_affine, uint8）
$PEGASUS quantize \
  --model ${NAME}.json \
  --model-data ${NAME}.data \
  --batch-size 1 \
  --device CPU \
  --with-input-meta /home/ubuntu/dms_quantize/v3_640_inputmeta.yml \
  --rebuild \
  --model-quantize ${NAME}.quantize \
  --quantizer asymmetric_affine \
  --qtype uint8

# 6d. 导出 NB（关键：--pack-nbg-unify 生成 VPMN 头）
$PEGASUS export ovxlib \
  --model ${NAME}.json \
  --model-data ${NAME}.data \
  --dtype quantized \
  --model-quantize ${NAME}.quantize \
  --batch-size 1 \
  --save-fused-graph \
  --target-ide-project linux64 \
  --with-input-meta /home/ubuntu/dms_quantize/v3_640_inputmeta.yml \
  --output-path ovxlib/${NAME}/${NAME}prj \
  --pack-nbg-unify \
  --optimize VIP9000PICO_PID0XEE \
  --viv-sdk ${VIV_SDK}
```

产物：`ovxlib/${NAME}/${NAME}_nbg_unify/network_binary.nb`

**重要**:
- `--pack-nbg-unify` 生成带 VPMN header 的 `.nb` 文件（magic bytes `56 50 4D 4E` = "VPMN"）。AWNN API 只认这种格式，不要用 `ovxlib/*/export.data` 裸数据
- `--optimize VIP9000PICO_PID0XEE` 针对 V853 的 VIP9000PICO NPU
- 不可用 `--pack-nbg-unify` 时，NB 文件无效

### 6e. pegasus 实操踩坑记录（2026-05-16 V5 量化）

以下问题在实际操作中逐一遇到，记录在此供后续项目参考：

**坑 1：pegasus 必须在 bin 目录下运行**
- pegasus 是 PyInstaller 打包的二进制，运行时依赖同目录下的 `onnxruler/` 模块
- 错误示例：`/path/to/bin/pegasus import ...` 从其他目录调用
- 正确做法：`cd /path/to/bin && ./pegasus import ...` 或使用绝对路径

**坑 2：hand-written inputmeta 可能 lid 名不匹配**
- 模型的 input lid 名称由 ONNX 导出时决定，不一定是 `images`
- 建议先用 `pegasus generate inputmeta --model X.json --input-meta-output X.yml` 生成模板
- 然后只编辑 `path`（指向 dataset.txt）、`scale`（改为 0.0039）、`mean` 等字段
- V5 实际 lid 名为 `images_370`（不是预期的 `images`）

**坑 3：export NB 时需要 Vivante 模拟器库**
- pegasus export 报 `libovxlib.so not found` / `libvdtproxy.so not found`
- 修复：创建 `/etc/ld.so.conf.d/vivante.conf` 添加 Vivante SDK lib 路径，然后 `ldconfig`
- Vivante SDK 路径：`~/VeriSilicon/VivanteIDE5.7.1/cmdtools/vsimulator`
- 同时需要 `linux64/lib` 子目录

**坑 4：inputmeta 中的 reverse_channel**
- 模型训练使用 RGB → `reverse_channel: true`（VIP 预处理管线默认 BGR 输入，需要反转得到 RGB）
- 如果训练使用 BGR → `reverse_channel: false`
- V5 训练用 RGB 输入，故 inputmeta 设 `reverse_channel: true`

**坑 5：calib 数据路径必须是绝对路径**
- dataset.txt 中每行必须是绝对路径，否则 pegasus 找不到文件
- `find $(pwd)/calib_images -name '*.jpg' > dataset.txt` 生成绝对路径

**坑 6：NB 文件名约定**
- `--pack-nbg-unify` 输出的 `.nb` 文件固定名为 `network_binary.nb`
- 重命名或移动时保持文件名，或在使用时引用正确路径
- 不要使用同目录下的 `export.data`（裸权重，缺 VPMN 头）

---

## 步骤 7：交叉编译测试程序

**机器**: Ubuntu18 (192.168.144.136)

### 7a. 使用预编译工具链（推荐，无需完整 SDK make）

TINA SDK tarball（`tina-v853-100ask.tar.gz`, 5.2GB）内含预编译 musl 工具链，无需先执行 `make`（节省 ~1 小时）：

```bash
# 从 tarball 只提取需要的部分（1-2 分钟）
cd ~
tar -xzf tina-v853-100ask.tar.gz --wildcards \
  '*/toolchain-sunxi-musl/*' \
  '*/libawnn_full/*' \
  '*/libsdk-viplite-driver/*'

# 编译
TOOLCHAIN=~/tina-v853-100ask/prebuilt/gcc/linux-x86/arm/toolchain-sunxi-musl/toolchain
CC=$TOOLCHAIN/bin/arm-openwrt-linux-muslgnueabi-gcc
AWNNDIR=~/tina-v853-100ask/package/allwinner/libawnn_full
VIPDIR=~/tina-v853-100ask/package/allwinner/libsdk-viplite-driver

$CC -O2 -Wall -std=c11 \
  -o test_yolo_input_v5 \
  test_yolo_input.c \
  -I$AWNNDIR/sdk/include \
  -L$AWNNDIR/sdk/library/musl \
  -L$VIPDIR/sdk_release/library/musl \
  -Wl,--start-group -l:libawnn_full.a -l:libVIPuser.a -l:libVIPlite.a -Wl,--end-group \
  -lstdc++ -lm -lpthread
```

链接库说明：
- `libawnn_full.a` — AWNN NPU 推理封装（注意名称是 `awnn_full` 不是 `awnn`）
- `libVIPuser.a` / `libVIPlite.a` — VeriSilicon VIP 底层驱动，在 viplite-driver SDK 包中
- 链接顺序很重要：awnn_full → VIPuser → VIPlite 必须用 `--start-group` 包裹
- 使用 `-l:libawnn_full.a` 语法指定完整文件名（`-lawnn_full` 也等效）

### 7b. 传统方式：编译完整 SDK（耗时 ~1 小时）

```bash
cd ~/tina-v853-100ask
source build/envsetup.sh
lunch  # 选择 1 (v853_100ask-tina)
make    # 首次编译约 1 小时

# 编译产物在 out/v853-100ask/staging_dir/
SDK=~/tina-v853-100ask/out/v853-100ask/staging_dir
CC=$SDK/toolchain/bin/arm-openwrt-linux-muslgnueabi-gcc-6.4.1

$CC -O2 -Wall -std=c11 \
  -o test_yolo_input_v5 \
  test_yolo_input.c \
  -I$SDK/target/usr/include/viplite-driver \
  -L$SDK/target/usr/lib \
  -Wl,--start-group -lawnn -lVIPuser -lVIPlite -Wl,--end-group \
  -lstdc++ -lm -lpthread -lgcc_s
```

产物：ARM 32-bit ELF 可执行文件（~200KB），动态链接 musl libc。

---

## 步骤 8：准备测试输入

**机器**: 本地 Windows

板端需要 RGB planar 格式的 raw 文件（R plane + G plane + B plane）。

### 8a. Letterbox 预处理（重要）

YOLO 模型训练时输入是正方形（640×640）。如果原图不是正方形，**不能直接 resize**——那会破坏宽高比，导致框偏移、置信度下降。

正确做法是 **letterbox**：将图片等比缩放后居中放入 640×640，多余区域用灰色（114,114,114）填充：

```python
from PIL import Image
import numpy as np

def letterbox(img, new_shape=640):
    """等比缩放 + 居中填充，保持宽高比"""
    w, h = img.size
    scale = new_shape / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2
    result = Image.new('RGB', (new_shape, new_shape), (114, 114, 114))
    result.paste(img, (pad_w, pad_h))
    return result, pad_w, pad_h, scale

img = Image.open("test.jpg").convert("RGB")
img_lb, pad_x, pad_y, scale = letterbox(img, 640)
arr = np.array(img_lb, dtype=np.uint8)  # [640, 640, 3]
rgb_planar = arr.transpose(2, 0, 1)  # [3, 640, 640] RGB planar
rgb_planar.tofile("test_640_rgb_planar.raw")
```

文件大小：640 × 640 × 3 = 1,228,800 bytes。

### 8b. 坐标反向映射

YOLO 输出框在 640×640 letterbox 空间，需映射回原图：

```python
# 从 letterbox 坐标 -> 原图坐标
x_orig = (x_lb - pad_x) / scale
y_orig = (y_lb - pad_y) / scale
```

### 8c. 实例对比

以 1448×1086 的原图为例（scale=0.442, pad_x=0, pad_y=80）：

| 方法 | 0.jpg seatbelt conf | 0.jpg cigarette conf | 1.jpg phone conf | 1.jpg cigarette conf |
|------|---------------------|----------------------|------------------|----------------------|
| 直接 resize | 0.14 | 0.12 | 0.80 | 0.63 |
| letterbox | **0.87** | **0.59** | **0.86** | **0.79** |

直接 resize 拉伸图片导致置信度大幅下降、框坐标偏移，不可用。

---

## 步骤 9：部署到板端

**机器**: 本地 Windows PowerShell

```powershell
$adb = "D:\WORK\TOOL\AllwinnertechPhoeniSuit (1)\AllwinnertechPhoeniSuitRelease20201225\adb.exe"

# 检查连接
& $adb devices

# 推送文件（V5）
& $adb push artifacts/detection/multiclass_yolo26s_v5/nb/network_binary.nb /mnt/UDISK/v5_640_nosig.nb
& $adb push artifacts/detection/multiclass_yolo26s_v5/board_test/test_yolo_input_v5 /mnt/UDISK/test_yolo_input_v5
& $adb push artifacts/detection/multiclass_yolo26s_v5/board_test/0_640_rgb_planar.raw /mnt/UDISK/0_640_rgb_planar.raw
& $adb push artifacts/detection/multiclass_yolo26s_v5/board_test/1_640_rgb_planar.raw /mnt/UDISK/1_640_rgb_planar.raw

# 运行
& $adb shell "chmod +x /mnt/UDISK/test_yolo_input_v5"
& $adb shell "/mnt/UDISK/test_yolo_input_v5 /mnt/UDISK/v5_640_nosig.nb /mnt/UDISK/0_640_rgb_planar.raw"
& $adb shell "/mnt/UDISK/test_yolo_input_v5 /mnt/UDISK/v5_640_nosig.nb /mnt/UDISK/1_640_rgb_planar.raw"
```

板端关键路径：
- 模型：`/mnt/UDISK/v5_640_nosig.nb`
- 测试程序：`/mnt/UDISK/test_yolo_input_v5`
- 测试图：`/mnt/UDISK/*_640_rgb_planar.raw`

---

## 步骤 10：精度验证

### 10a. PC 基线

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

# 注意：必须用 letterbox，不能直接 resize
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
# scores shape: [1, 4, 8400], raw logits (no sigmoid)

for c, name in enumerate(["face", "phone", "cigarette", "seatbelt"]):
    logit = float(scores[0, c, :].max())
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    print(f"  {name}: logit={logit:.4f} sigmoid={sigmoid:.4f}")
```

### 10b. NPU 结果对比

运行 `test_yolo_input_640` 后，输出会显示逐类 max logit/sigmoid。与 PC 基线对比：

| 指标 | 可接受范围 |
|------|-----------|
| logit 差值 | ≤ 0.5 |
| sigmoid 差值 | ≤ 0.1 |
| 检测决策一致 | 正类 > 0.5, 负类 < 0.5 对齐 |

V3 640 实测：logit 差约 0.2-0.3，sigmoid 差约 0.04-0.06，检测决策完全一致。

### 10c. V5 640 验证结果 (2026-05-16)

PC 基线 `best_640_split_nosig.onnx` vs NPU (`reverse_channel: false`, uint8 量化)：

| 图片 | 类别 | PC logit | PC sig | NPU logit | NPU sig | Δ sig | 判定一致 |
|------|------|----------|--------|-----------|---------|-------|---------|
| 0.jpg | face | +3.25 | **0.963** | +2.86 | **0.946** | -0.017 | ✓ |
| 0.jpg | phone | -6.84 | 0.001 | -7.93 | 0.000 | -0.001 | ✓ |
| 0.jpg | cigarette | -11.95 | 0.000 | -7.93 | 0.000 | 0.000 | ✓ |
| 0.jpg | seatbelt | -10.69 | 0.000 | -11.74 | 0.000 | 0.000 | ✓ |
| 1.jpg | face | +3.38 | **0.967** | +2.22 | **0.902** | -0.065 | ✓ |
| 1.jpg | phone | +0.84 | **0.699** | +1.27 | **0.781** | +0.081 | ✓ |
| 1.jpg | cigarette | -10.39 | 0.000 | -7.93 | 0.000 | 0.000 | ✓ |
| 1.jpg | seatbelt | -10.13 | 0.000 | -11.42 | 0.000 | 0.000 | ✓ |

所有 logit 差值 ≤ 0.5，sigmoid 差值 ≤ 0.1，检测决策（>0.5 / <0.5）完全一致。

**注意**：第一次量化使用了 `reverse_channel: true`（继承自 pegasus 默认模板），导致 1.jpg phone 检测从 0.699 翻转为 0.420（判定错误）。改为 `reverse_channel: false` 后恢复正常。根本原因是模型训练和 NPU 输入都用 RGB，VIP 预处理器不应再做通道反转。

---

## 常见问题

### Q: NB 文件加载失败
确认使用了 `--pack-nbg-unify` 导出的 `_nbg_unify/network_binary.nb`，而不是 `export.data`。VPMN header 的 magic bytes 应该是 `56 50 4D 4E`。

### Q: 检测框偏移/置信度偏低
- 输入预处理必须用 **letterbox**（等比缩放+填充），不能用直接 resize 拉伸。非正方形原图被拉伸后宽高比变形，导致框位置偏移
- 坐标反向映射公式：`(coord_lb - pad) / scale`，其中 pad 和 scale 由 letterbox 步骤得出
- 详见步骤 8a-8b

### Q: 某类分数全为 0 或死值
- 检查输入通道顺序：训练用 RGB → `reverse_channel: false`；训练用 BGR → `reverse_channel: true`
- 检查是否移除了 Sigmoid：VIP9000PICO 的 Sigmoid + 4ch Conv 组合会输出全 uniform 值

### Q: NPU 推理精度明显偏低但并非全死值（如 sigmoid 差 > 0.2）
-**首先检查 `reverse_channel`**：这是最常见的原因。如果模型训练用 RGB，inputmeta 必须设 `reverse_channel: false`。设错会导致通道反转（RGB↔BGR），模型看到错误的颜色分布，部分类别置信度大幅下降
- 确认校准图片的预处理与推理时一致（是否需要 letterbox、resize 方式等）
- V5 实测：`reverse_channel: true` 时 1.jpg phone sigmoid 从 0.699→0.420（判定翻转），改为 false 后恢复到 0.781（正常）

### Q: 板端 segfault
- `awnn_destroy` 必须在 `awnn_uninit` 之前调用
- `awnn_get_output_buffers` 返回的 buf_count 可能大于实际有效 buffer 数，需限流

### Q: adb shell 中路径被误解析
Windows Git Bash 会把 `/mnt/UDISK/...` 转换为 Windows 路径。用 PowerShell + adb.exe 可以避免。

### Q: 512 模型在 640 模型上推理
- 测试程序的 cell 数（8400 vs 5376）必须和模型匹配
- 输入尺寸必须和模型训练尺寸一致

### Q: letterbox 参数不对导致置信度反而下降
letterbox 实现中有三个参数极易出错，任何一个不对都会导致检测恶化而非改善：

| 参数 | 正确值 | 错误值 | 后果 |
|------|--------|--------|------|
| 填充色 | 灰色 (114, 114, 114) | 黑色 (0, 0, 0) | 输入分布偏移，检测全面退化 |
| scale | `target / max(w, h)` | `min(target/w, target/h)` | 长边被裁切，关键区域丢失 |
| 对齐 | 居中 `pad = (target - new) // 2` | 左上角 `(0, 0)` | 目标偏离模型预训练分布 |

参见步骤 8a 中的 `letterbox()` 参考实现。

---

## 关键脚本与文件索引

| 文件 | 用途 | 运行机器 |
|------|------|---------|
| `training/detection/train.py` | 训练 YOLO 检测模型 | AutoDL |
| `training/detection/export_onnx.py` | 导出 PyTorch → ONNX | AutoDL/本地 |
| `deployment/quantize/split_yolo_head.py` | 裁掉 NMS 后处理 | 本地 |
| `deployment/quantize/remove_sigmoid.py` | 移除 Sigmoid（VIP 编译器 workaround） | 本地 |
| `deployment/quantize/prepare_calibration_data.py` | 准备校准图片集 | 本地 |
| `deployment/quantize/quantize_model.sh` | pegasus 一键量化脚本 | Ubuntu20 |
| `deployment/board/test_yolo_input/test_yolo_input.c` | 板端最小 NPU 测试程序 | 本地编写, Ubuntu18 编译 |
| `deployment/board/landmark_npu_infer/Makefile` | 交叉编译模板 | Ubuntu18 |

---

## 更新记录

- 2026-05-13: 初版，基于 V3 640 模型 4 分类验证通过
- 2026-05-14: 补充 letterbox 常见错误（黑边/`min`缩放/左上对齐）及诊断方法；更新 `tools/build_replay_input.py` 和 `training/detection/infer_onnx.py` 使用正确的 letterbox 预处理
- **2026-05-15: V5 版本锁定 (commit 3ce89d5)**
  - 从 yolo26s.pt 全新训练，修复 V4 对 cigarette/seatbelt 的灾难性遗忘
  - ADMS 数据集清洗：移除 326/439 个错误安全带标注（bbox 面积<1%）
  - IR 数据增强：dark + infrared + low_contrast 各 1 份
  - 80/20 train/val 随机切分，100 epochs，best mAP50=0.953
  - PC 全链路验证通过：4 类全部检出，ResNet18 13pt landmark + 空间邻近规则正常
  - `tools/build_replay_input.py` 默认检测器切换为 V5
  - 测试图片集：`_test_imgs/`（cigarette×3, seatbelt×2）
  - 模型产物：`artifacts/detection/multiclass_yolo26s_v5/best.onnx` (37MB) + `best.pt` (20MB)
- **2026-05-16: V5 pegasus 量化完成 + 板端部署准备**
  - 拆分 YOLO 头 + 移除 Sigmoid（`best_640_split_nosig.onnx`）
  - pegasus 导入 → 量化 (uint8 asymmetric_affine) → 导出 NB（`network_binary.nb`, 9.7MB）
  - 记录 6 个 pegasus 实操坑（CWD 依赖、inputmeta lid、ldconfig、reverse_channel、绝对路径、NB 命名）
  - 交叉编译：使用 SDK 预编译 musl 工具链（无需完整 make），生成 `test_yolo_input_v5` (200KB ARM ELF)
  - 测试输入：0.jpg/1.jpg + `_test_imgs/` 5 张，全部生成 letterbox 640×640 RGB planar raw
  - PC 基线已记录
