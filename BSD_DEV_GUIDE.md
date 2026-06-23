# BSD 交通参与方检测报警系统开发手册

本手册记录当前 BSD 工程从数据准备、模型导出、量化到 V853 板端实时运行的主流程。以本地仓库当前文件树为准；训练服务器和量化服务器上的脚本/产物如果未同步到本仓库，会在对应章节标注为外部依赖。

## 1. 工程定位

BSD 是 4 类交通参与方检测与报警系统：

| 类别 ID | 类别 | 说明 |
|---:|---|---|
| 0 | person | 行人 |
| 1 | bicycle | 自行车/电瓶车 |
| 2 | motorcycle | 摩托车/三轮车 |
| 3 | vehicle | 车辆备用类，合并 car/bus/truck |

模型和板端约定：

| 项目 | 当前约定 |
|---|---|
| 输入尺寸 | 640x640 为精度基线；320x320 已完成速度验证；板端代码支持 320/416/512/640 |
| 输入格式 | RGB planar，uint8，R/G/B 三个平面连续存放 |
| 预处理 | NV21M/BGR camera frame -> letterbox model_size x model_size -> RGB planar |
| 输出 | `boxes [1,4,N]` + `scores [1,4,N]`，其中 640 输入 `N=8400`，320 输入 `N=2100` |
| scores | raw logit，板端手动 `sigmoid(logit)` |
| boxes | YOLO11/YOLO26 head 的 LTRB distances，按 stride 8/16/32 解码 |
| NPU | AWNN / VIP9000PICO，Pegasus `asymmetric_affine uint8` 量化 |

## 2. 机器分工

| 机器 | 地址/入口 | 用途 |
|---|---|---|
| 本地 Windows | `D:\WORK\CODE\BSD` | 工程整理、数据脚本、ONNX 拆头、ADB/串口操作 |
| AutoDL/训练服务器 | 训练平台控制台 | GPU 训练，导出原始 ONNX |
| Ubuntu18 编译机 | `192.168.144.137` | Tina/V853 ARM32 musl 交叉编译 |
| Ubuntu20 量化机 | `192.168.144.133` | Pegasus 量化、NB 导出、runtime 验证 |
| V853 板端 | ADB/串口/以太网 | 部署、摄像头、NPU、LCD 实测 |

注意：仓库里有不少历史脚本硬编码了 IP、用户名和密码。长期维护时建议改为环境变量或本地私有配置，不要把真实凭据提交。

## 3. 当前目录结构

主线目录：

```text
BSD/
├── BSD_DEV_GUIDE.md                 # 本手册
├── DEPLOYMENT_GUIDE.md              # DMS/V853 历史部署手册，作为参考
├── deployment/
│   ├── dataset/                     # COCO/BDD100K -> BSD YOLO 数据集脚本
│   ├── quantize/                    # ONNX 拆头、校准数据准备脚本
│   └── board/
│       ├── common/types.h           # DetObject/DetResult/AlarmEvent ABI
│       ├── detect_engine/           # 检测引擎、实时 live_bsd、Makefile
│       ├── alarm_engine/            # 区域、跟踪、连续帧报警
│       ├── test_v4/                 # 板端单测入口
│       └── BOARD_ENGINE.md          # 板端引擎说明
├── examples/demo_app.c              # 调用 detect/alarm 的示例入口
├── training/detection/configs/      # BSD 训练配置
├── tools/                           # PC/SSH/量化/验证辅助脚本，含大量实验脚本
├── artifacts/                       # 本地产物目录，已忽略
└── _toolchain/                      # 本地 SDK/工具链缓存，已忽略
```

当前需要特别区分：

| 类型 | 说明 |
|---|---|
| 可维护源码 | `deployment/board/common`、`detect_engine`、`alarm_engine`、`deployment/dataset`、`deployment/quantize` |
| 运行产物 | `artifacts/`、`*.pt`、`*.onnx`、`*.nb`、`*.o`、板端二进制 |
| 实验脚本 | 根目录和 `tools/`、`deployment/board/camera/` 下很多一次性排查脚本 |
| 当前未跟踪但建议审阅入库 | `live_bsd.c`、`compile_live.py`、`prepare_calibration_data.py`、`split_yolo_head_sig.py`、`training/detection/configs/` |
| 外部缺失项 | `training/detection/train.py`、`training/detection/export_onnx.py`、`deployment/quantize/remove_sigmoid.py` 当前不在本地文件树中 |

## 4. 标准开发流程

### 4.1 准备训练数据

当前本地仓库中的数据准备脚本在：

```bash
python deployment/dataset/prepare_bsd_dataset.py
```

该脚本从 COCO2017 中提取目标类，并映射为 BSD 4 类：

```text
person -> 0
bicycle -> 1
motorcycle -> 2
car/bus/truck -> 3
```

脚本内默认路径面向训练服务器：

```text
/root/autodl-tmp/BSD/datasets/coco_raw/annotations
/autodl-pub/data/COCO2017
/root/autodl-tmp/BSD/datasets/bsd
```

如果在本地 Windows 跑，需要先把这些路径参数化或改为本地路径。

### 4.2 训练模型

当前本地文件树只保留了配置文件；这些配置当前还未纳入 Git，需要审阅后添加：

```bash
training/detection/configs/bsd_yolo26s.yaml
training/detection/configs/bsd_yolo26s_small.yaml
```

配置约定：

| 项目 | 值 |
|---|---|
| 模型 | YOLO26s / YOLO11 系列检测头 |
| 输入 | 640 RGB |
| 类别数 | 4 |
| 数据路径 | `/root/autodl-tmp/BSD/datasets/bsd/train` 和 `val` |
| 输出根目录 | `/root/autodl-tmp/BSD/artifacts` |

手册历史命令如下，但 `train.py` 和 `export_onnx.py` 当前不在本仓库，需要从训练环境同步或按训练框架补齐：

```bash
python training/detection/train.py \
  --config training/detection/configs/bsd_yolo26s.yaml

python training/detection/export_onnx.py \
  --weights artifacts/detection/bsd_yolo26s/best.pt \
  --imgsz 640 \
  --output artifacts/detection/bsd_yolo26s/best.onnx
```

导出的原始 ONNX 通常带 NMS，输出类似 `[1,300,6]`，不适合直接给 Pegasus/VIP9000PICO。

### 4.3 拆分 YOLO 头

本地可用脚本：

```bash
python deployment/quantize/split_yolo_head.py \
  --input artifacts/detection/bsd_yolo26s/best.onnx \
  --output artifacts/detection/bsd_yolo26s/best_640_split_nosig.onnx
```

`split_yolo_head.py` 的当前行为：

| 输出 | 说明 |
|---|---|
| `boxes` | `[1,4,N]`，LTRB distances |
| `scores` | `[1,nc,N]`，raw logits，未加 Sigmoid |

`N` 随输入尺寸变化：`N=(S/8)^2 + (S/16)^2 + (S/32)^2`。例如 640 输入是 `8400`，320 输入是 `2100`。

历史手册曾写过单独执行 `remove_sigmoid.py`。当前仓库没有该文件，而且 `split_yolo_head.py` 已直接取 Sigmoid 之前的 `Concat_1` 输出，所以标准流程以 `split_yolo_head.py` 为准。

另有实验脚本：

```bash
python deployment/quantize/split_yolo_head_sig.py ...
```

该脚本会把 Sigmoid 放回 scores 输出，主要用于对比或绕量化实验。当前板端 `yolo_decode.c` 期望 raw logit，不应和 sigmoid 版 NB 混用。

### 4.4 PC 端快速验证

量化前建议先用 ONNX Runtime 验证 `best_640_split_nosig.onnx`：

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

def letterbox(img, target=640):
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    pad_x = (target - nw) // 2
    pad_y = (target - nh) // 2
    canvas = Image.new("RGB", (target, target), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas

session = ort.InferenceSession("best_640_split_nosig.onnx")
img = Image.open("test.jpg").convert("RGB")
arr = np.asarray(letterbox(img), dtype=np.float32) / 255.0
arr = arr.transpose(2, 0, 1)[None, ...]
boxes, scores = session.run(None, {"images": arr})

for c, name in enumerate(["person", "bicycle", "motorcycle", "vehicle"]):
    logit = float(scores[0, c].max())
    conf = 1.0 / (1.0 + np.exp(-logit))
    print(name, logit, conf)
```

仓库里也有一些 PC 验证脚本可参考：

```text
tools/verify_split_onnx.py
tools/verify_decode_fix.py
tools/pc_model_test.py
tools/compare_pt_vs_split.py
tools/compare_pt_vs_onnx_box.py
```

这些脚本多数含绝对路径，运行前需要检查路径。

### 4.5 准备校准数据

当前脚本：

```bash
python deployment/quantize/prepare_calibration_data.py
```

默认输出：

```text
artifacts/calib_camera/
```

它会读取已有 JPEG 和若干 BGR raw 文件，生成 letterbox 后的 640x640 RGB 校准图，并做亮度、对比度、色彩增强。校准集建议 200-300 张，并覆盖：

| 场景 | 目的 |
|---|---|
| 白天/夜间/低照 | 降低场景漂移 |
| 行人/电瓶车/摩托/车辆 | 覆盖全部类别 |
| 远近尺度 | 覆盖不同检测头 |
| 板端真实摄像头帧 | 对齐实际 ISP/白平衡表现 |

### 4.6 Pegasus 量化

在 Ubuntu20 上执行，核心参数：

```bash
PEGASUS=/home/ubuntu/VeriSilicon/acuity-toolkit-binary-6.9.3/bin/acuitylib/pegasus
VIV_SDK=$HOME/VeriSilicon/VivanteIDE5.7.1/cmdtools/vsimulator
NAME=bsd_640
REMOTE_DIR=/home/ubuntu/bsd_quantize
OUT_DIR=${REMOTE_DIR}/out_640

mkdir -p ${OUT_DIR}
cd $(dirname ${PEGASUS})

./pegasus import onnx \
  --model ${REMOTE_DIR}/best_640_split_nosig.onnx \
  --output-data ${OUT_DIR}/${NAME}.data \
  --output-model ${OUT_DIR}/${NAME}.json

./pegasus generate inputmeta \
  --model ${OUT_DIR}/${NAME}.json \
  --input-meta-output ${REMOTE_DIR}/${NAME}_inputmeta.yml
```

`inputmeta.yml` 关键项：

```yaml
input_meta:
  databases:
  - path: /home/ubuntu/bsd_quantize/dataset.txt
    type: TEXT
    ports:
    - category: image
      dtype: float32
      layout: nchw
      shape: [1, 3, 640, 640]
      fitting: scale
      preprocess:
        reverse_channel: false
        mean: [0, 0, 0]
        scale: 0.0039
        preproc_node_params:
          add_preproc_node: false
          preproc_type: IMAGE_RGB
          preproc_image_size: [640, 640]
```

继续量化和导出：

```bash
./pegasus quantize \
  --model ${OUT_DIR}/${NAME}.json \
  --model-data ${OUT_DIR}/${NAME}.data \
  --batch-size 1 \
  --device CPU \
  --with-input-meta ${REMOTE_DIR}/${NAME}_inputmeta.yml \
  --rebuild \
  --model-quantize ${OUT_DIR}/${NAME}.quantize \
  --quantizer asymmetric_affine \
  --qtype uint8

./pegasus export ovxlib \
  --model ${OUT_DIR}/${NAME}.json \
  --model-data ${OUT_DIR}/${NAME}.data \
  --dtype quantized \
  --model-quantize ${OUT_DIR}/${NAME}.quantize \
  --batch-size 1 \
  --save-fused-graph \
  --target-ide-project linux64 \
  --with-input-meta ${REMOTE_DIR}/${NAME}_inputmeta.yml \
  --output-path ${OUT_DIR}/ovxlib/${NAME}/${NAME}prj \
  --pack-nbg-unify \
  --optimize VIP9000PICO_PID0XEE \
  --viv-sdk ${VIV_SDK}
```

产物：

```text
${OUT_DIR}/ovxlib/${NAME}/${NAME}_nbg_unify/network_binary.nb
```

踩坑要点：

| 点 | 要求 |
|---|---|
| Pegasus 工作目录 | 尽量在 `acuitylib` bin 目录下执行 |
| 通道顺序 | `reverse_channel: false`，板端已经喂 RGB planar |
| 输入缩放 | `scale: 0.0039`，不要额外做 Z-score |
| Sigmoid | 当前标准 NB 输出 raw logit，板端 host 侧算 sigmoid |
| NB 文件 | 用 `network_binary.nb`，不要误用 `export.data` |

### 4.7 编译板端程序

核心目录：

```bash
cd deployment/board/detect_engine
make clean && make
```

Makefile 当前目标：

| 目标 | 用途 |
|---|---|
| `libdetect_engine.so` | 检测引擎共享库 |
| `bsd_detect` | 从 stdin 读取 BGR raw 的单帧/离线测试程序 |
| `live_bsd` | 摄像头 + NPU + framebuffer 实时管线 |
| `test_npu_direct` | 读取 BGR raw 文件并直接跑 NB/NPU，支持 `[width height]` 参数 |

依赖目录：

```text
deployment/board/common/types.h
deployment/board/alarm_engine/*.c
deployment/board/detect_engine/*.c
```

本地 Windows 可用自动上传编译脚本；该脚本当前也还未纳入 Git，需要审阅后添加：

```bash
python deployment/board/compile_live.py
```

或者使用历史工具：

```bash
python tools/upload_build.py
```

这些脚本会 SSH 到 Ubuntu18 编译机，上传源码，执行 `make`，再把二进制下载到 `artifacts/board_bin/`。

### 4.8 板端部署运行

ADB 可用时：

```powershell
$adb = "D:\WORK\TOOL\AllwinnertechPhoeniSuit (1)\AllwinnertechPhoeniSuitRelease20201225\adb.exe"

& $adb push artifacts/board_bin/live_bsd /mnt/UDISK/live_bsd
& $adb push artifacts/detection/bsd_yolo26s/nb/network_binary.nb /mnt/UDISK/bsd_640.nb
& $adb shell "chmod +x /mnt/UDISK/live_bsd"
& $adb shell "export LD_LIBRARY_PATH=/mnt/UDISK && /mnt/UDISK/live_bsd /mnt/UDISK/bsd_640.nb 0.5 0.45 0.5 0.5"
```

ADB 不可用时，可用以太网直连 + HTTP：

```powershell
netsh interface ip set address "以太网 3" static 192.168.144.100 255.255.255.0
cd artifacts\board_bin
python -m http.server 8888
```

板端串口：

```bash
ip addr add 192.168.144.200/24 dev eth0
/usr/bin/wget http://192.168.144.100:8888/live_bsd -O /mnt/UDISK/live_bsd
chmod +x /mnt/UDISK/live_bsd
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_640.nb 0.5 0.45 0.5 0.5
```

最近这条板端链路已经走通，当前更接近“可用默认值”而不是“实验值”：

| 参数 | 当前建议 |
|---|---|
| `det_conf` | `0.5` |
| `nms_thr` | `0.45` |
| `disp_conf` | `0.5` |
| `person_conf` | `0.5` |

其中 `det_conf` 和 `person_conf` 仍然分别控制基础类和 `person`，但在当前板端场景里不建议再把阈值压得很低，否则误检会明显增多。摄像头物理摆放已经调整为横向输入，后续不要再优先走软件旋转补偿路径。

## 5. 板端软件架构

### 5.1 detect_engine

文件：

```text
deployment/board/detect_engine/detect_engine.c
deployment/board/detect_engine/preprocess.c
deployment/board/detect_engine/yolo_decode.c
deployment/board/detect_engine/detect_engine.h
```

API：

```c
int  detect_init(const char* nb_path, const char* shm_name,
                 int cam_width, int cam_height,
                 float conf_thr, float nms_thr);
void detect_set_nv21_stride(int y_stride, int uv_stride);
int  detect_process_nv21m(const uint8_t* y_plane, const uint8_t* vu_plane);
int  detect_process_frame(const uint8_t* frame_buf);
const DetResult* detect_get_result(void);
const DetectProfile* detect_get_last_profile(void);
void detect_set_class_threshold(int class_id, float thr);
void detect_deinit(void);
```

当前生产热路径优先走 NV21M 直入：

```text
V4L2 MPLANE NV21M Y/VU planes
  -> letterbox_preprocess_nv21m()
  -> RGB planar uint8
  -> awnn_set_input_buffers()
  -> awnn_run()
  -> awnn_get_output_buffers()
  -> yolo_decode()
  -> yolo_nms()
  -> 坐标映射回原始图像
  -> 写入 /bsd_detect_shm
```

旧 BGR 输入仍可用于离线单帧测试：

```text
BGR interleaved camera frame
  -> letterbox_preprocess()
  -> RGB planar uint8
  -> awnn_set_input_buffers()
  -> awnn_run()
  -> awnn_get_output_buffers()
  -> yolo_decode()
  -> yolo_nms()
  -> 坐标映射回原始图像
  -> 写入 /bsd_detect_shm
```

### 5.2 yolo_decode

`deployment/board/detect_engine/yolo_decode.c` 当前按 YOLO11/YOLO26 LTRB distance 解码：

```text
head0: (model_size/8)  x (model_size/8),  stride 8
head1: (model_size/16) x (model_size/16), stride 16
head2: (model_size/32) x (model_size/32), stride 32
```

常用输入的 cell 数：

| model_size | head0 | head1 | head2 | total |
|---:|---:|---:|---:|---:|
| 320 | 40x40 | 20x20 | 10x10 | 2100 |
| 416 | 52x52 | 26x26 | 13x13 | 3549 |
| 512 | 64x64 | 32x32 | 16x16 | 5376 |
| 640 | 80x80 | 40x40 | 20x20 | 8400 |

每个 cell：

```text
best_cls = argmax(scores[class, cell])
conf = sigmoid(best_logit)
box = anchor +/- ltrb_distance
xyxy_norm = box_grid * stride / model_size
```

注意：这不是旧版 `cx/cy/w/h` 归一化解码。

### 5.3 alarm_engine

文件：

```text
deployment/board/alarm_engine/alarm_engine.c
deployment/board/alarm_engine/zone_mgr.c
deployment/board/alarm_engine/tracker.c
deployment/board/alarm_engine/alarm_engine.h
```

职责：

| 模块 | 职责 |
|---|---|
| zone_mgr | 最多 3 层矩形报警区域，归一化坐标 |
| tracker | IOU 多目标跟踪，连续帧计数 |
| alarm_engine | 读取共享内存，命中区域并触发回调 |

### 5.4 live_bsd

文件：

```text
deployment/board/detect_engine/live_bsd.c
```

实时管线：

```text
GC2053 / V4L2 MPLANE NV21M
  -> detect_process_nv21m()
  -> alarm_engine
  -> optional preview: NV21 to BGR + rotate 90deg + scale
  -> optional framebuffer /dev/fb0 draw boxes + flip
```

当前实现里还有两个固定点要记住：

1. `ENABLE_FRAME_DUMP` 默认是关闭的，排查图像时可以临时打开，但不要长期挂着。
2. `NPU_FRAME_INTERVAL` 不是 1，当前是间隔跑推理，不是每帧都进 NPU。这个设计是为了避免把显示和推理绑死在一起。
3. 默认模式是 `headless`，即不初始化 framebuffer、不画框、不做预览转换。`preview` 只用于调试。

命令行：

```bash
/mnt/UDISK/live_bsd <model.nb> [det_conf=0.5] [nms_thr=0.45] [disp_conf=0.5] [person_conf=0.5] [cam_w cam_h] [headless|preview] [model_size=640]
```

`det_conf` 是 bicycle/motorcycle/vehicle 的基础阈值，`person_conf` 单独控制 person，二者都会影响报警；`disp_conf` 只影响屏幕画框。
当前实测后，建议把默认调用改成 `0.5 0.45 0.5 0.5`，板端已经验证过这组比低阈值更稳。镜头放横后，person 识别恢复正常；之前那套竖屏补偿思路不要再作为主路径。

当前建议的性能测试命令：

```bash
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v4_640_bdd100k.nb 0.5 0.45 0.5 0.5 1280 720 headless
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v4_320.nb 0.5 0.45 0.5 0.5 1280 720 headless 320
```

FPS 日志字段说明：

| 字段 | 含义 |
|---|---|
| `FPS` | 主循环帧率，包含采集、抽帧推理、预览和 QBUF |
| `calls` | 最近 30 帧内实际调用 NPU 检测的次数，当前 `NPU_FRAME_INTERVAL=3` 时通常是 10 |
| `avg_ms pre` | 每帧摊销后的检测耗时，不等于单次 NPU 调用耗时 |
| `det_call_ms total` | 每次实际检测调用耗时 |
| `nv21` | NV21M fused letterbox + RGB planar 预处理 |
| `npu` | `awnn_run()` 和取输出耗时 |
| `decode` | YOLO decode + NMS |
| `preview` | framebuffer 预览、旋转缩放、画框和翻页 |

## 6. 常见问题和排查

### 6.1 NPU 输出置信度接近 0

优先检查：

| 检查项 | 正确状态 |
|---|---|
| ONNX 输出 | scores 是 raw logit，PC 侧 sigmoid 后正常 |
| inputmeta | `reverse_channel: false`，`scale: 0.0039` |
| C 端输入 | `uint8 RGB planar`，不要 Z-score |
| NB 文件 | `network_binary.nb`，不是中间文件 |
| 板端解码 | host 侧手动 sigmoid |

### 6.2 画面偏黄/偏绿

`live_bsd.c` 内有白平衡增益：

```c
#define WB_R_GAIN 285
#define WB_G_GAIN 205
#define WB_B_GAIN 307
```

如果换 sensor 或光照环境变化，先 dump 一帧 BGR raw，在 PC 看偏色方向，再调整增益。

### 6.3 live_bsd 跑一会儿冻屏

常见原因：

| 原因 | 处理 |
|---|---|
| 每帧 `printf` 打爆串口 | 日志限速，每 30 帧打印一次 |
| V4L2 QBUF/DQBUF 失败未处理 | 失败计数、退避、连续失败后退出 |
| 摄像头缓冲区耗尽 | 确保每次 DQBUF 后最终 QBUF 归还 |

补充一条：`ENABLE_FRAME_DUMP` 开着时会明显拖慢主循环，只适合短时间抓图，不适合持续跑性能测试。

### 6.4 LCD 竖屏坐标错位

`live_bsd` 对横向摄像头画面顺时针旋转 90 度显示到 480x800 LCD。图像旋转后，检测框坐标也必须同步旋转：

```text
source 1280x720 -> rotated 720x1280 -> display 450x800 at x=15
```

改摄像头分辨率时必须同步检查 `CAM_W`、`CAM_H`、`DISP_W`、`DISP_H` 和画框映射。

现在已经确认一件事：摄像头横放以后，检测效果比竖放时稳定很多。后续排查识别异常时，先确认摄像头物理姿态，再动代码。

### 6.5 PC 和 NPU 置信度有差异

uint8 量化后 logit 变小是正常现象。重点看判定是否一致，而不是绝对 logit 完全一致。如果某类持续偏低，优先补校准图和检查该类训练/验证覆盖。

当前同帧验证入口：

```bash
python tools/pc_model_test.py artifacts/frames/frame_000055_bgr.raw artifacts/v4_model/best_split.onnx
/mnt/UDISK/test_npu_direct /mnt/UDISK/bsd_v4_640_bdd100k.nb /mnt/UDISK/frame_000055_bgr.raw 1280 720
```

2026-05-28 的板端回灌结果：

| 帧 | PC ONNX | V853 NB/NPU | 结论 |
|---|---|---|---|
| `frame_000055_bgr.raw` | person 0.856, `[742,3,1166,710]` | person 0.719, `[744,5,1171,710]` | 位置一致，量化后置信度降低 |
| `frame_000042_bgr.raw` | person 0.764 + vehicle 0.383 | person 0.500 + vehicle 0.719 | vehicle 量化后偏高 |
| `frame_000069_bgr.raw` | 近全画幅 vehicle 被过滤 | 近全画幅 vehicle 0.804，修复后被过滤 | 板端需要 95% 宽高退化框过滤 |

因此当前主要问题不是 NB/NPU 解码完全错位，而是实际摄像头画面上的大框误检、量化分数漂移和场景域差异。继续优化时优先补板端真实帧做校准/训练验证。

### 6.6 板端性能不够跑 4 路

当前 `live_bsd` 的主循环已经打通，headless + NV21M fused 之后显示和颜色转换已经不是主瓶颈。2026-05-29 在 V853 板端用 `/mnt/UDISK/bsd_v4_640_bdd100k.nb` 实测，真正卡住的是 640 输入模型的 NPU 推理时间。

实测命令：

```bash
/mnt/UDISK/live_bsd.prof /mnt/UDISK/bsd_v4_640_bdd100k.nb 0.5 0.45 0.5 0.5 1280 720 headless
/mnt/UDISK/live_bsd.prof /mnt/UDISK/bsd_v4_640_bdd100k.nb 0.5 0.45 0.5 0.5 640 360 headless
/mnt/UDISK/live_bsd.prof /mnt/UDISK/bsd_v4_640_bdd100k.nb 0.5 0.45 0.5 0.5 1280 720 preview
```

结果摘要：

| 模式 | 主循环 FPS | 单次检测 total | NV21 预处理 | NPU | decode/NMS | preview |
|---|---:|---:|---:|---:|---:|---:|
| `1280x720 headless` | 10.00 | 223.8-224.1 ms | 16.1-16.3 ms | 206.6-206.8 ms | 1.1-1.2 ms | 0 ms |
| `640x360 headless` | 10.00 | 222.5-222.7 ms | 15.3-15.5 ms | 206.1-206.2 ms | 1.1 ms | 0 ms |
| `1280x720 preview` | 5.98 | 223.9-224.0 ms | 16.1-16.2 ms | 206.5-206.8 ms | 1.2 ms | 92.2-92.5 ms/frame |

320 输入速度验证：

```bash
# 本地导出和拆头
python -c "from ultralytics import YOLO; YOLO('artifacts/v4_best.pt').export(format='onnx', imgsz=320, opset=12, nms=True, simplify=False, dynamic=False, device='cpu')"
python deployment/quantize/split_yolo_head.py \
  --input artifacts/v4_model_320/best.onnx \
  --output artifacts/v4_model_320/best_320_split_nosig.onnx

# 板端运行
/mnt/UDISK/live_bsd320 /mnt/UDISK/bsd_v4_320.nb 0.5 0.45 0.5 0.5 1280 720 headless 320
```

产物：

| 产物 | 路径 |
|---|---|
| 320 raw-logit split ONNX | `artifacts/v4_model_320/best_320_split_nosig.onnx` |
| 320 NB | `artifacts/v4_model_320/bsd_v4_320.nb` |
| 320 校准图 | `artifacts/calib_camera_320/` |

320 板端实测：

| 模式 | 主循环 FPS | 单次检测 total | NV21 预处理 | NPU | decode/NMS | preview |
|---|---:|---:|---:|---:|---:|---:|
| `320, 1280x720 headless` | 19.99 | 50.5-50.6 ms | 4.2-4.4 ms | 45.9-46.0 ms | 0.3 ms | 0 ms |
| `640, 1280x720 headless` 回归 | 10.00 | 223.7-223.9 ms | 16.0-16.1 ms | 206.6-206.7 ms | 1.2-1.3 ms | 0 ms |

这次 320 只验证了速度链路。测试时画面里没有目标，`detect_count=0`，所以不能据此判断精度是否够用。

判读：

| 结论 | 说明 |
|---|---|
| 生产态必须 `headless` | `preview` 每帧约 92 ms，会把主循环从 10 FPS 拉到约 6 FPS |
| 只降摄像头采集分辨率收益很小 | 640x360 只把 NV21 预处理省约 0.8 ms，NPU 仍是 206 ms 级 |
| 当前 640 模型不适合直接 4 路全量实时 | 单次 NPU 调用约 224 ms，4 路轮询会明显排队 |
| 320 模型速度达标进入下一阶段 | 单次 NPU 调用约 46 ms，已经具备 4 路低频轮询的性能基础 |
| 下一步重点是精度验收 | 需要让人/车进入镜头，比较 320 和 640 的 person/vehicle 召回、误检和近距离稳定性 |

结论上先记住这几点：

| 项目 | 结论 |
|---|---|
| 4 路全量实时 | 640 不够；320 速度有机会，精度待验 |
| 单路 headless 验证 | 可用，约 10 FPS 主循环 |
| 低频抽帧告警 | 有机会 |
| 显示和推理同时满载 | 不建议，显示只做调试 |

后续如果要扩到 4 路，优先顺序应是：先用 320 NB 做真实人/车场景精度验收；如果 320 小目标召回明显回退，再做 416 作为折中；保持生产态 `headless`；再按业务要求给每路设置轮询频率。如果更小输入和 YOLO26n 精度都不够，再考虑多板或外部分流方案。

### 6.7 无板端实拍条件下的 v5 精度/耗时路线

当前暂时没有条件新增板端实拍图，所以不要把已有姿态异常的 raw/JPEG 当成正式精度结论。它们只用于链路、量化、回灌排查；最终上线前仍必须补 `board_val`。

当前主目标已经收敛为：训练出符合 BSD 精度标准的 `YOLO26n@512`。速度收益只是选择 `n@512` 的原因之一，但不是验收标准本身；如果 `n@512` 在 PC 侧精度和可视化检查没有达到要求，就继续从数据和训练策略上改，不进入最终板端候选。公开数据集路线、类别映射和验收门槛见：

```text
docs/BSD_PUBLIC_DATASET_PLAN.md
```

当前基线：

| 项目 | 值 |
|---|---|
| 5/20 主模型 | `/root/autodl-tmp/BSD/artifacts/bsd_v4_150ep/weights/best.pt` |
| MD5 | `bfc40b3fbfd6001e83c7de680412587d` |
| 模型/输入 | `YOLO26s@640` |
| 阈值 | `conf=0.5`, `nms=0.45` |
| 板端运行 | `live_bsd headless` 为生产路径，`preview` 只调试 |
| ABI | 不改 `DetResult` / `AlarmEvent` |

已知 v4 对照指标：

| 验证集 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `coco_val` | 0.818 | 0.557 | 0.652 | 0.425 |
| `bdd_proxy_val` | 0.678 | 0.502 | 0.556 | 0.336 |

`YOLO26n@512` 的验收原则：

| 门槛 | 要求 |
|---|---|
| 总体精度 | `coco_val` 接近 v4；如果 mAP50 回退，必须能用 BSD 场景收益解释 |
| 道路场景 | `bdd_proxy_val` 追平或超过 v4 |
| person | 近中距离行人不能明显漏检 |
| bicycle/motorcycle | 两轮车和小目标不能明显召回塌陷 |
| vehicle | 不能反复出现大框误检 |
| 阈值 | 固定 `conf=0.5`, `nms=0.45`，不靠低阈值换召回 |

v5 数据集构建脚本：

```bash
python deployment/dataset/build_bsd_v5_balanced.py
```

默认从 `/root/autodl-tmp/BSD/datasets/bsd_merged` 生成：

| 输出 | 说明 |
|---|---|
| `/root/autodl-tmp/BSD/datasets/bsd_v5_balanced/train` | 清洗并均衡后的训练集 |
| `dataset.yaml` | 训练 + COCO val，可复现主指标 |
| `bdd_proxy_val.yaml` | 训练集外切出的 BDD 代理验证 |
| `manifest.json` | 来源、类别、框面积、筛选参数 |
| `rejected_labels.csv` | 被剔除 label 明细 |

验证集临时分三类：

| 名称 | 用途 |
|---|---|
| `coco_val` | 当前 1,470 张 COCO/BSD 验证集，作为可复现指标 |
| `bdd_proxy_val` | BDD auto-label 中切出、不参与训练的道路场景代理验证 |
| `board_debug_frames` | 现有板端 raw/JPEG，只做链路/量化/回灌调试，不做精度结论 |

候选矩阵：

| 候选 | 配置 | 目的 |
|---|---|---|
| `v5_yolo26s_640` | `training/detection/configs/bsd_v5_yolo26s_640.yaml` | 新精度基线 |
| `v5_yolo26s_512` | `training/detection/configs/bsd_v5_yolo26s_512.yaml` | 降输入尺寸基线 |
| `v5_yolo26n_640` | `training/detection/configs/bsd_v5_yolo26n_640.yaml` | 小模型精度上限 |
| `v5_yolo26n_512` | `training/detection/configs/bsd_v5_yolo26n_512.yaml` | 主力压缩候选 |
| `v5_yolo26n_416` | `training/detection/configs/bsd_v5_yolo26n_416.yaml` | 4 路性能候选 |

320 暂时不作为重新训练主线。前面已经证明 320 速度明显快，但验证集可视化显示小目标损失明显；只有当 416 仍不够快时再回到 320。

公开数据导入当前先从 BDD100K 官方 detection labels 开始：

```bash
python deployment/dataset/import_bdd100k_detection.py \
  --bdd-root /root/autodl-tmp/BSD/datasets/bdd100k \
  --output /root/autodl-tmp/BSD/datasets/bdd100k_bsd_det20 \
  --include-empty \
  --empty-ratio 0.08 \
  --overwrite
```

该脚本会生成 BSD YOLO 格式的 `train/images`、`train/labels`、`val/images`、`val/labels`、`dataset.yaml` 和 `manifest.json`。类别映射为：

```text
person/pedestrian/rider -> person
bicycle/bike -> bicycle
motorcycle/motorbike/scooter -> motorcycle
car/bus/truck/van -> vehicle
```

导入后的公开数据不要直接全量混入训练，先检查 `manifest.json` 的类别分布、空标注比例和 vehicle 占比，再通过 `build_bsd_v6_precision.py --extra-set ...` 合成 `bsd_v6_public_precision`。

训练和导出入口：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v5_yolo26s_640.yaml \
  --mode all \
  --device 0
```

注意：这一步需要 GPU。没有 GPU 时只允许做数据构建、脚本检查、已有权重少量验证，不要启动完整训练。启动训练前先明确告诉操作者要跑哪个候选、预计占用哪台 GPU、预计耗时。

单独评估已有权重：

```bash
python tools/evaluate_bsd_candidates.py \
  --weights /root/autodl-tmp/BSD/artifacts/bsd_v5_yolo26n_512/weights/best.pt \
  --imgsz 512 \
  --coco-yaml /root/autodl-tmp/BSD/datasets/bsd_v5_balanced/dataset.yaml \
  --bdd-yaml /root/autodl-tmp/BSD/datasets/bsd_v5_balanced/bdd_proxy_val.yaml \
  --out-dir /root/autodl-tmp/BSD/artifacts/bsd_v5_yolo26n_512_eval \
  --device 0 \
  --conf 0.5 \
  --iou 0.45
```

每个候选必须保持板端流程不断：

```text
PT
  -> ONNX
  -> split raw-logit ONNX
  -> Pegasus NB
  -> test_npu_direct 回灌一致性
  -> live_bsd headless 单路性能
  -> 4 路轮询压力测试
```

拆头：

```bash
python deployment/quantize/split_yolo_head.py \
  --input /root/autodl-tmp/BSD/artifacts/bsd_v5_yolo26n_512/weights/best.onnx \
  --output /root/autodl-tmp/BSD/artifacts/bsd_v5_yolo26n_512/best_512_split_nosig.onnx
```

量化可用通用入口从本地发到 Ubuntu20 Pegasus 机器：

```bash
python tools/quantize_bsd_candidate.py \
  --onnx artifacts/detection/bsd_v5_yolo26n_512/best_512_split_nosig.onnx \
  --calib-dir artifacts/calib_camera_512 \
  --model-name bsd_v5_yolo26n_512 \
  --model-size 512
```

阶段性验收顺序：

| 阶段 | 硬要求 |
|---|---|
| PC 指标 | `coco_val` 和 `bdd_proxy_val` 同时看 overall/per-class Precision、Recall、mAP50、mAP50-95 |
| 可视化 | 固定输出对比图，重点看小人、两轮车、vehicle 大框误检 |
| 量化回灌 | `test_npu_direct` 与 PC ONNX 判定一致，允许量化分数漂移 |
| 板端性能 | `live_bsd headless` 记录单路 FPS、NPU、预处理、decode |
| 上线前 | 必须补正式 `board_val`，没有通过不能定最终模型 |

### 6.8 当前固定候选：v7 YOLO26n@640

当前已经上板跑通的候选是 `bsd_v7_yolo26n_640_public_kitti`。它不是最终量产签版模型，但已经是后续开发、量化和板端联调的默认候选。

| 项目 | 值 |
|---|---|
| 模型 | `YOLO26n@640` |
| 训练权重 | `/root/autodl-tmp/BSD/artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/weights/best.pt` |
| 本地 split ONNX | `artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx` |
| 本地 NB | `artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/bsd_v7_yolo26n_640_public_kitti.nb` |
| 板端 NB | `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |
| NB 大小 | `3,696,064` bytes，约 `3.52 MiB` |
| NB MD5 | `26c358cf22e9ba05f64e3c6e42c5ccaf` |
| NB magic | `VPMN` |
| 阈值 | `0.5 0.45 0.5 0.5` |

PC 指标对比：

| 验证集 | 模型 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `coco_val` | v4 `s@640` | 0.818 | 0.557 | 0.652 | 0.425 |
| `coco_val` | v7 `n@640` | 0.741 | 0.521 | 0.586 | 0.364 |
| `bdd_proxy_val` | v4 `s@640` | 0.678 | 0.502 | 0.556 | 0.336 |
| `bdd_proxy_val` | v7 `n@640` | 0.682 | 0.615 | 0.674 | 0.467 |

BSD 场景优先看 `bdd_proxy_val`，因此当前 v7 在道路代理验证集上已经超过 v4；但 `coco_val` 仍低于 v4，所以不能说通用精度全面追平。最终上线前仍必须补正式 `board_val`。

标准导出、拆头、量化：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v7_yolo26n_640_public_kitti_stage1.yaml \
  --mode export \
  --weights /root/autodl-tmp/BSD/artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/weights/best.pt \
  --device cpu

python deployment/quantize/split_yolo_head.py \
  --input artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best.onnx \
  --output artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx

python tools/quantize_bsd_candidate.py \
  --host 192.168.144.133 \
  --user ubuntu \
  --password "$BSD_QUANT_PASSWORD" \
  --remote-dir /home/ubuntu/bsd_quantize \
  --onnx artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx \
  --calib-dir artifacts/bdd100k/calib_images \
  --model-name bsd_v7_yolo26n_640_public_kitti \
  --model-size 640
```

标准板端部署：

```bash
python deployment/board/deploy_live_bsd_serial.py \
  --serial COM5 \
  --host-ip 192.168.144.100 \
  --port 8899 \
  --nb artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/bsd_v7_yolo26n_640_public_kitti.nb \
  --remote-nb bsd_v7_yolo26n_640.nb \
  --live-bsd artifacts/board_bin/live_bsd \
  --test-npu-direct artifacts/board_bin/test_npu_direct \
  --install-autostart \
  --run-mode headless
```

板端手动运行：

```sh
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v7_yolo26n_640.nb 0.5 0.45 0.5 0.5 1280 720 headless 640
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v7_yolo26n_640.nb 0.5 0.45 0.5 0.5 1280 720 preview 640
```

开机自启动脚本固定在：

```text
deployment/board/init.d/bsd_live
```

板端安装和启用：

```sh
cp deployment/board/init.d/bsd_live /etc/init.d/bsd_live
chmod +x /etc/init.d/bsd_live
/etc/init.d/bsd_live enable
ln -sf bsd_live /etc/init.d/S99bsd_live
```

这份 Tina 系统的 `rc.final` 实际遍历 `/etc/init.d/S??*`，不只看 `/etc/rc.d`，所以除了 `/etc/init.d/bsd_live enable` 生成 `/etc/rc.d/S99bsd_live` 外，还必须保留 `/etc/init.d/S99bsd_live -> bsd_live` 这个链接。

该服务默认启动 `headless`，等待 `/mnt/UDISK/live_bsd` 和 `/mnt/UDISK/bsd_v7_yolo26n_640.nb` 就绪后运行，日志写到 `/mnt/UDISK/bsd_live_autostart.log`。如果调试时需要屏幕显示，手动停服务后再跑 `preview`，不要把 `preview` 作为开机默认。

2026-06-12 V853 实测：

| 模式 | FPS | 单次检测 total | NV21 预处理 | NPU | decode | preview |
|---|---:|---:|---:|---:|---:|---:|
| `headless` | 约 15.0 | 约 109.5 ms | 约 16.0 ms | 约 92.4 ms | 约 1.2 ms | 0 ms |
| `preview` | 约 7.7 | 约 109.5 ms | 约 15.9 ms | 约 92.4 ms | 约 1.2 ms | 约 93.6 ms |

`preview` 慢不是模型变慢，而是额外做 framebuffer 转换、旋转缩放、画框和翻页。生产态必须用 `headless`。

## 7. 维护约定

建议以后按下面规则维护仓库：

| 内容 | 放置位置 |
|---|---|
| 可复用源码 | `deployment/`、`examples/`、`training/` |
| PC/SSH 辅助工具 | `tools/` |
| 一次性排查脚本 | `tools/experiments/` 或 `deployment/board/camera/experiments/` |
| 模型和二进制 | `artifacts/`，不要提交 |
| SDK/工具链 | `_toolchain/`，不要提交 |
| 真实密码/IP 私有配置 | `.env` 或本地配置，别写入源码 |

建议优先审阅并加入版本控制的文件见 `PROJECT_INVENTORY.md`。

提交前至少检查：

```bash
git status --short
git diff --check
```

如能访问编译机和板端，再跑：

```bash
python deployment/board/compile_live.py
```

## 8. 更新记录

| 日期 | 内容 |
|---|---|
| 2026-05-18 | 初版，BSD 4 分类模型 + 报警引擎 |
| 2026-05-27 | 补充 live_bsd 实时视频管线和板端踩坑记录 |
| 2026-05-28 | 按当前仓库文件树重整手册，修正数据脚本、拆头脚本、板端解码和产物管理说明 |
| 2026-05-28 | 完成 1280x720 同帧 PC ONNX vs V853 NB/NPU 回灌验证，修复近全画幅误检过滤 |
| 2026-05-29 | 增加 headless/NV21M profiling 记录，确认当前 640 模型瓶颈主要在 NPU 推理 |
| 2026-05-29 | 完成 320 输入非重训练版导出、量化、板端测速；NPU 单次约 46 ms，精度待实拍验收 |
| 2026-06-10 | 增加无板端实拍条件下的 v5 数据清洗、候选训练、PC 评估、量化和板端回归路线 |
| 2026-06-11 | 明确主目标为符合 BSD 精度标准的 `YOLO26n@512`，补充公开数据集路线和 BDD100K 官方 detection 导入入口 |
## 9. 2026-06-13 当前候选快照

保留稳定 v7 候选作为回退：

| 项目 | v7 稳定回退 | v8 no-attn 速度候选 | v9 ReLU from v8 |
|---|---:|---:|---:|
| 模型 | `YOLO26n@640` | `YOLO26n_noattn@640` | `YOLO26n_noattn_relu@640` |
| 板端 NB | `/mnt/UDISK/bsd_v7_yolo26n_640.nb` | `/mnt/UDISK/bsd_v8_yolo26n_noattn_640.nb` | `/mnt/UDISK/bsd_v9_relu_from_v8_640.nb` |
| NB size | `3.52 MiB` | `3.43 MiB` | `2.01 MiB` |
| `coco_val` mAP50 | `0.586` | `0.577` | `0.548` |
| `bdd_proxy_val` mAP50 | `0.674` | `0.680` | `0.673` |
| `bdd_proxy_val` recall | `0.615` | `0.589` | `0.577` |
| 板端 NPU | 约 `92.1 ms` | 约 `67.4 ms` | 约 `31.3 ms` |
| 检测总耗时 | 约 `109.2 ms` | 约 `84.5 ms` | 约 `48.5 ms` |

结论：v9 ReLU from v8 是当前实测速度最好的候选，因为它保持固定 `640x640`，满足硬性 `<5 MB` NB 门槛且有余量，并把板端 NPU 降到约 `31.3 ms`。它还不是生产替代模型，因为 `bdd_proxy_val` recall 仍低于 v7，且正式板端 `board_val` 验证缺失。继续保留 v7 用于对比和回退。不要覆盖 `/mnt/UDISK/bsd_v7_yolo26n_640.nb`；速度候选必须使用独立文件名部署。

重要区别：早期冷启动训练的 v9 ReLU 主要是速度证明，精度损失过大。当前有价值的 v9 是从 v8 微调得到的版本，记录在 `docs/candidates/bsd_v9_yolo26n_noattn_relu_640.md`。

## 10. 2026-06-13 v9.1 Person/Vehicle 优先路线

下一条精度路线冻结高速 v9 结构，只改数据和训练策略。BSD 验收应优先看 `person` 和 `vehicle`，而不是只看四类平均值，因为 bicycle/motorcycle 场景通常仍能看到人和车体。

入口：

```text
tools/evaluate_bsd_pv_priority.py
deployment/dataset/build_bsd_v9p1_pv_priority.py
training/detection/configs/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority.yaml
docs/candidates/bsd_v9p1_pv_priority_plan.md
```

当前逐类检查显示，v9 的主要差距是 recall，不是 precision。在 `bdd_proxy_val` 上，v9 person recall 为 `0.676`，v7 为 `0.731`；v9 vehicle recall 为 `0.806`，v7 为 `0.841`。v9.1 的目标是在保持 `YOLO26n_noattn + ReLU + 640`、NB `<5 MB`、板端 NPU 接近 `31-35 ms` 的同时追回这些指标。

第一版 v9.1 数据集基座：

```text
/root/autodl-tmp/BSD/datasets/bsd_v9p1_pv_priority
```

它基于 `bsd_v6_public_kitti` 构建，移除了重复标签，并可通过 `--extra-set name:/images:/labels` 接入经过审核的额外数据。真正的精度恢复应来自加入 person/vehicle-heavy 道路数据，而不是改板端运行路径。

## 11. 2026-06-13 阶段收尾

本阶段按以下模型角色收尾：

| 角色 | 候选 | 状态 |
|---|---|---|
| 稳定回退 | v7 `YOLO26n@640` | 保留在板端；不要覆盖 `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |
| 已归档中间版本 | v8 `YOLO26n_noattn@640` | 只保留文档/权重；不再做板端优化 |
| 板端速度基线 | v9 `YOLO26n_noattn_relu@640` | 保留为当前速度参考 |
| 精度恢复候选 | v9.1 base `YOLO26n_noattn_relu@640` | PC 侧精度工作的优先候选 |

最新 `bdd_proxy_val` person/vehicle 对比：

| 候选 | P | R | mAP50 | PV Recall | PV mAP50 |
|---|---:|---:|---:|---:|---:|
| v7 | `0.682` | `0.615` | `0.674` | `0.786` | `0.808` |
| v9 | `0.753` | `0.577` | `0.673` | `0.741` | `0.798` |
| v9.1 base | `0.720` | `0.602` | `0.677` | `0.760` | `0.797` |
| v9.1 bddval | `0.744` | `0.593` | `0.680` | `0.743` | `0.798` |

结论：

- v8 不再是活跃候选。它证明了 no-attention 路线，但慢于 v9，且优势不足以继续投入板端优化。
- v9 保持为实测速度基线：NB 约 `2.01 MiB`，板端 NPU 约 `31.3 ms`，live headless 约 `20 FPS`。
- v9.1 base 是当前精度恢复候选，因为它在不改变可部署架构的前提下提升了 person/vehicle recall。
- v9.1 bddval 不被选择：它略微提升 overall mAP50，但相对 v9.1 base 降低了 BSD 关键 person/vehicle recall。

当前提升上限：

- 在当前数据和固定架构下，继续只靠训练带来的收益大概率有限。重复 early-stop 变体没有实质改善 BSD 指标。
- 有意义的精度提升现在需要经过审核的 person/vehicle-heavy 道路数据，或正式板端验证集。
- 有意义的速度提升应作为单独的 runtime/NPU 调度任务处理。除非 `<5 MB` NB 门槛或四路调度失败，否则不要只为速度改变已选模型路径。

下一阶段进入条件：

1. 加入精选道路场景数据，尤其是小目标/遮挡 person 和侧后方 vehicle，然后重建 `bsd_v9p1_pv_priority` 数据集。
2. 从 v9.1 base 重新训练，先按 `bdd_proxy_val` person/vehicle recall 选择，再看 mAP50。
3. 具备板端采集条件后，构建真实 `board_val`；没有通过它之前，不选择最终 BSD 发布模型。
