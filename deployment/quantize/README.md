# 量化工具

用于把 BSD ONNX 模型整理成 Pegasus/VIP9000PICO 可部署 NB 的工具。

| 文件 | 用途 |
|---|---|
| `split_yolo_head.py` | 删除 NMS 图尾，导出 `boxes` + raw-logit `scores` |
| `split_yolo_head_sig.py` | 给 `scores` 添加 Sigmoid 的实验版本 |
| `prepare_calibration_data.py` | 从板端帧生成 640x640 RGB 校准图 |
| `../../tools/quantize_bsd_candidate.py` | 将 320/416/512/640 拆分后的 ONNX 上传到 Ubuntu20 Pegasus 并导出 NB |

当前标准路径：

```bash
python deployment/quantize/split_yolo_head.py \
  --input artifacts/detection/bsd_yolo26s/best.onnx \
  --output artifacts/detection/bsd_yolo26s/best_640_split_nosig.onnx
```

板端解码器期望输入 raw logits，并在 C 端执行 sigmoid。除非同步修改 `yolo_decode.c`，否则不要把 sigmoid-output ONNX/NB 直接用于当前板端解码器。

压缩候选也保持同样的 raw-logit 约定，并传入真实模型输入尺寸：

```bash
python tools/quantize_bsd_candidate.py \
  --onnx artifacts/detection/bsd_v5_yolo26n_416/best_416_split_nosig.onnx \
  --calib-dir artifacts/calib_camera_416 \
  --model-name bsd_v5_yolo26n_416 \
  --model-size 416
```

校准目录应提前准备好目标尺寸的 RGB letterbox 图片。不要把 640 校准图混进 416/512 量化流程。

当前已部署候选：

```bash
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

该候选已验证 NB 约 `3.52 MiB`，满足硬性 `<5 MB` 约束。

## 640 固定输入优化记录

当前优化阶段输入固定为 `640x640`。不要用 416/512/320 的结果解释这一轮优化。

V853 实测变体：

| 变体 | NB 大小 | 板端 NPU | 结果 |
|---|---:|---:|---|
| `bsd_v7_yolo26n_640_public_kitti` | `3.52 MiB` | `92.1-92.4 ms` | 当前基线 |
| `best_640_split_nosig` + `onnxsim` | `3.52 MiB` | `92.1-92.2 ms` | 无速度收益 |
| Pegasus `--force-remove-permute` | `3.52 MiB` | `92.1-92.2 ms` | 无速度收益 |
| Pegasus `--dtype quantized_strict` | `20.56 MiB` | 未测试 | 不满足 `<5 MB` |
| Pegasus `perchannel_symmetric_affine int8` | `4.37 MiB` | 未测试 | 大小满足，但模拟器更慢 |

Pegasus 融合图中仍包含 attention-like 和 layout-heavy 算子：

```text
CONV2D 102
SWISH 87
CONCAT 21
ADD 20
RESHAPE2 12
SPLIT 11
PERMUTE 4
MATRIXMUL 4
SOFTMAX 2
RESIZE 2
```

结论：普通 ONNX 简化和 Pegasus 导出参数无法明显降低这个 YOLO26n@640 图的板端 NPU 耗时。若要在保持 `640x640` 和 NB `<5 MB` 的前提下接近 `~50 ms`，下一步有意义的实验应是 640 输入的 NPU 友好 nano 模型，避开 attention block（`MatMul`/`Softmax`），并尽量减少 `Permute`/`Resize` 开销。

量化辅助脚本支持小规模导出矩阵：

```bash
python tools/quantize_bsd_candidate.py \
  --onnx artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx \
  --calib-dir artifacts/bdd100k/calib_images \
  --model-name bsd_v7_yolo26n_640_public_kitti_forceperm \
  --model-size 640 \
  --force-remove-permute

python tools/quantize_bsd_candidate.py \
  --onnx artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx \
  --calib-dir artifacts/bdd100k/calib_images \
  --model-name bsd_v7_yolo26n_640_public_kitti_qstrict \
  --model-size 640 \
  --export-dtype quantized_strict

python tools/quantize_bsd_candidate.py \
  --onnx artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx \
  --calib-dir artifacts/bdd100k/calib_images \
  --model-name bsd_v7_yolo26n_640_public_kitti_perch_int8 \
  --model-size 640 \
  --quantizer perchannel_symmetric_affine \
  --qtype int8
```
