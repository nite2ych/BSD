# Quantization Tools

Tools for preparing BSD ONNX models for Pegasus/VIP9000PICO.

| File | Purpose |
|---|---|
| `split_yolo_head.py` | Removes NMS graph tail and exports `boxes` + raw-logit `scores` |
| `split_yolo_head_sig.py` | Experimental variant that adds Sigmoid to `scores` |
| `prepare_calibration_data.py` | Generates 640x640 RGB calibration images from board frames |
| `../../tools/quantize_bsd_candidate.py` | Uploads 320/416/512/640 split ONNX candidates to Ubuntu20 Pegasus and exports NB |

Current standard path:

```bash
python deployment/quantize/split_yolo_head.py \
  --input artifacts/detection/bsd_yolo26s/best.onnx \
  --output artifacts/detection/bsd_yolo26s/best_640_split_nosig.onnx
```

The board decoder expects raw logits and applies sigmoid in C. Do not use the sigmoid-output ONNX/NB with the current `yolo_decode.c` unless the board decoder is changed accordingly.

For compressed candidates, keep the same raw-logit convention and pass the real model input size:

```bash
python tools/quantize_bsd_candidate.py \
  --onnx artifacts/detection/bsd_v5_yolo26n_416/best_416_split_nosig.onnx \
  --calib-dir artifacts/calib_camera_416 \
  --model-name bsd_v5_yolo26n_416 \
  --model-size 416
```

The calibration directory should already contain RGB letterbox images at the target size. Do not mix 640 calibration images into a 416/512 quantization run.

Current deployed candidate:

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

The verified NB for this candidate is about `3.52 MiB`, so it passes the hard `<5 MB` gate.

## 640 fixed optimization notes

Input size is fixed at `640x640` for the current optimization pass. Do not use
416/512/320 results to explain this pass.

Measured variants on V853:

| Variant | NB size | Board NPU | Result |
|---|---:|---:|---|
| `bsd_v7_yolo26n_640_public_kitti` | `3.52 MiB` | `92.1-92.4 ms` | Current baseline |
| `best_640_split_nosig` + `onnxsim` | `3.52 MiB` | `92.1-92.2 ms` | No speed gain |
| Pegasus `--force-remove-permute` | `3.52 MiB` | `92.1-92.2 ms` | No speed gain |
| Pegasus `--dtype quantized_strict` | `20.56 MiB` | Not tested | Fails `<5 MB` gate |
| Pegasus `perchannel_symmetric_affine int8` | `4.37 MiB` | Not tested | Passes size, simulator slower |

The fused Pegasus graph still contains attention-like and layout-heavy ops:

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

Conclusion: ordinary ONNX simplification and Pegasus export flags do not move
the board NPU time for this YOLO26n@640 graph. To approach a `~50 ms` target
while keeping `640x640` and NB `<5 MB`, the next meaningful experiment is an
NPU-friendly nano model at 640 that avoids attention blocks (`MatMul`/`Softmax`)
and minimizes `Permute`/`Resize` overhead.

The quantization helper supports small export matrices:

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
