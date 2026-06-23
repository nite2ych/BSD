# BSD v9 YOLO26n No-Attention ReLU 640 Candidate

This route tests whether replacing SiLU/Swish with ReLU improves VIPLite NPU
efficiency at fixed `640x640`. It does not replace v7 or v8 yet.

## Baselines

| Candidate | Role | Board NPU | Notes |
|---|---|---:|---|
| v7 `YOLO26n@640` | stable fallback | about `92.1 ms` | best current fallback recall |
| v8 no-attn `YOLO26n@640` | trained speed candidate | about `67.4 ms` | removes `MATRIXMUL/SOFTMAX/PERMUTE` |
| v9 no-attn ReLU probe | speed probe only | about `28.8 ms` | random/untrained weights |

Keep `/mnt/UDISK/bsd_v7_yolo26n_640.nb` unchanged.

## Probe Result

The probe uses the same no-attention YOLO26n topology as v8, but overrides
`Conv.default_act` to `ReLU` before model construction and ONNX export.

Split ONNX operator check:

```text
Conv 104
Relu 97
Concat 23
Add 17
Split 9
Reshape 6
MaxPool 3
Resize 2
Sigmoid 0
Mul 0
MatMul 0
Softmax 0
Transpose 0
```

Pegasus fused graph:

```text
CONV2D 104
RELU 97
CONCAT 23
ADD 17
SPLIT 9
RESHAPE2 6
POOL 3
RESIZE 2
```

Quantized probe NB:

| Item | Value |
|---|---|
| Local NB | `artifacts/experiments/bsd_v9_yolo26n_noattn_relu_640_probe/bsd_v9_yolo26n_noattn_relu_640_probe.nb` |
| Board NB | `/mnt/UDISK/bsd_v9_yolo26n_noattn_relu_640_probe.nb` |
| NB size | `458,624` bytes, about `0.44 MiB` |
| NB MD5 | `259312dad61f8f29d32d2574e7b0b4c5` |
| NB magic | `VPMN` |
| Pegasus simulator | `3.59 ms` |
| Board benchmark | `28.83 ms` NPU, `38.87 ms` total, `25.72 FPS` |

The probe NB is tiny because it uses random weights and compresses unusually
well. Do not use its size as the trained-model size estimate.

## Interpretation

ReLU is materially faster than SiLU/Swish on this VIPLite path. The trained v8
no-attn model still has `SWISH 97` in Pegasus fused graph and runs about
`67.4 ms` NPU. The ReLU probe runs about `28.8 ms` NPU. This makes a trained
v9 ReLU candidate worth running, with the main risk being accuracy regression.

## 2026-06-13 From-v8 Fine-tune Result

The cold-trained v9 ReLU route recovered too little accuracy. The useful v9
candidate is the ReLU model fine-tuned from trained v8 no-attention weights:

```text
training/detection/configs/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft.yaml
```

Accuracy:

| Dataset | Candidate | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| `coco_val` | v8 no-attn SiLU | 0.739 | 0.506 | 0.577 | 0.361 |
| `coco_val` | v9 ReLU from v8 | 0.733 | 0.479 | 0.548 | 0.331 |
| `bdd_proxy_val` | v8 no-attn SiLU | 0.744 | 0.589 | 0.680 | 0.473 |
| `bdd_proxy_val` | v9 ReLU from v8 | 0.753 | 0.577 | 0.673 | 0.460 |

Deployment artifacts:

| Item | Value |
|---|---|
| Local PT | `artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/weights/best.pt` |
| Local ONNX | `artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/weights/best.onnx` |
| Split ONNX | `artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/best_640_split_nosig.onnx` |
| Local NB | `artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft.nb` |
| Board NB | `/mnt/UDISK/bsd_v9_relu_from_v8_640.nb` |
| NB size | `2,110,400` bytes, about `2.01 MiB` |
| NB MD5 | `008c779ab66ad29bcb993df9a57d5a36` |
| Pegasus simulator | `4.43 ms` |

Split ONNX operator check:

```text
Conv 104
Relu 97
Concat 23
Add 17
Constant 10
Split 9
Reshape 6
MaxPool 3
Resize 2
```

Pegasus fused graph check:

```text
CONVOLUTION 104
RELU 97
CONCAT 23
ADD 17
SPLIT 9
RESHAPE 6
POOLING 3
IMAGE_RESIZE 2
```

No `SWISH`, `SIGMOID`, `MUL`, `MATRIXMUL`, `SOFTMAX`, `PERMUTE`, or
`TRANSPOSE` operators were found in the fused graph.

Board benchmark on V853:

| Mode | Result |
|---|---|
| `bench_npu_loop`, 100 loops, `640x640` BGR frame | `avg_total=45.83 ms`, `avg_pre=13.47 ms`, `avg_npu=31.21 ms`, `avg_decode=1.15 ms`, `fps=21.82` |
| `live_bsd headless`, 1280x720 NV21M camera | about `19.99 FPS`, detection call `48.5-49.4 ms`, NV21 preprocess `16.0-16.5 ms`, NPU `31.3-32.2 ms`, decode `1.1-1.2 ms` |

Current judgement: v9 ReLU from v8 is the best measured speed candidate so far.
It stays well below the hard `<5 MB` NB size gate and cuts board NPU time from
v8's about `67.4 ms` to about `31.3 ms`. The tradeoff is a small proxy accuracy
drop from v8, especially on `coco_val`; formal board-side validation is still
required before replacing the stable fallback.

## Training Config

Use:

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v9_yolo26n_noattn_relu_640_stage1.yaml \
  --mode all \
  --device 0
```

The training script supports:

```yaml
model:
  activation: ReLU
```

This must be applied before constructing the Ultralytics model; otherwise the
export will silently fall back to SiLU/Swish.

## Gates

| Gate | Requirement |
|---|---|
| Accuracy | Compare against v7 and v8 on `coco_val` and `bdd_proxy_val` |
| Size | Trained quantized NB `<5 MB` |
| Speed | Must beat v8 trained `67.4 ms`; target is `30-45 ms` NPU |
| Board flow | `split -> quantize -> bench_npu_loop -> live_bsd headless` |
| Baseline safety | Do not overwrite `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |
