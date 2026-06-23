# BSD v9 YOLO26n No-Attention ReLU 640 候选

该路线验证在固定 `640x640` 输入下，将 SiLU/Swish 替换为 ReLU 是否能提升 VIPLite NPU 效率。它曾是速度验证路线，后续由 v9.1 继承。

## 基线

| 候选 | 角色 | 板端 NPU | 备注 |
|---|---|---:|---|
| v7 `YOLO26n@640` | 稳定回退 | 约 `92.1 ms` | 当前回退模型召回最好 |
| v8 no-attn `YOLO26n@640` | 已训练速度候选 | 约 `67.4 ms` | 移除 `MATRIXMUL/SOFTMAX/PERMUTE` |
| v9 no-attn ReLU probe | 仅速度 probe | 约 `28.8 ms` | 随机/未训练权重 |

保持 `/mnt/UDISK/bsd_v7_yolo26n_640.nb` 不变。

## Probe 结果

probe 使用与 v8 相同的 no-attention YOLO26n 拓扑，但在构建模型和导出 ONNX 前把 `Conv.default_act` 覆盖为 `ReLU`。

拆分 ONNX 算子检查：

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

Pegasus 融合图：

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

量化 probe NB：

| 项目 | 值 |
|---|---|
| 本地 NB | `artifacts/experiments/bsd_v9_yolo26n_noattn_relu_640_probe/bsd_v9_yolo26n_noattn_relu_640_probe.nb` |
| 板端 NB | `/mnt/UDISK/bsd_v9_yolo26n_noattn_relu_640_probe.nb` |
| NB 大小 | `458,624` bytes，约 `0.44 MiB` |
| NB MD5 | `259312dad61f8f29d32d2574e7b0b4c5` |
| NB magic | `VPMN` |
| Pegasus 模拟器 | `3.59 ms` |
| 板端 benchmark | `28.83 ms` NPU，`38.87 ms` total，`25.72 FPS` |

probe NB 很小，是因为随机权重异常容易压缩。不要用这个大小估计已训练模型大小。

## 判断

ReLU 在这条 VIPLite 路径上明显快于 SiLU/Swish。已训练 v8 no-attn 模型的 Pegasus 融合图中仍有 `SWISH 97`，板端 NPU 约 `67.4 ms`；ReLU probe 约 `28.8 ms`。因此值得训练 v9 ReLU 候选，主要风险是精度回退。

## 2026-06-13 从 v8 微调结果

冷启动训练的 v9 ReLU 精度恢复不足。真正有用的是从已训练 v8 no-attention 权重微调出来的 ReLU 模型：

```text
training/detection/configs/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft.yaml
```

精度：

| 数据集 | 候选 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| `coco_val` | v8 no-attn SiLU | 0.739 | 0.506 | 0.577 | 0.361 |
| `coco_val` | v9 ReLU from v8 | 0.733 | 0.479 | 0.548 | 0.331 |
| `bdd_proxy_val` | v8 no-attn SiLU | 0.744 | 0.589 | 0.680 | 0.473 |
| `bdd_proxy_val` | v9 ReLU from v8 | 0.753 | 0.577 | 0.673 | 0.460 |

部署产物：

| 项目 | 值 |
|---|---|
| 本地 PT | `artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/weights/best.pt` |
| 本地 ONNX | `artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/weights/best.onnx` |
| 拆分 ONNX | `artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/best_640_split_nosig.onnx` |
| 本地 NB | `artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft.nb` |
| 板端 NB | `/mnt/UDISK/bsd_v9_relu_from_v8_640.nb` |
| NB 大小 | `2,110,400` bytes，约 `2.01 MiB` |
| NB MD5 | `008c779ab66ad29bcb993df9a57d5a36` |
| Pegasus 模拟器 | `4.43 ms` |

拆分 ONNX 算子检查：

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

Pegasus 融合图检查：

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

融合图中没有发现 `SWISH`、`SIGMOID`、`MUL`、`MATRIXMUL`、`SOFTMAX`、`PERMUTE` 或 `TRANSPOSE`。

V853 板端 benchmark：

| 模式 | 结果 |
|---|---|
| `bench_npu_loop`，100 loops，`640x640` BGR frame | `avg_total=45.83 ms`，`avg_pre=13.47 ms`，`avg_npu=31.21 ms`，`avg_decode=1.15 ms`，`fps=21.82` |
| `live_bsd headless`，1280x720 NV21M camera | 约 `19.99 FPS`，detection call `48.5-49.4 ms`，NV21 preprocess `16.0-16.5 ms`，NPU `31.3-32.2 ms`，decode `1.1-1.2 ms` |

当前判断：v9 ReLU from v8 是当时实测速度最好的候选。它明显低于 `<5 MB` 硬约束，并把板端 NPU 从 v8 的约 `67.4 ms` 降到约 `31.3 ms`。代价是 proxy 精度小幅下降，尤其是 `coco_val`。正式替换稳定回退前仍需要板端验证集。

## 训练配置

命令：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v9_yolo26n_noattn_relu_640_stage1.yaml \
  --mode all \
  --device 0
```

训练脚本支持：

```yaml
model:
  activation: ReLU
```

必须在构建 Ultralytics 模型前应用该配置；否则导出会悄悄回到 SiLU/Swish。

## 验收门槛

| 门槛 | 要求 |
|---|---|
| 精度 | 在 `coco_val` 和 `bdd_proxy_val` 上对比 v7/v8 |
| 大小 | 已训练量化 NB `<5 MB` |
| 速度 | 必须快于 v8 已训练模型 `67.4 ms`；目标 `30-45 ms` NPU |
| 板端流程 | `split -> quantize -> bench_npu_loop -> live_bsd headless` |
| 基线保护 | 不覆盖 `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |
