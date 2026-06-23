# BSD v10/v11 激活函数候选

这些候选保持当前 `640x640` 输入和 `YOLO26n_noattn` 拓扑，只替换 Conv 激活函数，用来验证 VIPLite 是否能在不出现 v9 纯 ReLU 精度损失的情况下获得更高速度。

稳定板端模型保持不变：

```text
/mnt/UDISK/bsd_v7_yolo26n_640.nb
```

## 基线

| 候选 | 角色 | coco mAP50 | bdd mAP50 | 板端 NPU |
|---|---|---:|---:|---:|
| v7 `YOLO26n@640` | 稳定回退 | `0.586` | `0.674` | 约 `92.1 ms` |
| v8 no-attn SiLU | 速度候选 | `0.577` | `0.680` | 约 `67.4 ms` |
| v9 no-attn ReLU | 速度证明，不可直接部署 | `0.463` | `0.596` | probe 约 `28.8 ms` |

## 新路线

| 候选 | 激活函数 | 意图 |
|---|---|---|
| v10 | `LeakyReLU(0.1)` | 首选。通常接近 ReLU 速度，同时减少死激活风险。 |
| v11 | `ReLU6` | 备选。有界激活可能追回部分精度，但必须确认板端算子映射。 |

## 命令

训练 v10：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v10_yolo26n_noattn_leakyrelu_640_stage1.yaml \
  --mode all \
  --device 0
```

训练 v11：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v11_yolo26n_noattn_relu6_640_stage1.yaml \
  --mode all \
  --device 0
```

## 验收门槛

| 门槛 | 要求 |
|---|---|
| 精度 | 先在 `bdd_proxy_val` 上对比 v8；v7 保持回退。 |
| 召回 | BSD 路线不能为了速度接受明显召回下降。 |
| 大小 | 量化压缩后的 NB 必须 `<5 MB`。 |
| 速度 | 必须快于 v8 `~67.4 ms` NPU；如果精度保住，目标是 `30-50 ms`。 |
| 板端流程 | `PT -> ONNX -> split raw-logit ONNX -> Pegasus NB -> bench_npu_loop -> live_bsd headless`。 |
