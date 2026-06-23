# BSD v8 YOLO26n No-Attention 640 候选

这是速度路线实验，不替代 v7 已部署稳定基线。

## 基线规则

保持 v7 不变用于对比：

| 项目 | 值 |
|---|---|
| 基线候选 | `bsd_v7_yolo26n_640_public_kitti` |
| 板端 NB | `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |
| 基线 NPU | 约 `92.1 ms` |
| 基线 NB | `3.52 MiB` |

## 模型变化

v8 保持 YOLO26 输入/输出约定，但移除了会产生低效 NPU 算子的 attention block。

| 项目 | 值 |
|---|---|
| 模型 YAML | `training/detection/models/yolo26n_noattn.yaml` |
| 训练配置 | `training/detection/configs/bsd_v8_yolo26n_noattn_640_stage1.yaml` |
| 输入 | `640x640` |
| 检测头 | YOLO26，`reg_max=1`，兼容 raw-logit split |
| ABI | 不改 `DetResult` / `AlarmEvent` |

原 v7 图包含 attention/layout-heavy 算子：

```text
MATRIXMUL 4
SOFTMAX 2
PERMUTE 4
```

v8 no-attention probe 图移除了这些算子：

```text
CONV2D 104
SWISH 97
CONCAT 23
ADD 17
SPLIT 9
RESHAPE2 6
POOL 3
RESIZE 2
```

## 速度 Probe

probe 由随机/未训练权重导出，只用于测板端耗时，不用于精度判断。

| 变体 | NB 大小 | 模拟器 | V853 板端 NPU | 检测总耗时 |
|---|---:|---:|---:|---:|
| v7 `YOLO26n@640` | `3.52 MiB` | `18.49 ms` | `92.1 ms` | `109.3 ms` |
| v8 no-attn probe | `1.69 MiB` | `14.66 ms` | `62.8-62.9 ms` | `79.8-80.0 ms` |

这是在固定 `640x640` 下的真实速度提升，但尚未达到 `~50 ms` 目标。它值得训练，因为保留了板端解码路径，并且明显低于 `<5 MB` NB 限制。

## 已训练 Stage1 结果

训练在 epoch 78 early stop，最佳 epoch 为 48。模型从 v7 public-KITTI 权重初始化，转移了 `635/720` 个权重项。

| 验证集 | 模型 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `coco_val` | v7 `n@640` | 0.741 | 0.521 | 0.586 | 0.364 |
| `coco_val` | v8 no-attn `n@640` | 0.739 | 0.506 | 0.577 | 0.361 |
| `bdd_proxy_val` | v7 `n@640` | 0.682 | 0.615 | 0.674 | 0.467 |
| `bdd_proxy_val` | v8 no-attn `n@640` | 0.744 | 0.589 | 0.680 | 0.473 |

解读：

- v8 在 `coco_val` 上略低于 v7。
- v8 在更贴近 BSD 道路场景的 `bdd_proxy_val` mAP 上略高。
- v8 在 `bdd_proxy_val` recall 上下降约 `0.026`，因此在没有板端验证集前，v7 仍是回退模型。

## 已训练量化

| 产物 | 值 |
|---|---|
| PT | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.pt` |
| ONNX | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.onnx` |
| 拆分 ONNX | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/best_640_split_nosig.onnx` |
| NB | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1.nb` |
| 板端 NB | `/mnt/UDISK/bsd_v8_yolo26n_noattn_640.nb` |
| NB 大小 | `3,598,080` bytes，约 `3.43 MiB` |
| NB MD5 | `ca0667ef858073ec9e93f0cb8b7c23f3` |
| NB magic | `VPMN` |
| 硬性大小门槛 | 满足 `<5 MB` |

已训练拆分 ONNX 算子检查：

```text
Conv 104
Sigmoid 97
Mul 97
Concat 23
Add 17
Split 9
Reshape 6
MaxPool 3
Resize 2
MatMul 0
Softmax 0
Transpose 0
```

Pegasus 融合图：

```text
CONV2D 104
SWISH 97
CONCAT 23
ADD 17
SPLIT 9
RESHAPE2 6
POOL 3
RESIZE 2
```

Pegasus 导出后模拟器：

```text
Run the 1 time: 17.12 ms
```

## 板端结果

在 V853 上使用与 v7 相同的 `bench_npu_loop` 和 `live_bsd headless` 流程测试。v7 NB 仍单独保留为 `/mnt/UDISK/bsd_v7_yolo26n_640.nb`。

| 变体 | NB 大小 | 连续 benchmark | `live_bsd headless` 检测总耗时 | `live_bsd headless` NPU |
|---|---:|---:|---:|---:|
| v7 `YOLO26n@640` | `3.52 MiB` | `91.66 ms` NPU，`9.83 FPS` | 约 `109.2 ms` | 约 `92.1 ms` |
| v8 no-attn trained | `3.43 MiB` | `67.28 ms` NPU，`12.93 FPS` | 约 `84.5 ms` | 约 `67.4 ms` |

速度变化：

- NPU 耗时从约 `92.1 ms` 降到约 `67.4 ms`。
- detection-call 总耗时从约 `109.2 ms` 降到约 `84.5 ms`。
- 在固定 `640x640` 下，板端 NPU 下降约 `26-27%`。

## 命令

训练和评估：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v8_yolo26n_noattn_640_stage1.yaml \
  --mode all \
  --device 0
```

训练后导出并拆分：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v8_yolo26n_noattn_640_stage1.yaml \
  --mode export \
  --weights /root/autodl-tmp/BSD/artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.pt \
  --device cpu

python deployment/quantize/split_yolo_head.py \
  --input artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.onnx \
  --output artifacts/bsd_v8_yolo26n_noattn_640_stage1/best_640_split_nosig.onnx
```

量化：

```bash
python tools/quantize_bsd_candidate.py \
  --host 192.168.144.133 \
  --user ubuntu \
  --password "$BSD_QUANT_PASSWORD" \
  --remote-dir /home/ubuntu/bsd_quantize \
  --onnx artifacts/bsd_v8_yolo26n_noattn_640_stage1/best_640_split_nosig.onnx \
  --calib-dir artifacts/bdd100k/calib_images \
  --model-name bsd_v8_yolo26n_noattn_640_stage1 \
  --model-size 640
```

## 量化矩阵补充

激活函数实验未能保住精度后，v8 又测试了一组低风险 Pegasus 量化/导出变体。这些测试不改变已训练 PT 权重。

| 变体 | 本地 NB | NB 大小 | Pegasus 模拟器 | 结果 |
|---|---|---:|---:|---|
| standard `asymmetric_affine` | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1.nb` | `3,598,080` bytes | `17.12 ms` | 当前最好 |
| `moving_average` algorithm | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1_mavg.nb` | `3,598,080` bytes | `17.28 ms` | 无速度收益 |
| `kl_divergence` algorithm | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1_kl.nb` | `3,598,080` bytes | `17.88 ms` | 更慢 |
| `perchannel_symmetric_affine int8` | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1_perch_int8.nb` | `4,427,776` bytes | `23.66 ms` | 更慢、更大 |
| sigmoid score output | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1_sig_v2.nb` | `3,602,752` bytes | `17.64 ms` | 无速度收益；需要修改解码器 |

sigmoid-output split 已按正确 BSD 类别元数据重新生成：

```bash
python deployment/quantize/split_yolo_head_sig.py \
  --input artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.onnx \
  --output artifacts/bsd_v8_yolo26n_noattn_640_stage1/best_640_split_sig.onnx
```

重要：sigmoid-output NB 不能直接替换当前板端解码器。当前 `yolo_decode.c` 期望 raw logits，并在 CPU 上执行 sigmoid。只有在板端解码器改成消费已 sigmoid 的分数后，才能使用 sigmoid NB。

结论：标准 v8 量化仍是该路线速度最好的候选。已测试的 Pegasus 量化变体没有带来模拟器速度收益，因此不作为优先板端部署候选。后续 v8 工作应转向板端运行时集成：输入预处理、队列和四路 NPU 调度。

## 验收门槛

| 门槛 | 要求 |
|---|---|
| 精度 | 在 `coco_val` 和 `bdd_proxy_val` 上对比 v7 |
| 大小 | 量化 NB `<5 MB` |
| 板端速度 | 必须快于 v7 `92 ms`，目标接近 `50 ms` |
| 板端流程 | 先 `test_npu_direct`，再 `live_bsd headless 640` |
| 基线保护 | 不覆盖 `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |

## 结论

v8 no-attn 是速度候选，不是稳定替代。它满足 NB 大小门槛，并在保留 `640x640` 输入和现有 raw-logit 解码路径的同时明显提升板端速度。由于 `bdd_proxy_val` recall 仍低于 v7，且正式 `board_val` 仍缺失，v7 保持生产回退。

## 2026-06-13 归档结论

当前判断：v8 已归档为中间候选。它应保留在仓库中，作为“移除 attention 能改善 VIPLite 图”的证据；新的板端部署和优化应走 v9 或 v9.1。

它有价值的原因：

- 大小门槛富余：`3.43 MiB`，低于硬性 `<5 MB` NB 限制。
- 固定 640 速度明显提升：板端 NPU 从约 `92.1 ms` 降到约 `67.4 ms`。
- 道路场景 proxy mAP 略高于 v7：`bdd_proxy_val` mAP50 为 `0.680`，v7 为 `0.674`。
- v8 避开了 v7 中映射到 VIPLite 较差的 attention 算子（`MATRIXMUL/SOFTMAX/PERMUTE`）。

它暂不替代 v7 的原因：

- `bdd_proxy_val` recall 从 v7 `0.615` 降到 v8 `0.589`，约 `-0.026`。BSD 更不能接受近处 person/two-wheel/vehicle 漏检。
- `coco_val` 也略低于 v7：mAP50 `0.577` vs `0.586`。
- 还没有正式板端 `board_val`；当前板端测试只能证明链路和速度。

最终定位：

| 用途 | 结论 |
|---|---|
| 历史对比 | 保留 |
| no-attention 证明 | 保留 |
| 速度候选 | 已被 v9 替代 |
| 四路调度基线 | 已被 v9 替代 |
| 生产回退替代 | 暂不替代 |
| 最终 BSD 模型 | 受 `board_val` 和召回复核阻塞 |

## 剩余优化空间

除非出现新的编译器参数或 SDK 路径，且 v9/v9.1 无法编译，否则不要继续在 v8 Pegasus 量化变体上投入时间。标准 v8 量化仅作为对比记录保留。

以下剩余工作已经转到 v9/v9.1 路线：

1. 板端运行时 profiling：拆分 `capture / preprocess / set_input / awnn_run / get_output / decode / qbuf`，只优化实测热点。
2. 四路设计：一个 NPU worker、四个 latest-frame slots、round-robin 推理、丢弃陈旧帧。不要跑四个独立模型进程，因为 VIPLite 会串行化 NPU 工作。
3. 预处理优化：保留现有 NV21M fused path，再减少行索引重复计算、重复 padding fill，并避免不必要的 buffer clear。
4. 精度恢复：保持 no-attention + `640x640`，加入更难道路负样本；如果 recall 仍落后，再从 v7/v4 蒸馏。
5. 更小架构属于单独 v12 路线，仅当 v8/v9 在调度和预处理优化后仍无法满足四路延迟时再考虑。简单替换激活函数已被否决，因为 v9/v10/v11 曾出现较大精度损失。
