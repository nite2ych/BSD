# BSD v9.1 YOLO26n No-Attention ReLU 512 BDD-Val 候选

## 状态

已归档为性能兜底候选。除非板端吞吐成为首要约束，否则不要用这个 512 模型替换当前 BSD 主模型 `YOLO26n_noattn + ReLU + 640`。

## 训练

| 项目 | 值 |
|---|---|
| Config | `training/detection/configs/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval.yaml` |
| 架构 | `YOLO26n_noattn + ReLU` |
| Input | `512x512` |
| Initial weights | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/weights/best.pt` |
| 数据集 | `datasets/bsd_v9p1_pv_priority` |
| Early-stop 验证集 | `bdd_proxy_val` |
| 最佳 epoch | 22 |
| 停止 | epoch 47 触发 early stop，patience 25 |

## 本地产物

| 产物 | 路径 |
|---|---|
| PT | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/weights/best.pt` |
| ONNX | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/weights/best.onnx` |
| 指标 | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/v5_eval_summary.json` |
| 训练 CSV | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/results.csv` |
| 冻结配置副本 | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/train_config.yaml` |

## 精度

| 模型 | 验证集 | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| v9.1 640 bddval | coco_val | 0.762 | 0.461 | 0.539 | 0.325 |
| v9.1 512 bddval | coco_val | 0.695 | 0.449 | 0.509 | 0.300 |
| v9.1 640 bddval | bdd_proxy_val | 0.744 | 0.593 | 0.680 | 0.456 |
| v9.1 512 bddval | bdd_proxy_val | 0.685 | 0.502 | 0.578 | 0.372 |

512 模型在 BDD proxy 上的逐类结果：

| 类别 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.677 | 0.585 | 0.646 | 0.379 |
| bicycle | 0.636 | 0.376 | 0.414 | 0.243 |
| motorcycle | 0.654 | 0.289 | 0.436 | 0.282 |
| vehicle | 0.771 | 0.757 | 0.817 | 0.582 |

## 板端性能背景

之前的 512 板端 probe 使用同样的高速图结构，但直接从 640 v9.1 权重导出，没有按 512 重新训练。实测结果：

| 输入 | FPS | 检测总耗时 | NV21 预处理 | NPU | 解码 |
|---|---:|---:|---:|---:|---:|
| v9.1 640 | about 19.99 | 48.7-49.0 ms | 15.8-16.0 ms | 31.7-31.8 ms | about 1 ms |
| v9.1 512 probe | 19.99 | 32.1-32.3 ms | 10.5-10.8 ms | 20.8-21.0 ms | 0.7-0.8 ms |

整体 FPS 仍接近 20，是因为当前串行循环受摄像头采集/等待节奏限制，而不是受 512 NPU 耗时限制。

## 结论

保留该候选作为存档兜底。如果 640 路线无法满足四路调度要求，它有参考价值；但目前 BDD proxy 回退过大，不适合作为主模型：

- `bdd_proxy_val mAP50`: `0.680 -> 0.578`
- `bdd_proxy_val recall`: `0.593 -> 0.502`

如果后续需要启用该候选，下一步流程是：

1. 用 `deployment/quantize/split_yolo_head.py` 拆分 raw-logit ONNX。
2. 使用 512 校准集量化。
3. 用 `test_npu_direct` 测试。
4. 用 `live_bsd ... headless 512` 部署。
