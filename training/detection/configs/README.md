# BSD 检测训练配置

v5/v6/v7 配置定义了“暂无新增板端采集图”条件下的训练矩阵。当前稳定回退候选是 `bsd_v7_yolo26n_640_public_kitti_stage1.yaml`，当前主线优先看 v9/v9.1 系列。

先在训练服务器构建均衡数据集：

```bash
python deployment/dataset/build_bsd_v5_balanced.py
```

候选顺序：

| 配置 | 用途 |
|---|---|
| `bsd_v5_yolo26s_640.yaml` | 对比 5/20 模型的精度基线 |
| `bsd_v5_yolo26s_512.yaml` | 同模型规模下降输入尺寸 |
| `bsd_v5_yolo26n_640.yaml` | 小模型精度上限 |
| `bsd_v5_yolo26n_512.yaml` | 如果 `n@640` 吞吐不够，用作压缩兜底 |
| `bsd_v5_yolo26n_416.yaml` | 四路吞吐候选 |
| `bsd_v7_yolo26n_640_public_kitti_stage1.yaml` | 已部署且满足 `<5 MB` 的稳定回退候选 |
| `bsd_v8_yolo26n_noattn_640_stage1.yaml` | 640 固定输入速度候选；移除 C2PSA attention，保留 YOLO26 head |

运行一个旧矩阵候选：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v5_yolo26n_512.yaml \
  --mode all \
  --device 0
```

训练需要 GPU。下游可视化检查和板端验证保持 `conf=0.5`、`nms=0.45`，避免把精度变化和阈值变化混在一起。

当前 v7 板端候选：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v7_yolo26n_640_public_kitti_stage1.yaml \
  --mode export \
  --weights /root/autodl-tmp/BSD/artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/weights/best.pt \
  --device cpu
```

当前 v8 速度候选：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v8_yolo26n_noattn_640_stage1.yaml \
  --mode all \
  --device 0
```

v8 模型定义是 `training/detection/models/yolo26n_noattn.yaml`。它刻意与已安装 Ultralytics 的 `yolo26.yaml` 分离，保证冻结的 v7 基线可以复现。
