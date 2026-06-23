# BSD v9.1 Person/Vehicle 优先路线

## 目标

保持当前高速 v9 部署形态，通过数据清洗和保守微调追回 BSD 最关键的 `person` / `vehicle` 召回。

冻结约束：

| 项目 | 值 |
|---|---|
| 架构 | `YOLO26n_noattn` |
| 激活函数 | `ReLU` |
| 输入 | `640x640` |
| 初始权重 | `artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/weights/best.pt` |
| 板端输出 | raw-logit `boxes/scores`，沿用现有 `yolo_decode.c` |
| NB 大小门槛 | `<5 MB` |
| 速度目标 | 板端 NPU 接近 v9，约 `31-35 ms` |

## 2026-06-13 Person/Vehicle 基线

生成命令：

```bash
python tools/evaluate_bsd_pv_priority.py \
  --candidate v4_s640:640:artifacts/bsd_v4_150ep/weights/best.pt \
  --candidate v7_n640:640:artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/weights/best.pt \
  --candidate v8_noattn_silu640:640:artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.pt \
  --candidate v9_relu_from_v8_640:640:artifacts/bsd_v9_yolo26n_noattn_relu_640_from_v8_ft/weights/best.pt \
  --coco-yaml datasets/bsd_v6_public_kitti/dataset.yaml \
  --bdd-yaml datasets/bsd_v6_public_kitti/bdd_proxy_val.yaml \
  --out-dir artifacts/pv_priority_eval_20260613 \
  --device 0
```

汇总：

| 数据集 | 候选 | P | R | mAP50 | PV Recall | PV mAP50 |
|---|---|---:|---:|---:|---:|---:|
| `coco_val` | v4 `s@640` | 0.811 | 0.565 | 0.651 | 0.608 | 0.706 |
| `coco_val` | v7 `n@640` | 0.743 | 0.517 | 0.589 | 0.589 | 0.653 |
| `coco_val` | v8 no-attn | 0.739 | 0.506 | 0.577 | 0.576 | 0.649 |
| `coco_val` | v9 ReLU from v8 | 0.733 | 0.479 | 0.548 | 0.547 | 0.627 |
| `bdd_proxy_val` | v4 `s@640` | 0.678 | 0.502 | 0.556 | 0.657 | 0.723 |
| `bdd_proxy_val` | v7 `n@640` | 0.682 | 0.615 | 0.674 | 0.786 | 0.808 |
| `bdd_proxy_val` | v8 no-attn | 0.744 | 0.589 | 0.680 | 0.754 | 0.813 |
| `bdd_proxy_val` | v9 ReLU from v8 | 0.753 | 0.577 | 0.673 | 0.741 | 0.798 |

BDD proxy 的 person/vehicle 细分：

| 候选 | Person R | Person mAP50 | Vehicle R | Vehicle mAP50 |
|---|---:|---:|---:|---:|
| v7 `n@640` | 0.731 | 0.746 | 0.841 | 0.869 |
| v8 no-attn | 0.693 | 0.751 | 0.814 | 0.876 |
| v9 ReLU from v8 | 0.676 | 0.730 | 0.806 | 0.867 |

判断：

- v9 的 Precision 可以接受，主要差距是 Recall。
- BSD 关键差距最大的是 person recall：v9 在 `bdd_proxy_val` 上比 v7 低 `0.055`。
- vehicle 接近 v7，但 recall 仍低约 `0.035`。
- `bicycle` 和 `motorcycle` 是辅助类别。只要 person/vehicle 稳定，它们不应主导模型选择。

## 数据路线

构建命令：

```bash
python deployment/dataset/build_bsd_v9p1_pv_priority.py \
  --base /root/autodl-tmp/BSD/datasets/bsd_v6_public_kitti \
  --output /root/autodl-tmp/BSD/datasets/bsd_v9p1_pv_priority \
  --coco-val /root/autodl-tmp/BSD/datasets/bsd/val/images \
  --link-mode hardlink
```

当前 base-only manifest：

| split | 图片数 | person 目标 | vehicle 目标 | bicycle 目标 | motorcycle 目标 |
|---|---:|---:|---:|---:|---:|
| train | 15,636 | 43,918 | 72,910 | 3,614 | 2,048 |
| bdd_proxy_val | 600 | 1,327 | 5,126 | 93 | 45 |

已做清理：

| split | 移除重复标签数 |
|---|---:|
| train | 261 |
| bdd_proxy_val | 23 |

base-only v9.1 数据集主要提供干净、可复现的起点。真正的精度提升应来自额外加入经过审核的 person/vehicle-heavy 道路场景数据，使用 `--extra-set name:/images:/labels` 接入。

额外数据优先级：

| 优先级 | 数据 | 原因 |
|---:|---|---|
| 1 | 道路场景 person，尤其是小目标、遮挡、侧后方、低照度 | 追回 person recall |
| 1 | 近距离/侧向/后方 vehicle，以及护栏、标志牌、杆体等 hard negative | 稳定 vehicle recall 和误检 |
| 2 | 能看到人的骑行者/电动车/摩托车图 | BSD 风险仍依赖人能否检出 |
| 3 | 纯 vehicle 图 | 控制比例，不能让 vehicle 主导数据集 |

## 训练配置

命令：

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority.yaml \
  --mode all \
  --device 0
```

配置文件：

```text
training/detection/configs/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority.yaml
```

这是从 v9 出发的保守微调：

| 项目 | 值 |
|---|---|
| epochs | 60 |
| batch | 48 |
| lr0 | 0.001 |
| mosaic | 0.20 |
| scale | 0.25 |
| erasing | 0.05 |

## 验收标准

PC 侧先过：

| 指标 | 目标 |
|---|---:|
| `bdd_proxy_val` PV Recall | 从 v9 `0.741` 提升，目标 `>=0.760`，拉伸目标 `>=0.786` |
| `bdd_proxy_val` person Recall | 从 v9 `0.676` 提升，目标 `>=0.700`，拉伸目标 `>=0.731` |
| `bdd_proxy_val` vehicle Recall | 保持 `>=0.806`，目标 `>=0.830` |
| `bdd_proxy_val` PV mAP50 | 接近 `0.798`，目标 `>=0.808` |
| `coco_val` mAP50 | 避免继续塌陷，目标 `>=0.548` |

PC 侧通过后再走板端：

```text
PT -> ONNX -> split raw-logit ONNX -> Pegasus NB -> bench_npu_loop -> live_bsd headless
```

不要覆盖 `/mnt/UDISK/bsd_v7_yolo26n_640.nb`，也不要覆盖当前 v9 板端 NB。

## 2026-06-13 结果

第一次数据清理后评估了两个 v9.1 微调版本：

| 候选 | 训练验证集 | `bdd_proxy_val` P | `bdd_proxy_val` R | `bdd_proxy_val` mAP50 | PV Recall | PV mAP50 |
|---|---|---:|---:|---:|---:|---:|
| v9 ReLU from v8 | 原路线 | `0.753` | `0.577` | `0.673` | `0.741` | `0.798` |
| v9.1 base | COCO early-stop，清洗 PV 数据集 | `0.720` | `0.602` | `0.677` | `0.760` | `0.797` |
| v9.1 bddval | BDD proxy early-stop | `0.744` | `0.593` | `0.680` | `0.743` | `0.798` |
| v7 fallback | 稳定参考 | `0.682` | `0.615` | `0.674` | `0.786` | `0.808` |

本阶段选择：**v9.1 base**。

原因：

- BSD 关键的 person/vehicle recall 从 v9 的 `0.741` 提升到 `0.760`。
- 仍保持可部署的 v9 结构：`YOLO26n_noattn + ReLU + 640`、raw-logit split ONNX、现有板端解码器，并且 NB 明显低于 `<5 MB`。
- bddval early-stop 版本整体 mAP50 略高，但 person/vehicle recall 回落到 `0.743`，不适合作为 BSD 首选。

当前上限判断：

- 仅用当前数据，距离 v7 仍有小但真实的差距：`PV Recall -0.026`，`PV mAP50 -0.011`。
- 继续调 early-stop 或学习率，大概率不会在无新数据条件下产生大幅收益。
- 下一步真正有效的精度工作是加数据：person-heavy 道路场景、侧后方 vehicle、以及杆体、标志牌、护栏、阴影、车形背景结构等 hard negative。
