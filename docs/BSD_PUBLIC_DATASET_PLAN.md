# BSD 公开数据集计划

本文档固定 BSD 检测器的公开数据路线。当前主线候选是 `YOLO26n_noattn + ReLU + 640` 的 v9.1，因为量化 NB 满足硬性 `<5 MB` 门槛，同时在 BSD proxy 上有更合理的 person/vehicle recall。5/20 的 `YOLO26s@640` 保留为固定精度基线，v7 保留为稳定回退。

## 验收目标

目标不是单纯更小或更快，而是训练出一个精度足够支撑 BSD 部署、且 Pegasus `network_binary.nb` 小于 5 MB 的量化模型。

当前主线是 `YOLO26n_noattn + ReLU + 640`。`YOLO26n@512` 仅在板端吞吐强制要求继续降尺寸时作为兜底。

| 门槛 | 要求 |
|---|---|
| 整体精度 | 在 `coco_val` 上接近 v4 `YOLO26s@640` 基线；若 mAP50 下降，必须能用更好的 BSD 场景表现解释 |
| 道路场景精度 | 在 `bdd_proxy_val` 上匹配或超过 v4 |
| person | 近距离和中距离 person 不能明显召回塌陷 |
| bicycle/motorcycle | 两轮车召回不能明显塌陷；小目标必须做可视化检查 |
| vehicle | 不能反复出现大框误检 |
| 阈值 | 固定 `conf=0.5`, `nms=0.45` 评估；不要靠降低阈值追回 recall |
| 板端就绪 | PC 精度通过后，才进行 export、split、quantize，并用 `test_npu_direct` 和 `live_bsd headless` 验证 |

如果当前候选过不了这些门槛，下一步应做数据/训练改进，而不是先优化板端。更小输入只作为吞吐兜底。

## 固定基线

| 项目 | 值 |
|---|---|
| 基线名 | `bsd_v4_150ep` |
| 权重 | `/root/autodl-tmp/BSD/artifacts/bsd_v4_150ep/weights/best.pt` |
| MD5 | `bfc40b3fbfd6001e83c7de680412587d` |
| 模型 | `YOLO26s@640` |
| 阈值 | `conf=0.5`, `nms=0.45` |
| 板端路径 | `live_bsd headless`；preview 只用于调试 |

已知 PC 指标：

| 验证集 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `coco_val` | 0.818 | 0.557 | 0.652 | 0.425 |
| `bdd_proxy_val` | 0.678 | 0.502 | 0.556 | 0.336 |

任何新候选在进入板端转换前，都必须先和这张表对比。

## 目标路线

当前路线：

```text
public/reviewed data
  -> bsd_v6_public_kitti / bsd_v9p1_pv_priority
  -> YOLO26n_noattn + ReLU + 640
  -> PT/ONNX accuracy check
  -> split raw-logit ONNX
  -> Pegasus NB
  -> test_npu_direct
  -> live_bsd headless
```

数据工作不改 `DetResult` 或 `AlarmEvent`。不要用降低阈值掩盖训练回退。

## 数据集优先级

| 优先级 | 数据集 | 主要价值 | 用法 |
|---:|---|---|---|
| 1 | BDD100K official detection labels | 驾驶场景；pedestrian、bicycle、motorcycle、car、bus、truck | 主要公开补充 |
| 2 | KITTI object detection | 干净道路标签；car、pedestrian、cyclist | 验证和小比例补充 |
| 2 | CityPersons | Cityscapes 上更高质量 pedestrian 框 | person recall 补充 |
| 2 | WiderPerson | 密集/遮挡人群、小目标 | person hard-case 补充 |
| 3 | MIO-TCD localization | 交通摄像头视角；vehicle-heavy；含 bicycle/motorcycle/pedestrian | license/download 明确后作为 vehicle/two-wheel 补充 |
| 3 | VisDrone DET | 小目标、拥挤城市视角 | hard-case 补充；视角不同，比例要低 |
| 3 | Open Images V7 subset | person/bicycle/motorcycle/vehicle-like 类别易下载 | 受控额外数据或 hard negative |
| 3 | nuImages | 自动驾驶 2D 框，非商用 | license 通过后可选道路场景补充 |

参考链接：

| 数据集 | 官方/参考 URL |
|---|---|
| BDD100K | https://bair.berkeley.edu/blog/2018/05/30/bdd/ |
| BDD100K download layout note | https://docs.voxel51.com/dataset_zoo/datasets/bdd100k.html |
| KITTI | https://www.cvlibs.net/datasets/kitti/ |
| KITTI object detection | https://www.cvlibs.net/datasets/kitti/eval_3dobject.php |
| CityPersons annotations | https://github.com/cvgroup-njust/CityPersons |
| Cityscapes images for CityPersons | https://www.cityscapes-dataset.com/ |
| CityPersons paper | https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf |
| EuroCity Persons | https://eurocity-dataset.tudelft.nl/ |
| EuroCity Persons license | https://eurocity-dataset.tudelft.nl/eval/license/ecplicense |
| WiderPerson | https://fmi-data-index.github.io/wider_person.html |
| MIO-TCD | https://tcd.miovision.com/challenge/dataset.html |
| VisDrone | https://github.com/VisDrone/VisDrone-Dataset |
| Open Images V7 | https://storage.googleapis.com/openimages/web/download_v7.html |
| nuImages | https://www.nuscenes.org/nuimages |
| Waymo Open Dataset | https://waymo.com/open/ |
| Audi A2D2 | https://www.a2d2.audi/a2d2/ |
| Mapillary Vistas | https://www.mapillary.com/dataset/vistas |

## 2026-06-13 v9.2 公开数据搜索

当前目标：在不改变可部署模型形态（`YOLO26n_noattn + ReLU + 640`）的情况下，提高 v9.1 base 对 BSD 关键 `person` 和 `vehicle` 的 recall。当前还没有板端真实采集条件，所以公开数据只能暂时补空。

数据源分层：

| 层级 | 数据集 | 状态 | 预期 BSD 价值 | 动作 |
|---:|---|---|---|---|
| 1 | BDD100K Detection 2020 | 已有 importer | 主要道路场景基座；person/rider/bike/motorcycle/vehicle | 作为 base 保留，但审核标签并限制纯 vehicle 图 |
| 1 | CityPersons | 需要 Cityscapes 图片和 annotation import | 干净城市 pedestrian/rider 框 | 增加 importer，作为 person-heavy 补充训练 |
| 1 | EuroCity Persons | 需要 license/application | 大规模 onboard 城市 person/cyclist/rider，含 day/night | license 允许后下载，作为 person-heavy 补充 |
| 2 | KITTI Object | 已有 importer | 干净 car/pedestrian/cyclist 标签，规模小但可信 | 保留为校准/验证风格补充，不加大权重 |
| 2 | nuImages | 需要账号/条款 | 93k 2D 自动驾驶标注图 | 可下载后再加；道路域匹配好 |
| 2 | Waymo Open Dataset 2D camera labels | 需要大型 TFRecord 流程 | 多城市 vehicle/pedestrian/cyclist 标签强 | 轻量来源之后再用；先抽样子集 |
| 3 | A2D2 | 下载很大；偏 3D/semantic | 侧后多摄 vehicle 场景 | 可选；转换成本高 |
| 3 | Mapillary Vistas | polygon segmentation，非 box-native | street-level hard negatives 和 person/vehicle 多样性 | 可选；需要 polygon-to-box 和 license review |
| 3 | Objects365/Open Images subset | 通用网络图 | 额外 hard negative 和少见 person/vehicle 案例 | 只用精选子集，避免域漂移 |

即时建议：

1. 不从 Waymo/A2D2 开始，因为下载和转换成本都高。
2. 先构建一个小而干净的 `bsd_v9p2_public_expand`：
   - 现有 BDD/KITTI base；
   - 如果 Cityscapes 可访问，加入 CityPersons；
   - 如果 EuroCity Persons license/download 通过，加入 EuroCity Persons；
   - 若误检仍严重，再加入小规模人工审核 Open Images 或 Objects365 hard-negative 子集。
3. 控制来源比例。通用数据集不能超过道路场景 base，纯 vehicle 图不能主导新增数据。

第一版 v9.2 目标混合：

| 新增数据桶 | 目标图片数 | 备注 |
|---|---:|---|
| 道路 person/rider | `3k-8k` | 小目标、遮挡、侧后方、低照度最有价值 |
| person + vehicle 同场景 | `2k-5k` | 最接近 BSD 交互风险 |
| vehicle hard negatives / difficult vehicle | `1k-3k` | 护栏、标志牌、杆体、阴影、局部车身 |
| bicycle/motorcycle/rider | `1k-3k` | 保留，但选择仍优先看可见 person 质量 |
| empty/hard-negative frames | train 的 `5%-10%` | 只用审核过的负样本，避免随机风景 |

需要的 importer：

| importer | 优先级 | 输出 |
|---|---:|---|
| `import_citypersons.py` | 1 | BSD YOLO 标签，`pedestrian/rider` 映射到 `person`；默认忽略 sitting/other/group |
| `import_eurocity_persons.py` | 1 | 标准 EuroCity 风格 `pedestrian/rider/cyclist` JSON 映射到 `person` |
| `import_nuimages.py` | 2 | pedestrian/bicycle/motorcycle/car/bus/truck/trailer-like 类别转 BSD YOLO |
| `import_waymo_2d.py` | 3 | 只采样 camera frames；映射 vehicle/pedestrian/cyclist，忽略 signs |

CityPersons 导入命令：

```bash
python deployment/dataset/import_citypersons.py \
  --citypersons-root /root/autodl-tmp/BSD/datasets/citypersons \
  --cityscapes-leftimg-root /root/autodl-tmp/BSD/datasets/cityscapes/leftImg8bit \
  --output /root/autodl-tmp/BSD/datasets/citypersons_bsd_person \
  --box-mode full \
  --link-mode hardlink \
  --overwrite
```

再加入 v9.1/v9.2 数据构建：

```bash
python deployment/dataset/build_bsd_v9p1_pv_priority.py \
  --base /root/autodl-tmp/BSD/datasets/bsd_v6_public_kitti \
  --output /root/autodl-tmp/BSD/datasets/bsd_v9p2_public_expand \
  --coco-val /root/autodl-tmp/BSD/datasets/bsd/val/images \
  --extra-set citypersons:/root/autodl-tmp/BSD/datasets/citypersons_bsd_person/train/images:/root/autodl-tmp/BSD/datasets/citypersons_bsd_person/train/labels \
  --link-mode hardlink
```

CityPersons 按 BSD 映射后是 person-only。它应作为 recall 补充，而不是新的验证源。仍以 `bdd_proxy_val` 和未来 `board_val` 作为 BSD 选择门槛。

EuroCity Persons 标准 JSON 导入命令：

```bash
python deployment/dataset/import_eurocity_persons.py \
  --ecp-root /root/autodl-tmp/BSD/datasets/eurocity_persons \
  --json-root /root/autodl-tmp/BSD/datasets/eurocity_persons_standard_json \
  --image-root /root/autodl-tmp/BSD/datasets/eurocity_persons/ECP/day/img \
  --output /root/autodl-tmp/BSD/datasets/eurocity_persons_bsd_person \
  --link-mode hardlink \
  --overwrite
```

再与 CityPersons 一起加入 v9.2：

```bash
python deployment/dataset/build_bsd_v9p1_pv_priority.py \
  --base /root/autodl-tmp/BSD/datasets/bsd_v6_public_kitti \
  --output /root/autodl-tmp/BSD/datasets/bsd_v9p2_public_expand \
  --coco-val /root/autodl-tmp/BSD/datasets/bsd/val/images \
  --extra-set citypersons:/root/autodl-tmp/BSD/datasets/citypersons_bsd_person/train/images:/root/autodl-tmp/BSD/datasets/citypersons_bsd_person/train/labels \
  --extra-set eurocity:/root/autodl-tmp/BSD/datasets/eurocity_persons_bsd_person/train/images:/root/autodl-tmp/BSD/datasets/eurocity_persons_bsd_person/train/labels \
  --link-mode hardlink
```

EuroCity importer 当前面向标准 per-image JSON。如果下载到的官方包使用不同原始 JSON 布局，应在 `import_eurocity_persons.py` 内扩展，而不是新开第二条数据路径。

下载可用后的执行顺序：

1. 将 Cityscapes `leftImg8bit` 放到 `/root/autodl-tmp/BSD/datasets/cityscapes/leftImg8bit`。
2. 将 CityPersons `gtBboxCityPersons.mat` 放到 `/root/autodl-tmp/BSD/datasets/citypersons`。
3. 运行 `import_citypersons.py`，检查 `citypersons_bsd_person/manifest.json`。
4. 按上述路径放置 EuroCity 图片和标准 JSON。
5. 运行 `import_eurocity_persons.py`，检查 `eurocity_persons_bsd_person/manifest.json`。
6. 使用两个 `--extra-set` 构建 `bsd_v9p2_public_expand`。
7. 训练前对比新 manifest 和 v9.1 base：
   - person objects 应明显增加；
   - vehicle-only 来源比例不能主导；
   - empty/hard-negative frames 比例受控。
8. 从 v9.1 base 微调，并用 `tools/evaluate_bsd_pv_priority.py` 评估。

v9.2 选择规则：

- 优先优化 `bdd_proxy_val` 的 `PV Recall`，其次看 `PV mAP50`。
- v9.1 base 是起点和对比目标。
- v7 仍是稳定回退；新数据必须缩小 v9.1 与 v7 的差距，同时不能增加明显大框误检。
- 所有公开数据改进在真实 `board_val` 出现前都只是暂定结论。

## 2026-06-13 v9.2 当前数据 Sanity Run

在 CityPersons/EuroCity 可用前，先跑了 current-data sanity run，用于确认最快可部署模型形态仍能正常训练：

| 项目 | 值 |
|---|---|
| 配置 | `training/detection/configs/bsd_v9p2_yolo26n_noattn_relu_640_currentdata.yaml` |
| 架构 | `YOLO26n_noattn + ReLU + 640` |
| 初始权重 | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/weights/best.pt` |
| 数据集 | `datasets/bsd_v9p1_pv_priority` |
| 服务器产物 | `artifacts/bsd_v9p2_yolo26n_noattn_relu_640_currentdata` |
| Early stop | epoch 21，best epoch 9 |

独立 PV 评估：

| 数据集 | 候选 | P | R | mAP50 | PV Recall | PV mAP50 |
|---|---|---:|---:|---:|---:|---:|
| `bdd_proxy_val` | v9 ReLU from v8 | `0.753` | `0.577` | `0.673` | `0.741` | `0.798` |
| `bdd_proxy_val` | v9.1 base | `0.720` | `0.602` | `0.677` | `0.760` | `0.797` |
| `bdd_proxy_val` | v9.2 currentdata | `0.750` | `0.578` | `0.668` | `0.739` | `0.795` |

结论：**不选择 v9.2 currentdata**。它证明训练/导出路径正常，但没有提升 BSD 关键 person/vehicle recall。保留 v9.1 base 作为当前精度恢复候选，直到真正的公开数据扩充可用。

## 类别映射

BSD 类别：

| ID | 名称 |
|---:|---|
| 0 | `person` |
| 1 | `bicycle` |
| 2 | `motorcycle` |
| 3 | `vehicle` |

来源映射：

| 来源标签 | BSD 标签 | 说明 |
|---|---|---|
| `person`, `pedestrian`, `adult`, `child`, `people` | `person` | `rider` 默认映射为 `person`，除非后续 importer 能干净拆出 rider 和 vehicle 框 |
| `bicycle`, `bike` | `bicycle` | 源数据有 e-bike 时纳入 |
| `motorcycle`, `motorbike`, `motor`, `scooter` | `motorcycle` | `tricycle` 需要视觉复核后再定 motorcycle vs vehicle |
| `car`, `bus`, `truck`, `van`, `lorry`, `pickup` | `vehicle` | 默认不启用 train/trailer/caravan，除非部署场景需要 |

## 数据混合规则

目标是提升 `YOLO26n@640` 精度，同时不破坏道路场景 recall。更小输入是兜底，不是主精度路线。

| 规则 | 默认 |
|---|---|
| 保留全部高质量 `person`, `bicycle`, `motorcycle` 样本 | 是 |
| 对纯 vehicle 公开图降采样 | 是 |
| hard negative empty images | 训练图的 5% 到 12% |
| COCO/BDD base 之后单一新增来源最大占比 | 35%，除非人工复核 |
| 使用验证集训练 | 否 |
| 用旧异常板端 raw 帧做精度结论 | 否 |
| 用 board debug frames 做量化/链路调试 | 是 |

训练公开数据候选前，必须写入 `manifest.json`：

- 来源名称和版本；
- 来源图片数；
- 复制/硬链接图片数；
- 空标签数；
- 逐类目标数；
- 过滤参数；
- train/val 划分策略。

## 当前部署候选

| 项目 | 值 |
|---|---|
| 候选 | `bsd_v9p1_yolo26n_noattn_relu_640_pv_priority` |
| 模型 | `YOLO26n_noattn + ReLU + 640` |
| 数据集 | `bsd_v9p1_pv_priority` |
| BDD proxy 指标 | P `0.720`, R `0.602`, mAP50 `0.677`, PV Recall `0.760`, PV mAP50 `0.797` |
| 量化 NB 大小 | `2,111,168` bytes，约 `2.01 MiB` |
| 板端结果 | headless detect call 约 `48.7-49.0 ms`，NPU 约 `31.7-31.8 ms`；preview 约 `9.4-9.5 FPS` |
| 详细记录 | `docs/candidates/bsd_v9p1_pv_priority_plan.md` |

## 当前速度背景

v7 保留为稳定回退。v8 证明了移除 attention 能显著改善 VIPLite 图；v9/v9.1 进一步把激活函数换成 ReLU，并把 NPU 耗时压到约 `31-32 ms`。

| 候选 | 作用 | 板端 NPU | 说明 |
|---|---|---:|---|
| v7 `YOLO26n@640` | 稳定回退 | 约 `92.1 ms` | recall 强，但速度慢 |
| v8 no-attn SiLU | attention 移除验证 | 约 `67.4 ms` | 速度提升，recall 略降 |
| v9 ReLU from v8 | ReLU 速度验证 | 约 `31.3 ms` | 速度最佳，PV Recall `0.741` |
| v9.1 base | 当前优先候选 | 约 `31.7-31.8 ms` | PV Recall 提升到 `0.760` |

preview 慢主要是 framebuffer 转换、旋转/缩放、画框和 page flip，不代表 NPU 慢。真实性能判断以 `headless` 为准。

## 即时实施顺序

1. 保持 v9.1 base 为当前板端候选，v7 为稳定回退。
2. 搜罗并下载 CityPersons/EuroCity 等公开 person-heavy 道路数据。
3. 先做 importer 和 manifest，确认 person/vehicle 分布变好。
4. 构建 `bsd_v9p2_public_expand`，从 v9.1 base 微调。
5. 在 `coco_val`、`bdd_proxy_val` 和固定可视化对比上验证。
6. 只有 PC 指标和可视化通过，才 export/quantize 并上板 `live_bsd headless`。

## 停止条件

候选出现以下情况时暂停并检查数据：

- `coco_val` precision 比 v4 下降超过 0.05，而 BDD 提升很小；
- `person` recall 在 `coco_val` 和 `bdd_proxy_val` 上都下降；
- 纯 vehicle 图主导新增数据；
- 可视化对比出现反复大框误检；
- 新公开数据加入后 `n@640` 丢失 BDD proxy 收益；
- 512 等更小兜底模型在加入 BDD official 后仍明显落后 v4。
