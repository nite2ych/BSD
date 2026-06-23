# BSD Public Dataset Plan

This document fixes the public-data route for the BSD detector. The current
deployment candidate is `YOLO26n@640` because its quantized NB fits the hard
`<5 MB` gate while preserving better BSD proxy accuracy. The 5/20
`YOLO26s@640` model is kept as the fixed accuracy baseline.

## Acceptance Target

The target is not just a smaller or faster model. The target is a quantized
`YOLO26n` model whose accuracy is good enough for BSD deployment and whose
Pegasus `network_binary.nb` is smaller than 5 MB.

`YOLO26n@640` is the current main candidate. `YOLO26n@512` remains a fallback
only if board throughput forces another size reduction.

| Gate | Requirement |
|---|---|
| Overall accuracy | Close to the v4 `YOLO26s@640` baseline on `coco_val`; any mAP50 regression must be explained by better BSD-scene behavior |
| Road-scene accuracy | Match or exceed v4 on `bdd_proxy_val` |
| Person | No obvious recall collapse on close-range and medium-range people |
| Bicycle/motorcycle | No obvious two-wheel recall collapse; small targets must be checked visually |
| Vehicle | No recurring large false-positive boxes |
| Thresholds | Evaluated at `conf=0.5`, `nms=0.45`; do not lower thresholds to recover recall |
| Board readiness | Only after PC accuracy passes: export, split, quantize, then validate with `test_npu_direct` and `live_bsd headless` |

If the current `n@640` candidate regresses on these gates, the next action is
data/training improvement, not board optimization. Smaller inputs are only
throughput fallbacks.

## Fixed Baseline

| Item | Value |
|---|---|
| Baseline name | `bsd_v4_150ep` |
| Weight | `/root/autodl-tmp/BSD/artifacts/bsd_v4_150ep/weights/best.pt` |
| MD5 | `bfc40b3fbfd6001e83c7de680412587d` |
| Model | `YOLO26s@640` |
| Thresholds | `conf=0.5`, `nms=0.45` |
| Board path | `live_bsd headless`; preview is debug only |

Known PC metrics:

| Validation set | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `coco_val` | 0.818 | 0.557 | 0.652 | 0.425 |
| `bdd_proxy_val` | 0.678 | 0.502 | 0.556 | 0.336 |

Every new candidate must be compared against this table before any board-side
conversion work is treated as useful.

## Target Route

The current route is:

```text
public/reviewed data
  -> bsd_v6_public_kitti
  -> YOLO26n@640 stage1
  -> PT/ONNX accuracy check
  -> split raw-logit ONNX
  -> Pegasus NB
  -> test_npu_direct
  -> live_bsd headless
```

Do not change `DetResult` or `AlarmEvent` for dataset work. Do not lower
thresholds to hide training regressions.

## Dataset Priority

| Priority | Dataset | Main value | Use |
|---:|---|---|---|
| 1 | BDD100K official detection labels | Driving scenes; pedestrian, bicycle, motorcycle, car, bus, truck | Main public supplement |
| 2 | KITTI object detection | Clean road-scene labels; car, pedestrian, cyclist | Validation and small supplement |
| 2 | CityPersons | Higher quality pedestrian boxes on Cityscapes | Person recall supplement |
| 2 | WiderPerson | Dense/occluded people, small targets | Person hard-case supplement |
| 3 | MIO-TCD localization | Traffic-camera viewpoint; vehicle-heavy; includes bicycles/motorcycles/pedestrians | Vehicle/two-wheel supplement if license/download is clear |
| 3 | VisDrone DET | Small objects, crowded urban views | Hard-case supplement; use limited ratio because viewpoint differs |
| 3 | Open Images V7 subset | Easy class-subset download for person/bicycle/motorcycle/vehicle-like classes | Controlled extra data or hard negatives |
| 3 | nuImages | Autonomous-driving 2D boxes, non-commercial | Optional road-scene supplement after license check |

Reference links:

| Dataset | Official/reference URL |
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

## 2026-06-13 v9.2 Public Data Search

Current objective: improve v9.1 base recall for BSD-critical `person` and
`vehicle` without changing the selected deployable model shape
(`YOLO26n_noattn + ReLU + 640`). Board-side real capture is not available yet,
so public data must fill the gap temporarily.

Use the following source tiers:

| Tier | Dataset | Status | Expected BSD value | Action |
|---:|---|---|---|---|
| 1 | BDD100K Detection 2020 | Already has importer | Main road-scene base; person/rider/bike/motorcycle/vehicle | Keep as base, but audit labels and cap vehicle-only images |
| 1 | CityPersons | Needs Cityscapes images plus annotation import | Clean urban pedestrian/rider boxes | Add importer; train as person-heavy supplement |
| 1 | EuroCity Persons | License/application required | Large on-board urban person/cyclist/rider set, day/night diversity | Apply/download if license allows; add as person-heavy supplement |
| 2 | KITTI Object | Already has importer | Clean car/pedestrian/cyclist labels, small but trusted | Keep as calibration/validation-like supplement, not large weight |
| 2 | nuImages | Account/terms required | 93k 2D annotated autonomous-driving images | Add later if download is available; good road-scene domain match |
| 2 | Waymo Open Dataset 2D camera labels | Large TFRecord pipeline needed | Strong vehicle/pedestrian/cyclist labels; multi-city domain | Use only after lightweight sources; sample subset first |
| 3 | A2D2 | Very large download; mostly 3D/semantic sources | Side/rear multi-camera vehicle context | Optional; high conversion cost |
| 3 | Mapillary Vistas | Polygon segmentation, not box-native | Street-level hard negatives and person/vehicle diversity | Optional; polygon-to-box conversion and license review required |
| 3 | Objects365/Open Images subset | Generic web images | Extra hard negatives and uncommon vehicle/person cases | Use only curated subset; avoid domain drift |

Immediate recommendation:

1. Do **not** start from Waymo/A2D2 because they are heavy and conversion-heavy.
2. First build a small, clean `bsd_v9p2_public_expand` from:
   - existing BDD/KITTI base;
   - CityPersons, if Cityscapes access is available;
   - EuroCity Persons, if license/download approval is available;
   - a small reviewed hard-negative subset from Open Images or Objects365 only
     if false positives remain a problem.
3. Keep source share controlled. Do not let a generic dataset exceed the road
   scene base, and do not let vehicle-only images dominate the added data.

Target mix for the first v9.2 experiment:

| Added data bucket | Target images | Notes |
|---|---:|---|
| Road-scene person/rider | `3k-8k` | Small, occluded, side/rear, low-light are most valuable |
| Person + vehicle in same road scene | `2k-5k` | Best proxy for BSD interaction risk |
| Vehicle hard negatives / difficult vehicle | `1k-3k` | Guardrails, signs, poles, shadows, partial cars |
| Bicycle/motorcycle/rider | `1k-3k` | Keep, but selection still follows visible person quality |
| Empty/hard-negative frames | `5%-10%` of train | Only reviewed negatives; avoid random scenery |

Importer work needed:

| Importer | Priority | Output |
|---|---:|---|
| `import_citypersons.py` | 1 | BSD YOLO labels with `pedestrian/rider` mapped to `person`; sitting/other/group ignored by default |
| `import_eurocity_persons.py` | 1 | BSD YOLO labels for standardized EuroCity-style `pedestrian/rider/cyclist` JSON mapped to `person` |
| `import_nuimages.py` | 2 | BSD YOLO labels for pedestrian/bicycle/motorcycle/car/bus/truck/trailer-like classes |
| `import_waymo_2d.py` | 3 | Sampled camera frames only; map vehicle/pedestrian/cyclist and ignore signs |

CityPersons import command:

```bash
python deployment/dataset/import_citypersons.py \
  --citypersons-root /root/autodl-tmp/BSD/datasets/citypersons \
  --cityscapes-leftimg-root /root/autodl-tmp/BSD/datasets/cityscapes/leftImg8bit \
  --output /root/autodl-tmp/BSD/datasets/citypersons_bsd_person \
  --box-mode full \
  --link-mode hardlink \
  --overwrite
```

Then add it to v9.1/v9.2 dataset construction:

```bash
python deployment/dataset/build_bsd_v9p1_pv_priority.py \
  --base /root/autodl-tmp/BSD/datasets/bsd_v6_public_kitti \
  --output /root/autodl-tmp/BSD/datasets/bsd_v9p2_public_expand \
  --coco-val /root/autodl-tmp/BSD/datasets/bsd/val/images \
  --extra-set citypersons:/root/autodl-tmp/BSD/datasets/citypersons_bsd_person/train/images:/root/autodl-tmp/BSD/datasets/citypersons_bsd_person/train/labels \
  --link-mode hardlink
```

CityPersons is person-only after BSD mapping. It should be used as a recall
supplement, not as a new validation source. Keep `bdd_proxy_val` and later
`board_val` as the BSD-oriented selection gates.

EuroCity Persons standardized JSON import command:

```bash
python deployment/dataset/import_eurocity_persons.py \
  --ecp-root /root/autodl-tmp/BSD/datasets/eurocity_persons \
  --json-root /root/autodl-tmp/BSD/datasets/eurocity_persons_standard_json \
  --image-root /root/autodl-tmp/BSD/datasets/eurocity_persons/ECP/day/img \
  --output /root/autodl-tmp/BSD/datasets/eurocity_persons_bsd_person \
  --link-mode hardlink \
  --overwrite
```

Then add it to the v9.2 dataset together with CityPersons:

```bash
python deployment/dataset/build_bsd_v9p1_pv_priority.py \
  --base /root/autodl-tmp/BSD/datasets/bsd_v6_public_kitti \
  --output /root/autodl-tmp/BSD/datasets/bsd_v9p2_public_expand \
  --coco-val /root/autodl-tmp/BSD/datasets/bsd/val/images \
  --extra-set citypersons:/root/autodl-tmp/BSD/datasets/citypersons_bsd_person/train/images:/root/autodl-tmp/BSD/datasets/citypersons_bsd_person/train/labels \
  --extra-set eurocity:/root/autodl-tmp/BSD/datasets/eurocity_persons_bsd_person/train/images:/root/autodl-tmp/BSD/datasets/eurocity_persons_bsd_person/train/labels \
  --link-mode hardlink
```

The EuroCity importer currently targets standardized per-image JSON. If the
downloaded official package uses a different native JSON layout, extend
`import_eurocity_persons.py` in place rather than creating a second data path.

Execution order after downloads are available:

1. Place Cityscapes `leftImg8bit` under
   `/root/autodl-tmp/BSD/datasets/cityscapes/leftImg8bit`.
2. Place CityPersons `gtBboxCityPersons.mat` under
   `/root/autodl-tmp/BSD/datasets/citypersons`.
3. Run `import_citypersons.py` and inspect
   `citypersons_bsd_person/manifest.json`.
4. Place EuroCity images and standardized JSON under the paths shown above.
5. Run `import_eurocity_persons.py` and inspect
   `eurocity_persons_bsd_person/manifest.json`.
6. Build `bsd_v9p2_public_expand` with both `--extra-set` inputs.
7. Compare the new manifest against v9.1 base before training:
   - person objects should increase clearly;
   - vehicle-only source share should not dominate;
   - empty/hard-negative frames should stay controlled.
8. Fine-tune from v9.1 base and evaluate using
   `tools/evaluate_bsd_pv_priority.py`.

Selection rule for v9.2:

- Optimize for `bdd_proxy_val` `PV Recall` first, then `PV mAP50`.
- v9.1 base is the starting point and comparison target.
- v7 remains the stable fallback; new data must narrow the v9.1-v7 gap without
  increasing obvious large false positives.
- Any public-data improvement remains provisional until a real `board_val`
  exists.

## 2026-06-13 v9.2 Current-Data Sanity Run

Before CityPersons/EuroCity data became available, a current-data sanity run was
started to verify that the fastest deployable model shape still trains cleanly:

| Item | Value |
|---|---|
| Config | `training/detection/configs/bsd_v9p2_yolo26n_noattn_relu_640_currentdata.yaml` |
| Architecture | `YOLO26n_noattn + ReLU + 640` |
| Initial weights | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/weights/best.pt` |
| Dataset | `datasets/bsd_v9p1_pv_priority` |
| Server artifact | `artifacts/bsd_v9p2_yolo26n_noattn_relu_640_currentdata` |
| Early stop | epoch 21, best epoch 9 |

Independent PV evaluation:

| Dataset | Candidate | P | R | mAP50 | PV Recall | PV mAP50 |
|---|---|---:|---:|---:|---:|---:|
| `bdd_proxy_val` | v9 ReLU from v8 | `0.753` | `0.577` | `0.673` | `0.741` | `0.798` |
| `bdd_proxy_val` | v9.1 base | `0.720` | `0.602` | `0.677` | `0.760` | `0.797` |
| `bdd_proxy_val` | v9.2 currentdata | `0.750` | `0.578` | `0.668` | `0.739` | `0.795` |

Decision: do **not** select v9.2 currentdata. It proves the training/export path
is healthy but does not improve BSD-critical person/vehicle recall. Keep v9.1
base as the selected accuracy recovery candidate until real public-data
expansion is available.

## Class Mapping

BSD classes:

| ID | Name |
|---:|---|
| 0 | `person` |
| 1 | `bicycle` |
| 2 | `motorcycle` |
| 3 | `vehicle` |

Source mapping:

| Source label | BSD label | Notes |
|---|---|---|
| `person`, `pedestrian`, `adult`, `child`, `people` | `person` | `rider` is mapped to `person` unless a later importer can split rider and vehicle boxes cleanly |
| `bicycle`, `bike` | `bicycle` | Include e-bike if the source has that class |
| `motorcycle`, `motorbike`, `motor`, `scooter` | `motorcycle` | `tricycle` needs visual review before deciding motorcycle vs vehicle |
| `car`, `bus`, `truck`, `van`, `lorry`, `pickup` | `vehicle` | Keep train/trailer/caravan disabled by default unless the deployment scene needs them |

## Data Mix Rules

The goal is to improve `YOLO26n@640` precision without destroying road-scene
recall. Smaller input sizes are fallback options, not the main accuracy route.

| Rule | Default |
|---|---|
| Keep all high-quality `person`, `bicycle`, `motorcycle` samples | Yes |
| Downsample vehicle-only public images | Yes |
| Hard negative empty images | 5% to 12% of training images |
| Max single-source share after COCO/BDD base | 35% unless reviewed |
| Use validation data for training | No |
| Use old abnormal board raw frames for accuracy conclusion | No |
| Use board debug frames for quantization/link debug | Yes |

Before training a public-data candidate, write a `manifest.json` with:

- source name and version;
- source image count;
- copied/linked image count;
- empty-label count;
- per-class object count;
- filter parameters;
- train/val split policy.

## Current Deployed Candidate

| Item | Value |
|---|---|
| Candidate | `bsd_v7_yolo26n_640_public_kitti` |
| Model | `YOLO26n@640` |
| Dataset | `bsd_v6_public_kitti` |
| BDD proxy metrics | P `0.682`, R `0.615`, mAP50 `0.674`, mAP50-95 `0.467` |
| Quantized NB size | `3,696,064` bytes, about `3.52 MiB` |
| Board result | `headless` about 15 FPS; `preview` about 7.7 FPS |
| Detail record | `docs/candidates/bsd_v7_yolo26n_640_public_kitti.md` |

## Current Speed Experiment

The v7 candidate is frozen as the comparison baseline. The current speed
experiment keeps `640x640` and the YOLO26 raw-logit board decode path, but
removes C2PSA attention blocks from the model graph.

| Item | Value |
|---|---|
| Candidate | `bsd_v8_yolo26n_noattn_640_stage1` |
| Model YAML | `training/detection/models/yolo26n_noattn.yaml` |
| Detail record | `docs/candidates/bsd_v8_yolo26n_noattn_640.md` |
| Probe NB size | `1.69 MiB` |
| Probe board NPU | `62.8-62.9 ms` |

The speed probe used untrained/random weights and is not an accuracy result.
Train and evaluate v8 before using it for any detection comparison.

## Immediate Implementation Order

1. Import BDD100K official detection labels to BSD YOLO format.
2. Build `bsd_v6_public_precision` from `bsd_v5_balanced` plus reviewed BDD official data.
3. Compare class/object distribution against `bsd_v5_balanced`.
4. Train or fine-tune `YOLO26n@640` first; only try `512/416` if board throughput forces it.
5. Evaluate against `coco_val` and `bdd_proxy_val`, then generate fixed visual comparisons against v4.
6. Only if PC metrics and visual checks pass the acceptance target, export/quantize and test on board.

## Stop Conditions

Pause and inspect data if any candidate shows:

- `coco_val` precision drops more than 0.05 from v4 while BDD improves only slightly;
- `person` recall drops on both `coco_val` and `bdd_proxy_val`;
- vehicle-only images dominate the new dataset;
- visual comparison shows repeated large false-positive boxes;
- `n@640` loses the BDD proxy gains after adding new public data;
- a smaller fallback such as `n@512` trails v4 by a large margin after BDD official data is added.
