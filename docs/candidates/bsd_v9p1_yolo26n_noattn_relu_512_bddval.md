# BSD v9.1 YOLO26n No-Attention ReLU 512 BDD-Val Candidate

## Status

Archived as a fallback performance candidate. Do not replace the current
`YOLO26n_noattn + ReLU + 640` main BSD model with this 512 model unless board
throughput becomes the primary constraint.

## Training

| Item | Value |
|---|---|
| Config | `training/detection/configs/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval.yaml` |
| Architecture | `YOLO26n_noattn + ReLU` |
| Input | `512x512` |
| Initial weights | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/weights/best.pt` |
| Dataset | `datasets/bsd_v9p1_pv_priority` |
| Early-stop validation | `bdd_proxy_val` |
| Best epoch | 22 |
| Stop | Early stop at epoch 47, patience 25 |

## Local Artifacts

| Artifact | Path |
|---|---|
| PT | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/weights/best.pt` |
| ONNX | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/weights/best.onnx` |
| Metrics | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/v5_eval_summary.json` |
| Training CSV | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/results.csv` |
| Frozen config copy | `artifacts/bsd_v9p1_yolo26n_noattn_relu_512_pv_priority_bddval/train_config.yaml` |

## Accuracy

| Model | Val set | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| v9.1 640 bddval | coco_val | 0.762 | 0.461 | 0.539 | 0.325 |
| v9.1 512 bddval | coco_val | 0.695 | 0.449 | 0.509 | 0.300 |
| v9.1 640 bddval | bdd_proxy_val | 0.744 | 0.593 | 0.680 | 0.456 |
| v9.1 512 bddval | bdd_proxy_val | 0.685 | 0.502 | 0.578 | 0.372 |

BDD proxy per-class result for the 512 model:

| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.677 | 0.585 | 0.646 | 0.379 |
| bicycle | 0.636 | 0.376 | 0.414 | 0.243 |
| motorcycle | 0.654 | 0.289 | 0.436 | 0.282 |
| vehicle | 0.771 | 0.757 | 0.817 | 0.582 |

## Board Performance Context

The earlier 512 board probe used the same fast graph shape but was exported
from the 640 v9.1 weights without 512 retraining. It measured:

| Input | FPS | Detect total | NV21 preprocess | NPU | Decode |
|---|---:|---:|---:|---:|---:|
| v9.1 640 | about 19.99 | 48.7-49.0 ms | 15.8-16.0 ms | 31.7-31.8 ms | about 1 ms |
| v9.1 512 probe | 19.99 | 32.1-32.3 ms | 10.5-10.8 ms | 20.8-21.0 ms | 0.7-0.8 ms |

The overall FPS stayed near 20 because the current serial loop is limited by
camera capture/wait timing, not by the 512 NPU time.

## Decision

Keep this candidate as a stored fallback. It is useful if the 640 route cannot
meet four-channel scheduling requirements, but the BDD proxy regression is too
large to promote it as the main model now:

- `bdd_proxy_val mAP50`: `0.680 -> 0.578`
- `bdd_proxy_val recall`: `0.593 -> 0.502`

If this candidate is needed later, the next steps are:

1. Split raw-logit ONNX with `deployment/quantize/split_yolo_head.py`.
2. Quantize with a 512 calibration set.
3. Test with `test_npu_direct`.
4. Deploy with `live_bsd ... headless 512`.
