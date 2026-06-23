# BSD Detection Configs

The v5/v6/v7 configs define the no-new-board-capture training matrix. The
current deployed candidate is `bsd_v7_yolo26n_640_public_kitti_stage1.yaml`.

Build the balanced dataset first on the training server:

```bash
python deployment/dataset/build_bsd_v5_balanced.py
```

Candidate order:

| Config | Purpose |
|---|---|
| `bsd_v5_yolo26s_640.yaml` | Accuracy baseline against the 5/20 model |
| `bsd_v5_yolo26s_512.yaml` | Input-size reduction with the same model scale |
| `bsd_v5_yolo26n_640.yaml` | Small-model accuracy ceiling |
| `bsd_v5_yolo26n_512.yaml` | Fallback compressed candidate if `n@640` throughput is not enough |
| `bsd_v5_yolo26n_416.yaml` | Four-channel throughput candidate |
| `bsd_v7_yolo26n_640_public_kitti_stage1.yaml` | Current deployed `<5 MB` candidate |
| `bsd_v8_yolo26n_noattn_640_stage1.yaml` | 640 fixed speed candidate; removes C2PSA attention while keeping YOLO26 head |

Run one legacy matrix candidate:

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v5_yolo26n_512.yaml \
  --mode all \
  --device 0
```

Training requires GPU. Keep `conf=0.5` and `nms=0.45` for downstream visual checks and board validation so precision changes are not mixed with threshold changes.

Current v7 board candidate:

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v7_yolo26n_640_public_kitti_stage1.yaml \
  --mode export \
  --weights /root/autodl-tmp/BSD/artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/weights/best.pt \
  --device cpu
```

Current v8 speed candidate:

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v8_yolo26n_noattn_640_stage1.yaml \
  --mode all \
  --device 0
```

The v8 model definition is `training/detection/models/yolo26n_noattn.yaml`.
It is intentionally separate from the installed Ultralytics `yolo26.yaml` so
the frozen v7 baseline remains reproducible.
