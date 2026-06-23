# Training

This directory currently contains BSD training configuration files only.

| File | Purpose |
|---|---|
| `detection/configs/bsd_yolo26s.yaml` | Main BSD 4-class training config |
| `detection/configs/bsd_yolo26s_small.yaml` | Smaller/alternate BSD training config |

Planned next-step config:

| File | Purpose |
|---|---|
| `detection/configs/bsd_yolo26n.yaml` | Candidate smaller BSD model for board-side throughput comparison |

The training and export entry scripts referenced by `BSD_DEV_GUIDE.md` are not currently present in this local repository:

```text
training/detection/train.py
training/detection/export_onnx.py
```

They need to be restored from the training environment or recreated before end-to-end training can run from this checkout.
