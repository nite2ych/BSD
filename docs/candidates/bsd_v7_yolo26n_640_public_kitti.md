# BSD v7 YOLO26n@640 Public KITTI Candidate

This is the current deployed BSD candidate.

## Summary

| Item | Value |
|---|---|
| Candidate | `bsd_v7_yolo26n_640_public_kitti` |
| Model | `YOLO26n@640` |
| Training artifact | `/root/autodl-tmp/BSD/artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/weights/best.pt` |
| Local artifact dir | `artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/` |
| Board NB | `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |
| Thresholds | `conf=0.5`, `nms=0.45`, `disp_conf=0.5`, `person_conf=0.5` |
| Board mode | `headless` for production, `preview` only for debug |
| ABI | Keep `DetResult` / `AlarmEvent` unchanged |

## Accuracy

Fixed v4 baseline: `bsd_v4_150ep`, `YOLO26s@640`.

| Validation set | Model | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `coco_val` | v4 `s@640` | 0.818 | 0.557 | 0.652 | 0.425 |
| `coco_val` | v7 `n@640` | 0.741 | 0.521 | 0.586 | 0.364 |
| `bdd_proxy_val` | v4 `s@640` | 0.678 | 0.502 | 0.556 | 0.336 |
| `bdd_proxy_val` | v7 `n@640` | 0.682 | 0.615 | 0.674 | 0.467 |

BSD ranking should use `bdd_proxy_val` first because it better matches the road-scene deployment domain. `coco_val` remains a generalization check.

Per-class `bdd_proxy_val` comparison:

| Class | v4 R | v7 R | v4 mAP50 | v7 mAP50 |
|---|---:|---:|---:|---:|
| `person` | 0.622 | 0.731 | 0.686 | 0.746 |
| `bicycle` | 0.293 | 0.419 | 0.347 | 0.535 |
| `motorcycle` | 0.400 | 0.467 | 0.429 | 0.544 |
| `vehicle` | 0.692 | 0.841 | 0.761 | 0.869 |

## Quantization

| Artifact | Value |
|---|---|
| Split ONNX | `artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx` |
| NB | `artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/bsd_v7_yolo26n_640_public_kitti.nb` |
| NB size | `3,696,064` bytes, about `3.52 MiB` |
| NB MD5 | `26c358cf22e9ba05f64e3c6e42c5ccaf` |
| NB magic | `VPMN` |
| Hard size gate | Passes `<5 MB` |

Standard split:

```bash
python deployment/quantize/split_yolo_head.py \
  --input artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best.onnx \
  --output artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx
```

Standard Pegasus export:

```bash
python tools/quantize_bsd_candidate.py \
  --host 192.168.144.133 \
  --user ubuntu \
  --password "$BSD_QUANT_PASSWORD" \
  --remote-dir /home/ubuntu/bsd_quantize \
  --onnx artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx \
  --calib-dir artifacts/bdd100k/calib_images \
  --model-name bsd_v7_yolo26n_640_public_kitti \
  --model-size 640
```

The board decoder expects raw logits. Do not deploy a sigmoid-output NB unless `yolo_decode.c` is changed accordingly.

## Board Deployment

Deploy via serial + temporary HTTP server:

```bash
python deployment/board/deploy_live_bsd_serial.py \
  --serial COM5 \
  --host-ip 192.168.144.100 \
  --port 8899 \
  --nb artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/bsd_v7_yolo26n_640_public_kitti.nb \
  --remote-nb bsd_v7_yolo26n_640.nb \
  --live-bsd artifacts/board_bin/live_bsd \
  --test-npu-direct artifacts/board_bin/test_npu_direct \
  --install-autostart \
  --run-mode preview
```

Production command:

```sh
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v7_yolo26n_640.nb 0.5 0.45 0.5 0.5 1280 720 headless 640
```

Debug preview command:

```sh
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v7_yolo26n_640.nb 0.5 0.45 0.5 0.5 1280 720 preview 640
```

Boot autostart:

```sh
cp deployment/board/init.d/bsd_live /etc/init.d/bsd_live
chmod +x /etc/init.d/bsd_live
/etc/init.d/bsd_live enable
ln -sf bsd_live /etc/init.d/S99bsd_live
```

This Tina image runs `/etc/init.d/S??*` from `rc.final`, so the `S99bsd_live`
symlink is required in addition to the OpenWrt-style `/etc/rc.d/S99bsd_live`
link created by `enable`.

The autostart service runs `headless` and writes logs to `/mnt/UDISK/bsd_live_autostart.log`.

## Board Results

Measured on V853 with `1280x720` NV21M camera input:

| Mode | FPS | Detection total | NV21 preprocess | NPU | Decode | Preview |
|---|---:|---:|---:|---:|---:|---:|
| `headless` | about 15.0 | about 109.5 ms | about 16.0 ms | about 92.4 ms | about 1.2 ms | 0 ms |
| `preview` | about 7.7 | about 109.5 ms | about 15.9 ms | about 92.4 ms | about 1.2 ms | about 93.6 ms |

`preview` is slower because it adds framebuffer conversion, rotation/scale, box drawing, and page flip. It does not mean NPU inference is slower.

## 640 Fixed Speed Experiments

The current optimization pass keeps the model input fixed at `640x640` and the
hard deployment gate fixed at NB `<5 MB`.

| Variant | Local artifact | NB size | Board NPU | Decision |
|---|---|---:|---:|---|
| Baseline | `bsd_v7_yolo26n_640_public_kitti.nb` | `3.52 MiB` | `92.1-92.4 ms` | Current stable model |
| ONNX simplified | `bsd_v7_yolo26n_640_public_kitti_sim.nb` | `3.52 MiB` | `92.1-92.2 ms` | No speed gain |
| Pegasus `--force-remove-permute` | `bsd_v7_yolo26n_640_public_kitti_forceperm.nb` | `3.52 MiB` | `92.1-92.2 ms` | No speed gain |
| Pegasus `--dtype quantized_strict` | `bsd_v7_yolo26n_640_public_kitti_qstrict.nb` | `20.56 MiB` | Not tested | Reject, fails `<5 MB` |
| Pegasus `perchannel_symmetric_affine int8` | `bsd_v7_yolo26n_640_public_kitti_perch_int8.nb` | `4.37 MiB` | Not tested | Size passes, simulator slower |

The split ONNX simplification reduced nodes only from `375` to `357`; the
heavy operator set did not change:

```text
Conv 102
Mul/Sigmoid(Swish) 89/87
MatMul 4
Softmax 2
Transpose/Permute 4
Resize 2
```

Pegasus fused graph keeps the same non-conv operators:

```text
CONV2D 102, SWISH 87, MATRIXMUL 4, SOFTMAX 2, PERMUTE 4, RESIZE 2
```

Conclusion: the current `YOLO26n@640` is already small enough, but its graph is
not NPU-friendly enough to reach a `~50 ms` target through normal ONNX
simplification or Pegasus export flags. The next speed-focused model experiment
should keep `640x640` but replace the attention-style blocks with a mostly
Conv/BN/activation nano architecture, then rerun the same accuracy, NB size, and
board NPU gates.

## Remaining Gate

This candidate is deployable for flow testing and current BSD proxy validation, but final release still needs a formal board-side validation set when capture conditions are available.
