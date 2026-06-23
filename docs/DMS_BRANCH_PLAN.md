# DMS/Cockpit Detection Branch Plan

This document defines how to reuse the BSD detector work for a new cockpit/DMS
branch. The goal is to preserve the proven V853 deployment pipeline while
changing only the product-specific data, labels, post-processing, and alarm
logic.

## Goal

Create a new branch for cockpit/DMS detection based on the BSD pipeline:

```text
dataset -> YOLO26 candidate -> ONNX export -> raw-logit split ONNX
  -> Pegasus NB -> V853 live video -> temporal post-processing -> event output
```

The first DMS milestone should be a running single-camera cockpit detector on
the same board path, not a full driver-monitoring product. Advanced behavior
analysis can be layered after the detection path is stable.

## Current BSD Baseline Snapshot

As of 2026-06-17, the BSD board path has moved from the older v7/v9 comparison
state to a selected v9.1 base candidate:

| Item | Current value |
|---|---|
| Selected BSD candidate | `bsd_v9p1_yolo26n_noattn_relu_640_pv_priority` |
| Architecture | `YOLO26n_noattn + ReLU + 640` |
| PC selection reason | Better BSD-critical person/vehicle recall than v9 |
| Local PT | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/weights/best.pt` |
| Local split ONNX | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/best_640_split_nosig.onnx` |
| Local NB | `artifacts/bsd_v9p1_yolo26n_noattn_relu_640_pv_priority/bsd_v9p1_pv_base_640.nb` |
| Board NB | `/mnt/UDISK/bsd_v9p1_pv_base_640.nb` |
| NB size | `2,111,168 bytes`, below the hard `<5 MB` gate |
| Board runtime | `/mnt/UDISK/live_bsd` with v9.1 NB, `preview` and `headless` modes |

The key v9 to v9.1 comparison on `bdd_proxy_val`:

| Candidate | P | R | mAP50 | PV Recall | PV mAP50 |
|---|---:|---:|---:|---:|---:|
| v9 ReLU from v8 | `0.753` | `0.577` | `0.673` | `0.741` | `0.798` |
| v9.1 base | `0.720` | `0.602` | `0.677` | `0.760` | `0.797` |

The v9.1 base candidate trades some precision for better recall, which matches
the BSD priority of missing fewer people and vehicles. v7 remains the stable
fallback and comparison anchor, but v9.1 is the current deployable BSD
candidate.

## Recommended Branch

Use a separate branch so BSD model history and board defaults remain stable:

```bash
git switch -c codex/dms-cockpit-detector
```

Keep BSD board models and startup scripts untouched until the DMS branch has its
own deployment filenames.

## Reusable BSD Assets

| BSD asset | Reuse in DMS | Notes |
|---|---|---|
| `training/detection/run_bsd_candidate.py` | Yes, rename/generalize later | Already handles train/eval/export and activation override |
| `training/detection/models/yolo26n_noattn.yaml` | Yes | Fastest deployable shape proven on V853 |
| `deployment/quantize/split_yolo_head.py` | Yes | Keep raw-logit `boxes/scores` output |
| Pegasus quantization flow | Yes | Same `<5 MB` NB concern applies |
| `deployment/board/detect_engine` | Mostly yes | Rename only when DMS ABI is defined |
| `live_bsd` camera/NPU loop | Yes as template | Split product-specific overlay/alarm from camera/NPU loop |
| headless/preview modes | Yes | Production remains headless; preview debug only |
| profiling fields | Yes | Keep `capture/preprocess/npu/decode/postprocess/display` timing |
| display-frame cleanup fix | Yes | Preview overlays must not hold stale boxes or leave framebuffer residue |

## Product Differences

BSD is road-scene object detection. DMS/cockpit is cabin-scene occupant/behavior
detection. Do not reuse BSD classes or alarm rules directly.

Candidate DMS class sets:

| Version | Classes | Use |
|---|---|---|
| dms_v1_min | `driver`, `face`, `phone`, `smoking`, `seatbelt` | Practical first detector |
| dms_v1_occ | `person`, `face`, `hand`, `phone`, `seatbelt` | Better if downstream behavior logic uses body parts |
| dms_v2_behavior | `eyes_closed`, `yawn`, `head_down`, `phone`, `smoking`, `seatbelt` | Requires stronger labels and temporal logic |

Recommended first class set:

```text
driver/person, face, phone, smoking, seatbelt
```

Reason:

- `driver/person` provides occupancy and ROI anchor.
- `face` enables later gaze/fatigue models.
- `phone/smoking/seatbelt` are common DMS product events.
- Avoid starting with fine-grained eye/yawn labels unless the dataset quality is
  already known.

## Model Recommendation

Start with the fastest BSD-proven structure:

```text
YOLO26n_noattn + ReLU + 640
```

Rationale:

- NB size should remain comfortably under `5 MB`.
- Board NPU behavior is already measured: v9/v9.1 route is around `31-32 ms`
  NPU time and around `48-49 ms` total detect call time on one 1280x720 stream.
- It avoids attention and SiLU/Swish operators that were slow on VIPLite.
- DMS usually has fewer classes and a tighter ROI, so this model should be a
  good first deployment candidate.
- The newest BSD v9.1 NB is about `2.1 MB`, leaving useful room under the hard
  `<5 MB` deployment limit.

Do not start from BSD traffic weights for DMS unless no better pretrained weight
is available. Prefer generic YOLO26n pretrained weights or a DMS/cabin
checkpoint. BSD weights are road-domain biased.

## Dataset Plan

Data must be rebuilt for cabin scenes. Public road data does not transfer.

Potential sources to evaluate:

| Dataset type | Examples | Value |
|---|---|---|
| Driver monitoring | public DMS distraction/fatigue datasets | Phone, smoking, gaze, fatigue labels |
| Cabin occupancy | in-cabin person/seat datasets | Occupant and seat ROI robustness |
| Generic detection | COCO/OpenImages subsets | Phone/person/face bootstrapping only |
| Internal captures | target camera and IR conditions | Required before production selection |

Dataset rules:

- Keep train/val/test split by video/session/person, not by frame.
- Avoid near-duplicate adjacent frames dominating training.
- Keep day/night/IR exposure variants separate in manifest.
- Record camera placement, lens, resolution, and IR/visible mode.
- Build a `dms_board_val` before selecting a final model.

Expected dataset layout:

```text
datasets/dms_v1/
  train/images
  train/labels
  val/images
  val/labels
  board_val/images
  board_val/labels
  dataset.yaml
  board_val.yaml
  manifest.json
```

## Training Route

Create DMS-specific configs rather than editing BSD configs in place:

```text
training/detection/configs/dms_v1_yolo26n_noattn_relu_640.yaml
training/detection/configs/dms_v1_yolo26n_noattn_relu_512.yaml  # fallback only
```

Initial training defaults:

| Item | Value |
|---|---|
| architecture | `training/detection/models/yolo26n_noattn.yaml` |
| activation | `ReLU` |
| imgsz | `640` |
| batch | start with `48` on RTX 4090 |
| epochs | `80-120`, early stop enabled |
| conf/nms eval | fixed `0.5 / 0.45` for deployment comparison |
| main metric | per-class recall for safety events, not only overall mAP |

For DMS model selection, copy the BSD v9.1 lesson: choose by product-critical
event recall first, then check precision and mAP. A candidate with slightly
lower precision can still be the right deployment model if it materially
improves recall for phone, smoking, driver/face, or seatbelt events.

## Board Runtime Route

Keep the camera/NPU loop product-neutral where possible:

```text
live_video_core
  -> detect_engine
  -> product_postprocess
  -> product_event_sink
```

For the first DMS branch, copying `live_bsd.c` to `live_dms.c` is acceptable,
but the follow-up should extract shared camera/NPU code so BSD and DMS do not
drift.

Deployment filenames must be DMS-specific:

```text
/mnt/UDISK/live_dms
/mnt/UDISK/dms_v1_yolo26n_noattn_relu_640.nb
/mnt/UDISK/dms_live_autostart.log
/etc/init.d/dms_live
```

Do not overwrite:

```text
/mnt/UDISK/live_bsd
/mnt/UDISK/bsd_v7_yolo26n_640.nb
/mnt/UDISK/bsd_v9_relu_from_v8_640.nb
/mnt/UDISK/bsd_v9p1_pv_base_640.nb
```

The current BSD preview-mode measurements are:

| Segment | Typical time |
|---|---:|
| NV21 preprocess | `15.8-16.0 ms` |
| NPU | `31.7-31.8 ms` |
| decode | `1.1-1.2 ms` |
| detect total | `48.7-49.0 ms` |
| framebuffer preview | about `89 ms` |
| preview FPS | about `9.4-9.5 FPS` |

Do not interpret preview FPS as production FPS. Production runtime must be
measured in `headless` mode.

## Runtime Architecture Direction

The current `live_bsd` path was built to prove camera, NPU, display, and alarm
integration. It is still mostly single-threaded:

```text
DQBUF -> preprocess -> NPU -> decode -> preview draw -> QBUF
```

This is acceptable for single-camera debugging but not the target shape for
four-channel BSD or a DMS product runtime. The next shared runtime should move
to a producer/consumer pipeline:

```text
camera capture thread(s)
  -> latest-frame slots
  -> preprocess worker
  -> single NPU worker
  -> postprocess/alarm worker
  -> optional low-frequency preview worker
```

Rules for the shared runtime:

- Keep only the latest frame per channel; drop stale frames instead of building
  a queue of old images.
- Let CPU preprocessing overlap with NPU inference where possible.
- Keep NPU access serialized unless the SDK proves concurrent submissions are
  safe and faster.
- Keep preview off the detection critical path. Debug preview can run at low
  FPS and draw the most recent result.
- Keep product event state separate from display state. Display may show only
  the latest inference; alarm logic can still use short missed-frame hold and
  temporal confirmation.

## Video Post-Processing

DMS should rely on video-level logic. Single-frame detection is not reliable
enough for alerts.

Recommended first post-processing:

| Layer | DMS use |
|---|---|
| ROI gating | Driver seat / face area / hand-phone area |
| temporal confirm | Event must persist for N frames before alert |
| short hold | Keep target active through 1-3 missed frames |
| lightweight tracking | IoU + center distance is enough initially |
| event state machine | `pending -> active -> cooldown` |
| class-specific rules | Phone/smoking need persistence; seatbelt may need longer confirmation |

The latest BSD preview issue is a useful warning: display smoothing and alarm
tracking are different concerns. BSD had stale boxes when preview rendering kept
its own held/smoothed boxes and the framebuffer page was not fully cleared. DMS
preview should therefore draw only the current/latest inference result, while
temporal hold and smoothing live inside event logic.

Do not bake these rules into the detector model. Keep them as product logic so
the same detection engine can support BSD, DMS, and other products.

## Suggested Code Layout

Short term:

```text
deployment/board/dms_engine/
  live_dms.c
  dms_postprocess.c
  dms_postprocess.h
  Makefile
```

Longer term:

```text
deployment/board/video_core/
  v4l2_capture.c
  awnn_runner.c
  framebuffer_preview.c
  profiler.c
  pipeline_scheduler.c

deployment/board/products/
  bsd/
  dms/
```

Only extract shared code after DMS is running. Do not block the first branch on
a large refactor.

## First Milestone Checklist

1. Create `codex/dms-cockpit-detector`.
2. Define DMS class list and `dataset.yaml`.
3. Prepare a small DMS validation set before training.
4. Train `dms_v1_yolo26n_noattn_relu_640`.
5. Export ONNX and split with raw-logit `boxes/scores`.
6. Quantize with Pegasus and confirm NB `<5 MB`.
7. Run `test_npu_direct` equivalent on board.
8. Run `live_dms headless`.
9. Add ROI + temporal confirmation post-processing.
10. Build `dms_board_val` before declaring any model final.

## Carry-Over Lessons From BSD

- Fixed `640x640` can still meet size and speed when the graph is NPU-friendly.
- Removing attention matters more than shrinking input when VIPLite maps ops
  poorly.
- ReLU is materially faster than SiLU/Swish on the current board path.
- Preview mode can halve apparent FPS; always measure headless for production.
- On the current board, preview drawing can dominate runtime at about `89 ms`
  per frame even when NPU inference is only about `32 ms`.
- CPU preprocess and NPU inference should be pipelined; otherwise the effective
  detect call time is `preprocess + NPU + decode`, currently about `49 ms`.
- For four-channel runtime, keep per-channel latest-frame slots and avoid stale
  frame queues.
- Do not select models by overall mAP only. Use product-critical per-class
  recall and event-level validation.
- v9.1 was selected over v9 because BSD-critical person/vehicle recall improved
  from `0.741` to `0.760` on `bdd_proxy_val`, despite lower precision.
- Keep stable fallbacks and candidate filenames separate on the board.
- No final model without real board-domain validation data.
