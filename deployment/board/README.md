# Board-Side Code

V853 board-side source for the BSD detector and alarm pipeline.

| Path | Purpose |
|---|---|
| `common/` | Shared C ABI types used by detect and alarm engines |
| `detect_engine/` | AWNN model loading, preprocessing, YOLO decode, NMS, shared-memory output |
| `alarm_engine/` | Zone hit testing, IOU tracking, short hold, smoothing, cooldown, consecutive-frame alarm logic |
| `test_v4/` | Board-side model test code |
| `test_bdd100k/` | BDD100K board test code, currently untracked |
| `camera/` | Camera bring-up and debugging scripts, currently experimental |

Main entry points:

| File | Purpose |
|---|---|
| `detect_engine/main.c` | stdin BGR raw test runner, builds as `bsd_detect` |
| `detect_engine/live_bsd.c` | realtime V4L2 camera -> NPU -> framebuffer runner |
| `detect_engine/test_npu_direct.c` | raw BGR file -> NB/NPU direct comparison runner |
| `compile_engines.py` | legacy remote build helper |
| `compile_live.py` | remote build helper for `live_bsd` and `test_npu_direct`, currently untracked |

Generated objects, shared libraries, board executables, and copied SDK libraries are ignored by Git.

## Video Reliability Layer

Model output is single-frame detection. The BSD runtime adds a lightweight
video reliability layer in `alarm_engine` before emitting alarms:

| Step | Purpose | Default |
|---|---|---|
| Zone hit test | Only alarm inside configured blind-zone ROIs | zone 0 center, zone 1 full frame |
| IoU association | Keep a target identity across frames | `match_iou=0.30` |
| BBox smoothing | Reduce box jitter before zone/alarm output | `smooth_alpha=0.65` |
| Consecutive confirmation | Avoid one-frame false alarms | `alarm_frames=3` |
| Short hold | Survive brief detector misses | `max_missed=3` |
| Cooldown | Suppress repeated alarms after leaving ROI | `cooldown=15` |
| Re-emit interval | Repeat long active alarms at controlled rate | `reemit=15` |
| Class enable mask | Control which classes can alarm | person/bicycle/motorcycle/vehicle enabled in `live_bsd`; vehicle can be disabled |

`DetResult` and `AlarmEvent` ABI are unchanged. The reliability layer is
configured through `alarm_set_tracker_params()` and `alarm_set_class_enabled()`.

`live_bsd` keeps old arguments compatible and appends optional alarm parameters:

```sh
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v9_relu_from_v8_640.nb \
  0.5 0.45 0.5 0.5 1280 720 headless 640 \
  3 3 15 15 0.65 1
```

The last six values are:

```text
alarm_frames max_missed cooldown reemit smooth_alpha vehicle_alarm
```

For early board tests, keep `alarm_frames=3` and `max_missed=3`. Increase
`alarm_frames` only if false alarms remain after ROI tuning.
