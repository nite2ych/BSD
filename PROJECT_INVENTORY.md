# BSD Project Inventory

This file separates the current project into maintainable source, local generated artifacts, and experimental scripts.

## Maintainable Core

These files/directories are the main code and documentation that should generally stay under version control:

| Path | Role |
|---|---|
| `README.md` | Project entry point |
| `BSD_DEV_GUIDE.md` | Main development manual |
| `DEPLOYMENT_GUIDE.md` | Historical V853/DMS deployment reference |
| `deployment/board/BOARD_ENGINE.md` | Board-side engine reference |
| `deployment/board/common/` | Shared ABI types |
| `deployment/board/detect_engine/` | Detection engine and realtime `live_bsd` pipeline |
| `deployment/board/alarm_engine/` | Zone, tracker, and alarm engine |
| `deployment/board/test_v4/` | Board-side test code |
| `deployment/dataset/` | Dataset conversion/preparation scripts |
| `deployment/quantize/` | ONNX split and calibration scripts |
| `examples/` | Integration examples |
| `training/detection/configs/` | Training configs |

## Candidate Source To Review

These are currently untracked but look like source or useful tooling. Review before adding them to Git:

| Path | Notes |
|---|---|
| `deployment/board/detect_engine/live_bsd.c` | Realtime camera/NPU/framebuffer pipeline |
| `deployment/board/detect_engine/awnn.h` | Local AWNN declarations needed for board build |
| `deployment/board/detect_engine/test_npu_direct.c` | Direct NPU smoke test |
| `deployment/board/detect_engine/*.sh` | Board build helper scripts |
| `deployment/board/compile_live.py` | Upload and compile `live_bsd` remotely |
| `deployment/board/check_camera.py` | Board camera diagnostic helper |
| `deployment/board/test_bdd100k/` | BDD100K board test |
| `deployment/quantize/prepare_calibration_data.py` | Calibration image generator |
| `deployment/quantize/split_yolo_head_sig.py` | Sigmoid-output split experiment |
| `tools/` | PC, SSH, validation, and quantization helpers |
| `training/detection/configs/bsd_yolo26s*.yaml` | BSD training configs |

Suggested first add set:

```text
README.md
PROJECT_INVENTORY.md
deployment/board/detect_engine/awnn.h
deployment/board/detect_engine/live_bsd.c
deployment/board/detect_engine/test_npu_direct.c
deployment/board/detect_engine/*.sh
deployment/board/compile_live.py
deployment/board/check_camera.py
deployment/board/test_bdd100k/
deployment/quantize/prepare_calibration_data.py
deployment/quantize/split_yolo_head_sig.py
training/detection/configs/
```

Review before adding:

```text
tools/
deployment/board/camera/
root-level *.py scripts
DEBUG_RECORD.md
```

The review bucket contains useful debugging history, but many files contain hardcoded machine paths, hostnames, or credentials.

## Local Artifacts

These are intentionally ignored and should not be committed:

| Path/Pattern | Contents |
|---|---|
| `artifacts/` | Models, NB files, board binaries, test frames, calibration images |
| `_toolchain/` | SDK/toolchain cache |
| `*.pt`, `*.onnx`, `*.nb`, `*.raw` | Model and validation artifacts |
| `*.o`, `*.so`, board executables | Build outputs |
| `__pycache__/` | Python cache |

## Experimental And Cleanup Candidates

The root directory contains many one-off server/model scripts such as `check_server.py`, `download_v4_model.py`, `quick_val.py`, and `test_pt_model.py`. They may still be useful, but they duplicate scripts under `tools/` and often contain hardcoded hosts or credentials.

Recommended cleanup path:

1. Keep reusable scripts under `tools/`.
2. Move one-off scripts to `tools/experiments/` or delete them after confirming they are obsolete.
3. Replace hardcoded credentials with environment variables.
4. Keep generated outputs under `artifacts/`.
