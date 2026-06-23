# BSD Traffic Participant Detection

V853 board-side blind spot detection project for 4 traffic participant classes:

- `person`
- `bicycle`
- `motorcycle`
- `vehicle`

Start here:

| Document | Purpose |
|---|---|
| `BSD_DEV_GUIDE.md` | Main development manual: data, ONNX, quantization, board build, deployment |
| `docs/BSD_PUBLIC_DATASET_PLAN.md` | Public dataset route and BSD accuracy target |
| `docs/candidates/bsd_v7_yolo26n_640_public_kitti.md` | Current deployed `YOLO26n@640` candidate, metrics, quantization, and board commands |
| `docs/candidates/bsd_v8_yolo26n_noattn_640.md` | 640 fixed no-attention speed experiment; keeps v7 as baseline |
| `deployment/board/BOARD_ENGINE.md` | Board-side detect/alarm engine details |
| `DEPLOYMENT_GUIDE.md` | Historical V853 deployment notes from the DMS project |

Core directories:

| Path | Purpose |
|---|---|
| `deployment/dataset/` | COCO/BDD100K to BSD dataset scripts |
| `deployment/quantize/` | ONNX head splitting and calibration data tools |
| `deployment/board/common/` | Shared C ABI types |
| `deployment/board/detect_engine/` | NPU detection engine and `live_bsd` pipeline |
| `deployment/board/alarm_engine/` | Zone hit test, tracking, and alarm callbacks |
| `training/detection/configs/` | BSD training configs |
| `tools/` | PC, SSH, quantization, and validation helper scripts |

Generated models, board binaries, SDKs, and local toolchains are intentionally ignored by Git. Keep them in `artifacts/` or `_toolchain/`.
