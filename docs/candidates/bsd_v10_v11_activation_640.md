# BSD v10/v11 Activation Candidates

These candidates keep the current `640x640` input and `YOLO26n_noattn`
topology, then replace the Conv activation to test whether VIPLite can run
faster without the accuracy loss seen in v9 pure ReLU.

Keep the stable board model unchanged:

```text
/mnt/UDISK/bsd_v7_yolo26n_640.nb
```

## Baselines

| Candidate | Role | coco mAP50 | bdd mAP50 | Board NPU |
|---|---|---:|---:|---:|
| v7 `YOLO26n@640` | stable fallback | `0.586` | `0.674` | about `92.1 ms` |
| v8 no-attn SiLU | speed candidate | `0.577` | `0.680` | about `67.4 ms` |
| v9 no-attn ReLU | speed proof, not deployable | `0.463` | `0.596` | probe about `28.8 ms` |

## New Routes

| Candidate | Activation | Intent |
|---|---|---|
| v10 | `LeakyReLU(0.1)` | First choice. Usually closer to ReLU speed while avoiding dead activations. |
| v11 | `ReLU6` | Fallback. Bounded activation may recover some accuracy, but board op mapping must be checked. |

## Commands

Train v10:

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v10_yolo26n_noattn_leakyrelu_640_stage1.yaml \
  --mode all \
  --device 0
```

Train v11:

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v11_yolo26n_noattn_relu6_640_stage1.yaml \
  --mode all \
  --device 0
```

## Gates

| Gate | Requirement |
|---|---|
| Accuracy | First compare against v8 on `bdd_proxy_val`; v7 remains the fallback. |
| Recall | BSD route should not accept a large recall drop just for speed. |
| Size | Quantized compressed NB must be `<5 MB`. |
| Speed | Must beat v8 `~67.4 ms` NPU; target is `30-50 ms` if accuracy holds. |
| Board flow | `PT -> ONNX -> split raw-logit ONNX -> Pegasus NB -> bench_npu_loop -> live_bsd headless`. |
