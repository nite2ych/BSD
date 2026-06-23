# BSD v8 YOLO26n No-Attention 640 Candidate

This is the speed-route experiment. It does not replace the v7 deployed
baseline.

## Baseline Rule

Keep v7 unchanged for comparison:

| Item | Value |
|---|---|
| Baseline candidate | `bsd_v7_yolo26n_640_public_kitti` |
| Board NB | `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |
| Baseline NPU | about `92.1 ms` |
| Baseline NB | `3.52 MiB` |

## Model Change

v8 keeps the YOLO26 input/output convention but removes the attention block that
created bad NPU ops.

| Item | Value |
|---|---|
| Model YAML | `training/detection/models/yolo26n_noattn.yaml` |
| Train config | `training/detection/configs/bsd_v8_yolo26n_noattn_640_stage1.yaml` |
| Input | `640x640` |
| Detect head | YOLO26, `reg_max=1`, raw-logit split compatible |
| ABI | No `DetResult` / `AlarmEvent` change |

The original v7 graph contains attention/layout-heavy ops:

```text
MATRIXMUL 4
SOFTMAX 2
PERMUTE 4
```

The v8 no-attention probe graph removes them:

```text
CONV2D 104
SWISH 97
CONCAT 23
ADD 17
SPLIT 9
RESHAPE2 6
POOL 3
RESIZE 2
```

## Speed Probe

The probe was exported from random/untrained weights only to measure board
runtime. Do not use it for accuracy.

| Variant | NB size | Simulator | V853 board NPU | Detection total |
|---|---:|---:|---:|---:|
| v7 `YOLO26n@640` | `3.52 MiB` | `18.49 ms` | `92.1 ms` | `109.3 ms` |
| v8 no-attn probe | `1.69 MiB` | `14.66 ms` | `62.8-62.9 ms` | `79.8-80.0 ms` |

This is a real speed improvement at fixed `640x640`, but not yet the `~50 ms`
target. It is worth training because it preserves the board decode path and
stays far below the `<5 MB` NB limit.

## Trained Stage1 Result

Training completed with early stopping at epoch 78. Best epoch was 48. The
model was initialized from the v7 public-KITTI weights, with
`635/720` weight items transferred.

| Validation set | Model | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `coco_val` | v7 `n@640` | 0.741 | 0.521 | 0.586 | 0.364 |
| `coco_val` | v8 no-attn `n@640` | 0.739 | 0.506 | 0.577 | 0.361 |
| `bdd_proxy_val` | v7 `n@640` | 0.682 | 0.615 | 0.674 | 0.467 |
| `bdd_proxy_val` | v8 no-attn `n@640` | 0.744 | 0.589 | 0.680 | 0.473 |

Interpretation:

- v8 is slightly worse on `coco_val`.
- v8 is slightly better on `bdd_proxy_val` mAP, which is more relevant to BSD.
- v8 recall drops on `bdd_proxy_val` by about `0.026`, so v7 remains the
  fallback until board-side validation exists.

## Trained Quantization

| Artifact | Value |
|---|---|
| PT | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.pt` |
| ONNX | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.onnx` |
| Split ONNX | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/best_640_split_nosig.onnx` |
| NB | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1.nb` |
| Board NB | `/mnt/UDISK/bsd_v8_yolo26n_noattn_640.nb` |
| NB size | `3,598,080` bytes, about `3.43 MiB` |
| NB MD5 | `ca0667ef858073ec9e93f0cb8b7c23f3` |
| NB magic | `VPMN` |
| Hard size gate | Passes `<5 MB` |

Trained split ONNX operator check:

```text
Conv 104
Sigmoid 97
Mul 97
Concat 23
Add 17
Split 9
Reshape 6
MaxPool 3
Resize 2
MatMul 0
Softmax 0
Transpose 0
```

Pegasus fused graph:

```text
CONV2D 104
SWISH 97
CONCAT 23
ADD 17
SPLIT 9
RESHAPE2 6
POOL 3
RESIZE 2
```

Pegasus simulator after export:

```text
Run the 1 time: 17.12 ms
```

## Board Result

Measured on V853 with the same `bench_npu_loop` and `live_bsd headless`
procedure as v7. The v7 NB remains deployed separately as
`/mnt/UDISK/bsd_v7_yolo26n_640.nb`.

| Variant | NB size | Continuous benchmark | `live_bsd headless` detection total | `live_bsd headless` NPU |
|---|---:|---:|---:|---:|
| v7 `YOLO26n@640` | `3.52 MiB` | `91.66 ms` NPU, `9.83 FPS` | about `109.2 ms` | about `92.1 ms` |
| v8 no-attn trained | `3.43 MiB` | `67.28 ms` NPU, `12.93 FPS` | about `84.5 ms` | about `67.4 ms` |

Speed delta:

- NPU time improves from about `92.1 ms` to about `67.4 ms`.
- Detection-call total improves from about `109.2 ms` to about `84.5 ms`.
- This is about a `26-27%` board-side NPU reduction at fixed `640x640`.

## Commands

Train and evaluate:

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v8_yolo26n_noattn_640_stage1.yaml \
  --mode all \
  --device 0
```

After training, export and split:

```bash
python training/detection/run_bsd_candidate.py \
  --config training/detection/configs/bsd_v8_yolo26n_noattn_640_stage1.yaml \
  --mode export \
  --weights /root/autodl-tmp/BSD/artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.pt \
  --device cpu

python deployment/quantize/split_yolo_head.py \
  --input artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.onnx \
  --output artifacts/bsd_v8_yolo26n_noattn_640_stage1/best_640_split_nosig.onnx
```

Quantize:

```bash
python tools/quantize_bsd_candidate.py \
  --host 192.168.144.133 \
  --user ubuntu \
  --password "$BSD_QUANT_PASSWORD" \
  --remote-dir /home/ubuntu/bsd_quantize \
  --onnx artifacts/bsd_v8_yolo26n_noattn_640_stage1/best_640_split_nosig.onnx \
  --calib-dir artifacts/bdd100k/calib_images \
  --model-name bsd_v8_yolo26n_noattn_640_stage1 \
  --model-size 640
```

## Quantization Matrix Follow-Up

After the activation experiments failed to preserve accuracy, v8 was retested
with low-risk Pegasus quantization/export variants. These do not change the
trained PT weights.

| Variant | Local NB | NB size | Pegasus simulator | Result |
|---|---|---:|---:|---|
| standard `asymmetric_affine` | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1.nb` | `3,598,080` bytes | `17.12 ms` | Current best |
| `moving_average` algorithm | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1_mavg.nb` | `3,598,080` bytes | `17.28 ms` | No speed gain |
| `kl_divergence` algorithm | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1_kl.nb` | `3,598,080` bytes | `17.88 ms` | Slower |
| `perchannel_symmetric_affine int8` | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1_perch_int8.nb` | `4,427,776` bytes | `23.66 ms` | Slower, larger |
| sigmoid score output | `artifacts/bsd_v8_yolo26n_noattn_640_stage1/bsd_v8_yolo26n_noattn_640_stage1_sig_v2.nb` | `3,602,752` bytes | `17.64 ms` | No speed gain; decoder change required |

The sigmoid-output split was regenerated with corrected BSD class metadata:

```bash
python deployment/quantize/split_yolo_head_sig.py \
  --input artifacts/bsd_v8_yolo26n_noattn_640_stage1/weights/best.onnx \
  --output artifacts/bsd_v8_yolo26n_noattn_640_stage1/best_640_split_sig.onnx
```

Important: sigmoid-output NB is not drop-in compatible with the current board
decoder. The current `yolo_decode.c` expects raw logits and applies sigmoid on
the CPU. Only use the sigmoid NB after changing the board decoder to consume
already-sigmoided scores.

Conclusion: standard v8 quantization remains the best speed candidate. The
tested Pegasus quantization variants do not show a simulator speed improvement,
so they are not priority board-deploy candidates. Further v8 work should focus
on board runtime integration: input preprocessing, queueing, and 4-channel NPU
scheduling.

## Gates

| Gate | Requirement |
|---|---|
| Accuracy | Compare against v7 on `coco_val` and `bdd_proxy_val` |
| Size | Quantized NB `<5 MB` |
| Board speed | Must beat v7 `92 ms`; target is closer to `50 ms` |
| Board flow | `test_npu_direct` then `live_bsd headless 640` |
| Baseline safety | Do not overwrite `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |

## Decision

v8 no-attn is the current speed candidate, not the stable replacement yet. It
passes the NB size gate and clearly improves board speed, while preserving
`640x640` input and the existing raw-logit decoder path. Keep v7 as the
production fallback because its recall is still better on `bdd_proxy_val` and
formal `board_val` is still missing.

## 2026-06-13 Archive Decision

Current judgement: v8 is now an archived intermediate candidate. It should stay
in the repository as the proof that removing attention improves the VIPLite
graph, but new board deployment and optimization should use v9 or v9.1 instead.

Why it was useful:

- Size gate passes with margin: `3.43 MiB`, below the hard `<5 MB` NB limit.
- Fixed-640 speed improves clearly: board NPU drops from about `92.1 ms` to
  about `67.4 ms`.
- Road-scene proxy mAP is slightly better than v7: `bdd_proxy_val` mAP50
  `0.680` vs v7 `0.674`.
- v8 avoids the v7 attention ops that mapped poorly to VIPLite
  (`MATRIXMUL/SOFTMAX/PERMUTE`).

Why it should not replace v7 yet:

- `bdd_proxy_val` recall drops from v7 `0.615` to v8 `0.589`, about `-0.026`.
  BSD cares about missed nearby person/two-wheel/vehicle targets more than
  small precision gains.
- `coco_val` is also slightly lower than v7: mAP50 `0.577` vs `0.586`.
- There is still no formal board-side `board_val`; current board tests only
  prove runtime flow and speed.

Final position:

| Use | Decision |
|---|---|
| Historical comparison | Keep |
| No-attention proof | Keep |
| Speed candidate | Superseded by v9 |
| 4-channel scheduler baseline | Superseded by v9 |
| Production fallback replacement | Not yet |
| Final BSD model | Blocked by `board_val` and recall review |

## Remaining Optimization Space

Do not spend more time on v8 Pegasus quantization variants unless a new compiler
flag or SDK path appears and v9/v9.1 cannot be compiled. Standard v8
quantization remains documented only for comparison.

The remaining work below has moved to the v9/v9.1 route:

1. Board runtime profiling with v8: split `capture / preprocess / set_input /
   awnn_run / get_output / decode / qbuf`, then optimize only the measured hot
   stages.
2. Four-channel design: one NPU worker, four latest-frame slots, round-robin
   inference, drop stale frames. Do not run four independent model processes
   because VIPLite serializes NPU work.
3. Preprocess optimization: keep the existing NV21M fused path, then reduce row
   index recomputation, repeated padding fill, and avoid unnecessary buffer
   clears.
4. Accuracy recovery without losing v8 speed: keep no-attention + `640x640`,
   train with harder road-scene negatives and distill from v7/v4 if recall
   remains behind.
5. A smaller architecture is a separate v12 route only if v8 still cannot meet
   4-channel latency after scheduler/preprocess work. Simple activation swaps
   are rejected because v9/v10/v11 lost too much accuracy.
