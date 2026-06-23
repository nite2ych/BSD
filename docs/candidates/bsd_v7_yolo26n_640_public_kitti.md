# BSD v7 YOLO26n@640 Public KITTI 候选

这是当前稳定回退 BSD 候选。

## 汇总

| 项目 | 值 |
|---|---|
| 候选 | `bsd_v7_yolo26n_640_public_kitti` |
| 模型 | `YOLO26n@640` |
| 训练产物 | `/root/autodl-tmp/BSD/artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/weights/best.pt` |
| 本地产物目录 | `artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/` |
| 板端 NB | `/mnt/UDISK/bsd_v7_yolo26n_640.nb` |
| 阈值 | `conf=0.5`, `nms=0.45`, `disp_conf=0.5`, `person_conf=0.5` |
| 板端模式 | 生产用 `headless`，`preview` 只用于调试 |
| ABI | 保持 `DetResult` / `AlarmEvent` 不变 |

## 精度

固定 v4 基线：`bsd_v4_150ep`，`YOLO26s@640`。

| 验证集 | 模型 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `coco_val` | v4 `s@640` | 0.818 | 0.557 | 0.652 | 0.425 |
| `coco_val` | v7 `n@640` | 0.741 | 0.521 | 0.586 | 0.364 |
| `bdd_proxy_val` | v4 `s@640` | 0.678 | 0.502 | 0.556 | 0.336 |
| `bdd_proxy_val` | v7 `n@640` | 0.682 | 0.615 | 0.674 | 0.467 |

BSD 排名优先看 `bdd_proxy_val`，因为它更接近道路场景部署域。`coco_val` 保留为泛化检查。

`bdd_proxy_val` 逐类对比：

| 类别 | v4 R | v7 R | v4 mAP50 | v7 mAP50 |
|---|---:|---:|---:|---:|
| `person` | 0.622 | 0.731 | 0.686 | 0.746 |
| `bicycle` | 0.293 | 0.419 | 0.347 | 0.535 |
| `motorcycle` | 0.400 | 0.467 | 0.429 | 0.544 |
| `vehicle` | 0.692 | 0.841 | 0.761 | 0.869 |

## 量化

| 产物 | 值 |
|---|---|
| 拆分 ONNX | `artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx` |
| NB | `artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/bsd_v7_yolo26n_640_public_kitti.nb` |
| NB 大小 | `3,696,064` bytes，约 `3.52 MiB` |
| NB MD5 | `26c358cf22e9ba05f64e3c6e42c5ccaf` |
| NB magic | `VPMN` |
| 硬性大小门槛 | 满足 `<5 MB` |

标准拆分：

```bash
python deployment/quantize/split_yolo_head.py \
  --input artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best.onnx \
  --output artifacts/bsd_v7_yolo26n_640_public_kitti_stage1/best_640_split_nosig.onnx
```

标准 Pegasus 导出：

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

板端解码器期望 raw logits。除非同步修改 `yolo_decode.c`，否则不要部署 sigmoid-output NB。

## 板端部署

通过串口 + 临时 HTTP server 部署：

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

生产命令：

```sh
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v7_yolo26n_640.nb 0.5 0.45 0.5 0.5 1280 720 headless 640
```

调试预览命令：

```sh
/mnt/UDISK/live_bsd /mnt/UDISK/bsd_v7_yolo26n_640.nb 0.5 0.45 0.5 0.5 1280 720 preview 640
```

上电自启动：

```sh
cp deployment/board/init.d/bsd_live /etc/init.d/bsd_live
chmod +x /etc/init.d/bsd_live
/etc/init.d/bsd_live enable
ln -sf bsd_live /etc/init.d/S99bsd_live
```

这个 Tina 镜像会从 `rc.final` 执行 `/etc/init.d/S??*`，所以除了 `enable` 创建的 OpenWrt 风格 `/etc/rc.d/S99bsd_live` 链接，还必须有 `/etc/init.d/S99bsd_live` 这个软链接。

自启动服务运行 `headless`，日志写入 `/mnt/UDISK/bsd_live_autostart.log`。

## 板端结果

V853 上使用 `1280x720` NV21M 摄像头输入实测：

| 模式 | FPS | 检测总耗时 | NV21 预处理 | NPU | 解码 | 预览 |
|---|---:|---:|---:|---:|---:|---:|
| `headless` | 约 15.0 | 约 109.5 ms | 约 16.0 ms | 约 92.4 ms | 约 1.2 ms | 0 ms |
| `preview` | 约 7.7 | 约 109.5 ms | 约 15.9 ms | 约 92.4 ms | 约 1.2 ms | 约 93.6 ms |

`preview` 更慢是因为额外做 framebuffer 转换、旋转/缩放、画框和 page flip，并不代表 NPU 推理变慢。

## 640 固定输入速度实验

当前优化阶段保持模型输入 `640x640`，硬部署门槛保持 NB `<5 MB`。

| 变体 | 本地产物 | NB 大小 | 板端 NPU | 结论 |
|---|---|---:|---:|---|
| 基线 | `bsd_v7_yolo26n_640_public_kitti.nb` | `3.52 MiB` | `92.1-92.4 ms` | 当前稳定模型 |
| ONNX simplified | `bsd_v7_yolo26n_640_public_kitti_sim.nb` | `3.52 MiB` | `92.1-92.2 ms` | 无速度收益 |
| Pegasus `--force-remove-permute` | `bsd_v7_yolo26n_640_public_kitti_forceperm.nb` | `3.52 MiB` | `92.1-92.2 ms` | 无速度收益 |
| Pegasus `--dtype quantized_strict` | `bsd_v7_yolo26n_640_public_kitti_qstrict.nb` | `20.56 MiB` | 未测试 | 拒绝，不满足 `<5 MB` |
| Pegasus `perchannel_symmetric_affine int8` | `bsd_v7_yolo26n_640_public_kitti_perch_int8.nb` | `4.37 MiB` | 未测试 | 大小满足，但模拟器更慢 |

拆分 ONNX 简化只把节点从 `375` 降到 `357`，重算子集合没有变化：

```text
Conv 102
Mul/Sigmoid(Swish) 89/87
MatMul 4
Softmax 2
Transpose/Permute 4
Resize 2
```

Pegasus 融合图仍保留同样的非 conv 算子：

```text
CONV2D 102, SWISH 87, MATRIXMUL 4, SOFTMAX 2, PERMUTE 4, RESIZE 2
```

结论：当前 `YOLO26n@640` 已足够小，但图结构对 NPU 不够友好，无法靠普通 ONNX 简化或 Pegasus 导出参数达到 `~50 ms` 目标。下一步速度模型应保持 `640x640`，但用主要由 Conv/BN/activation 组成的 nano 架构替换 attention-style block，然后重新跑同一套精度、NB 大小和板端 NPU 门槛。

## 剩余门槛

该候选可用于流程测试和当前 BSD proxy 验证，但最终发布前仍需要在具备采集条件后补正式板端验证集。
