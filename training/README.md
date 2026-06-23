# 训练

该目录当前主要保存 BSD 训练配置文件。

| 文件 | 用途 |
|---|---|
| `detection/configs/bsd_yolo26s.yaml` | BSD 4 类主训练配置 |
| `detection/configs/bsd_yolo26s_small.yaml` | 更小/备用 BSD 训练配置 |

后续候选配置：

| 文件 | 用途 |
|---|---|
| `detection/configs/bsd_yolo26n.yaml` | 用于板端吞吐对比的小模型候选 |

`BSD_DEV_GUIDE.md` 中引用的训练和导出入口脚本当前不在本地仓库内：

```text
training/detection/train.py
training/detection/export_onnx.py
```

如果要从当前 checkout 直接跑通端到端训练，需要先从训练环境恢复这些脚本，或重新实现等价入口。
