# PT 缺陷检测项目上下文

供后续 Codex 和开发者快速恢复项目状态。执行训练前先阅读本文档。

## 目标与方案

- 固定工业场景中的圆环工件缺陷检测，原图分辨率 `5472×3648`。
- 采用两阶段架构：第一阶段实例分割得到圆环 ROI，第二阶段在 ROI 内进行小目标缺陷检测。
- 当前阶段只训练第一阶段，使用官方预训练权重 `yolov8s-seg.pt`。

## 代码与环境

- 仓库：`git@github.com:BigFang007/ultralytics.git`
- 分支：`project/pt-defect`
- Mac Conda 环境：`ultralytics`；本机无 CUDA/MPS，仅用于数据准备。
- AutoDL 计划目录：代码 `/root/autodl-tmp/ultralytics`，数据 `/root/autodl-tmp/yolo_seg`。
- 推荐单卡 RTX 4090 24GB；先训练 `imgsz=640` 基线，边界精度不足再对比 `imgsz=1024`。

## 分割数据集

数据集根目录为 `yolo_seg`，内部结构：

```text
yolo_seg/
├── images/{train,val}       # 50 / 19 张
├── labelme/{train,val}      # 原始 Labelme JSON，50 / 19 份
├── labels/{train,val}       # YOLO-seg TXT，50 / 19 份
├── pt_ring_seg.yaml
├── split_manifest.json
└── conversion_report.json
```

- 69 张图随机划分为 train 50、val 19，固定随机种子 `42`。
- 类别固定为 `0: out`、`1: in`；每张图各有一个外圆和内圆实例。
- 每个 Labelme 圆转换为 72 点归一化多边形，框架检查结果为 69 张、138 个实例、0 损坏。
- 训练保持 `overlap_mask=True`：内圆覆盖外圆的重叠像素，使 `out` 监督区域成为圆环，`in` 为内部圆盘。
- 数据集 YAML 不含绝对 `path`，必须放在 `yolo_seg` 根目录以便跨机器迁移。

## 相关文件

- `my_tool/split_yolo_seg_dataset.py`：严格配对、随机划分并移动数据。
- `my_tool/labelme_circle_to_yolo_seg.py`：Labelme circle 转 YOLO-seg。
- `my_tool/pt_ring_seg.yaml`：可迁移的数据集配置模板。
- `my_tool/analyze_seg_dataset.py`：本地数据分析工具，当前未纳入 Git。

## AutoDL 首次训练

上传数据并安装当前仓库后先执行冒烟训练：

```bash
yolo segment train \
  model=yolov8n-seg.pt \
  data=/root/autodl-tmp/yolo_seg/pt_ring_seg.yaml \
  epochs=3 imgsz=640 batch=8 device=0 workers=4 \
  overlap_mask=True mosaic=0.0 \
  project=/root/autodl-tmp/runs/pt_ring_seg name=smoke
```

确认标签、类别和预测正常后，再用 `yolov8s-seg.pt` 训练约 150 epochs。训练结果和 `best.pt` 放在数据盘并及时备份。第二阶段缺陷类别、标签和切片策略尚未确定。
