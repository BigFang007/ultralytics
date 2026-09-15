# PT 缺陷检测项目上下文

供后续 Codex 和开发者快速恢复项目状态。执行训练前先阅读本文档。

## 目标与方案

- 固定工业场景中的圆环工件缺陷检测，原图分辨率 `5472×3648`。
- 采用两阶段架构：第一阶段实例分割得到圆环 ROI，第二阶段在 ROI 内进行小目标缺陷检测。

## 代码与目录

- 仓库：`git@github.com:BigFang007/ultralytics.git`
- 分支：`project/pt-defect`
- Mac Conda 环境：`ultralytics`；本机无 CUDA/MPS，仅用于数据准备。
- AutoDL 实际代码目录：`/root/ultralytics`；仓库 HEAD 为
  `38e876eacb7dc66e620ffea49279e04bf4401336`，工作区在训练检查时保持干净。
- AutoDL 数据目录：YAML 为 `/root/autodl-tmp/pt_ring_seg.yaml`，图像和标签分别位于
  `/root/autodl-tmp/{images,labels}/{train,val}`。
- 训练日志根目录：`/root/tf-logs`。注意该目录位于 30 GB 系统盘，不是 50 GB 数据盘，训练权重需及时备份。

## 训练环境

- 服务器系统为 Ubuntu 22.04.3 LTS，128 个逻辑 CPU、约 1.0 TiB 内存，无 Swap。
- GPU 为单卡 NVIDIA GeForce RTX 4090 24 GB（24564 MiB），驱动 `560.35.03`，`nvidia-smi` 显示最高支持
  CUDA 12.6。
- Python 环境位于 `/root/miniconda3`：Python 3.10.8、Ultralytics 8.4.115、PyTorch 2.1.2+cu121、
  torchvision 0.16.2+cu121、NumPy 1.26.3、CUDA Runtime 12.1、cuDNN 8.9.2。

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
- AutoDL 上采用扁平部署：`pt_ring_seg.yaml` 与 `images/`、`labels/` 同在 `/root/autodl-tmp`；实测数量为
  train 图像/标签各 50，val 图像/标签各 19。

## 相关文件

- `my_tool/split_yolo_seg_dataset.py`：严格配对、随机划分并移动数据。
- `my_tool/labelme_circle_to_yolo_seg.py`：Labelme circle 转 YOLO-seg。
- `my_tool/pt_ring_seg.yaml`：可迁移的数据集配置模板。
- `my_tool/analyze_seg_dataset.py`：本地数据分析工具，当前未纳入 Git。

## 分割阶段首次训练（已完成）

执行命令：

```bash
yolo segment train \
  model=yolov8n-seg.pt \
  data=/root/autodl-tmp/pt_ring_seg.yaml \
  epochs=150 imgsz=1024 batch=-1 device=0 overlap_mask=True \
  project=/root/tf-logs/yolo_seg name=yolov8n_1024
```

- 因同名目录已由前一次启动创建且 `exist_ok=False`，实际有效输出目录为
  `/root/tf-logs/yolo_seg/yolov8n_1024-2`；`/root/tf-logs/yolo_seg/yolov8n_1024` 不是本次有效结果。
- 实际参数还包括：`workers=8`、`optimizer=auto`、`amp=True`、`deterministic=True`、`mosaic=1.0`、
  `close_mosaic=10`、`overlap_mask=True`。
- 训练时间：2026-09-15 09:56 至 10:06（Asia/Shanghai），150 epochs 总耗时 617.648 秒，进程正常退出。
- 训练期间显存占用约 6.07 GiB（`nvidia-smi` 采样约 6831 MiB），未发现 OOM、标签损坏或训练中断。
- 最后一轮 box 指标：P 0.99693、R 1.0、mAP50 0.995、mAP50-95 0.96878。
- 最后一轮 mask 指标：P 0.99693、R 1.0、mAP50 0.995、mAP50-95 0.87731。
- 单项最佳值：box mAP50-95 为 0.97225（epoch 149），mask mAP50-95 为 0.89051（epoch 93）。
  `best.pt` 由框架综合 fitness 选取，不应仅凭某个单项最大值推断对应 epoch。
- 权重：`weights/best.pt` 和 `weights/last.pt` 均约 6.8 MB；`best.pt` SHA-256 为
  `7873182ff4f8118fde8e6e38428f588e1ff7b2c9716fc4b4d6b7a1964244938d`。
- 训练产物齐全，包括 `results.csv`、`results.png`、PR/F1/P/R 曲线、混淆矩阵和验证预测图。

下一步应使用 `best.pt` 对原始高分辨率图像做可视化验证，重点检查圆环边界和内外圆掩膜，再决定是否训练
`yolov8s-seg.pt` 对照实验。第二阶段缺陷类别、标签和切片策略尚未确定。

## 未标注样本推理与伪标签筛选

- 使用分割阶段首次训练的 `best.pt`，以 `imgsz=1024`、`conf=0.25` 对 914 张未标注图像完成推理。
- 推理结果位于 `datasets/pt/PT缺陷/PT缺陷/yolo_seg_predict/yolov8n_1024-2`（相对于 `practice/`）。
- 当前可用预测标签为 913 份；`1 (2622).jpg` 没有对应预测标签，不参与筛选。
- 两个实例的最低置信度分布为：最小值 0.442596、P25 0.956871、中位数 0.965152、P75 0.971343、
  最大值 0.983949。
- 优质伪标签筛选条件：每张图恰好包含一个 `out` 和一个 `in`；两个实例的最低置信度不低于 0.965；
  内外多边形面积比处于 0.45–0.60；归一化中心偏移不大于 0.05；至少 99% 的内圆多边形点位于外圆
  多边形内。归一化中心偏移定义为内外多边形中心距离除以外圆等效半径。
- 上述阈值共选出 200 张。拒绝原因允许重叠计数：低置信度 449、中心偏移超限 525、面积比超限 11、
  包含率不足 7、缺少标签 1。
- 筛选脚本：`my_tool/select_seg_pseudo_labels.py`。
- 筛选结果位于 `datasets/pt/PT缺陷/PT缺陷/yolo_detect`（相对于 `practice/`），内含 `images/` 200 张、
  `labels/` 200 份、`selection_manifest.csv` 和 `selection_report.json`，总计约 250 MB。
- 输出标签已移除 `save_conf=True` 添加的行尾置信度，仅保留标准 YOLO-seg 类别和归一化多边形坐标；图片与
  标签已验证严格同名配对，每份标签均为 `out`、`in` 各一个实例。
