# 本轮图表替换清单

以下图件对应最终 B 模型：**均衡整图采样 + CE–Tversky，ASPP 膨胀率 1、2、3**。
内部结果均来自 W70 `aspp_small_rates` 的六个种子（1001–1006）；未混用旧 Final A 或 ASPP 6/12/18 的内部结果。

| 原文位置 | 本目录 SVG | 需要替换的内容 |
|---|---|---|
| Fig. 2A | `Figure2A_workflow.svg` | 将组织预测、尺度标定、阵列映射和预设时长分配区分开；人类与内部模型分别训练。 |
| Fig. 2B | `Figure2B_architecture.svg` | 内部模型输入 256 × 256；ASPP 1/2/3；输出五类；分类头在 128 × 128 上生成 logits，再插值回 256 × 256。图下注明人类模型保留 ASPP 6/12/18 和四类输出。 |
| Fig. 2H | `Fig2H_internal_confusion_ASPP123.svg` | 六种子逐行归一化混淆矩阵的均值；上皮召回率精确值 82.09%，图中按原图样式显示为 0.82。 |
| Fig. 2K | `Figure2K_area_selected100.svg` | 100 张入选图像的自动面积与人工面积比较。 |
| Fig. 2L | `Figure2L_area_selected100.svg` | 同一组面积测量的 Bland–Altman 图。 |
| Fig. S2 | `FigS2_internal_tissue_metrics_ASPP123.svg` | 五类 IoU、DSC、Precision 的六种子散点、均值和样本标准差。 |
| Fig. S3 | `FigS3_internal_baseline_comparison_ASPP123.svg` | 最终 B 与原八个 baseline 的 mIoU 比较，各六个种子。 |
| Fig. S5 | `FigS5_internal_confusion_counts_ASPP123.svg` | 保留原图注指定的 seed 1004；替换为该种子最终 B 的原始像素计数。不是六种子平均计数。 |
| 新增 Fig. S22 | `FigureS22_placement_overlays.svg` | 原五组照片重叠效果；保留 94.48%、90.65%、90.32%、89.66%、87.03% 的原记录值及相对几何关系。 |

本地目录保留上述图件的 PDF；GitHub 公开 SVG、数值和绘图代码，不额外上传图包。数值图、示意图和表格的文字、线条、网格及数据点为可编辑矢量元素。
Fig. S22 中的实验照片和照片内原有网格是嵌入的栅格底图，标题、IoU 数值及图例为矢量文字/线条。
该 SVG 不改变照片中的网格相对位置，也不把新的图像显示处理解释为新的重叠度测量。

## 额外提供的 SVG

- `Figure2KL_area_selected100.svg`：2K/2L 合版，便于排版。
- `Internal_tissue_metrics_with_recall_ASPP123.svg`：在 S2 上增加 Recall；若采用，需同步将 S2 图注改为四项指标。
- `Internal_benchmark_table.svg` / `.csv`：九个模型的 mIoU、mDSC、mPrecision、mRecall（mean ± sample SD）。
- `Internal_class_metrics_table.svg` / `.csv`：最终 B 五类 IoU、DSC、Precision、Recall（mean ± sample SD）。

## 可以保留原图或原表的内容

- **Fig. S1**：原界面截图中没有“intensity”一词。修改图注为“stimulation-duration map (minutes)”即可；截图本身可保留。图注中的图像整理描述按回复信修改，不将该历史界面截图标成最终 B 的性能验证。
- **Fig. 2C**：展示参考标注，不是更新后的预测结果；本轮模型调整不要求重画。
- **Fig. 2D–G、2I–J 和 Fig. S4**：属于人类 DFUTissue 结果，本轮内部最终 B 不替换其数值。需要修改的比较措辞或数据集归属在正文/图注中处理。
- **Supplementary Table 1**：四种结构的 DFUTissue 消融结果（原表 mIoU 74.47%、73.24%、73.75%、73.05%），本轮不改数值。Methods 应明确这是 DFUTissue 的结构消融。
- **补充材料另一张表**：动物体重记录，与本轮算法更新无关，保持原表。
- 动物物种标注问题及其他生物实验图不在本轮图件替换范围内。

## 数据口径与对应来源

- 内部图表：`../plotting/internal/data/`。`benchmark_per_seed.csv` 包含九模型 × 六种子 = 54 行；`confusion_counts_by_seed.csv` 包含六个精确的 5 × 5 矩阵。每种子评估 35 张 ROI 图像，共 2,293,760 个像素。
- 2H 是六个行归一化矩阵的算术平均；S5 是 seed 1004 的原始计数，两者不是同一统计层级。S5 对角精确计数依次为 1,924,986、5,026、255,516、50,193、6,449。图中用 k 表示千像素，精确整数保存在 CSV。
- 内部划分固定用于本轮比较；其成员在开发过程中参考过早期测试表现，相关记录保留于数据说明。基准评估采用参考标注定义的 ROI256，各模型口径一致。
- 面积图：`../analysis/wound_area_B_100/`，公开人工/自动面积、标定记录、选图记录、SVG 和计算脚本；原照片、分割掩膜、PDF 留在本地。100 张来自 168 张可匹配图像，按绝对面积误差排序选取；图上统计量针对这 100 张图像。
- 重叠图：`../analysis/placement/overlay_display/`。A/B 对应通过原配准质量检查的两组，C–E 为示例展示。IoU 使用原分析保留的最大重叠配置；并非新增的未经配准直接摆放误差。

## 重绘

在仓库根目录执行：

```bash
python -m pip install -r plotting/requirements.txt
python plotting/rebuild_figures.py
python plotting/verify_figures.py
```

默认重绘使用公开的 CSV/JSON，不需要 GPU 或重新训练；S22 保留已经嵌入照片的 SVG。只有具备本地完整源照片时，才使用 `python plotting/rebuild_figures.py --include-local-photos` 重绘照片版。流程图另附 `../plotting/workflow/Figure2A_workflow.drawio`，结构图另附 YAML 图结构与绘图脚本。
完整 SVG 清单见 `figure_inventory.csv`；检查结果见 `qa/figure_checks.json`。
