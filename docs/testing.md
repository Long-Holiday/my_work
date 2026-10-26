# 快速测试代码使用说明

以更新后的 `model_test.md` 为准：这是单卡两周时间盒内的可行性检查，不是投稿级完整实验。所有命令在仓库根目录运行，使用 `pytorch` conda 环境；本机只运行单元测试及合成数据的单步冒烟检查。

## 1. 本机检查

```bash
conda run -n pytorch python -m unittest discover -s tests -v
conda run -n pytorch python -m roadflow.cli stage0 --output runs/stage0-synthetic
```

研究默认使用标准宽度 ResNet-18 的 ImageNet 预训练权重，并在模型内做 ImageNet normalization。首次训练会通过 torchvision 获取权重，离线机器需预先放入 torch 缓存。`--no-pretrained` 仅供无需下载的本地冒烟检查。旧半宽 checkpoint 可以加载，但不符合新版底座协议，不能与新模型混做公平对照。

## 2. 三组空间划分与数据适用性

```bash
conda run -n pytorch python -m roadflow.cli prepare \
  --root SpaceNet3_AOI_2_Vegas --output prepared/vegas-v2 \
  --train-limit 400 --dev-limit 60 --holdout-limit 40 \
  --spatial-buffer-tiles 1 --crop-size 256 --seed 0
conda run -n pytorch python -m roadflow.cli stage0 \
  --manifest prepared/vegas-v2/manifest.json --output runs/stage0-real --limit 8
```

必须重新准备旧的 train/val 数据；新实验使用 train/dev/holdout，不能把旧 val 同时当开发和最终测试。按原始影像的地理区域先划分，划分间保留空间缓冲再裁块，数量不足如实报告，不以相邻样本凑数。

准备步骤缓存 GT 查询池，训练时从池中抽样，同组候选共享查询。开发评价使用独立查询种子。适用性报告包含查询数、路径长度、交叉点/连通分量/环路，以及道路复杂度分布；完整可见比例须查看具体定义和可测范围，不把“端点在块内”等同于“原生全图路径可见”。若256块主要为单直路，应先生成独立的512方案再决定是否训练，不能训练后在留出集挑裁块。

## 3. 底座、判别式参照与采样扫描

下面训练命令供训练机器使用，本机不执行这些预算。

```bash
conda run -n pytorch python -m roadflow.cli train \
  --manifest prepared/vegas-v2/manifest.json --output runs/b0 \
  --steps 3000 --batch-size 4 --eval-every 100 --apls-every 300 --device cuda
conda run -n pytorch python -m roadflow.cli train \
  --manifest prepared/vegas-v2/manifest.json --output runs/unet \
  --steps 3000 --batch-size 4 --eval-every 100 --device cuda --discriminative
conda run -n pytorch python -m roadflow.cli scan \
  --manifest prepared/vegas-v2/manifest.json --checkpoint runs/b0/best.pt \
  --output runs/sampling-scan --sample-steps 16 --noise-levels .1 .2 .3 --device cuda
```

验证只用 dev；记录固定间隔和预算1/3、2/3、末尾节点，保存 `step_*.pt`、`last.pt`、`validation.jsonl`。底座 `best.pt` 用预先固定的 Dice+BF1+path 联合开发分排序，不声称该分数证明收敛或等价官方拓扑指标。应人工查看结构有效性、匹配失败和不可达比例以及开发曲线平台期。

`scan` 测 ODE 8/16/32，以及固定步数的三档 SDE 噪声，记录几何/路径、短末端分支密度、小连通域面积、16候选耗时和 CUDA 峰值显存。CPU显存字段为 null；短支路不等于错误毛刺。选定主采样器后，所有方法单候选评价使用相同 `--sample-steps` 与 `--noise`，H1 覆盖 GRPO 的 SDE。

## 4. H1：开发集几何合格候选

```bash
conda run -n pytorch python -m roadflow.cli candidates \
  --manifest prepared/vegas-v2/manifest.json --checkpoint runs/b0/best.pt \
  --output runs/h1 --split dev --group-size 16 --sample-steps 16 --noise .2 \
  --reference-samples 3 --delta-dice .005 --delta-bf1 .01 --device cuda
```

参考几何取独立固定种子的少量采样均值，不从候选池择优。合格门槛默认 Dice 下降最多0.005、BF1最多0.01，写入缓存，不得为了扩大 oracle 差距事后放宽。记录合格率、合格池平均/最优 APLS、奖励选优 APLS 及相对池均值增益；不足两个不计算内部差距。按已有结构类别分层统计，不自动伪造“遮挡”标签。

官方 APLS 使用 `--apls-adapter package.module:function`，接口为：

```python
def score(gt_mask, predicted_mask, metadata) -> float:
    # metadata 包括 id、source_id、manifest，关联原生道路图。
    # 必须调用独立、锁定版本的官方实现。
    ...
```

没有适配器时输出 null 与 `blocked_missing_official_apls`，不能用 sampled path 代替官方 APLS 宣布 H1 成立。已有 APLS 但合格候选不足时单列证据不足。诊断不使用未经验证的统一相关系数阈值，也不自动判定 idea 失败。允许一次有依据的配置修正，超预算记录证据不足。

## 5. 同预算 B1、B3、B3-random

必须在训练集重新生成候选，使用全训练集，不得复用开发候选。

```bash
conda run -n pytorch python -m roadflow.cli candidates \
  --manifest prepared/vegas-v2/manifest.json --checkpoint runs/b0/best.pt \
  --output runs/train-pool --split train --group-size 16 --sample-steps 16 --noise .2 --device cuda
conda run -n pytorch python -m roadflow.cli train \
  --manifest prepared/vegas-v2/manifest.json --checkpoint runs/b0/best.pt \
  --output runs/b1 --steps 300 --batch-size 4 --seed 0 --eval-every 100 --device cuda
conda run -n pytorch python -m roadflow.cli distill \
  --manifest prepared/vegas-v2/manifest.json --checkpoint runs/b0/best.pt \
  --candidate-cache runs/train-pool/candidates.json --selection reward \
  --output runs/b3 --steps 300 --batch-size 4 --seed 0 --eval-every 100 --device cuda
conda run -n pytorch python -m roadflow.cli distill \
  --manifest prepared/vegas-v2/manifest.json --checkpoint runs/b0/best.pt \
  --candidate-cache runs/train-pool/candidates.json --selection random \
  --output runs/b3-random --steps 300 --batch-size 4 --seed 0 --eval-every 100 --device cuda
```

三者冻结同一编码器，使用相同优化器、学习率、更新数、batch-size、seed和逐步影像安排。B3在合格池取总奖励最高的一个；random从同池等概率固定选择一个。无合格候选两组均回退GT，`selection.json`记录候选索引和回退率。拒绝不完整缓存、非训练缓存、几何容忍量不一致。旧 `--top-k` 已删除，以免改变本轮对照含义。

示例省略主采样参数时沿用ODE16评价，正式比较需显式给三者同一主采样参数。B3持平不是GRPO的硬停止条件。

## 6. 最小 GRPO

```bash
conda run -n pytorch python -m roadflow.cli grpo \
  --manifest prepared/vegas-v2/manifest.json --checkpoint runs/b0/best.pt \
  --output runs/grpo --steps 300 --group-size 4 --sample-steps 16 --noise .2 \
  --lr 1e-5 --beta .01 --eval-every 50 --apls-every 150 --device cuda
```

仅允许G=4/8与8/16步，冻结编码器，每次更新前检查同轨迹概率比；确定性末步不参与高斯策略损失。组内标准差低于1e-4时优势置零，防止浮点噪声被放大。低方差类型、分量统计、几何合格率、KL、clipping和成本写入日志。

开发集独立查询奖励与训练查询分开。官方APLS只在 `--apls-every` 预定节点与末尾运行，不进入训练奖励。配置官方APLS后，GRPO仅在开发Dice/BF1不越固定底座容忍量时保存最高APLS为`best.pt`。无APLS或无达标checkpoint不伪造最佳结果。

## 7. 最终留出确认

先锁定开发集选出的checkpoint、采样器、阈值、几何门槛和比较方法，再执行一次：

```bash
conda run -n pytorch python -m roadflow.cli confirm \
  --manifest prepared/vegas-v2/manifest.json --output runs/final-confirmation \
  --models B0=runs/b0/best.pt B1=runs/b1/best.pt B3=runs/b3/best.pt GRPO=runs/grpo/best.pt \
  --sample-steps 16 --noise .2 --inference-seeds 0 1 2 --confirm-holdout --device cuda
```

必须恰有三个不同的固定推理种子，完整评价holdout，分别单候选，不进行集成或择优。`frozen_protocol.json`保存checkpoint/manifest哈希及协议，已有同一路径的协议不允许更改。`confirmation.json`对每个模型相对B0/B1按种子、原始影像聚合配对差值；缺APLS不下结论。各次evaluation保留耗时和显存。不要换目录反复调参后仍将该留出集称为独立测试。

## 尚需接入或人工判断

- 官方 APLS 仍需独立适配器与版本锁定；代码不将本地奖励冒充官方指标。
- SDE仍为线性FM score恒等式加漂移修正的实验实现，尚未逐步对齐固定版本官方Flow-GRPO；不能声称完整复现。
- 图压缩保留折线和长度；交叉点采用逻辑聚合并保留内部端口，避免把路口内部距离直接归零。它不等同于把整个交叉区合并为单个无长度节点。
- 查询分层当前按单图长度，中心线precision为骨架像素采样，尚非物理等距折线采样。
- `applicability.json`的完整可见比例是裁块内mask骨架路径不接触最外圈的代理，不能检测全图最短路离开裁块再返回；该代理不是原生全图完整可见比例。正式训练前仍需原生图上下文检查，不能仅据此字段宣布数据适用性通过。
- 空间划分、场景复杂度和可见性统计应检查真实数据。原始矢量图的立交连接、裁块表示误差无法只凭mask单元测试排除。
- 平台期、一次修正预算、跨种子一致改善以及原始失败案例仍由实验者判断；程序不把一次配置失败自动归因为idea失败。
- 未执行完整训练、第二数据集或多训练种子实验；本机测试不触发预训练权重下载。
