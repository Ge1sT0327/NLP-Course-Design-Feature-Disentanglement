# 特征解耦（Feature Disentanglement）—— 基于 β-VAE 的实现与可视化

> 自然语言处理课程设计 · 第六组
> 核心模型：β-VAE（β-Variational Autoencoder）· 实验平台：PyTorch

本项目完整复现了特征解耦（Feature Disentanglement）的核心思想：**通过 β-VAE
强化隐空间的正则约束，迫使模型将「数字类别、笔画粗细、旋转角度」等相互纠缠的
潜在变化因子分配到不同的隐变量维度上**，并给出定性与定量的验证。

---

## 1. 背景：深度学习的“黑箱”困境

深度学习模型虽然能从海量数据中自动提取复杂特征，但这些特征往往是高度**耦合**
（entangled）的——单个维度可能同时混合了类别、粗细、旋转、位置等多种信息，导致：

- **难解释**：决策逻辑像“黑箱”，无法理解模型内部在表征什么；
- **难控制**：无法精确操控生成结果（想只改“表情”却连“身份”一起变）。

**特征解耦**旨在打破这种纠缠，追求更结构化、更具语义的表示：让不同维度尽量
一一对应不同的独立变化因子（如 `z1 = 数字类别`、`z2 = 笔画粗细`、`z3 = 旋转角度`），
从而带来**可解释、可控制、可迁移**三大价值。

## 2. 方法：从 VAE 到 β-VAE

| 模型 | 关键思想 | 目标函数 | 局限 / 优势 |
| ---- | -------- | -------- | ----------- |
| 自编码器（AE） | Encoder + Decoder，最小化重构损失 | `L = L_recon` | 隐空间不连续、无结构，难解耦 |
| 变分自编码器（VAE） | 引入概率生成，隐变量服从先验分布 | `L = L_recon + KL` | 隐空间连续规整，为解耦奠定基础 |
| **β-VAE** | 给 KL 项乘系数 β，强化约束 | `L = L_recon + β·KL` | **β>1 时迫使因子分配到不同维度，实现解耦** |

β-VAE 的核心改动只有一个超参数 β：

```
Loss_β-VAE = 重构损失(Reconstruction Loss) + β × KL散度约束
```

当 `β > 1` 时，模型为了满足更强的隐空间规整性约束，被迫以更“结构化”的方式编码
信息，从而将不同语义因子“逼”到不同维度。本项目直接验证了这一点。

## 3. 项目结构

```
NLP-Course-Design-Feature-Disentanglement/
├── config.py            # 全局超参数配置（β、隐变量维度、训练轮数等）
├── model.py             # β-VAE 模型（CNN 编码器 + 转置卷积解码器）
├── dataset.py           # FactorMNIST 数据集（在 MNIST 上叠加旋转/粗细因子）
├── metrics.py           # 解耦指标：MIG + 因子探针
├── train.py             # 训练脚本
├── evaluate.py          # 评估与可视化脚本
├── requirements.txt     # 依赖
├── outputs/             # 训练产物（权重 + 可视化 + 指标）
│   ├── beta_vae.pt          # 训练好的模型权重
│   ├── history.json         # 训练曲线数据
│   ├── metrics.json         # 定量指标
│   ├── reconstructions.png  # 原始 vs 重构
│   ├── latent_traversal.png # 逐维度隐变量遍历（核心图）
│   ├── factor_traversal.png # 旋转/粗细/类别因子的遍历
│   ├── factor_importance.png# 因子探针重要度热力图
│   ├── latent_space.png     # PCA + t-SNE 隐空间
│   └── training_curves.png  # 训练曲线
└── README.md
```

## 4. 环境安装

- Python ≥ 3.9，推荐配合 CUDA 使用（CPU 也可运行，仅更慢）。

```bash
pip install -r requirements.txt
```

依赖：`torch`、`torchvision`、`numpy`、`scikit-learn`、`matplotlib`。
MNIST 数据集会在首次运行时自动下载到 `data/` 目录。

## 5. 快速开始

### 5.1 训练

```bash
# 使用默认参数（β=4.0，隐变量 20 维，40 epochs）
python train.py

# 自定义 β 和训练轮数
python train.py --beta 8 --epochs 30

# 快速冒烟测试（1 个 epoch）
python train.py --quick
```

训练结束后生成 `outputs/beta_vae.pt`（模型权重）与 `outputs/history.json`（训练曲线）。

### 5.2 评估与可视化

```bash
python evaluate.py
```

脚本会一次性生成全部可视化图片，并输出定量指标到 `outputs/metrics.json`。

## 6. 实验结果

### 6.1 定性验证：Latent Traversal（隐变量遍历）

固定其余维度、只改变单个隐变量维度，观察生成结果的变化——这是验证特征解耦最直观
的方法。若某一维度只改变“笔画粗细”而其余不变，则说明该维度独立编码了“粗细”因子。

`outputs/latent_traversal.png` 展示了全部 20 个维度的遍历结果，其中部分维度明显独立
对应旋转 / 粗细 / 类别因子；`outputs/factor_traversal.png` 进一步放大了旋转、粗细、
类别三个因子各自的最优维度。

### 6.2 定量验证：MIG 与因子探针

- **MIG（Mutual Information Gap）**：衡量最重要维度与次重要维度之间的互信息差距，
  越大说明解耦越彻底。
- **因子探针（Factor Probe）**：用线性回归 / 线性分类器从隐变量预测已知因子，
  R² / 准确率越高、且重要度集中在少数维度上，说明该因子被“干净地”编码。

> 具体数值见 `outputs/metrics.json`，以下为本次训练的实测结果：

| 指标 | 数值 | 对应最优维度 | 解读 |
| ---- | ---- | ------------ | ---- |
| MIG（数字类别） | 0.0231 | — | 无监督下类别因子解耦较弱，符合“完全解耦的难度” |
| MIG（笔画粗细） | 0.2265 | — | 粗细因子被显著解耦，远高于类别因子 |
| 旋转因子探针 R² | 0.4836 | `z12` | 旋转信息被较好地编码到单一维度 |
| 粗细因子探针 R² | **0.8361** | `z2` | 粗细因子几乎被完整、干净地编码，解耦最成功 |
| 类别因子分类准确率 | **82.39%** | `z4` | 类别信息高度集中，但无监督下仍与风格耦合 |

**结论**：通过因子探针可以定量看到，隐空间的 **`z2` 主要编码“笔画粗细”、`z12` 主要编码
“旋转角度”、`z4` 主要编码“数字类别”**——三个原本纠缠的因子被分配到了不同维度，直观地
复现了课程设计 PPT 中 “改第 5 维→粗细变化、改第 7 维→旋转变化” 的 Latent Traversal 现象。
同时，类别因子 MIG 明显低于粗细因子，也印证了 β-VAE 无监督解耦“并非必然成功、受因子
天然相关性制约”这一已知挑战（对应 PPT 中“面临的挑战”一节）。

### 6.3 隐空间结构

`outputs/latent_space.png` 展示了隐空间的 PCA 与 t-SNE 投影：不同数字类别在隐空间中
形成相对分离的簇，说明类别信息被较好地结构化表示。

## 7. 核心代码说明

- **重参数化技巧**（`model.py` 的 `reparameterize`）：`z = μ + ε·exp(0.5·log σ²)`，
  使采样过程可反向传播。
- **β 系数**（`model.py` 的 `BetaVAE.loss`）：`L = L_recon + β·KL`，β 是解耦强度的唯一旋钮。
- **FactorMNIST**（`dataset.py`）：用 `max_pool2d` 近似形态学膨胀模拟“笔画变粗”，
  用 `torchvision.transforms.functional.rotate` 施加旋转，从而获得带已知因子的数据。

## 8. 参考文献

1. Higgins, I., et al. *β-VAE: Learning Basic Visual Concepts with a Constrained
   Variational Framework.* ICLR, 2017.
2. Kim, H., & Mnih, A. *Disentangling by Factorising.* PMLR, 2018.
3. Chen, R. T. Q., et al. *Isolating Sources of Disentanglement in Variational
   Autoencoders.* NeurIPS, 2018.
4. Google Research. [disentanglement_lib](https://github.com/google-research/disentanglement_lib).
5. YannDubs. [disentangling-vae](https://github.com/YannDubs/disentangling-vae).
6. Olah, C., et al. *Feature Visualization.* Distill, 2017.
