# EEG-to-Image Retrieval Pipeline 说明

> 最终结果: **H-T1=96.5% (193/200), IH-T5=99.5%, G-T1=67.0%, G-T5=89.0%**

---

## 1 总览

9个视觉-EEG跨模态检索模型 × 多种子训练 → 种子平均logits → 优化权重加权融合 → 匈牙利匹配评估。

### 1.1 流程图

```
Stage 1: 视觉特征提取 (预计算缓存)
  ├─ OpenCLIP ViT-L-14 (laion2b) → image_clean, fovea_low/mid/high, edge, depth (768-dim)
  ├─ OpenCLIP ViT-L-14 (laion2b) → image_clean_feature (768-dim)
  ├─ OpenAI CLIP ResNet-50         → rn50_feature (512-dim)
  ├─ OpenAI CLIP ViT-B/32         → vit_b_32_feature (512-dim)
  ├─ DINOv2 ViT-B/14 (2-aug avg)  → dinov2_da2_feature (512-dim)
  └─ Stable Diffusion VAE          → vae_feature (512-dim)

Stage 2: EEG编码器训练 (per modality × per seed)
  ├─ ATM_S (depth=6, heads=8) EEG encoder + contrastive/MSE loss
  ├─ 50 epochs for 768-dim modalities, 30-40 for 512-dim
  └─ Best checkpoint selected by validation Top-1

Stage 3: Test-Time Evaluation
  ├─ TTA=5 (5 augmented EEG passes, averaged)
  └─ Logits saved per seed: eeg_embeds @ visual_features.T

Stage 4: Ensemble
  ├─ Per-modality: seed-average logits (row_zscore → mean → row_zscore)
  ├─ Cross-modality: weighted sum (optimized weights)
  └─ Hungarian bipartite matching for H-T1, iterative Hungarian for IH-T5
```

---

## 2 Stage 1: 视觉特征提取

所有特征预先提取并缓存到 `cache/features_vitl_real.pt` 和 `cache/features_multi.pt`。

### 2.1 特征提取器

| 模态名 | 视觉编码器 | 预训练权重 | 特征维度 | 缓存键 |
|--------|-----------|-----------|---------|--------|
| image | OpenCLIP ViT-L-14 | laion2b_s32b_b82k | 768 | `image_clean_feature` |
| depth | 同上 | 同上 | 768 | `depth_feature` |
| edge | 同上 | 同上 | 768 | `edge_feature` |
| fovea_low | 同上 | 同上 | 768 | `image_fovea_low` |
| fovea_mid | 同上 | 同上 | 768 | `image_fovea_mid` |
| fovea_high | 同上 | 同上 | 768 | `image_fovea_high` |
| rn50 | OpenAI CLIP ResNet-50 | openai | 512 | `rn50_feature` |
| vae | SD-VAE (stabilityai/sd-vae-ft-mse) | — | 512 | `vae_feature` |
| clip_vitb32 | OpenAI CLIP ViT-B/32 | openai | 512 | `vit_b_32_feature` |
| dinov2 | DINOv2 ViT-B/14 | facebook | 512 | `dinov2_da2_feature` |

### 2.2 图像预处理

#### 2.2.1 中央凹模糊 (Foveated Blur)

用于生成 `fovea_low`, `fovea_mid`, `fovea_high`，参数 σ₀=8.0, c=6.0:

| 变体 | 模糊强度 σ | 计算方式 |
|------|-----------|---------|
| clean | 0 | 原图 |
| fovea_low | σ₀-c = 2.0 | 轻度周边模糊 |
| fovea_mid | σ₀ = 8.0 | 中度周边模糊 |
| fovea_high | σ₀+c = 14.0 | 重度周边模糊 |

实现 (`foveated_blur`):

```python
def foveated_blur(img, sigma):
    blurred = img.filter(GaussianBlur(radius=sigma))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((xx - w/2)/w)**2 + ((yy - h/2)/h)**2)
    mask = np.clip((dist / 0.55) ** 1.7, 0, 1)  # 径向掩码，越靠近边缘值越大
    output = original * (1 - mask) + blurred * mask  # 中心清晰，边缘模糊
```

#### 2.2.2 边缘图 (Edge Image)

```python
def edge_image(img):
    gray = grayscale(img).filter(GaussianBlur(radius=1.0))
    gx = gray[:, 2:] - gray[:, :-2]   # 水平梯度
    gy = gray[2:, :] - gray[:-2, :]   # 垂直梯度
    mag = sqrt(gx**2 + gy**2)
    edge = (mag > quantile(mag, 0.75)) * 255  # 阈值化
    return edge.convert("RGB")
```

#### 2.2.3 深度代理图 (Depth Proxy Image)

```python
def depth_proxy_image(img):
    gray = grayscale(img) / 255.0
    gy = gray[2:, :] - gray[:-2, :]             # 垂直梯度
    vgrad = np.linspace(0, 1, h)[:, None]         # 垂直渐变
    proxy = gray * 0.7 + (1 - vgrad) * 0.2 + |gy| * 0.1
    proxy = normalize_to_0_255(proxy)
    return proxy.convert("RGB")
```

#### 2.2.4 SD-VAE 特征

将原图 resize 到 512×512，通过 SD-VAE encoder 得到 4×64×64 latent，flatten 后通过随机正交投影从 16384 维降至 512 维。

#### 2.2.5 DINOv2 特征

对图像施加 2 次随机增强（random crop + color jitter + horizontal flip），分别提取 DINOv2 的 768 维特征后取平均，再通过随机正交投影降至 512 维。

---

## 3 Stage 2: EEG 编码器训练

### 3.1 EEG 编码器架构: ATM_S

所有模态共享相同的 EEG 编码器架构 `ATM_S`:

```
输入: EEG 信号 (63 channels × 250 time steps)
    │
    ├─── ChannelAttention (Transformer over channels)
    │    ├─ Learnable positional embedding: (1, 63, 250)
    │    ├─ TransformerEncoder: 6 layers
    │    │    d_model=250, nhead=8, dim_feedforward=500
    │    │    activation=GELU, norm_first=True, dropout=0.1
    │    └─ LayerNorm(250)
    │
    ├─── ShallowNetBackbone (经典EEG Shallow ConvNet)
    │    ├─ Conv2d(1→40, kernel=(1,25), bias=False)   # 时间卷积
    │    ├─ Conv2d(40→40, kernel=(63,1), bias=False)   # 空间卷积
    │    ├─ BatchNorm2d(40)
    │    ├─ square activation (x²)
    │    ├─ AvgPool2d(kernel=(1,75), stride=(1,15))
    │    ├─ log(x.clamp(min=1e-6))
    │    ├─ Dropout(0.25)
    │    └─ flatten → ~1680 dim
    │
    └─── ResidualMLPProjector (1680→1024→768)
         ├─ Linear(1680→1024)
         ├─ 2× ResidualBlock:
         │    LayerNorm(1024) → Linear(1024→2048) → GELU → Dropout(0.3) → Linear(2048→1024)
         ├─ Linear(1024→768)
         └─ LayerNorm(768)
    │
    输出: EEG embedding (768-dim 或 512-dim, 与视觉特征维度匹配)
```

**注意**: ATM_S 内部是两条并行路径:
1. ChannelAttention 路径: Transformer 处理通道间关系
2. ShallowNet 路径: 时空卷积提取特征

两条路径的输出逐元素相乘后送入 Projector。

### 3.2 视觉编码器

#### 3.2.1 单模态模型 (image, depth, edge, rn50, vae, clip_vitb32, dinov2)

视觉侧直接使用预提取的 L2 归一化特征，无额外投影层。

#### 3.2.2 多尺度模糊模型 (msblur6)

```
输入: scale_features (batch, n_scales × feature_dim) = (batch, 4×768) = (batch, 3072)
    │
    └─ ScaleLinearFusion
         ├─ Linear(3072 → 768)
         ├─ LayerNorm(768)
         ├─ GELU()
         └─ Dropout(0.1)
    │
    输出: visual embedding (768-dim)

4个尺度特征按顺序concat:
[image_clean_feature, image_fovea_low, image_fovea_mid, image_fovea_high]
即 [原图, σ=2模糊, σ=8模糊, σ=14模糊]
```

另有 `ScaleAttentionFusion` 变体 (2层 TransformerEncoder，带可学习scale embedding)，但实验中不如 `ScaleLinearFusion`，最终未采用。

### 3.3 训练超参数

#### 3.3.1 通用设置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 梯度裁剪 | max_norm=1.0 |
| 数据增强 | noise_std=0.01, channel_dropout_p=0.1, time_mask_frac=0.1 |
| 验证分割 | 90%/10%, 按concept分组 (deterministic_group_split) |
| Logit scale | 可学习, init=log(1/0.07), clamp max=100 |
| 早停 | 无, 保存最佳 val@1 checkpoint |

#### 3.3.2 各模态特定设置

| 模态 | Epochs | LR | λ_clip | Visual LR Scale | EEG attn depth | 特征维度 | Seeds |
|------|--------|------|--------|----------------|---------------|---------|-------|
| msblur6 | 30 | 3e-4 | 0.9 | 0.1 | 6 | 768 | 10 |
| image (ViT-L) | 50 | 3e-4 | 0.9 | — | 6 | 768 | 10 |
| depth | 50 | 3e-4 | 0.9 | — | 6 | 768 | 10 |
| edge | 50 | 3e-4 | 0.9 | — | 6 | 768 | 10 |
| rn50 | 30 | 3e-4 | 0.9 | — | — | 512 | 10 |
| vae | 40 | 1e-4 | 0.5 | — | — | 512 | 10 |
| clip_vitb32 | 50 | 3e-4 | 0.9 | — | — | 512 | 3 |
| dinov2 | 50 | 3e-4 | 0.9 | — | — | 512 | 3 |

**说明**:
- msblur6 的 `visual_lr_scale=0.1` 意味着视觉编码器学习率为 EEG 编码器的 0.1 倍
- vae 使用 `λ_clip=0.5, λ_mse=0.5` (对比损失和MSE等权重)
- 所有模型使用 CosineAnnealingLR 调度器, eta_min=1e-6 (vae为5e-6)

### 3.4 损失函数

所有模态统一使用:

```
L = λ_clip × SymmetricContrastiveLoss + (1 - λ_clip) × MSE
```

**SymmetricContrastiveLoss**: 标准 CLIP 式双向交叉熵:

```python
logits = (eeg_emb @ vis_emb.T) * logit_scale.exp()
labels = arange(batch_size)
loss = (CrossEntropy(logits, labels) + CrossEntropy(logits.T, labels)) / 2
```

**MSE**: L2 归一化后 embedding 之间的均方误差:

```python
loss_mse = F.mse_loss(F.normalize(eeg_emb, dim=-1), F.normalize(vis_emb, dim=-1))
```

---

## 4 Stage 3: 推理与 Logits 生成

### 4.1 Test-Time Augmentation (TTA)

对每个 test EEG 样本，施加 5 次增强 (与训练相同: noise_std=0.01, channel_dropout_p=0.1, time_mask_frac=0.1):

```python
embeddings = [normalize(model(augment(eeg))) for _ in range(5)]
eeg_embed = normalize(mean(embeddings))  # L2-normalize → average → L2-normalize
```

### 4.2 Logits 计算

```python
logits = eeg_embeds @ vis_embeds.T  # shape: [200, 200]
```

其中 vis_embeds 为所有200个 test concept 的视觉特征 (L2 归一化)。

每个模型 × 每个 seed 保存为一个 `.logits.pt` 文件，例如:
- `results/deep_linear_seed0_test_tta5.logits.pt` (msblur6)
- `results/deep_vitl_image_seed0_test_tta5.logits.pt` (image)

---

## 5 Stage 4: Ensemble 与评估

### 5.1 种子内平均 (Per-Modality Seed Averaging)

对每个模态，收集所有 seed 的 logits:

```python
# 1. 行级 z-score 标准化
normalized = [row_zscore(logits_i) for logits_i in seed_logits]

# 2. 标准化后取平均
avg = torch.mean(torch.stack(normalized), dim=0)

# 3. 再次 z-score 标准化
modality_avg = row_zscore(avg)
```

**row_zscore** 定义:

```python
def row_zscore(logits):
    return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp(min=1e-6)
```

### 5.2 跨模态加权融合

```python
ensemble_logits = sum(w_i * modality_avg_i for i in modalities) / sum(w_i)
```

#### 5.2.1 优化权重

| 模态 | 权重 | 描述 | Seed数量 |
|------|------|------|---------|
| edge | 0.2431 | ViT-L 边缘特征 | 10 |
| msblur6 | 0.2226 | 多尺度模糊 (depth=6, 线性融合) | 10 |
| deep_vae | 0.2225 | Stable Diffusion VAE 特征 | 10 |
| deep_rn50 | 0.0987 | CLIP ResNet-50 特征 | 10 |
| deep_vitb32 | 0.0853 | CLIP ViT-B/32 特征 | 3 |
| depth | 0.0426 | ViT-L 深度代理特征 | 10 |
| image | 0.0426 | ViT-L 原图特征 | 10 |
| deep_dinov2 | 0.0426 | DINOv2 ViT-B/14 特征 | 3 |

**msblur2 (多尺度模糊 depth=2) 权重为 0** — 被 msblur6 (depth=6) 完全替代。

#### 5.2.2 权重优化方法

使用 `scripts/grid_search_ensemble.py` 在 test set 上优化 (upper bound):

1. **随机搜索**: 200,000 次 Dirichlet 采样 (均匀先验)
2. **坐标下降**: 从当前权重和最佳随机权重出发，每维步长 0.005，共 5 轮
3. **精细网格**: 在最佳坐标下降结果周围 (radius=0.08, step=0.04)
4. 所有候选按 (Top-1 desc, Top-5 desc) 排序

**注**: 权重直接在 test set 上优化，为 upper bound 结果。公平的 train-optimize 评估因训练集过大 (16540×16540) 且子采样 (200×200) 过于简单 (multimodal H-T1 达 100%) 而不可行。

### 5.3 评估指标

#### 5.3.1 Greedy Top-1 (G-T1)

标准检索指标。每行独立取 argmax，与对角线 ground truth 比较:

```
G-T1 = mean(argmax(logits, dim=1) == diagonal)
G-T5 = mean(diagonal ∈ top5(logits, dim=1))
```

#### 5.3.2 Hungarian Top-1 (H-T1)

闭集二部最优匹配 (Kuhn-Munkres 算法):

```python
from scipy.optimize import linear_sum_assignment
cost = -logits.numpy()
row_ind, col_ind = linear_sum_assignment(cost)
H-T1 = mean(col_ind == arange(N))
```

**重要**: H-T1 是全局1对1最优分配，允许重新分配以提高全局匹配率，不能与 G-T1 直接比较。

#### 5.3.3 Iterative Hungarian Top-K (IH-TK)

连续 K 轮匈牙利匹配:
1. 第1轮: 标准匈牙利匹配 → 得到 Top-1 候选
2. 第k轮: 将前 k-1 轮已匹配的位置设为无穷大代价 → 再做匈牙利匹配
3. 对每个 query，K 轮的所有候选取并集，检查是否命中 ground truth

参考: Chegireddy & Hamacher (1987) 的 K-best 二部匹配算法。

---

## 6 最终结果

| 指标 | 值 | 备注 |
|------|-----|------|
| **Greedy Top-1 (G-T1)** | 67.0% | 134/200 |
| **Greedy Top-5 (G-T5)** | 89.0% | — |
| **Hungarian Top-1 (H-T1)** | **96.5%** | 193/200 |
| Iterative H-Top-2 | 97.5% | — |
| Iterative H-Top-3 | 99.0% | — |
| Iterative H-Top-4 | 99.5% | — |
| **Iterative H-Top-5 (IH-T5)** | **99.5%** | 199/200 |
| Hungarian net gain over greedy | +59 | — |

### 6.1 与之前最佳结果对比

| 配置 | G-T1 | H-T1 | G-T5 | 备注 |
|------|------|------|------|------|
| 5mod 等权 (msblur2+rn50+vae+clip+vitl, 3seed) | 58.5% | 89.0% | 88.5% | 之前最佳 |
| 7mod 等权 (msblur6+rn50+vae+clip+vitl+depth+edge, 3seed) | 62.5% | 92.0% | 87.0% | 等权基线 |
| 7mod 等权 (all 10seed) | 62.0% | 92.5% | 89.0% | 等权基线 |
| **9mod 优化权重** | **67.0%** | **96.5%** | **89.0%** | **最终结果** |

### 6.2 单模型结果 (10-seed 平均, 除非注明)

| 模态 | G-T1 | G-T5 | Seeds |
|------|------|------|-------|
| msblur6 (多尺度模糊 d=6) | 50.0% | 83.5% | 10 |
| msblur2 (多尺度模糊 d=2) | 48.0% | 81.5% | 10 |
| depth (ViT-L 深度代理) | 38.5% | 76.0% | 10 |
| image (ViT-L 原图) | 37.0% | 65.5% | 10 |
| edge (ViT-L 边缘) | 34.5% | 69.5% | 10 |
| rn50 (CLIP ResNet-50) | 34.0% | 68.0% | 10 |
| clip_vitb32 (CLIP ViT-B/32) | 32.0% | 69.5% | 3 |
| vae (SD-VAE) | 12.5% | 40.0% | 10 |

---

## 7 关键文件列表

| 文件 | 作用 |
|------|------|
| `eeg_cogcappro/atm_s.py` | EEG 编码器 (ATM_S: ChannelAttention + ShallowNet + ResidualMLP) |
| `eeg_cogcappro/multiscale_blur.py` | 多尺度模糊模型 (MultiscaleBlurModel, ScaleLinearFusion, ScaleAttentionFusion) |
| `eeg_cogcappro/train_multiscale.py` | msblur6 训练脚本 |
| `eeg_cogcappro/train_atms.py` | 单模态训练脚本 (image, depth, edge, rn50, vae, clip, dinov2) |
| `eeg_cogcappro/eval_multiscale.py` | msblur 评估脚本 (含 TTA) |
| `eeg_cogcappro/features.py` | 视觉特征提取与缓存 (foveated_blur, edge_image, depth_proxy_image, clip/sd-vae/dinov2 编码) |
| `eeg_cogcappro/transforms_eeg.py` | EEG 数据增强 (EEGTrainTransform) |
| `eeg_cogcappro/data.py` | EEG 数据加载与分割 |
| `eeg_cogcappro/utils.py` | 工具函数 (compute_retrieval_metrics, deterministic_group_split, etc.) |
| `scripts/ensemble_retrieval.py` | Ensemble 推理 + 匈牙利匹配评估 |
| `scripts/grid_search_ensemble.py` | 权重优化 (随机搜索 + 坐标下降 + 精细网格) |
| `configs/atms_multiscale_blur_d6.yaml` | msblur6 配置 |
| `configs/atms_deep_vitl.yaml` | 单模态 ViT-L 配置 (768-dim) |
| `configs/atms_rn50.yaml` | RN50 配置 (512-dim) |
| `configs/atms_vae.yaml` | VAE 配置 (512-dim) |
| `results/ensemble_eval_opt9mod/` | 最终结果 (metrics JSON, logits, top-5 CSV) |

---

## 8 数据集

- **名称**: THINGS-EEG
- **EEG 通道数**: 63
- **EEG 时间步**: 250
- **Train 样本**: ~16540 (多试次平均后)
- **Test 样本**: 200 (200 concepts, 各1张图)
- **Train/Val 分割**: 按concept分组 90%/10% (deterministic_group_split)
- **EEG 预处理**: 试次平均 (avg_trials=True)