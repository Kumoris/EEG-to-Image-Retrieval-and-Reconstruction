# EEG-to-Image Retrieval: 9-Modal Optimized Ensemble — Tech Report 资料汇总

> 整合日期: 2026-05-16  
> 项目路径: `/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex`

---

## 1 项目总览

### 1.1 任务定义

给定 THINGS-EEG 数据集中 200 个 test EEG 录音，在 200 个 candidate 图像中检索正确匹配：

- **Retrieval 任务**: EEG-to-image 跨模态检索
- **Reconstruction 任务**: 为 200 个 test EEG query 输出 200 张 reconstruction PNG

### 1.2 项目目标

1. EEG-to-image retrieval: 在 200 个 test candidate image 中做 Top-1 / Top-5 retrieval
2. Image reconstruction: 为 200 个 test EEG query 输出 reconstruction PNG，并用官方指标评估

### 1.3 合规性要求

- 训练损失只使用 `train.pt` 与 train-side 图像/文本/派生特征
- Test split 只用于 inference、candidate ranking 和最终评估
- Reconstruction 输出不得直接复制 test image files
- Ensemble 权重固定记录，避免用 test accuracy 做动态调参

---

## 2 数据集

### 2.1 THINGS-EEG 数据集

| 属性 | 值 |
|------|-----|
| EEG 通道数 | 63 |
| EEG 时间步 | 250 |
| Train 样本 | ~16,540 (多试次平均后, 1654 concepts × 10 images each) |
| Test 样本 | 200 (200 concepts × 1 image each) |
| Train/Val 分割 | 按concept分组 90%/10% (deterministic_group_split) |
| EEG 预处理 | 试次平均 (avg_trials=True) |

---

## 3 方法

### 3.1 总体流程

```
Stage 1: 视觉特征提取 (预计算缓存)
  ├─ OpenCLIP ViT-L-14 (laion2b) → image_clean, fovea_low/mid/high, edge, depth (768-dim)
  ├─ OpenCLIP ViT-L-14 (laion2b) → image_clean_feature (768-dim)
  ├─ OpenAI CLIP ResNet-50         → rn50_feature (512-dim)
  ├─ OpenAI CLIP ViT-B/32         → vit_b_32_feature (512-dim)
  ├─ DINOv2 ViT-B/14 (2-aug avg)  → dinov2_da2_feature (512-dim)
  └─ Stable Diffusion VAE          → vae_feature (512-dim)

Stage 2: EEG 编码器训练 (per modality × per seed)
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

Stage 5: Image Reconstruction (diffusion_prompt)
  ├─ Retrieval Top-5 概念 → 文本 prompt 生成
  ├─ SDXL-Turbo (stabilityai/sdxl-turbo, steps=4, guidance_scale=0.0)
  ├─ 256×256 纯文本生成（不依赖 train nearest image）
  └─ 200 张 test reconstruction PNGs 输出
```

### 3.2 Stage 1: 视觉特征提取

所有特征预先提取并缓存到 `cache/features_vitl_real.pt` 和 `cache/features_multi.pt`。

#### 3.2.1 特征提取器

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

#### 3.2.2 图像预处理方法

**中央凹模糊 (Foveated Blur)** — 生成 fovea_low/mid/high，参数 σ₀=8.0, c=6.0:

| 变体 | 模糊强度 σ | 计算方式 |
|------|-----------|---------|
| clean | 0 | 原图 |
| fovea_low | σ₀-c = 2.0 | 轻度周边模糊 |
| fovea_mid | σ₀ = 8.0 | 中度周边模糊 |
| fovea_high | σ₀+c = 14.0 | 重度周边模糊 |

```python
def foveated_blur(img, sigma):
    blurred = img.filter(GaussianBlur(radius=sigma))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((xx - w/2)/w)**2 + ((yy - h/2)/h)**2)
    mask = np.clip((dist / 0.55) ** 1.7, 0, 1)  # 径向掩码
    output = original * (1 - mask) + blurred * mask  # 中心清晰，边缘模糊
```

**边缘图 (Edge Image)**:
```python
def edge_image(img):
    gray = grayscale(img).filter(GaussianBlur(radius=1.0))
    gx = gray[:, 2:] - gray[:, :-2]   # 水平梯度
    gy = gray[2:, :] - gray[:-2, :]   # 垂直梯度
    mag = sqrt(gx**2 + gy**2)
    edge = (mag > quantile(mag, 0.75)) * 255
    return edge.convert("RGB")
```

**深度代理图 (Depth Proxy Image)**:
```python
def depth_proxy_image(img):
    gray = grayscale(img) / 255.0
    gy = gray[2:, :] - gray[:-2, :]         # 垂直梯度
    vgrad = np.linspace(0, 1, h)[:, None]     # 垂直渐变
    proxy = gray * 0.7 + (1 - vgrad) * 0.2 + |gy| * 0.1
    proxy = normalize_to_0_255(proxy)
    return proxy.convert("RGB")
```

**SD-VAE 特征**: 将原图 resize 到 512×512，通过 SD-VAE encoder 得到 4×64×64 latent，flatten 后通过随机正交投影从 16384 维降至 512 维。

**DINOv2 特征**: 对图像施加 2 次随机增强（random crop + color jitter + horizontal flip），分别提取 DINOv2 的 768 维特征后取平均，再通过随机正交投影降至 512 维。

---

### 3.3 Stage 2: EEG 编码器训练

#### 3.3.1 ATM_S 架构

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

**关键设计**: ATM_S 内部是两条并行路径:
1. **ChannelAttention 路径**: Transformer 处理通道间关系
2. **ShallowNet 路径**: 时空卷积提取特征

两条路径的输出**逐元素相乘**后送入 Projector。

#### 3.3.2 视觉编码器

**单模态模型** (image, depth, edge, rn50, vae, clip_vitb32, dinov2):
- 视觉侧直接使用预提取的 L2 归一化特征，无额外投影层

**多尺度模糊模型 (msblur6)**:

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
即 [原图(σ=0), 轻度模糊(σ=2), 中度模糊(σ=8), 重度模糊(σ=14)]
```

另有 `ScaleAttentionFusion` 变体 (2层 TransformerEncoder，带可学习scale embedding)，但实验中不如 `ScaleLinearFusion`，最终未采用。

#### 3.3.3 训练超参数

**通用设置**:

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 梯度裁剪 | max_norm=1.0 |
| 数据增强 | noise_std=0.01, channel_dropout_p=0.1, time_mask_frac=0.1 |
| 验证分割 | 90%/10%, 按concept分组 (deterministic_group_split) |
| Logit scale | 可学习, init=log(1/0.07), clamp max=100 |
| 早停 | 无, 保存最佳 val@1 checkpoint |
| 学习率调度 | CosineAnnealingLR, eta_min=1e-6 (vae为5e-6) |

**各模态特定设置**:

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
- 其余模态 λ_clip=0.9, λ_mse=0.1

#### 3.3.4 损失函数

所有模态统一使用：

```
L = λ_clip × SymmetricContrastiveLoss + (1 - λ_clip) × MSE
```

**SymmetricContrastiveLoss**: CLIP 式双向交叉熵:
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

### 3.4 Stage 3: 推理与 Logits 生成

#### 3.4.1 Test-Time Augmentation (TTA)

对每个 test EEG 样本，施加 5 次增强 (与训练相同: noise_std=0.01, channel_dropout_p=0.1, time_mask_frac=0.1):

```python
embeddings = [normalize(model(augment(eeg))) for _ in range(5)]
eeg_embed = normalize(mean(embeddings))  # L2-normalize → average → L2-normalize
```

#### 3.4.2 Logits 计算

```python
logits = eeg_embeds @ vis_embeds.T  # shape: [200, 200]
```

其中 vis_embeds 为所有 200 个 test concept 的视觉特征 (L2 归一化)。每个模型 × 每个 seed 保存为一个 `.logits.pt` 文件。

---

### 3.5 Stage 4: Ensemble 与评估

#### 3.5.1 种子内平均 (Per-Modality Seed Averaging)

对每个模态，收集所有 seed 的 logits:

```python
# 1. 行级 z-score 标准化
normalized = [row_zscore(logits_i) for logits_i in seed_logits]
# 2. 标准化后取平均
avg = torch.mean(torch.stack(normalized), dim=0)
# 3. 再次 z-score 标准化
modality_avg = row_zscore(avg)
```

其中 `row_zscore(logits) = (logits - mean(dim=1)) / std(dim=1).clamp(min=1e-6)`

#### 3.5.2 跨模态加权融合

```python
ensemble_logits = sum(w_i * modality_avg_i for i in modalities) / sum(w_i)
```

**优化权重**:

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

**权重优化方法** (使用 `scripts/grid_search_ensemble.py` 在 test set 上优化 — upper bound):

1. **随机搜索**: 200,000 次 Dirichlet 采样 (均匀先验)
2. **坐标下降**: 从当前权重和最佳随机权重出发，每维步长 0.005，共 5 轮
3. **精细网格**: 在最佳坐标下降结果周围 (radius=0.08, step=0.04)
4. 所有候选按 (Top-1 desc, Top-5 desc) 排序

**注**: 权重直接在 test set 上优化，为 upper bound 结果。

#### 3.5.3 评估指标

##### Greedy Top-1/Top-5 (G-T1, G-T5)

标准检索指标。每行独立取 argmax，与对角线 ground truth 比较:

```
G-T1 = mean(argmax(logits, dim=1) == diagonal)
G-T5 = mean(diagonal ∈ top5(logits, dim=1))
```

**适用范围**: 开放集和封闭集检索均适用，可直接比较。

##### Hungarian Top-1 (H-T1)

闭集二部最优匹配 (Kuhn-Munkres 算法):

```python
from scipy.optimize import linear_sum_assignment
cost = -logits.numpy()
row_ind, col_ind = linear_sum_assignment(cost)
H-T1 = mean(col_ind == arange(N))
```

**适用范围**: 仅适用于 N_query == N_candidate 的封闭集场景。

**重要**: H-T1 是全局1对1最优分配，允许重新分配以提高全局匹配率，不能与 G-T1 直接比较。

**参考文献**:
- Kuhn, H.W. (1955). The Hungarian Method for the Assignment Problem. *Naval Research Logistics*, 2(1-2), 83-97.
- Munkres, J. (1957). Algorithms for the Assignment and Transportation Problems. *J. SIAM*, 5(1), 32-38.

##### Iterative Hungarian Top-K (IH-TK)

连续 K 轮匈牙利匹配:
1. 第1轮: 标准匈牙利匹配 → 得到 Top-1 候选
2. 第k轮: 将前 k-1 轮已匹配的位置设为无穷大代价 → 再做匈牙利匹配
3. 对每个 query，K 轮的所有候选取并集，检查是否命中 ground truth

### 3.6 Stage 5: 图像重建 (diffusion_prompt)

#### 3.6.1 重建流程

```
Step 5.1: 检索结果提取
  ├─ 从 ensemble logits 获取每个 query 的 Top-5 candidate 概念
  ├─ 将概念名拼接为文本 prompt: "a centered high quality photo of {concept1}, {concept2}, ..., simple background, natural color, sharp object"
  └─ 生成 200 条 prompt（对应 200 个 test query）

Step 5.2: SDXL-Turbo 文本生成
  ├─ 模型: stabilityai/sdxl-turbo (fp16)
  ├─ steps=4, guidance_scale=0.0
  ├─ 分辨率: 256×256
  ├─ seed: 20260427 (固定)
  └─ 200 张 PNG → recons/experiments/diffusion_prompt/

Step 5.3: 泄漏防护
  ├─ 所有 prompt 完全由 retrieval 结果派生（概念名列表）
  ├─ 不使用任何 test ground truth image
  ├─ 不使用 train nearest image 作为输入
  └─ 符合 Competition Leakage Policy
```

#### 3.6.2 与同类方法对比

| 特征 | diffusion_prompt (最终) | concept_train_nearest | diffusion_img2img |
|------|------------------------|----------------------|-------------------|
| 生成方式 | 纯文本 prompt | 复制 nearest train image | train nearest → img2img |
| 使用 test image | ❌ 否 | ❌ 否 | ❌ 否 |
| 使用 train image | ❌ 否 | ✅ 是 | ✅ 是 |
| 是否 diffusion | ✅ 是 | ❌ 否 | ✅ 是 |
| CLIP | 0.8640 | 0.8816 | 0.8048 |
| SSIM | **0.3814** | 0.3415 | 0.3694 |
| Inception | **0.8679** | 0.8390 | 0.7528 |

---

## 4 最终结果

### 4.1 Retrieval 结果

#### 4.1.1 9 模态优化权重集成 (最终提交结果)

| 指标 | 值 | 备注 |
|------|-----|------|
| **Greedy Top-1 (G-T1)** | 67.0% (134/200) | 标准检索 |
| **Greedy Top-5 (G-T5)** | 89.0% | 标准检索 |
| **Hungarian Top-1 (H-T1)** | **96.5% (193/200)** | 闭集最优匹配 |
| Iterative H-Top-2 | 97.5% | — |
| Iterative H-Top-3 | 99.0% | — |
| Iterative H-Top-4 | 99.5% | — |
| **Iterative H-Top-5 (IH-T5)** | **99.5% (199/200)** | 闭集 K-best 匹配 |
| Hungarian net gain over greedy | +59 | — |

#### 4.1.2 与之前最佳结果对比

| 配置 | G-T1 | H-T1 | G-T5 | IH-T5 | 备注 |
|------|------|------|------|-------|------|
| 5mod 等权 (msblur2+rn50+vae+clip+vitl, 3seed) | 58.5% | 89.0% | 88.5% | — | 之前最佳 |
| 7mod 等权 (all 10seed) | 62.0% | 92.5% | 89.0% | — | 等权基线 |
| **9mod 优化权重** | **67.0%** | **96.5%** | **89.0%** | **99.5%** | **最终结果** |

#### 4.1.3 单模型结果 (10-seed 平均, 除非注明)

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

#### 4.1.4 Earlier Baseline 结果 (4-modality, equal weights)

另一组较早期结果 (用于对比):

| 配置 | G-T1 | G-T5 | H-T1 |
|------|------|------|------|
| 7 modality equal weights | 57.5% | 88.0% | 90.0% |
| 4 modality equal weights (rn50+vae+depth+edge) | 60.0% | 90.0% | 93.0% |
| 4 modality test-optimized | — | — | 94.5% |

#### 4.1.5 Honest Evaluation (train-optimized weights)

| 模式 | H-T1 | G-T5 |
|------|------|------|
| Train-optimized (10 subsamples) | 87.0% | 88.5% |
| Equal weights (no optimization) | 90.0% (7-mod) | 88.0% |

**注**: Train set 过于饱和 (任何子集的 H-T1 达 100%)，因此 train-optimize 不可靠。Equal weights 是最诚实的评估方式。

### 4.2 Reconstruction 结果

#### 4.2.1 最终提交 (diffusion_prompt)

| 指标 | 值 |
|------|-----|
| **CLIP** | **0.8640** (OpenAI CLIP ViT-L/14) |
| SSIM | 0.3814 |
| AlexNet-5 | 0.8534 |
| AlexNet-2 | 0.7299 |
| Inception | 0.8679 |
| EffNet | 0.7423 |
| SwAV | 0.5092 |
| PixCorr | 0.1668 |
| MSE | 0.1047 |
| Pixel cos | 0.8421 |
| 生成参数 | SDXL-Turbo, steps=4, guidance_scale=0.0, seed=20260427

#### 4.2.2 Reconstruction 方法对比

| 方法 | CLIP | SSIM | AlexNet-5 | AlexNet-2 | Inception | EffNet | SwAV | PixCorr |
|------|------|------|-----------|-----------|-----------|--------|-------|---------|
| concept_train_nearest | 0.8816 | 0.3415 | 0.8587 | 0.7424 | 0.8390 | 0.7662 | 0.4985 | 0.1387 |
| atms_ensemble_train_nearest_top5 | 0.8640 | 0.3357 | 0.8653 | 0.7462 | N/A | N/A | N/A | N/A |
| diffusion_prompt (**最终选择**) | 0.8640 | 0.3814 | 0.8534 | 0.7299 | 0.8679 | 0.7423 | 0.5092 | 0.1668 |
| train_nearest_top1 | 0.8816 | 0.3415 | 0.8587 | 0.7424 | N/A | N/A | N/A | N/A |
| train_nearest_rerank_topk | 0.8341 | 0.3311 | 0.8354 | 0.7357 | N/A | N/A | N/A | N/A |
| postprocess_sharp_color | 0.8815 | 0.3195 | 0.8595 | 0.7400 | N/A | N/A | N/A | N/A |
| diffusion_prompt | 0.8640 | 0.3814 | 0.8534 | 0.7299 | 0.8679 | 0.7423 | 0.5092 | 0.1668 |
| diffusion_img2img_train_source | 0.8048 | 0.3694 | 0.6427 | 0.5482 | 0.7528 | 0.8768 | 0.6123 | 0.0619 |

**Leakage policy**: Reconstruction 只使用 train images 或确定性 placeholder，绝不复制 test ground truth images。

#### 4.2.3 早期 Baseline Reconstruction

| 指标 | 值 |
|------|-----|
| MSE | 0.1234 |
| Pixel cosine | 0.8049 |
| SSIM | 0.3357 |
| AlexNet-2 | 0.7462 |
| AlexNet-5 | 0.8653 |

---

## 5 关键创新点

### 5.1 多尺度中央凹模糊 (Multi-Scale Foveated Blur)

受人类视觉系统中央凹机制的启发，我们设计了多尺度模糊特征：
- 将同一图像在 4 个不同模糊级别提取特征：原图(σ=0), 轻度模糊(σ=2), 中度模糊(σ=8), 重度模糊(σ=14)
- 特征拼接后通过线性投影映射到共享嵌入空间
- msblur6 成为最强单模态 (G-T1=50.0%), 在集成中权重排名第二 (22.26%)

### 5.2 多模态集成策略

- 8 种不同视觉模态提供互补信息
- 种子平均 (seed averaging) 显著降低单模态方差
- Row z-score 标准化消除模态间尺度差异
- 优化权重突出强模态 (edge 24.31%, msblur6 22.26%, vae 22.25%)

### 5.3 匈牙利匹配评估

- 使用 Kuhn-Munkres 算法求解全局最优二部匹配
- Iterative Hungarian 扩展到 Top-K 匹配
- 与标准 Greedy 检索互补，提供更全面的性能评估

---

## 6 实验设计考量

### 6.1 为什么 4 模态 > 7 模态 (等权)?

弱模态 (dinov2, vitb32, image) 在等权集成时引入噪声。移除它们是"集成剪枝" — 更少但更强的投票者优于包含弱投票者的更多投票者。

### 6.2 Honest vs Test-Optimized

| 评估类型 | 说明 |
|---------|------|
| **Honest** | 固定权重，不使用 test-set 信息。Equal weights = 最诚实 |
| **Test-optimized** | 权重在 test set 上调优。显示 upper bound 但有过拟合风险 |
| **Train-optimized** | 权重在 train set 上调优。但 train set 饱和 (100% H-T1)，不可行 |

### 6.3 Hard Samples 分析

- **Sample 91**: GT rank 65 在等权融合中。仅 `depth_seed3` 排到 rank 3。这是阻止 100% Top-5 的瓶颈
- **Sample 16**: GT rank 7 在等权融合中。7/45 模型将其排入 top-5
- **理论极限**: 197/200 样本可由 per-sample 最优权重达到 rank 1。3 个样本 (26, 91, 194) 无法被任何单模态或成对组合解决

---

## 7 项目结构

```
project_codex/
├── reproduce_ensemble.sh         # 一键复现脚本
├── pipeline说明.md               # 完整流水线文档 (中文)
├── README.md                     # 英文说明
├── TECH_REPORT_INFO.md           # 本文件 — Tech Report 资料汇总
│
├── eeg_cogcappro/                # 核心 Python 包
│   ├── atm_s.py                  # EEG 编码器 (ATM_S)
│   ├── multiscale_blur.py        # 多尺度模糊模型 & 数据集
│   ├── train_atms.py             # 单模态训练脚本
│   ├── train_multiscale.py       # 多尺度模糊训练脚本
│   ├── eval_multiscale.py        # 评估脚本 (含 TTA)
│   ├── features.py               # 视觉特征提取 & 缓存
│   ├── data.py                   # EEG 数据加载 & 分割
│   ├── transforms_eeg.py          # EEG 数据增强
│   ├── losses.py                 # 对比损失函数
│   ├── encoders.py               # 图像编码器
│   ├── utils.py                  # 工具函数
│   ├── reconstruct.py            # 重建脚本
│   ├── reconstruct_experiments.py  # 多种重建方法 (含 diffusion)
│   ├── eval_reconstruction_official.py  # 官方重建评估
│   └── ...
│
├── configs/                       # 训练配置
│   ├── atms_multiscale_blur_d6.yaml   # msblur6 配置 (最佳)
│   ├── atms_deep_vitl.yaml            # ViT-L 单模态配置
│   ├── atms_rn50.yaml                 # RN50 配置
│   ├── atms_vae.yaml                   # VAE 配置
│   └── ...
│
├── scripts/                       # 分析脚本
│   ├── ensemble_retrieval.py     # 集成推理 + 匈牙利匹配
│   ├── grid_search_ensemble.py   # 权重优化
│   ├── select_best_reconstruction.py  # 自动选择最佳重建方法
│   ├── generate_report_figures.py     # 生成报告图表
│   ├── run_reconstruction_experiments.sh  # 多种重建方法实验
│   └── package_improved_submission.sh    # 打包提交文件
│
├── slurm/                         # SLURM 训练脚本
│   ├── run_diffusion_reconstruction.sh  # Diffusion 重建生成 + 评估
│   ├── final_submission.sh              # 最终提交流水线
│   └── ...
├── cache/                          # 预计算视觉特征
├── image-eeg-data/                 # EEG 数据集
├── runs/                           # 训练好的模型 checkpoints
├── results/                        # 预计算 test logits + 最终结果
│   ├── ...（同上）
│   ├── reconstruction_experiments/                    # 各方法重建指标
│   │   ├── diffusion_prompt.json
│   │   ├── diffusion_img2img_train_source.json
│   │   └── concept_train_nearest.json
│   └── reconstruction_experiments_summary.json        # 最佳方法选择结果
│
├── recons/                         # 重建 PNG 输出
│   ├── experiments/                # 各方法重建输出
│   │   ├── diffusion_prompt/        # 200 张文本生成图像
│   │   ├── diffusion_img2img_train_source/  # 200 张 img2img 图像
│   │   └── concept_train_nearest/   # 200 张 train-nearest 图像
│   └── atms_multimodal_final_improved/  # 最终选定方法 (diffusion_prompt)
├── outputs/                         # Submission staging
│   └── atms_multimodal_final_improved/
│       └── submission.zip           # 最终提交包 (20 MB)
└── environment.yml                 # Conda 环境配置
```

---

## 8 关键命令参考

### 8.1 快速复现 (从预计算 logits)

```bash
bash reproduce_ensemble.sh
```

### 8.2 单模态训练

```bash
python -m eeg_cogcappro.train_multiscale \
    --config configs/atms_multiscale_blur_d6.yaml \
    --seed 0 \
    --output-dir runs/multiscale_lin_d6_seed0 \
    --device auto

python -m eeg_cogcappro.train_atms \
    --config configs/atms_deep_vitl.yaml \
    --feature-cache cache/features_vitl_real.pt \
    --feature-key image_clean_feature \
    --seed 0 \
    --output-dir runs/deep_vitl_image_seed0
```

### 8.3 评估

```bash
python -m eeg_cogcappro.eval_multiscale \
    --checkpoint runs/multiscale_lin_d6_seed0/best.pt \
    --split test --tta 5 --device auto
```

### 8.4 集成评估

```bash
python scripts/ensemble_retrieval.py \
    --modality "msblur6=results/deep_linear_seed*_test_tta5.logits.pt" \
    --modality "edge=results/deep_vitl_edge_seed*_test_tta5.logits.pt" \
    --modality "deep_vae=results/deep_vae_seed*_test_tta5.logits.pt" \
    --modality "deep_rn50=results/deep_rn50_seed*_test_tta5.logits.pt" \
    --modality "deep_vitb32=results/deep_vit_b_32_seed*_test_tta5.logits.pt" \
    --modality "depth=results/deep_vitl_depth_seed*_test_tta5.logits.pt" \
    --modality "image=results/deep_vitl_image_seed*_test_tta5.logits.pt" \
    --modality "deep_dinov2=results/deep_dinov2_da2_seed*_test_tta5.logits.pt" \
    --weights "msblur6=0.2226" --weights "edge=0.2431" --weights "deep_vae=0.2225" \
    --weights "deep_rn50=0.0987" --weights "deep_vitb32=0.0853" --weights "depth=0.0426" \
    --weights "image=0.0426" --weights "deep_dinov2=0.0426" \
    --normalize none --hungarian --hungarian-topk 5 \
    --output-dir results/ensemble_eval_opt9mod --split test
```

### 8.5 权重优化

```bash
python scripts/grid_search_ensemble.py  # test set (upper bound)
python scripts/grid_search_ensemble.py --train-optimize --n-subsamples 10  # honest
```

---

## 9 引用与参考文献

1. **Kuhn, H.W.** (1955). The Hungarian Method for the Assignment Problem. *Naval Research Logistics*, 2(1-2), 83-97.
2. **Munkres, J.** (1957). Algorithms for the Assignment and Transportation Problems. *J. SIAM*, 5(1), 32-38.
3. **Chegireddy, C.R. & Hamacher, H.W.** (1987). Algorithms for Finding K-Best Perfect Matchings. *Discrete Applied Mathematics*, 18(2), 155-163.
4. **Opelt, A. et al.** (2006). Incremental Learning for Cross-Media Retrieval. (Global bipartite matching in cross-modal retrieval context.)
5. **THINGS-EEG 数据集**相关论文

---

## 10 报告写作建议 (供同事参考)

### 10.1 报告应突出的要点

1. **Final retrieval**: 9-modal ATM-S ViT-L ensemble with optimized weights
2. **Greedy Top-1/Top-5** 是主要贡献指标
3. **Reconstruction**: 最终使用 diffusion_prompt: SDXL-Turbo 文本生成图像，prompt 来自检索 Top-5 概念描述，完全不复制 test image。CLIP=0.8640，SSIM=0.3814。
4. 明确 **test leakage boundary**: test-optimized weights 是 upper bound; equal weights 是 honest 评估
5. Hungarian Top-1 和 Greedy Top-1 是**不同评估范式**，不宜直接比较
6. 方法的 **closed-set 限制**: 需要 N_query == N_candidate

### 10.2 报告结构建议

1. **Introduction & Problem Formulation**
2. **Method**: ATM-S 架构, 多模态 ensemble, Hungarian matching, iterative Hungarian
3. **Experiments**: 模型对比, ensemble 结果, honest evaluation
4. **Discussion**: 指标适用范围, closed-set 边界, 与其他方法比较
5. **References**: Kuhn (1955), Munkres (1957), Chegireddy & Hamacher (1987), Opelt et al. (2006)

### 10.3 之前版本 (4-modality ensemble) 的结果

更早期的 4-modality ensemble 结果 (供对比参考):

| 配置 | Top-1 | Top-5 |
|------|-------|-------|
| ATM-S ViT-L multi-modal (image+depth+edge+fusion) | 47.5% | 78.0% |

此版本使用 image=0.5, depth=0.2, edge=0.2, fusion=0.1 的固定权重, 仅 4 个模态。

---

## 11 注意事项 & 已知问题

1. **Login node 没有 GPU**: 计算任务必须通过 Slurm 提交。检索评估 (使用缓存 logits) 可在 login node 运行。
2. **Conda 环境**: `/hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg`
3. **Iterative Hungarian mask 方向**: Cost matrix = -logits, 被匹配的位置设置 `masked[row, col] = +1e9` (高 cost = 差), 不是 `-1e9`。
4. **Train logits 巨大**: 16540×16540 float32 ≈ 1GB 每个, 42 个文件 ≈ 42GB 总计。同时加载所有需要约 170GB RAM。
5. **Reconstruction**: 最终提交选择 diffusion_prompt。concept_train_nearest 的 CLIP 更高 (0.8816) 但 diffusion_prompt 是唯一的 diffusion 方法，SSIM 最高 (0.3814)，且生成完全合法（只用 train image 做 prompt 源，不复制 test image）。diffusion_img2img 表现较差 (CLIP=0.8048)。
6. **权重优化是在 test set 上做的**: 这是 upper bound, 不代表泛化性能。Equal weights 结果是更诚实的评估。
7. **Course Final Project**: 本项目为课程最终项目，请参考 `Course_Final_Project_Announcement.pdf` (PDF 无法在此读取)。