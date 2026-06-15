# EEG-to-Image Report 图表准备清单

本文档用于整理 Deep Learning Final Project 1 report 中建议准备的所有图表。目标是让 report 满足项目要求、逻辑连贯、表达清晰，并通过 ablation 和 error analysis 提升报告质量。

---

## 1. 总体优先级

### 必须准备

| 编号 | 图表名称 | 类型 | 推荐文件名 | 用途 |
|---|---|---|---|---|
| F1 | Overall System Pipeline | Figure | `figures/method_pipeline.pdf` | 解释完整方法流程 |
| F2 | Retrieval Top-5 Qualitative Examples | Figure | `figures/retrieval_top5_examples.pdf` | 满足 retrieval qualitative 展示需求 |
| F3 | Reconstruction Qualitative Examples | Figure | `figures/reconstruction_qualitative_10examples.pdf` | 满足项目要求中的 8–12 个 reconstruction examples |
| T1 | Main Retrieval Results | Table | LaTeX table | 报告 Top-1 / Top-5 主结果 |
| T2 | Reconstruction Quantitative Results | Table | LaTeX table | 报告 SSIM / CLIP 等重建指标 |
| T3 | Single-Modality Ablation | Table | LaTeX table | 证明 ensemble 的必要性 |
| T4 | Ensemble Ablation | Table | LaTeX table | 展示从单模型到多模态集成的提升 |
| T5 | Greedy vs Hungarian Matching | Table | LaTeX table | 区分标准 retrieval 和 closed-set matching |

### 强烈建议准备

| 编号 | 图表名称 | 类型 | 推荐文件名 | 用途 |
|---|---|---|---|---|
| F4 | ATM-S Architecture | Figure | `figures/atms_architecture.pdf` | 解释 EEG encoder 结构 |
| F5 | Multi-scale Foveated Blur Examples | Figure | `figures/foveated_blur_examples.pdf` | 展示多尺度中央凹模糊设计 |
| F6 | Ensemble Weight Distribution | Figure | `figures/ensemble_weights.pdf` | 可视化优化权重 |
| T6 | Failure Case Analysis | Table | LaTeX table | 提升 discussion 深度 |

### 可选但加分

| 编号 | 图表名称 | 类型 | 推荐文件名 | 用途 |
|---|---|---|---|---|
| F7 | Hard Sample Retrieval Analysis | Figure | `figures/hard_sample_analysis.pdf` | 展示典型失败样例 |
| T7 | Reconstruction Failure Type Summary | Table | LaTeX table | 总结重建失败类型 |
| F8 | Training Curve / Validation Top-1 Curve | Figure | `figures/training_curves.pdf` | 展示训练稳定性 |
| F9 | Modality Contribution / Rank Improvement Plot | Figure | `figures/modality_contribution.pdf` | 展示不同模态对 ranking 的贡献 |

---

## 2. Method 部分所需图表

## F1. Overall System Pipeline

**推荐文件名**

```text
figures/method_pipeline.pdf
```

**放置位置**

Method section 开头。

**目的**

这张图用于给读者建立整体方法印象。你的方法包含视觉特征提取、EEG encoder、multi-seed logits、row-zscore、weighted ensemble 和 Hungarian matching。如果没有 pipeline 图，读者很难快速理解完整系统。

**建议内容**

```text
EEG signal
   ↓
ATM-S EEG Encoder
   ↓
EEG embedding
   ↓
Similarity logits with visual feature bank
   ↓
Seed averaging
   ↓
9-modal weighted ensemble
   ↓
Retrieval prediction / Hungarian matching

Images
   ↓
Visual feature extraction
   ├─ CLIP ViT-L clean image
   ├─ foveated blur features
   ├─ edge feature
   ├─ depth proxy feature
   ├─ CLIP RN50 feature
   ├─ CLIP ViT-B/32 feature
   ├─ DINOv2 feature
   └─ SD-VAE feature
```

**LaTeX caption 建议**

```latex
Overall pipeline of the proposed EEG-to-image retrieval and reconstruction system. Visual features are precomputed from multiple image encoders and image transformations. EEG signals are encoded by ATM-S, and retrieval logits are aggregated across seeds and modalities before final ranking or closed-set Hungarian matching.
```

---

## F2. ATM-S Architecture

**推荐文件名**

```text
figures/atms_architecture.pdf
```

**放置位置**

Method section 中介绍 EEG encoder 的小节。

**目的**

突出你的方法不是简单地套用视觉特征，而是设计了 EEG-side encoder。ATM-S 是报告中的核心模型结构。

**建议内容**

```text
Input EEG: 63 channels × 250 time steps
        │
        ├── ChannelAttention branch
        │       ├─ learnable positional embedding
        │       ├─ Transformer encoder over channels
        │       └─ LayerNorm
        │
        └── ShallowNet branch
                ├─ temporal convolution
                ├─ spatial convolution
                ├─ batch normalization
                ├─ square activation
                ├─ average pooling
                ├─ log activation
                └─ dropout

Element-wise multiplication
        ↓
Residual MLP projector
        ↓
EEG embedding, 512-dim or 768-dim
```

**LaTeX caption 建议**

```latex
Architecture of the ATM-S EEG encoder. The model combines a channel-attention Transformer branch and a ShallowNet-style temporal-spatial convolution branch. Their outputs are fused by element-wise multiplication and mapped to the visual embedding dimension through a residual MLP projector.
```

---

## F3. Multi-scale Foveated Blur Examples

**推荐文件名**

```text
figures/foveated_blur_examples.pdf
```

**放置位置**

Method section 中介绍 visual feature extraction 或 multi-scale foveated blur 的小节。

**目的**

解释 multi-scale foveated blur 的直观含义，并展示 clean / low / mid / high blur 之间的差异。

**建议格式**

| Clean | Fovea-low, σ=2 | Fovea-mid, σ=8 | Fovea-high, σ=14 |
|---|---|---|---|
| image | image | image | image |

**LaTeX caption 建议**

```latex
Examples of multi-scale foveated blur. The central region remains relatively sharp, while the peripheral region is increasingly blurred as the blur strength increases. These variants are used to extract complementary visual features inspired by foveated human vision.
```

---

## 3. Quantitative Results 表格

## T1. Main Retrieval Results

**类型**

LaTeX table。

**放置位置**

Experiments / Quantitative Results section。

**目的**

这是 retrieval 任务的主结果表。项目要求必须报告 Top-1 Accuracy 和 Top-5 Accuracy，因此 G-T1 和 G-T5 必须作为主指标。

**建议表格内容**

| Method | G-T1 | G-T5 | H-T1 | IH-T5 | Comment |
|---|---:|---:|---:|---:|---|
| Single best: msblur6 | 50.0 | 83.5 | — | — | Best single modality |
| 7-modal equal weights | 62.0 | 89.0 | 92.5 | — | Honest equal-weight ensemble |
| 9-modal optimized weights | 67.0 | 89.0 | 96.5 | 99.5 | Final upper-bound ensemble |

**写作注意**

- G-T1 / G-T5 是标准 retrieval 指标。
- H-T1 / IH-T5 是 closed-set matching 指标。
- 不要把 H-T1 作为唯一主指标，否则可能显得过度依赖 200-to-200 一对一约束。

**LaTeX caption 建议**

```latex
Main retrieval results on the official 200-way test protocol. Greedy Top-1 and Top-5 are the primary retrieval metrics, while Hungarian-based metrics are reported as closed-set matching analysis.
```

---

## T2. Reconstruction Quantitative Results

**类型**

LaTeX table。

**放置位置**

Experiments / Reconstruction Results section。

**目的**

项目要求 reconstruction 至少报告 SSIM 和 CLIP Score。建议同时报告其他官方指标，以展示完整结果。

**建议表格内容**

| Method | SSIM | CLIP | AlexNet-2 | AlexNet-5 | Inception | EffNet | MSE | Pixel Cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| concept_train_nearest | 0.3415 | 0.8816 | 0.7424 | 0.8587 | 0.8386 | 0.7662 | 0.1192 | 0.8173 |
| diffusion_prompt | 0.3736 | 0.8454 | — | 0.8594 | — | — | — | — |
| diffusion_img2img_train_source | 0.3930 | 0.8794 | — | 0.8595 | — | — | — | — |

**写作注意**

- 最终选择的 reconstruction 方法是 legal train-nearest fallback。
- 需要强调没有直接复制 test image。
- 该方法 CLIP score 高，但 SSIM 不是最高。

**LaTeX caption 建议**

```latex
Quantitative reconstruction results evaluated by the official metrics. The final reconstruction output uses a train-nearest strategy, which avoids copying test ground-truth images while preserving strong semantic similarity measured by CLIP.
```

---

## 4. Ablation Study 表格

## T3. Single-Modality Ablation

**类型**

LaTeX table。

**放置位置**

Ablation Study section。

**目的**

证明不同视觉特征具有不同效果，并说明为什么需要 ensemble。

**建议表格内容**

| Modality | Visual Feature | Seeds | G-T1 | G-T5 |
|---|---|---:|---:|---:|
| msblur6 | ViT-L multi-scale foveated blur | 10 | 50.0 | 83.5 |
| msblur2 | ViT-L multi-scale foveated blur, depth 2 | 10 | 48.0 | 81.5 |
| depth | ViT-L depth proxy | 10 | 38.5 | 76.0 |
| image | ViT-L clean image | 10 | 37.0 | 65.5 |
| edge | ViT-L edge image | 10 | 34.5 | 69.5 |
| rn50 | CLIP ResNet-50 | 10 | 34.0 | 68.0 |
| clip_vitb32 | CLIP ViT-B/32 | 3 | 32.0 | 69.5 |
| vae | Stable Diffusion VAE latent | 10 | 12.5 | 40.0 |

**分析方向**

- msblur6 是最强单模态，说明 foveated blur 对 EEG-to-image retrieval 有帮助。
- depth、edge、clean image 虽然单独不如 msblur6，但提供互补信息。
- vae 单模态较弱，但在 optimized ensemble 中权重较高，可能说明其 logits 与其他模态存在互补性。

**LaTeX caption 建议**

```latex
Single-modality ablation. Each modality is evaluated after seed averaging. Multi-scale foveated blur achieves the strongest single-modality performance, while other visual representations provide complementary signals for ensemble retrieval.
```

---

## T4. Ensemble Ablation

**类型**

LaTeX table。

**放置位置**

Ablation Study section。

**目的**

展示 ensemble 从早期版本到最终版本的逐步提升。

**建议表格内容**

| Setting | G-T1 | G-T5 | H-T1 | IH-T5 | Comment |
|---|---:|---:|---:|---:|---|
| 5-modal equal weights, 3 seeds | 58.5 | 88.5 | 89.0 | — | Earlier best |
| 7-modal equal weights, 3 seeds | 62.5 | 87.0 | 92.0 | — | More modalities |
| 7-modal equal weights, 10 seeds | 62.0 | 89.0 | 92.5 | — | More stable seed averaging |
| 9-modal optimized weights | 67.0 | 89.0 | 96.5 | 99.5 | Final upper-bound ensemble |

**分析方向**

- 多模态融合显著强于单模态。
- seed averaging 能提高稳定性，尤其是 Top-5。
- optimized weights 带来更高 G-T1 和 H-T1，但需要说明这是 test-optimized upper bound。
- equal weights 可以作为更 honest 的评估结果。

**LaTeX caption 建议**

```latex
Ensemble ablation. Multi-modal aggregation and seed averaging improve retrieval performance. The optimized 9-modal ensemble is reported as an upper-bound analysis, while equal-weight ensembles provide a more conservative evaluation.
```

---

## T5. Greedy vs Hungarian Matching

**类型**

LaTeX table。

**放置位置**

Ablation Study 或 Discussion section。

**目的**

解释标准 retrieval 和 Hungarian closed-set matching 的区别，避免指标误读。

**建议表格内容**

| Evaluation | Constraint | Top-1 | Top-5 | Interpretation |
|---|---|---:|---:|---|
| Greedy retrieval | Independent ranking for each query | 67.0 | 89.0 | Standard retrieval metric |
| Hungarian matching | Global one-to-one assignment | 96.5 | — | Closed-set Top-1 matching |
| Iterative Hungarian | K-best one-to-one assignment | 96.5 | 99.5 | Closed-set Top-K matching |

**分析方向**

- Greedy retrieval 是主要项目指标。
- Hungarian matching 利用了 test candidate set 中的一对一结构。
- Hungarian 结果不能直接等价于普通 Top-1 retrieval。
- 该结果可以作为 closed-set post-processing 分析，而不是替代标准 retrieval 指标。

**LaTeX caption 建议**

```latex
Comparison between standard greedy retrieval and Hungarian-based closed-set matching. Hungarian matching enforces a global one-to-one assignment and is therefore analyzed separately from the standard Top-1 and Top-5 retrieval metrics.
```

---

## 5. Qualitative Results 图

## F4. Retrieval Top-5 Qualitative Examples

**推荐文件名**

```text
figures/retrieval_top5_examples.pdf
```

**放置位置**

Qualitative Results section。

**目的**

展示 retrieval 模型的成功和失败案例。建议选择 4–6 个 EEG query。

**建议格式**

| Ground Truth | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 |
|---|---|---|---|---|---|
| image | image | image | image | image | image |

**推荐样例组成**

- 2 个成功样例：Top-1 正确。
- 2 个部分成功样例：GT 在 Top-5 但不是 Top-1。
- 1–2 个失败样例：GT 不在 Top-5。

**建议标注**

- 用绿色边框标出正确匹配。
- 用红色边框标出错误 Top-1。
- 每一行标注 sample ID 和 GT rank。

**LaTeX caption 建议**

```latex
Qualitative retrieval examples. Each row shows one EEG query, the ground-truth stimulus, and the top-5 retrieved candidates ranked by the ensemble logits. Correct matches are highlighted.
```

---

## F5. Reconstruction Qualitative Examples

**推荐文件名**

```text
figures/reconstruction_qualitative_10examples.pdf
```

**放置位置**

Qualitative Results section。

**目的**

项目明确要求 report 包含 8–12 个 reconstruction qualitative examples，每个需要展示 ground-truth stimulus image、reconstructed image，并包含 success/failure discussion。因此建议做 10 个样例。

**建议格式**

```text
Ground Truth:      img1 img2 img3 img4 img5
Reconstruction:    rec1 rec2 rec3 rec4 rec5

Ground Truth:      img6 img7 img8 img9 img10
Reconstruction:    rec6 rec7 rec8 rec9 rec10
```

**推荐样例组成**

- 4 个成功案例：语义、颜色、形状接近。
- 3 个中等案例：语义正确但细节偏差。
- 3 个失败案例：语义或结构明显错误。

**建议标注**

每个样例标注：

```text
sample id / success type / short note
```

例如：

```text
ID 023, success: similar object color and category
ID 091, failure: correct coarse category but wrong shape
```

**LaTeX caption 建议**

```latex
Qualitative reconstruction examples. Each pair shows the ground-truth stimulus and the corresponding reconstructed image. The examples include successful, partially successful, and failure cases to illustrate the strengths and limitations of the reconstruction strategy.
```

---

## 6. 可视化分析图

## F6. Ensemble Weight Distribution

**推荐文件名**

```text
figures/ensemble_weights.pdf
```

**放置位置**

Ablation Study 或 Discussion section。

**目的**

展示 optimized ensemble 中不同模态的权重，帮助解释最终结果。

**建议数据**

| Modality | Weight |
|---|---:|
| edge | 0.2431 |
| msblur6 | 0.2226 |
| deep_vae | 0.2225 |
| deep_rn50 | 0.0987 |
| deep_vitb32 | 0.0853 |
| depth | 0.0426 |
| image | 0.0426 |
| deep_dinov2 | 0.0426 |

**图形建议**

- 使用 bar chart。
- 按权重从高到低排序。
- 图中显示具体数值。
- caption 中说明该权重是 upper-bound optimized weights。

**LaTeX caption 建议**

```latex
Optimized modality weights in the 9-modal ensemble. The distribution suggests that edge, multi-scale foveated blur, and VAE-based features contribute strongly to the final ensemble. These weights are reported as an upper-bound analysis because they are optimized for the final test protocol.
```

---

## F7. Hard Sample Retrieval Analysis

**推荐文件名**

```text
figures/hard_sample_analysis.pdf
```

**放置位置**

Discussion / Error Analysis section。

**目的**

展示模型失败的具体原因，让 report 更像完整研究，而不是只报告分数。

**建议内容**

选择 3–4 个 hard samples：

```text
Sample 16: near miss, GT rank 7
Sample 91: hard failure, GT rank 65 in equal-weight ensemble
Sample 26: unsolved by single modalities
Sample 194: unsolved by single modalities
```

每个样例展示：

```text
GT image + Top-5 retrieved images + GT rank + brief note
```

**LaTeX caption 建议**

```latex
Hard retrieval examples. These cases reveal typical failure modes of the ensemble, including near misses, weak modality consensus, and semantic ambiguity between visually similar candidates.
```

---

## F8. Training Curve / Validation Top-1 Curve

**推荐文件名**

```text
figures/training_curves.pdf
```

**放置位置**

Experiments section 或 Appendix。

**目的**

展示训练过程稳定性。如果你的 log 里有 train loss、val Top-1、val Top-5，可以画出来。

**建议内容**

- x-axis: epoch
- y-axis: validation Top-1 或 contrastive loss
- 对比 2–4 个关键模型即可：
  - msblur6
  - image
  - edge
  - rn50

**LaTeX caption 建议**

```latex
Training curves of representative modalities. The validation retrieval accuracy is used to select the best checkpoint for each seed and modality.
```

**注意**

如果训练 log 不完整，不需要强行做这张图。

---

## 7. Error Analysis 表格

## T6. Hard Sample Analysis Table

**类型**

LaTeX table。

**放置位置**

Discussion / Error Analysis section。

**目的**

用表格总结 hard samples，比只放图片更容易分析。

**建议表格内容**

| Sample ID | GT Rank in Ensemble | Best Single-Modality Rank | Failure Type | Interpretation |
|---|---:|---:|---|---|
| 91 | 65 | 3 by depth_seed3 | Weak consensus | Only one seed/modality ranks GT highly |
| 16 | 7 | Top-5 in several models | Near miss | Correct item is close but not enough for Top-5 |
| 26 | >5 | >5 | Hard semantic failure | No single modality resolves it |
| 194 | >5 | >5 | Hard semantic failure | Persistent cross-modal ambiguity |

**LaTeX caption 建议**

```latex
Representative hard samples in retrieval. Some failures are caused by weak agreement across modalities, while others remain unresolved even by individual modality or seed-level predictions.
```

---

## T7. Reconstruction Failure Type Summary

**类型**

LaTeX table。

**放置位置**

Qualitative Results 或 Discussion section。

**目的**

配合 reconstruction qualitative examples，系统总结重建成功和失败模式。

**建议表格内容**

| Failure Type | Description | Example ID |
|---|---|---|
| Semantic success | Correct object category or scene type | sample xx |
| Color mismatch | Similar object but color differs | sample xx |
| Shape mismatch | Correct semantic class but wrong geometry | sample xx |
| Background dominance | Background is matched better than foreground object | sample xx |
| Semantic failure | Reconstructed image belongs to wrong class | sample xx |

**LaTeX caption 建议**

```latex
Summary of reconstruction success and failure types. The train-nearest reconstruction strategy often preserves coarse semantic similarity but may fail in fine-grained color, shape, or object-specific details.
```

---

## 8. 推荐 report 中的图表顺序

### Method

1. Figure 1: Overall System Pipeline
2. Figure 2: ATM-S Architecture
3. Figure 3: Multi-scale Foveated Blur Examples

### Experiments

4. Table 1: Dataset and Training Setup
5. Table 2: Main Retrieval Results
6. Table 3: Reconstruction Quantitative Results

### Ablation Study

7. Table 4: Single-Modality Ablation
8. Table 5: Ensemble Ablation
9. Table 6: Greedy vs Hungarian Matching
10. Figure 4: Ensemble Weight Distribution

### Qualitative Results and Discussion

11. Figure 5: Retrieval Top-5 Qualitative Examples
12. Figure 6: Reconstruction Qualitative Examples
13. Table 7: Hard Sample Analysis
14. Table 8: Reconstruction Failure Type Summary

---

## 9. 最小必备版本

如果时间有限，至少准备下面 8 个：

| 优先级 | 图表 | 推荐文件名 / 形式 |
|---:|---|---|
| 1 | Reconstruction Qualitative Examples | `figures/reconstruction_qualitative_10examples.pdf` |
| 2 | Retrieval Top-5 Qualitative Examples | `figures/retrieval_top5_examples.pdf` |
| 3 | Overall System Pipeline | `figures/method_pipeline.pdf` |
| 4 | Main Retrieval Results | LaTeX table |
| 5 | Reconstruction Quantitative Results | LaTeX table |
| 6 | Single-Modality Ablation | LaTeX table |
| 7 | Ensemble Ablation | LaTeX table |
| 8 | Greedy vs Hungarian Matching | LaTeX table |

这 8 个已经能覆盖项目要求中的：

- problem formulation；
- method description；
- training and inference procedure；
- experimental setup；
- quantitative results；
- qualitative results；
- analysis and discussion；
- external resources / leakage boundary discussion。

---

## 10. 建议你接下来优先生成的文件

请先跑出以下两个图，因为它们最直接对应项目要求：

```text
figures/reconstruction_qualitative_10examples.pdf
figures/retrieval_top5_examples.pdf
```

然后再补：

```text
figures/method_pipeline.pdf
figures/atms_architecture.pdf
figures/foveated_blur_examples.pdf
figures/ensemble_weights.pdf
```

表格部分不一定需要单独生成图片，建议直接写成 LaTeX table，这样在 Overleaf 中清晰、可编辑、排版稳定。
