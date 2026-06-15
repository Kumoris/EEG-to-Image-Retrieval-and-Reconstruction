# Reconstruction 算法详情
 
> 对应代码: `eeg_cogcappro/reconstruct_experiments.py`

---

## 1 Train-Nearest 算法

### 1.1 伪代码

```
输入: retrieval_logits (200×200, test→candidate 相似度矩阵)
      train_features (16540×768, 所有 train image 的 ViT-L 特征)
      candidate_features (200×768, 200 个 test candidate 的 ViT-L 特征)

算法: concept_train_nearest
─────────────────────────────────────
for each test query i ∈ [0, 199]:
    # Step 1: 取 retrieval top-1 candidate
    pred_idx = logits[i].argmax()
    pred_concept = concept_from_image_id(candidate_ids[pred_idx])

    # Step 2: 取该 candidate 的视觉特征作为 query
    query_feat = candidate_features[pred_idx]

    # Step 3: 计算 query_feat 与所有 train features 的余弦相似度
    sims[i] = query_feat @ train_features.T    # (1, 16540)

    # Step 4: 在共享 predicted concept 的 train images 中找最近的
    pool = [j for j, c in enumerate(train_concepts) if c == pred_concept]
    if pool:
        idx = argmax(sims[i, pool])
        output = train_images[idx]
    else:
        # concept 不在 train 中（如 submarine），fallback 到全局最近
        idx = argmax(sims[i])
        output = train_images[idx]
        note = "predicted_concept_missing_in_train"
```

### 1.2 Retrieval Pool

| 属性 | 说明 |
|------|------|
| **候选池** | 200 个 test candidate images（与 test query ——对应） |
| **特征池** | 全量 train images（16540 张）的 ViT-L `image_clean_feature` |
| **logits 来源** | 9-modal 优化权重集成: `results/ensemble_eval_opt9mod/retrieval_test_logits.pt` |
| **特征编码器** | OpenCLIP ViT-L-14 (laion2b_s32b_b82k), 768-dim |

### 1.3 相似度计算

```
similarity = cosine(query_feat, train_feat)
           = (query_feat @ train_feat.T) / (||query_feat|| · ||train_feat||)
```

其中:
- **train_nearest_top1**: `query_feat = candidate_features[top-1 predicted candidate]`
- **concept_train_nearest**: 同上，但只在 predicted concept 对应的 train images 子集中搜索
- **train_nearest_rerank_topk**: `query_feat = weighted_avg(candidate_features[top-k], weights=softmax(logits[top-k]))`

### 1.4 Top-K Rerank

仅 `train_nearest_rerank_topk` 方法使用：

```
rerank_score = 0.65 × query_train_sim
             + 0.25 × train_to_candidate_max_sim
             + 0.10 × retrieval_prior
```

其中:
- `query_train_sim`: query 特征与每个 train candidate 的相似度
- `train_to_candidate_max_sim`: train candidate 与所有 top-k retrieval candidates 的最大相似度
- `retrieval_prior`: 原始 retrieval logits 的 softmax 最大值

### 1.5 方法对比

| 方法 | 检索池范围 | Rerank | Postprocess |
|------|-----------|--------|-------------|
| train_nearest_top1 | 全量 train | 否 | 否 |
| concept_train_nearest | predicted concept 子集 | 否 | 否 |
| train_nearest_rerank_topk | 全量 train | ✅ top-25 | 否 |
| postprocess_sharp_color | 全量 train | 否 | ✅ contrast+color+sharpness |

---

## 2 SDXL-Turbo 超参表

### 2.1 diffusion_prompt（最终提交方法）

| 参数 | 值 | 说明 |
|------|-----|------|
| **模型** | `stabilityai/sdxl-turbo` | HuggingFace Diffusers |
| **精度** | fp16 | 混合精度推理 |
| **推理步数** | 4 | Turbo 蒸馏关键，1-4 步即可 |
| **guidance_scale** | 0.0 | Turbo 无需 classifier-free guidance |
| **生成分辨率** | 512×512 | Diffusers 生成原生分辨率 |
| **输出分辨率** | 256×256 | resize 后输出 |
| **随机种子** | 20260427 | 固定 seed + query_index 偏移 |
| **prompt template** | `"a centered high quality photo of {concept1}, {concept2}, {concept3}, simple background, natural color, sharp object"` | Top-3 去重概念拼接 |

### 2.2 diffusion_img2img

| 参数 | 值 | 说明 |
|------|-----|------|
| **模型** | `stabilityai/sdxl-turbo` | 同上 |
| **推理步数** | 4 | |
| **guidance_scale** | 0.0 | |
| **strength** | 0.55 | 输入图像保留 45%，55% 被扩散重绘 |
| **源图像** | train nearest image | `train_nearest_top1` 的结果，resize 到 512×512 |
| **prompt** | 同上 template | |

### 2.3 Prompt Template 生成逻辑

```python
def _prompt_for(candidate_ids, ranks, i):
    """从 retrieval top-3 unique concepts 构建 prompt"""
    concepts = []
    for idx in ranks.indices[i][:3]:
        concept = concept_from_image_id(candidate_ids[idx])
        concept = concept.replace("_", " ")
        if concept not in concepts:
            concepts.append(concept)
    joined = ", ".join(concepts)
    return f"a centered high quality photo of {joined}, simple background, natural color, sharp object"
```

示例:
- aircraft_carrier → `"a centered high quality photo of submarine, sailboat, ferry, simple background, natural color, sharp object"`
- basketball → `"a centered high quality photo of basketball, volleyball, soccer ball, simple background, natural color, sharp object"`

---

## 3 Concept 来源说明

### 3.1 数据流

```
Test EEG (200 queries)
    ↓ 9-modal optimized ensemble retrieval
Retrieval logits (200×200, test→candidate similarity)
    ↓ top-3 candidates
Candidate image_id → concept_from_image_id()
    ↓ 去重
Top-3 unique concept names (e.g., "submarine", "sailboat", "ferry")
    ↓ prompt template
SDXL-Turbo text-to-image generation
    ↓
200 reconstruction PNGs
```

### 3.2 concept_from_image_id 逻辑

```python
# 来自 eeg_cogcappro/data.py
def concept_from_image_id(image_id: str) -> str:
    """
    输入: "aircraft_carrier_06s"
    输出: "aircraft_carrier"     # 去掉后缀 _\d+[a-z]
    """
    import re
    return re.sub(r"_\d+[a-zA-Z]$", "", image_id)
```

### 3.3 Concept 来源对比

| 方法 | Concept 来源 | 是否用 train image |
|------|-------------|-------------------|
| **concept_train_nearest** | retrieval top-1 → concept → train 同 concept 子集搜索 | ✅ 是 |
| **diffusion_prompt** | retrieval top-3 → concept 去重 → text prompt | ❌ 否 |
| **diffusion_img2img** | retrieval top-3 → concept 去重 → text prompt + train nearest 初始化 | ✅ 是 |

**关键**: diffusion_prompt **完全不依赖 train image 的图像内容**，只使用 retrieval 结果提取的 concept text 作为 prompt。这是三种方法中泄漏风险最低的。

---

## 4 泄漏防护 (Leakage Policy)

```
所有方法均满足: "uses only train images or deterministic placeholders as 
reconstruction sources; never copies test ground truth images"

- concept_train_nearest: 复制 train image（合法）
- diffusion_prompt: 纯文本生成（完全独立于 image files）
- diffusion_img2img: 以 train image 为源做 img2img（合法）
```
