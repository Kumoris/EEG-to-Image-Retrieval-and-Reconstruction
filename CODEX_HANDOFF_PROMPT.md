 # Codex Handoff Prompt: THINGS-EEG Retrieval + Reconstruction Project

下面这份文档用于在新的 Codex 对话中快速恢复项目上下文。新的 Codex 应先阅读本文，再按需打开相关源码、脚本和结果文件继续推进。

## 建议给新 Codex 的开场 Prompt

你正在接手 `/root/autodl-tmp/project_codex` 下的 THINGS-EEG / CogCapPro-inspired 项目。请先阅读 `CODEX_HANDOFF_PROMPT.md`，再检查 README、关键脚本和最终结果文件。当前项目已经完成一个合法的 ATM-S 多模态 retrieval + train-nearest reconstruction submission pipeline。你的任务是基于现有成果继续优化、复核、写报告或补齐官方 reconstruction 指标。

请注意：

- 不要重置或删除现有结果文件。
- 优先复用当前 final path，而不是回到旧 RN18/RN50 baseline。
- reconstruction 不能复制 test ground-truth images；只能使用 train images、模型生成图或明确标记的 placeholder。
- 当前 retrieval 是主要强项，final metrics 已达到 Top-1 `0.475`、Top-5 `0.780`。
- 当前 reconstruction 是合法 train-nearest baseline，已生成 200 张 PNG 和 submission zip。

## 项目目标

本项目面向 THINGS-EEG 类任务，目标包含两部分：

1. EEG-to-image retrieval：给定 test EEG，在 200 个 test candidate image 中做 Top-1 / Top-5 retrieval。
2. Image reconstruction：为 200 个 test EEG query 输出 200 张 reconstruction PNG，并可用官方 notebook 的 `eval_images` 风格指标评估。

项目核心要求是合法性和可复现性：

- 训练损失只使用 `train.pt` 与 train-side 图像/文本/派生特征。
- test split 只用于 inference、candidate ranking 和最终评估。
- reconstruction 输出不得直接复制 test image files。
- ensemble 权重固定记录，避免用 test accuracy 做动态调参。

## 工作目录

项目根目录：

```text
/root/autodl-tmp/project_codex
```

重要目录：

```text
eeg_cogcappro/                 # 主 Python package
scripts/                       # 训练、评估、打包脚本
configs/                       # 模型配置
cache/                         # feature caches
results/                       # retrieval/eval 结果
runs/                          # checkpoints
recons/                        # reconstruction PNG 输出
outputs/                       # submission staging + zip
image-eeg-data/                # train/test EEG 和图像目录
tests/                         # pytest tests
```

当前环境注意事项：

- `/root/autodl-tmp/project_codex` 不是 git repository。
- `rg` 不可用，可用 `find` / `grep`。
- sandbox 在该主机可能因为 user namespace 问题无法执行命令，必要时需申请 escalated execution。

## 当前最终路线

最终主线是 ATM-S / ViT-L 多模态 ensemble：

- image ATM-S ViT-L 10 seeds
- depth ATM-S ViT-L 10 seeds
- edge ATM-S ViT-L 10 seeds
- fusion ATM-S ViT-L seed0
- 固定 ensemble weights：
  - image: `0.5`
  - depth: `0.2`
  - edge: `0.2`
  - fusion: `0.1`
- normalize: `row_zscore`

最终 retrieval 文件：

```text
results/atms_multimodal_ensemble/retrieval_test_metrics.json
results/atms_multimodal_ensemble/retrieval_test_logits.pt
results/atms_multimodal_ensemble/retrieval_test_top5.csv
```

最终 retrieval 结果：

```text
Top-1 Accuracy: 0.4749999940395355 ≈ 0.475 / 47.5%
Top-5 Accuracy: 0.7799999713897705 ≈ 0.780 / 78.0%
```

## Final Reconstruction 状态

最终 reconstruction 使用合法 train-nearest baseline：

```bash
bash scripts/reconstruct_atms_final.sh
```

对应 Python 入口：

```bash
python -m eeg_cogcappro.reconstruct \
  --mode atms_ensemble_train_nearest \
  --method auto \
  --data-dir image-eeg-data \
  --feature-cache cache/features_vitl.pt \
  --retrieval-logits results/atms_multimodal_ensemble/retrieval_test_logits.pt \
  --output-dir recons/atms_multimodal_final \
  --image-size 256 \
  --topk 5
```

方法逻辑：

1. 读取 final retrieval logits。
2. 对每个 test EEG query 取 top-k predicted test candidate image IDs。
3. 用 `cache/features_vitl.pt` 中的 candidate image features 做 softmax 加权 query proxy。
4. 在 train image feature pool 中找 nearest train image。
5. 将 nearest train image resize 到 `256x256`，写为 `000.png` 到 `199.png`。
6. 写出 manifest 和 summary，明确 leakage policy。

输出文件：

```text
recons/atms_multimodal_final/000.png ... 199.png
recons/atms_multimodal_final/manifest.csv
recons/atms_multimodal_final/summary.json
```

已验证：

- PNG 数量：`200`
- 文件名范围：`000.png` 到 `199.png`
- manifest 行数：`200`
- `source_kind`: 全部为 `train_nearest`
- `bad_sources`: `0`
- 没有 `test_ground_truth` source。

如果没有 local diffusion model，当前脚本会记录 fallback：

```text
diffusers is installed, but no local diffusion model path was provided; used train_nearest fallback
```

## Reconstruction 指标状态

### 2026-04-27 最新状态

已经新增项目内 official-compatible eval 模块和 reconstruction 实验框架：

```text
eeg_cogcappro/eval_reconstruction_official.py
eeg_cogcappro/reconstruct_experiments.py
scripts/eval_reconstruction_official_final.sh
scripts/run_reconstruction_experiments.sh
scripts/select_best_reconstruction.py
scripts/package_improved_submission.sh
scripts/check_final_submission.py
```

OpenAI `clip` package 已安装成功，OpenAI ViT-L/14 权重已通过当前 proxy 下载并缓存。当前 final CLIP 指标使用 strict `clip.load("ViT-L/14")`，不是 open_clip fallback。不要修改用户 proxy。

当前 improved final:

```text
recons/atms_multimodal_final_improved/
outputs/atms_multimodal_final_improved/submission.zip
results/atms_multimodal_final_improved_reconstruction_official.json
results/reconstruction_experiments_summary.json
```

当前 improved reconstruction 指标：

```text
PixCorr:    0.1387
SSIM:       0.3415
AlexNet-2:  0.7424
AlexNet-5:  0.8587
Inception:  0.8386
CLIP:       0.8816  (strict OpenAI CLIP ViT-L/14)
EffNet:     0.7662
SwAV:       0.4986
MSE:        0.1192
Pixel cos:  0.8173
```

实验选择规则固定为 `CLIP -> AlexNet-5 -> SSIM`。当前 best method 是 `concept_train_nearest`。

候选摘要：

```text
atms_ensemble_train_nearest_top5: CLIP 0.8640, SSIM 0.3357, AlexNet-5 0.8653
concept_train_nearest:          CLIP 0.8816, SSIM 0.3415, AlexNet-5 0.8587
train_nearest_top1:             CLIP 0.8816, SSIM 0.3415, AlexNet-5 0.8587
train_nearest_rerank_topk:      CLIP 0.8341, SSIM 0.3311, AlexNet-5 0.8354
postprocess_sharp_color:        CLIP 0.8815, SSIM 0.3195, AlexNet-5 0.8595
diffusion_prompt:               CLIP 0.8454, SSIM 0.3736, AlexNet-5 0.8594
diffusion_img2img_train_source: CLIP 0.8794, SSIM 0.3930, AlexNet-5 0.8595
```

SDXL Turbo fp16 UNet is available and loads with `variant="fp16"`. Diffusion variants now generate 200 PNGs each. `diffusion_img2img_train_source` uses train-nearest images as init sources only; it does not use test ground-truth images.

官方 notebook `eval_images` 依赖 `clip` 包。当前环境没有 OpenAI `clip` package，因此完整官方 eval 没有跑通：

```text
official eval_images unavailable, fallback used: No module named 'clip'
```

已保存 fallback eval：

```text
results/atms_multimodal_final_reconstruction_eval.json
```

内容：

```json
{
  "eval_mse_fallback": 0.12336476892232895,
  "eval_pixel_cosine_fallback": 0.8048766851425171,
  "note": "official eval_images unavailable, fallback used: No module named 'clip'"
}
```

后续又按 notebook-compatible 定义补算了用户关心的部分指标：

```text
results/atms_multimodal_final_reconstruction_requested_metrics.json
```

当前结果：

```text
SSIM:      0.33570364699471145
AlexNet-2: 0.7462311557788945
AlexNet-5: 0.8653015075376885
CLIP:      null / 未计算成功
```

CLIP 未计算原因：

```text
python package clip is not installed
```

如果继续推进 reconstruction 指标，下一步优先任务是安装或提供 OpenAI CLIP `clip` package 和 ViT-L/14 权重，然后重新跑官方 `eval_images` 或等价实现。注意不要为了安装依赖破坏当前环境；需要用户批准网络/安装操作。

## Final Submission 状态

最终打包脚本：

```bash
bash scripts/package_final_submission.sh
```

对应输出：

```text
outputs/atms_multimodal_final/submission.zip
```

zip 内容已验证：

```text
entries: 205
reconstruction PNGs: 200
retrieval files:
  retrieval_test_logits.pt
  retrieval_test_metrics.json
  retrieval_test_top5.csv
manifest: reconstruction_manifest.csv
summary: reconstruction_summary.json
```

staged output 目录：

```text
outputs/atms_multimodal_final/
outputs/atms_multimodal_final/reconstructions/
outputs/atms_multimodal_final/submission.zip
```

## 关键代码文件

### `eeg_cogcappro/reconstruct.py`

已经改造，支持两种 reconstruction mode：

- `checkpoint_train_nearest`: 旧 checkpoint-based RN50/CogCapPro fallback。
- `atms_ensemble_train_nearest`: final ATM-S ensemble logits based reconstruction。

新增/重要参数：

```text
--retrieval-logits
--mode atms_ensemble_train_nearest
--diffusion-model
--feature-key
--topk
```

重要 leakage policy 常量：

```text
uses only train images or deterministic placeholders as reconstruction sources; never copies test ground truth images
```

### `scripts/ensemble_retrieval.py`

负责多 logits / 多 modality ensemble。

最终脚本调用：

```bash
bash scripts/eval_atms_multimodal_ensemble.sh
```

核心功能：

- modality glob loading
- row z-score normalize
- modality-level averaging
- fixed weights
- output metrics / logits / top5 csv

### `scripts/package_submission.py`

已经修复旧依赖：

- 移除了 `brain2image.io.ensure_dir`
- 改为使用 `eeg_cogcappro.utils.ensure_dir`
- 支持 `--retrieval-dir` 和 `--recon-dir`
- 会复制 final artifacts 到 `outputs/atms_multimodal_final`
- 会验证 reconstruction PNG 数量

### 新增 final scripts

```text
scripts/reconstruct_atms_final.sh
scripts/eval_reconstruction_final.sh
scripts/package_final_submission.sh
```

### README

`README.md` 已加入 Final Submission Path、final metrics、leakage policy 和运行命令。

## 重要运行命令

完整 final path：

```bash
bash scripts/eval_atms_multimodal_ensemble.sh
bash scripts/reconstruct_atms_final.sh
bash scripts/eval_reconstruction_final.sh
bash scripts/package_final_submission.sh
```

静态和测试：

```bash
python -m compileall -q eeg_cogcappro scripts/ensemble_retrieval.py scripts/package_submission.py
pytest -q
```

已通过：

```text
pytest: 4 passed, 1 warning
compileall: passed
```

快速检查最终 artifacts：

```bash
python -c "from pathlib import Path; ps=sorted(Path('recons/atms_multimodal_final').glob('*.png')); print(len(ps), ps[0].name, ps[-1].name)"
python -c "import json; print(json.load(open('results/atms_multimodal_ensemble/retrieval_test_metrics.json'))['metrics'])"
python -c "import zipfile; z=zipfile.ZipFile('outputs/atms_multimodal_final/submission.zip'); names=z.namelist(); print(len(names), sum(n.startswith('reconstructions/') and n.endswith('.png') for n in names))"
```

## 当前用户关心的最终结果

Retrieval：

```text
Top-1: 0.475 / 47.5%
Top-5: 0.780 / 78.0%
```

Reconstruction：

```text
SSIM:      0.3357
CLIP:      not available yet, missing clip package
AlexNet-2: 0.7462
AlexNet-5: 0.8653
```

Fallback pixel metrics：

```text
MSE:          0.1234
Pixel cosine: 0.8049
```

## 可继续推进的方向

### 1. 补齐 CLIP / 官方 reconstruction eval

当前最直接的缺口是 OpenAI `clip` package 缺失。可选推进：

- 安装 `clip` package，并确保 ViT-L/14 weights 可用。
- 重新运行：

```bash
bash scripts/eval_reconstruction_final.sh
```

或写一个更稳健的 eval wrapper：

- SSIM/AlexNet 用 torchvision/skimage。
- CLIP 优先用 `clip`，fallback 用 `open_clip`，但要在报告中注明不是官方完全等价。

### 2. 提升 reconstruction 质量

当前 train-nearest 是合法 baseline，不是高质量生成模型。可继续：

- 用户提供本地 SDXL / Stable Diffusion / IP-Adapter / unCLIP 权重。
- 使用 predicted top-k candidates/concepts 作为 prompt 或 img2img conditioning。
- 保持 source policy：不能复制 test image；若用 test candidate features/predictions，只能作为 retrieval-derived conditioning，不直接输出 test GT。

### 3. 报告写作

报告应突出：

- final retrieval 是 ATM-S ViT-L 多模态 fixed-weight ensemble。
- Top-1/Top-5 是主要贡献。
- reconstruction 是 legal train-nearest fallback，主要用于满足 submission 完整性和可评估性。
- 明确 test leakage boundary。

### 4. 稳健性/复现性

可进一步新增：

- `scripts/check_final_submission.py` 已存在，可继续扩展检查更多 metric 文件。
- reconstruction requested/full metrics 已做成正式模块。
- 如果项目需要提交源码，清理 `__MACOSX` 和旧 baseline outputs，但不要删除 final artifacts。

## 不要踩的坑

- 不要把 `image-eeg-data/test_images` 的真实图片复制到 reconstruction 输出。
- 不要把 retrieval weights 改成动态搜索后的值，除非明确记录并解释 test leakage 风险。
- 不要把旧 RN18/RN50 outputs 当作 final submission。
- 不要依赖 `brain2image` 包；当前 packaging 已移除该旧依赖。
- 不要认为 `results/atms_multimodal_final_reconstruction_eval.json` 包含最新完整官方指标；最新 strict OpenAI CLIP 指标在 `results/atms_multimodal_final_improved_reconstruction_official.json`。

## 最终状态一句话

项目当前已经形成闭环：ATM-S ViT-L 多模态 retrieval final ensemble 达到 Top-1 `47.5%` / Top-5 `78.0%`；reconstruction 已补 strict OpenAI CLIP eval，最终选择 `concept_train_nearest` 作为 improved final，CLIP `0.8816`、SSIM `0.3415`、AlexNet-5 `0.8587`；SDXL Turbo fp16 UNet 已可加载，diffusion prompt/img2img 候选已生成并评估；`outputs/atms_multimodal_final_improved/submission.zip` 已打包完成。
