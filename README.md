# AffectAgent

Official implementation of **AffectAgent: Collaborative Multi-Agent Reasoning for Retrieval-Augmented Multimodal Emotion Recognition** (ACM Multimedia 2026).

AffectAgent combines a shared multimodal language model with three collaborative roles (Query Planner, Evidence Filter, and Emotion Generator), a frozen dual-channel retriever, Retrieval-Augmented Adaptive Fusion (RAAF), and a Modality-Balancing Mixture of Experts (MB-MoE). The trainable components are optimized with a shared affective reward using the repository's MAPPO-style training loop.

Paper: [ACM DOI](https://doi.org/10.1145/3767308.3835848)

## Code-to-paper map

| Paper component | Implementation |
| --- | --- |
| Query Planner, Evidence Filter, Emotion Generator | `retrieval/mmoa_lite/prompts.py`, orchestrated by `MmoaOrchestrator.full_rollout` |
| Cognitive semantic retrieval | `DoubleChannelRetriever.retrieve_channel_A` |
| Perceptual audiovisual retrieval | `DoubleChannelRetriever.retrieve_channel_B` |
| RAAF | `SupportFusion` in `retrieval/mmoa_lite/fusion_modules.py` |
| MB-MoE | `ModalityMoE` in `retrieval/mmoa_lite/fusion_modules.py` |
| Shared and local affective rewards | `EmotionRewardComputer.compute_pipeline_rewards` |
| MAPPO-style optimization | `retrieval/mmoa_lite/train_ppo.py` |

`SupportFusion` is the implementation name retained for the module described as RAAF in the paper.

## Repository layout

```text
configs/                         portable AffectAgent configuration
my_affectgpt/                    AffectGPT multimodal backbone
retrieval/faiss/                 audiovisual index construction
retrieval/mmoa_lite/             agents, retriever, fusion, rewards, training, evaluation
toolkit/                         dataset utilities required by the AffectGPT loaders
config.py                        model and dataset roots
tests/                           lightweight parser tests
```

Large datasets, model weights, retrieval indexes, checkpoints, and generated outputs are intentionally not included.

## Installation

The code was developed with Python 3.10, PyTorch 2.4.0, CUDA 12.1, and Transformers 4.49.0. A Linux CUDA environment is recommended.

```bash
conda create -n affectagent python=3.10 -y
conda activate affectagent

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For GPU FAISS, replace `faiss-cpu` with a CUDA-compatible FAISS package from your Conda/PyTorch environment. `flash-attn` is optional and should be installed only after PyTorch is available.

## Models and data

Keep large assets outside Git and point the code to them with environment variables:

```bash
export AFFECTAGENT_MODEL_ROOT=/path/to/models
export AFFECTAGENT_DATA_ROOT=/path/to/datasets
export AFFECTAGENT_OUTPUT_ROOT=/path/to/outputs
```

Expected model directories:

```text
$AFFECTAGENT_MODEL_ROOT/
├── Qwen2.5-7B-Instruct/
├── bert-base-uncased/
├── chinese-hubert-large/
├── clip-vit-large-patch14/
└── multilingual-e5-base/        # optional local E5 cache
```

The MER-Caption+ retrieval corpus is expected at `$AFFECTAGENT_DATA_ROOT/mer2025-dataset/` with:

```text
track2_train_mercaptionplus.csv  # name, openset
track3_train_mercaptionplus.csv  # name, reason
subtitle_chieng.csv              # name and a supported subtitle column
video/<sample_id>.mp4
audio/<sample_id>.wav
openface_face/<sample_id>/<sample_id>.npy
```

The dataset itself is not redistributed. Obtain MER-UniBench/MER-Caption+ and the pretrained AffectGPT checkpoint under their original terms.

## Build the retrieval datastore

Run commands from the repository root.

Channel A builds normalized multilingual-E5 embeddings over subtitle, description, and open-vocabulary label metadata:

```bash
python -m retrieval.mmoa_lite.build_semantic_index \
  --model-name intfloat/multilingual-e5-base \
  --output-dir retrieval/mmoa_lite/artifacts/semantic_index
```

Channel B extracts AffectGPT video/audio features, builds FAISS indexes, and stores the token sequences used by RAAF:

```bash
python -m retrieval.faiss.build_mercaptionplus_faiss \
  --cfg-path configs/affectagent.yaml \
  --options model.ckpt_3=/path/to/affectgpt_checkpoint.pth \
  --output-root retrieval/faiss/artifacts/mercaptionplus
```

Use `--max-samples 32` for a smoke run. Add `--overwrite` only when replacing an existing generated index.

## Train

```bash
python -m retrieval.mmoa_lite.train_ppo \
  --cfg-path configs/affectagent.yaml \
  --options model.ckpt_3=/path/to/affectgpt_checkpoint.pth \
  --semantic-index-dir retrieval/mmoa_lite/artifacts/semantic_index \
  --multimodal-index-dir retrieval/faiss/artifacts/mercaptionplus \
  --output-dir output/affectagent_mappo \
  --gpu 0
```

For a short pipeline check, add `--max-samples 8 --num-epochs 1 --batch-size 1`.

## Evaluate

```bash
python -m retrieval.mmoa_lite.evaluate \
  --cfg-path configs/affectagent.yaml \
  --options model.ckpt_3=/path/to/affectgpt_checkpoint.pth \
  --ckpt-dir output/affectagent_mappo/best_model \
  --dataset mer2023 \
  --semantic-index-dir retrieval/mmoa_lite/artifacts/semantic_index \
  --multimodal-index-dir retrieval/faiss/artifacts/mercaptionplus \
  --output-dir output/evaluation \
  --gpu 0
```

The evaluator reports WAR, UAR, macro-F1, per-class recall, and per-sample diagnostics. Checkpoint subdirectory names depend on the saved training step; pass the directory that directly contains `affectgpt_trainable.pth`, `support_fusion.pth`, and `modality_moe.pth`.

Supported evaluation loader names are `mer2023`, `mer2024`, `meld`, `iemocap`, and `iemocapfour`. The `mercaptionplus` loader exposes its available annotation corpus and does not define a held-out test split in this release, so use a benchmark loader for reported test metrics.

## Optional SFT-data generation

`build_sft_data.py`, `build_sft_data_s.py`, and `build_sft_data_g.py` prepare warm-start records through an OpenAI-compatible API. Set credentials only through environment variables; never commit them:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...  # optional compatible endpoint
```

The release assumes an existing AffectGPT/SFT checkpoint for MAPPO initialization; it does not redistribute training data or checkpoints.

## Validation

Lightweight tests do not require model weights:

```bash
python -m unittest discover -s tests -v
python -m compileall -q config.py my_affectgpt retrieval toolkit tests
```

An end-to-end run additionally requires the licensed datasets, pretrained encoders, an AffectGPT checkpoint, generated FAISS artifacts, and a CUDA GPU.

## Citation

```bibtex
@inproceedings{wang2026affectagent,
  title     = {AffectAgent: Collaborative Multi-Agent Reasoning for Retrieval-Augmented Multimodal Emotion Recognition},
  author    = {Wang, Zeheng and Yu, Zitong and Zhu, Yijie and Zhao, Bo and Liang, Haochen and Wang, Taorui and Xia, Wei and Zhang, Jiayu and Liu, Zhishu and Ma, Hui and Ma, Fei and Tian, Qi},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3835848}
}
```

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE). Third-party components retain their original notices in source headers.
