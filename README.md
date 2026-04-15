<div align="center">

# MOESCORE
### Mixture-of-Experts-Based Text-Audio Relevance Score Prediction for Text-to-Audio System Evaluation

[![Paper](https://img.shields.io/badge/Paper-[Paper_Link]-blue)]([https://arxiv.org/abs/2601.06829])

</div>

---

## Overview

**MOESCORE** is an objective evaluator for **text-audio relevance** in Text-to-Audio (TTA) systems.  
It is designed to assess whether a generated audio clip is semantically aligned with its input text prompt, with particular focus on:

- global semantic consistency,
- fine-grained event correspondence,
- temporal structure alignment.

Our method combines multiple complementary experts under a **Mixture-of-Experts (MoE)** framework and incorporates **Sequential Cross-Attention (SeqCoAttn)** to better model local temporal matching between audio and text.

---

## Highlights

- **1st place** in **XACLE Challenge 2026 (ICASSP 2026)**
- **SRCC: 0.6402** on the official blind test set
- **+30.6% SRCC improvement** over the challenge baseline
- A **4-expert MoE design** combining global semantic alignment and temporal correspondence
- Built for **automatic evaluation of Text-to-Audio systems**

---




## Table of Contents

- [MOESCORE](#moescore)
    - [Mixture-of-Experts-Based Text-Audio Relevance Score Prediction for Text-to-Audio System Evaluation](#mixture-of-experts-based-text-audio-relevance-score-prediction-for-text-to-audio-system-evaluation)
  - [Overview](#overview)
  - [Highlights](#highlights)
  - [Table of Contents](#table-of-contents)
  - [Method](#method)
    - [Motivation](#motivation)
    - [Architecture](#architecture)
  - [Main Results](#main-results)
    - [Official Blind Test Set](#official-blind-test-set)
    - [Validation Set Ablation](#validation-set-ablation)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Train](#train)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Metrics](#metrics)
  - [Acknowledgements](#acknowledgements)
  - [Contributors](#contributors)

---

## Method

### Motivation

Although modern Text-to-Audio models can generate perceptually plausible audio, they often fail to fully preserve the semantic content of the input text. This mismatch may appear in:

- missing or hallucinated sound events,
- incorrect temporal ordering,
- weak contextual consistency.

MOESCORE addresses this by combining several experts with different inductive biases, instead of relying on a single cross-modal scoring model.

### Architecture

Our framework consists of **four specialized experts**:

1. **Expert 1: LAION-CLAP-based expert**  
   Provides strong global semantic alignment in a shared audio-text embedding space.

2. **Expert 2: MGA-CLAP-based expert**  
   Enhances multi-grained alignment between local audio patterns and global text semantics.

3. **Expert 3: M2D-CLAP-based expert**  
   Improves robustness and generalization with strong audio-language representations.

4. **Expert 4: SeqCoAttn-based expert**  
   Uses **BEATs + RoBERTa + bidirectional Sequential Cross-Attention** to model fine-grained temporal correspondence.

A **gating network** dynamically fuses expert outputs for final score prediction.

![alt text](overall.png)
![alt text](image.png)

---

## Main Results

### Official Blind Test Set

| Method              | SRCC ↑     | LCC ↑      | KTAU ↑     | MSE ↓      |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Baseline            | 0.3345     | 0.3420     | 0.2290     | 4.8110     |
| **MOESCORE (Ours)** | **0.6402** | **0.6873** | **0.4612** | **3.0111** |

### Validation Set Ablation

| Method              | SRCC ↑     | LCC ↑      | KTAU ↑     | MSE ↓      |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Expert 1            | 0.6302     | 0.6492     | 0.4542     | 3.5382     |
| Expert 2            | 0.6297     | 0.6562     | 0.4550     | 3.7208     |
| Expert 3            | 0.6257     | 0.6438     | 0.4496     | 3.7369     |
| Expert 4            | 0.5808     | 0.5865     | 0.4147     | 3.8082     |
| MoE (3 Experts)     | 0.6480     | 0.6745     | 0.4693     | 3.3592     |
| **MoE (4 Experts)** | **0.6680** | **0.6845** | **0.4861** | **3.3462** |

---

## Installation
1. Clone the repository
```bash
conda create -n mosscore python=3.12
git clone https://github.com/S-Orion/MOESCORE.git
cd MoEScore
pip install -r requirements.txt
```
2. Download Pretrained Models (Future release)

| Model    | Description              | Link              |
| -------- | ------------------------ | ----------------- |
| MOESCORE | Final 4-expert MoE model | [Checkpoint_Link] |
| Expert 1 | LAION-CLAP-based expert  | [Checkpoint_Link] |
| Expert 2 | MGA-CLAP-based expert    | [Checkpoint_Link] |
| Expert 3 | M2D-CLAP-based expert    | [Checkpoint_Link] |
| Expert 4 | SeqCoAttn-based expert   | [Checkpoint_Link] |



## Data Preparation
- Download the XACLE Challenge 2026 dataset from [here](https://github.com/XACLE-Challenge/the_first_XACLE_challenge_dataset_train_validation).
- Organize the dataset directory structure as follows:
```bash
  XACLE_dataset
├── meta_data
│   ├── train_average.csv
│   ├── train.csv
│   ├── validation_average.csv
│   └── validation.csv
└── wav
    ├── train
    │   ├── 00000.wav
    │   ├── 00001.wav
    │   ├── 00002.wav
    │   ├──   .
    │   ├──   .
    │   └──   .
    └── validation
        ├── 07500.wav
        ├── 07501.wav
        ├── 07502.wav
        ├──   .
        ├──   .
        └──   .
```

## Train
```bash
python train.py --config config.yaml
```

## Inference
Given an audio-text pair, MOESCORE predicts a relevance score:
```bash
python inference.py \
    --config configs/infer.yaml \
    --split test \
    --ckpt checkpoints/best_model.pt \
    --output results/test_pred.csv \
    --device cuda:0
```

## Evaluation
```bash
python evaluate.py <inference_csv_path> <ground_truth_csv_path> <save_results_dir>
```

- Cmd-Line argument descriptions
  - <inference_csv_path>: Path to the CSV file containing the inference results for the validation data.
  - <ground_truth_csv_path>: Path to the CSV file containing the ground-truth scores for the validation data in XACLE dataset (validation_average.csv).
  - <save_results_dir>: Directory where the evaluation result will be saved (the output file name is fixed as evaluation_result.csv)

- Using the predicted scores and ground-truth scores for the validation data, it calculates SRCC, LCC, KTAU, and MSE.
  - *This program cannot be used for predicting scores on test data because ground-truth is required.

- The results for SRCC, LCC, KTAU, MSE, and the number of evaluation data are written to a file named evaluation_result.csv inside <save_results_dir>.



## Metrics
The repository supports the following evaluation metrics:
- SRCC
- LCC
- KTAU
- MSE


## Acknowledgements
This project builds upon or is inspired by:

- LAION-CLAP (https://github.com/LAION-AI/CLAP)
- MGA-CLAP (https://github.com/Ming-er/MGA-CLAP)
- M2D-CLAP (https://github.com/nttcslab/m2d?tab=readme-ov-file)
- BEATs (https://github.com/microsoft/unilm/tree/master/beats)
- RoBERTa (https://huggingface.co/FacebookAI/roberta-base)


## Contributors
- Bochao Sun （Northwestern Polytechnical University, Xi’an, China）
- Yang Xiao （The University of Melbourne, Melbourne, Australia）
- Han Yin  （KAIST, Daejeon, Republic of Korea）