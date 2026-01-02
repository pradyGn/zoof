# 🦁 Zoof (v1.2)

<div align="center">

![Zoof Badge](https://img.shields.io/badge/Zoof-v1.2-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Params](https://img.shields.io/badge/Params-394M-yellow?style=for-the-badge)

**A clean, optimized, and interpretable implementation of a decoder-only Transformer in PyTorch.**

[Run in Colab ⚡](https://colab.research.google.com/drive/1KUGAwqIZZtnQbBUYZjoxrsS4v2QNELoE#scrollTo=jbcAcx8ONVim) | [Hugging Face ☁️](https://huggingface.co/Jiraya/zoof-v1.2-394M-chat)

</div>

**Zoof** is a high-efficiency Small Language Model (SLM) engineered from scratch. It demonstrates how modern architectural choices and high-quality data can yield competitive performance in the sub-400M parameter regime, even with limited compute.

## ⚡ Key Features

- **Pre-Norm Architecture:** Applies `RMSNorm` before self-attention and MLP blocks for better gradient flow and training stability.
- **Rotary Positional Embeddings (RoPE):** Replaces absolute learned positional embeddings from v1 with `RoPE`, enabling better generalization to longer contexts.
- **Flash Attention:** Automatically uses PyTorch's `F.scaled_dot_product_attention`, leveraging Flash Attention kernels when available for efficient $O(N^2)$ computing.
- **Smart Initialization:** Implements a specific weight initialization strategy (scaling projections by $1/\sqrt{2L}$) to stabilize variance in deep residual paths.
- **Extensive Pre-training:** Trained on approximately **67 Billion tokens** from the `FineWeb-Edu` dataset, focusing on reasoning-dense content.

## ☁️ Quick Start (Google Colab)

You can prompt the Zoof model using Google Colab's free T4 GPUs. This is the fastest way to try the model without installing anything locally.

[**Click here to open the Interactive Notebook**](https://colab.research.google.com/drive/1KUGAwqIZZtnQbBUYZjoxrsS4v2QNELoE#scrollTo=jbcAcx8ONVim)

The notebook handles:
- Cloning the repository.
- Installing dependencies (torch, transformers).
- Loading the model on the GPU (cuda).
- Running the interactive chat loop.

## 🛠️ Local Installation

This project uses `uv` for fast package management, but standard `pip` works as well.

### Prerequisites
- Python 3.8+
- PyTorch (CUDA required for Flash Attention)

### Setup
```bash
git clone https://github.com/yourusername/zoof.git
cd zoof

uv sync

```


## 🎮 Usage: CLI Chat

I've provided a script to chat with a pre-trained & fine-tuned version of the model (zoof-v1.2-394M-chat) hosted on Hugging Face.

Run the following to prompt the model:
```
python prompt_zoof.py
```

This script will:

- Download the config and model weights from `Jiraya/zoof-250M-chat`.
- Download the tokenizer from `Jiraya/zoof-tokenizer`.
- Launch an interactive session.

## 📊 Performance & Benchmarks

Despite being trained on significantly less data than industry baselines, **zoof-v1.2-394M** demonstrates competitive performance, particularly in tasks requiring boolean logic and physical commonsense.

| Benchmark | Metric | **Zoof-v1.2-394M** | SmolLM-360M | SmolLM2-360M | Qwen2.5-0.5B |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Training Tokens** | *Data Efficiency* | **67B** | 600B | 4T | 18T |
| **PIQA** | Physical Commonsense | 69.4 | 71.6 | 71.7 | 69.9 |
| **BoolQ** | Boolean Reasoning | 61.9 | - | - | - |
| **WinoGrade** | Pronoun Resolution | 54.1 | 52.8 | 52.5 | 54.1 |
| **HellaSwag** | Commonsense NLI | 46.0 | 51.8 | 54.5 | 51.2 |
| **OBQA** | OpenBookQA | 35.6 | 37.2 | 37.4 | 37.4 |
| **ARC-E** | Science (Easy) | 42.9 | - | - | - |
| **ARC-C** | Science (Challenge) | 30.5 | - | - | 35.6 |
| **SIQA** | Social Commonsense | 39.2 | - | - | - |
| **MMLU** | General Knowledge (cloze) | 28.6 | 34.4 | 35.8 | 33.7 |

> **Note:** Zoof achieves these scores with **~1.7% of the training compute** used for SmolLM2 (67B vs 4T tokens), highlighting the efficiency of the architecture and FineWeb-Edu dataset.

## Directory Structure

```
│
├── src/
│   ├── zoof_v1
│   │   └── model.py    # Model definition for v1
│   ├── zoof_v1_2
│   │   └── model.py    # Model definition for v1.2
│   ├── config.py       # Configuration dataclass
│   ├── prompt_zoof.py  # Interactive CLI chat script
│   └── utils.py        # Helper utilities
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
└── uv.lock             # Dependency lock file
```

## Model Weights & Tokenizer

- [`zoof-v1.2-394M`](https://huggingface.co/Jiraya/zoof-v1.2-394M)
- [`zoof-v1.2-394M-chat`](https://huggingface.co/Jiraya/zoof-v1.2-394M-chat)
- [`zoof-tokenizer`](https://huggingface.co/Jiraya/zoof-tokenizer)
