# zoof (v1.2)

Zoof is a clean, and optimized implementation of a decoder-only Transformer language model in PyTorch.

## ⚡ Key Features

- Pre-Norm Architecture: Applies `RMSNorm` before self-attention and MLP blocks for better gradient flow and training stability.
- Rotary Positional Embeddings (RoPE): Uses `RoPE` in each attention block replacing learnt positional encoding from zoof v1.
- Flash Attention: Automatically uses PyTorch's `F.scaled_dot_product_attention`, leveraging Flash Attention kernels when available for efficient $O(N^2)$ computing.
- Smart Initialization: Implements a specific weight initialization strategy (scaling projections by $1/\sqrt{2L}$) to stabilize variance in deep residual paths.
- Extensive Pre-training: Trained on approximately 59 Billion tokens from the `FineWeb-Edu` dataset.

## ☁️ Run on Google Colab

You can prompt the Zoof model using Google Colab's free T4 GPUs. This is the fastest way to try the model without installing anything locally.

[Click here to open the Interactive Notebook.](https://colab.research.google.com/drive/1KUGAwqIZZtnQbBUYZjoxrsS4v2QNELoE#scrollTo=jbcAcx8ONVim)

The notebook handles:

- Cloning the repository.
- Installing dependencies (torch, transformers, etc.).
- Loading the model on the GPU (cuda).
- Running the interactive chat loop.

## 📂 Directory Structure

```
/
├── src/
│   ├── config.py    # Configuration dataclass
│   ├── model.py     # Main GPT-style model, SelfAttention, and MLP classes
│   ├── prompt_zoof.py  # Interactive CLI chat script
│   └── utils.py     # Helper utilities
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
└── uv.lock          # Dependency lock file
```

## 🛠️ Installation

This project uses uv for fast package management.

### Prerequisites

- Python 3.8+
- PyTorch (CUDA for Flash Attention)

### Setup

```
git clone https://github.com/yourusername/zoof.git
cd zoof

# Syncs the environment based on uv.lock
uv sync
```

## 🎮 Usage: Chat with Zoof on a Linux Box

I've provided a script to chat with a pre-trained & fine-tuned version of the model (zoof-v1.2-394M-chat) hosted on Hugging Face.

Run the following to prompt the model:
```
python prompt_zoof.py
```

This script will:

- Download the config and model weights from `Jiraya/zoof-250M-chat`.
- Download the tokenizer from `Jiraya/zoof-tokenizer`.

## Model Weights (v1.2)

- [`zoof-v1.2-394M`](https://huggingface.co/Jiraya/zoof-v1.2-394M)
- [`zoof-v1.2-394M-chat`](https://huggingface.co/Jiraya/zoof-v1.2-394M-chat)

## Tokenizer

- [`zoof-tokenizer`](https://huggingface.co/Jiraya/zoof-tokenizer)
