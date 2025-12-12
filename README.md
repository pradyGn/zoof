# zoof

Zoof is a clean, and optimized implementation of a decoder-only Transformer language model in PyTorch. It follows the GPT architecture with modern enhancements for training stability and performance.

## ⚡ Key Features

- Pre-Norm Architecture: Applies `LayerNorm` before self-attention and MLP blocks (standard in GPT-2/3) for better gradient flow and training stability.
- Flash Attention: Automatically uses PyTorch's `F.scaled_dot_product_attention`, leveraging Flash Attention kernels when available for efficient $O(N^2)$ computing.
- Weight Tying: Shares weights between the token embedding layer and the final output logic head, reducing parameter count.
- Smart Initialization: Implements a specific weight initialization strategy (scaling projections by $1/\sqrt{2L}$) to stabilize variance in deep residual paths.
- Optimizer Groups: Custom parameter grouping for AdamW to apply weight decay only to 2D tensors (embeddings, matmuls), skipping biases and layer norms.
- HuggingFace Integration: Inherits from PyTorchModelHubMixin for easy saving/pushing to the HF Hub.

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

We provide a script to chat with a pre-trained & fine-tuned version of the model (zoof-250M-chat) hosted on Hugging Face.

Run the following to prompt the model:
```
python src/prompt_zoof.py
```

This script will:

- Download the config and model weights from `Jiraya/zoof-250M-chat`.
- Download the tokenizer from `Jiraya/zoof-tokenizer`.

## Model Weights

- [`zoof-250M-base`](https://huggingface.co/Jiraya/zoof-250M-base)
- [`zoof-250M-chat`](https://huggingface.co/Jiraya/zoof-250M-chat)
- Launch a terminal-based chat session.
