import inspect
import math

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch.nn import functional as F


class SelfAttention(nn.Module, PyTorchModelHubMixin):
    """
    Multi-head self-attention mechanism using PyTorch's optimized functional attention.

    This layer performs the query, key, value projections, computes the attention scores
    (using Flash Attention if available via F.scaled_dot_product_attention), and projects
    the output back to the embedding dimension.

    Attributes:
        c_attn (nn.Linear): Combined projection for Query, Key, and Value.
        c_proj (nn.Linear): Output projection layer.
        resid_dropout (nn.Dropout): Dropout applied to the output projection.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Embedding_Dim).
            mask (torch.Tensor, optional): Attention mask. If None, causal masking is applied implicitly.

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Time, Embedding_Dim).
        """

        B, T, _ = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None if mask is None else mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True if mask is None else False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) block.

    Standard point-wise feed-forward network with GELU activation.
    Expands the dimensionality by a factor before projecting back.
    """

    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, config.hidden_size * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.hidden_size * config.n_embd, config.n_embd, bias=config.bias)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Embedding_Dim).

        Returns:
            t
        """

        x = self.gelu(self.c_fc(x))
        x = self.dropout(self.c_proj(x))
        return x


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block using Pre-Norm architecture.

    This block applies LayerNorm *before* the self-attention and MLP layers.
    This is a standard deviation from the original Transformer paper (Post-Norm)
    but is common in GPT-style architectures (GPT-2, GPT-3) for training stability.

    Structure:
    x = x + SelfAttention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        x = self.ln_1(x)
        x = x + self.attn(x, mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class zoofv1(nn.Module):
    """
    GPT-style Decoder-only Transformer Language Model.

    Features:
    - Learned positional embeddings.
    - Weight tying between token embeddings and the final linear head.
    - Custom weight initialization scheme.
    - Support for Flash Attention (via SelfAttention class).

    Attributes:
        config (object): Configuration object containing model hyperparameters.
        LangModel (nn.ModuleDict): Container for embeddings and transformer blocks.
        lm_head (nn.Linear): Final projection layer to vocabulary size.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert isinstance(config.dropout, float)
        assert isinstance(config.n_head, int) and config.n_head > 0
        assert isinstance(config.n_embd, int) and config.n_embd > 0
        assert isinstance(config.n_layer, int) and config.n_layer > 0

        self.config = config

        self.LangModel = nn.ModuleDict(
            dict(
                w_token_embd=nn.Embedding(config.vocab_size, config.n_embd),
                w_pos_embd=nn.Embedding(config.block_size, config.n_embd),
                dropout=nn.Dropout(config.dropout),
                DecoderStack=nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.LangModel.w_token_embd.weight = self.lm_head.weight

        self.apply(self.__init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.LangModel.w_pos_embd.weight.numel()
        return n_params

    def __init_weights(self, module):
        """
        Custom weight initialization logic.
        - Linears and Embeddings: Normal(0, 0.02)
        - Biases: Zero
        - LayerNorm: PyTorch default (mean=0, std=1)
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, mask=None):
        """
        Forward pass of the language model.

        Args:
            idx (torch.Tensor): Indices of input sequence tokens. Shape (Batch, Time).
            targets (torch.Tensor, optional): Indices of target tokens. Shape (Batch, Time).
            mask (torch.Tensor, optional): Optional mask for attention.

        Returns:
            tuple: (logits, loss)
                logits (torch.Tensor): Prediction scores. Shape (B, T, Vocab_Size).
                loss (torch.Tensor or None): CrossEntropy loss if targets are provided.
        """

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        tok_emb_out = self.LangModel.w_token_embd(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb_out = self.LangModel.w_pos_embd(pos)  # position embeddings of shape (t, n_embd)
        x = self.LangModel.dropout(tok_emb_out + pos_emb_out)
        for DeBlock in self.LangModel.DecoderStack:
            x = DeBlock(x, mask)
        x = self.LangModel.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configures the AdamW optimizer with distinct parameter groups for weight decay.

        Strategy:
        - Apply weight decay to 2D+ tensors (matrix weights: embeddings, attention projections, MLP layers).
        - Do NOT apply weight decay to 1D tensors (biases, layer norm parameters).
        - Use FusedAdamW if available on CUDA for performance.

        Args:
            weight_decay (float): Strength of weight decay.
            learning_rate (float): Learning rate.
            betas (tuple): Beta coefficients for AdamW.
            device_type (str): 'cuda' or 'cpu'.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [{"params": decay_params, "weight_decay": weight_decay}, {"params": nodecay_params, "weight_decay": 0.0}]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_tok=None, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_tok is not None and idx_next == eos_tok:
                break

        return idx
