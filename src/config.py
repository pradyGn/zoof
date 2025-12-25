from dataclasses import dataclass


@dataclass
class zoofConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool
    hidden_size: int
