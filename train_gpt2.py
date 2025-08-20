import math, os
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken   # GPT-2 tokenizer (fast and simple)
import numpy as np

# ----------------------------
# Config: keep all settings here
# ----------------------------
@dataclass
class GPTConfig:
    block_size: int = 128      # max tokens model can see
    vocab_size: int = 50257    # GPT-2 BPE vocab size
    n_layer: int = 6           # small GPT-2 (easy to train)
    n_head: int = 6
    n_embd: int = 256
    lr: float = 3e-4           # learning rate
    batch_size: int = 8        # tokens per batch
    grad_accum_steps: int = 4  # accumulate gradients for bigger effective batch

# ----------------------------
# Flash Attention (fast built-in in PyTorch)
# ----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # split into heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # PyTorch flash attention (does causal masking automatically if causal=True)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        # merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# ----------------------------
# Transformer Block
# ----------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # attention with residual
        x = x + self.mlp(self.ln2(x))   # feedforward with residual
        return x

# ----------------------------
# GPT Model
# ----------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)
        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ----------------------------
# Dataset (Cosmopedia-100k placeholder)
# ----------------------------
def load_data():
    # In real case, use Cosmopedia-100k (tokenized, sharded)
    # Here we simulate with some text
    text = "This is a simple dataset for testing GPT training. " * 1000
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)

    # Save as numpy shards (simulate real pipeline)
    shard = np.array(tokens, dtype=np.int32)
    np.save("shard0.npy", shard)
    print("Dataset prepared with", len(tokens), "tokens.")

    return torch.tensor(tokens, dtype=torch.long)

# ----------------------------
# Training loop
# ----------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)
    config = GPTConfig()
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # load dataset
    data = load_data().to(device)
    block_size = config.block_size

    scaler = torch.cuda.amp.GradScaler(enabled=True)  # mixed precision

    for step in range(200):  # training steps
        # gradient accumulation: simulate bigger batch
        optimizer.zero_grad()
        for micro_step in range(config.grad_accum_steps):
            ix = torch.randint(len(data) - block_size - 1, (config.batch_size,))
            x = torch.stack([data[i:i+block_size] for i in ix])
            y = torch.stack([data[i+1:i+block_size+1] for i in ix])
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=True):
                logits, loss = model(x, y)

            scaler.scale(loss / config.grad_accum_steps).backward()

        scaler.step(optimizer)
        scaler.update()

        if step % 20 == 0:
            print(f"Step {step}: loss {loss.item():.4f}")

if __name__ == "__main__":
    train()
