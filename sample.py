import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass
from train_gpt2 import GPT, GPTConfig

import argparse

# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser(description="GPT-2 Sampling Script with Styles")
parser.add_argument("--style", type=str, default="focused",
                    choices=["focused", "creative", "wild"],
                    help="Choose sampling style: focused / creative / wild")
parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,",
                    help="Custom text prompt for generation.")
parser.add_argument("--max_new_tokens", type=int, default=64,
                    help="Maximum tokens to generate.")
args = parser.parse_args()

# ============================================================
# 1. GPT-2 model definition (copied from train_gpt2.py)
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
                # NOTE: deleted
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        ) # lower-triangular mask
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ============================================================
# 2. Load checkpoint
# ============================================================

ckpt_path = "log/model_01999.pt"
print(f"Loading checkpoint from {ckpt_path} ...")
ckpt = torch.load(ckpt_path, map_location="cpu")
config = ckpt["config"]
model = GPT(config)
model.load_state_dict(ckpt["model"])
model.eval()

# ============================================================
# 3. Inference setup
# ============================================================

device = "cuda" if torch.cuda.is_available() else \
         "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
model.to(device)
enc = tiktoken.get_encoding("gpt2")

prompt = args.prompt
num_return_sequences = 4
max_new_tokens = args.max_new_tokens

# style presets
if args.style == "focused":
    temperature, top_p, top_k = 0.7, 0.85, 50
elif args.style == "creative":
    temperature, top_p, top_k = 1.1, 0.95, 100
elif args.style == "wild":
    temperature, top_p, top_k = 1.3, 0.98, 200
else:
    temperature, top_p, top_k = 1.0, 0.9, 50

print(f"\n🎨 Style = {args.style} | temp={temperature} | top_p={top_p} | top_k={top_k}")
print(f"🪄 Prompt: \"{prompt}\"\n")


tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, :]
tokens = tokens.repeat(num_return_sequences, 1)
xgen = tokens.clone()

torch.manual_seed(42)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)

print("\nGenerating...\n")

while xgen.size(1) < max_new_tokens:
    with torch.no_grad():
        logits, _ = model(xgen)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # # ---- 🔥 Sampling settings ----
        # temperature = 0.8   # lower -> more focused, higher -> more random
        # top_k = 50          # keep top 50 tokens
        # top_p = 0.9         # nucleus sampling, keep cumulative prob <= 0.9
        # # --------------------------------

        # Apply temperature
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Top-k filter
        if top_k is not None:
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            probs = torch.zeros_like(probs).scatter_(1, topk_indices, topk_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Top-p (nucleus) filter
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0
            probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Multinomial sample
        ix = torch.multinomial(probs, 1, generator=sample_rng)

        xgen = torch.cat((xgen, ix), dim=1)


for i in range(num_return_sequences):
    decoded = enc.decode(xgen[i].tolist())
    print(f"[Sample {i+1}] {decoded}\n")

print("✅ Generation complete.")
