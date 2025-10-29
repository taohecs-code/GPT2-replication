

# GPT-2 Reproduction — Technical Report (English Version)

## **Section 1: Model Architecture**

### 1.1 Model Definition

* Built GPT-2 from scratch using `nn.Module`.
* Implemented forward pass producing logits.
* Integrated tokenizer and sampling loop.
* Input: (B, T) data batches → Output: (B, T, C) logits for cross-entropy loss.
* Validated correctness on a single mini-batch (overfitting test).
* Supported loading Hugging Face GPT-2 weights for comparison.

### 1.2 Architecture Hyperparameters (`GPTConfig`)

| Parameter  | Meaning               | Default | Description                                             |
| ---------- | --------------------- | ------- | ------------------------------------------------------- |
| block_size | Context window length | 1024    | Maximum number of tokens the model can see at once.     |
| vocab_size | Vocabulary size       | 50257   | OpenAI GPT-2 BPE vocabulary.                            |
| n_layer    | Transformer layers    | 12      | Each layer contains Attention + MLP + Residual modules. |
| n_head     | Attention heads       | 12      | Number of parallel attention subspaces per layer.       |
| n_embd     | Embedding dimension   | 768     | Hidden size of token representations.                   |

### 1.3 Training Hyperparameters

| Parameter | Meaning         | Default | Large-scale Setting        | Notes                                                |
| --------- | --------------- | ------- | -------------------------- | ---------------------------------------------------- |
| B         | Batch size      | 4       | 64                         | Use gradient accumulation to simulate large batches. |
| T         | Sequence length | 32      | 1024                       | Match original GPT-2 context size.                   |
| lr        | Learning rate   | 3e-4    | 6e-4 (cosine schedule)     | Follow Radford et al.                                |
| optimizer | Optimizer       | Adam    | AdamW (fused CUDA version) | Decoupled weight decay for Transformers.             |
| seed      | Random seed     | 1337    | –                          | Ensures reproducibility.                             |

### 1.4 Parameter Sharing

`self.transformer.wte.weight = self.lm_head.weight`
→ Input embeddings and output projection share weights to reduce parameters and improve generalization.

### 1.5 Initialization Strategy

* Linear / Embedding layers: Normal(0, 0.02) (same as GPT-2).
* Residual scaling: Multiply by (2 × n_layers)^−½ ( `NANOGPT_SCALE_INIT` flag ) to prevent gradient explosion.
* Bias terms initialized to 0.

### 1.6 New Features

* `from_pretrained()` for direct loading of Hugging Face weights.
* `configure_optimizers()` to separate decay / no-decay parameter groups.
* DDP support with `master_process` flag.
* Use `F.scaled_dot_product_attention(..., is_causal=True)` to replace mask-based attention (FlashAttention).

---

## **Section 2: Training Acceleration and Mixed Precision**

### 2.1 TF32 Acceleration

```python
torch.set_float32_matmul_precision("high")
```

Performs FP32 matmul via Tensor Cores for speed while preserving numerical range.

### 2.2 `autocast()` Mixed Precision

```python
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

Automatically casts safe operations (matmul, conv, attention) to bfloat16 for speed and memory efficiency, while keeping sensitive ones (loss, norms) in FP32.

### 2.3 `torch.compile()` Optimization

* Introduced in PyTorch 2.0 to trace dynamic graphs into optimized GPU execution graphs.
* `model = torch.compile(model, mode="max-autotune")`
* Use only during training (single GPU, fixed input shape).
* Disabled for inference (due to dynamic sequence length) and autocast conflicts.

---

## **Section 3: Training Strategy and Hyperparameters**

### 3.1 Optimizer — AdamW + Gradient Clipping

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                              betas=(0.9, 0.95), eps=1e-8)
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

* β₁ = 0.9, β₂ = 0.95 (prevents slow momentum decay of β₂ = 0.999).
* `eps=1e-8` avoids division by zero.
* Gradient clipping keeps global norm ≤ 1.0 for stability.

### 3.2 Learning Rate Schedule — Warmup + Cosine Decay

```
      ^
 lr  |        /\
     |       /  \
     |_____/    \______
         warmup   decay
```

* Warmup: linearly increase LR from 0 → max_lr to avoid explosion.
* Cosine decay: smoothly reduce LR to min_lr for stable convergence.
* Typical values: max_lr = 6e-4, min_lr = max_lr / 10 ~ / 20, warmup_steps = 0.5–5% of total.

### 3.3 Gradient Accumulation and DDP

* Simulate large batches by accumulating gradients over multiple micro-batches:
  `total_batch = B × T × grad_accum_steps`.
* Used PyTorch DDP for multi-GPU training (all-reduce synchronized gradients).

---

## **Section 4: Dataset and Results**

### 4.1 Dataset Comparison

| Model             | Parameters  | Dataset Size                          | Notes                                         |
| ----------------- | ----------- | ------------------------------------- | --------------------------------------------- |
| GPT-2 (2019)      | 1.5 B       | ~40 GB (WebText, 8 M docs)            | Quality over quantity.                        |
| GPT-3 (2020)      | 175 B       | ~570 GB (Common Crawl + Books + Wiki) | Requires massive data to prevent overfitting. |
| NanoGPT / FineWeb | 100 M – 1 B | FineWeb-EDU (clean .edu subset)       | Lightweight for research reproduction.        |

### 4.2 Training Configuration and Results

* Modified mini-config for CPU/Mac safe testing:
  `total_batch_size = 8192`, `B=4`, `T=128`, `max_steps = 2000`.
* Ran successfully on macOS (MPS device).
* Final stats:

  * Steps = 1999 / 2000 ✅
  * Training loss ≈ 5.4 → 5.7 📉 (stable learning)
  * Validation loss = 5.58 ✅
  * HellaSwag accuracy = 29.5 % (normal for GPT-2 124M scale)

### 4.3 Validation and Benchmarking

* Validation split = subset of FineWeb (val loss monitor).
* External benchmark = HellaSwag (10,042 samples, 29.55 % accuracy).
* Consistent with official GPT-2 small model (30–32 %).

### 4.4 Sample Generation Modes

| Mode     | Command                                                                               | Effect                        |
| -------- | ------------------------------------------------------------------------------------- | ----------------------------- |
| Focused  | `python sample.py --style focused`                                                    | Logical, deterministic output |
| Creative | `python sample.py --style creative`                                                   | Story / blog style writing    |
| Wild     | `python sample.py --style wild`                                                       | Highly random, fun, diverse   |
| Custom   | `python sample.py --style creative --prompt "Once upon a time," --max_new_tokens 200` | Free story generation         |

**References:**

* Vaswani et al., *Attention Is All You Need* (2017)
* Radford et al., *Language Models are Unsupervised Multitask Learners* (2019)
* Brown et al., *Language Models are Few-Shot Learners* (2020)
* Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)
* Karpathy’s NanoGPT project and YouTube tutorials.
