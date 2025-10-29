"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.

Usage:
    $ python fineweb.py
Saves shards to the local directory "edu_fineweb10B".
"""

import os
import platform
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# ============================================================
# 1. Path and dataset setup
# ============================================================

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
# shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards
shard_size = int(5e7)  #  100M token => 50M token  total of 200 shards
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name=remote_name,
    split="train"
)

# ============================================================
# 2. Tokenizer
# ============================================================

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    """Tokenize one document into uint16 numpy array."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint32)

    # safety check: all tokens < 2^16
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    """Save token array to disk."""
    np.save(filename, tokens_np)


# ============================================================
# 3. Token accumulation logic
# ============================================================

def process_stream(token_stream):
    """Iterate tokenized documents and write shard files."""
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in token_stream:
        # enough space in current shard?
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                )
            progress_bar.update(len(tokens))
        else:
            # write current shard
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None

            # leftover to next shard
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(
            DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
        )
        write_datafile(filename, all_tokens_np[:token_count])


# ============================================================
# 4. Main function
# ============================================================

def main():
    system_name = platform.system().lower()
    # use half of gpus
    nprocs = max(1, os.cpu_count() // 2)
    print(f"🧠 Detected system: {system_name}")
    print(f"🧩 Using up to {nprocs} processes")

    # macOS → spawn mode & fallback to single thread
    if "darwin" in system_name:
        print("🍎 macOS detected → switching to safe single-process mode")
        token_stream = map(tokenize, tqdm(fw, desc="Tokenizing (1 proc)"))
        process_stream(token_stream)
    else:
        print("🐧 Linux detected → using multiprocessing")
        with mp.get_context("spawn").Pool(nprocs) as pool:
            token_stream = pool.imap(tokenize, fw, chunksize=16)
            process_stream(token_stream)


# ============================================================
# 5. Safe entrypoint
# ============================================================

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    mp.freeze_support()
    main()
