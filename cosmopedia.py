"""
cosmopedia.py
------------
This script downloads the Cosmopedia-100k dataset, tokenizes it with GPT-2 tokenizer,
and saves the tokens into `.npy` files (shards). These shards are later used in training.

Run with:
$ python cosmopedia.py
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset   # pip install datasets
from tqdm import tqdm               # pip install tqdm

# ------------------------------------------
# Where to save dataset
local_dir = "cosmopedia"
os.makedirs(local_dir, exist_ok=True)

# Each shard will have this many tokens
shard_size = int(1e7)  # 10M tokens per shard (smaller for simplicity)

# Load the dataset (Cosmopedia-100k)
print("Downloading Cosmopedia dataset...")
dataset = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")

# GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
end_token = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    """Convert a text sample into tokens."""
    tokens = [end_token]  # start with special end-of-text token
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens = np.array(tokens, dtype=np.uint16)  # store as uint16 to save memory
    return tokens

def save_shard(filename, tokens):
    """Save numpy array of tokens into file."""
    np.save(filename, tokens)

def process_dataset():
    """Tokenize dataset and split into shards."""
    shard_index = 0
    all_tokens = []
    token_count = 0

    for doc in tqdm(dataset, desc="Tokenizing"):
        tokens = tokenize(doc)
        all_tokens.extend(tokens)
        token_count += len(tokens)

        # If shard is full -> save it
        if token_count >= shard_size:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(local_dir, f"cosmopedia_{split}_{shard_index:06d}.npy")
            save_shard(filename, np.array(all_tokens[:shard_size], dtype=np.uint16))

            # leftover tokens move to next shard
            all_tokens = all_tokens[shard_size:]
            token_count = len(all_tokens)
            shard_index += 1

    # Save last incomplete shard
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(local_dir, f"cosmopedia_{split}_{shard_index:06d}.npy")
        save_shard(filename, np.array(all_tokens, dtype=np.uint16))

    print("✅ Tokenization finished. Files saved in:", local_dir)


if __name__ == "__main__":
    mp.freeze_support()  # needed for Windows
    process_dataset()
