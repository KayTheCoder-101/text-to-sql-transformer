import json
import os
import sys

import matplotlib
matplotlib.use("Agg")              # save to file, no window needed
import matplotlib.pyplot as plt
import sentencepiece as spm

sys.path.insert(0, "starter")      # so we can import the starter's module
from starter.embeddings import PositionalEncoding

sp = spm.SentencePieceProcessor()
sp.load("starter/sql_sp.model")                      # fix 1: correct model file
splits = ["train", "dev", "test"]

for split in splits:
    file_name = f"starter/{split}_pairs.jsonl"       # fix 2: correct path
    src_lengths = []                                 # fix 3: fresh lists per split
    tgt_lengths = []

    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pair = json.loads(line)
            src = pair["src"]                        # fix 4: indented into the line loop
            tgt = pair["tgt"]

            # Compute lengths with special token adjustments
            src_len = len(sp.encode(src)) + 1        # + </s>
            tgt_len = len(sp.encode(tgt)) + 2        # + <s> and </s>

            src_lengths.append(src_len)
            tgt_lengths.append(tgt_len)

    # --- After completing the file loop for the split ---
    count = len(src_lengths)
    mean_src = sum(src_lengths) / count if count > 0 else 0
    max_src = max(src_lengths) if count > 0 else 0
    mean_tgt = sum(tgt_lengths) / count if count > 0 else 0
    max_tgt = max(tgt_lengths) if count > 0 else 0

    # Pairs that dataset.py would drop in training (> 160 src or > 64 tgt)
    too_long = sum(1 for s, t in zip(src_lengths, tgt_lengths) if s > 160 or t > 64)

    print(f"{split}: pairs={count}, src_mean={mean_src:.1f}, src_max={max_src}, "
          f"tgt_mean={mean_tgt:.1f}, tgt_max={max_tgt}, too_long={too_long}")

# ---- Task 1.4: positional-encoding heat-map ----
pe = PositionalEncoding(d_model=256, max_len=100).pe[0]   # (100, 256)

plt.figure(figsize=(10, 5))
plt.imshow(pe.numpy(), cmap="RdBu", aspect="auto")
plt.xlabel("Embedding dimension")
plt.ylabel("Position")
plt.title("Sinusoidal positional encoding (first 100 positions × 256 dimensions)")
plt.colorbar(label="Value")
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/pe_heatmap.png", dpi=150)
print("Saved results/pe_heatmap.png")