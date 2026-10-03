import json
import os
import sys

import sentencepiece as spm

# Same path setup as decode.py: resolve "starter" relative to this file.
STARTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter")
sys.path.insert(0, STARTER_DIR)

from decode import parse_sql


def gold_roundtrip(out_path="results/dev_gold_roundtrip.jsonl"):
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(STARTER_DIR, "sql_sp.model"))

    written = 0
    failed = 0

    # Make sure the output directory exists (open() won't create it).
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(STARTER_DIR, "dev_pairs.jsonl"), "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue  # skip blank lines so they don't count as pairs
            tgt = json.loads(line)["tgt"]
            text = sp.decode(sp.encode(tgt))  # same path real model output will take
            query = parse_sql(text)

            if query is None:
                fout.write(json.dumps({"error": "parse"}) + "\n")
                failed += 1
            else:
                fout.write(json.dumps({"query": query}) + "\n")
            written += 1  # one line written either way

    print(f"lines written: {written} (expected 8421)")
    print(f"failed to parse: {failed}")


