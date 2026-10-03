import json
import os
import sys

import sentencepiece as spm

ROOT = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.join(ROOT, "starter")
sys.path.insert(0, STARTER_DIR)

from decode import parse_sql, snap_query

SRC_SEP = " <sep> "  # separator between the question and the columns in "src"


def gold_roundtrip(out_path="results/dev_gold_roundtrip.jsonl", snap=False):
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
            pair = json.loads(line)
            src, tgt = pair["src"], pair["tgt"]
            question = src.split(SRC_SEP)[0]

            text = sp.decode(sp.encode(tgt))  # same path real model output will take
            query = parse_sql(text)
            if snap:
                query = snap_query(query, question)  # passes None through

            if query is None:
                fout.write(json.dumps({"error": "parse"}) + "\n")
                failed += 1
            else:
                fout.write(json.dumps({"query": query}) + "\n")
            written += 1  # one line written either way

    print(f"[{out_path}] lines written: {written} (expected 8421), failed to parse: {failed}")


def _norm_conds(conds):
    """Lowercase string values and sort, so condition order is ignored."""
    return sorted((int(c), int(o), str(v).lower()) for c, o, v in conds)


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def show_roundtrip_failures(pred_path, n=10):
    gold_path = os.path.join(ROOT, "WikiSQL", "data", "dev.jsonl")
    total = 0
    shown = 0
    by_kind = {"parse error": 0, "sel/agg": 0, "conds": 0}
    seen = 0

    for idx, (gold_row, pred_row) in enumerate(
        zip(_read_jsonl(gold_path), _read_jsonl(pred_path))
    ):
        seen += 1
        gold = gold_row["sql"]
        g_conds = _norm_conds(gold["conds"])

        if "error" in pred_row:
            kind = "parse error"
            p_conds = None
        else:
            pred = pred_row["query"]
            p_conds = _norm_conds(pred["conds"])
            if (gold["sel"], gold["agg"]) != (pred["sel"], pred["agg"]):
                kind = "sel/agg"
            elif g_conds != p_conds:
                kind = "conds"
            else:
                continue  # match

        total += 1
        by_kind[kind] += 1
        if shown < n:
            shown += 1
            print(f"--- line {idx} ({kind})")
            print(f"  gold: sel={gold['sel']} agg={gold['agg']} conds={g_conds}")
            if p_conds is None:
                print(f"  pred: {pred_row['error']}")
            else:
                print(f"  pred: sel={pred['sel']} agg={pred['agg']} conds={p_conds}")

    print(f"\ncompared {seen} pairs; {total} mismatches {by_kind}")
    print(f"showed {shown} of {total}")


if __name__ == "__main__":
    gold_roundtrip("results/dev_gold_roundtrip.jsonl", snap=False)
    gold_roundtrip("results/dev_gold_roundtrip_snap.jsonl", snap=True)
    show_roundtrip_failures("results/dev_gold_roundtrip_snap.jsonl")