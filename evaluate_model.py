import json
import os
import subprocess
import sys

import torch
import sentencepiece as spm

ROOT = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.join(ROOT, "starter")
sys.path.insert(0, STARTER_DIR)

from tokenizer import PAD_ID
from decode import parse_sql, snap_query, predict   # predict: adjust if it lives elsewhere
from data_prep import load_split   # placeholder: import from wherever it lives

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SRC_SEP = " <sep> "  # separator between the question and the columns in "src"


# ---------------- Shared helpers ----------------
def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _norm_conds(conds):
    """Lowercase string values and sort, so condition order is ignored."""
    return sorted((int(c), int(o), str(v).lower()) for c, o, v in conds)


# ---------------- Gold round-trip ----------------
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


# ---------------- Step 2: load_model ----------------
def load_model(ckpt_paths):
    from model.transformer import Transformer

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(STARTER_DIR, "sql_sp.model"))
    model = Transformer(sp.get_piece_size(), pad_id=PAD_ID)

    states = [torch.load(p, map_location="cpu") for p in ckpt_paths]
    averaged = {}
    for key in states[0]:
        first = states[0][key]
        if first.is_floating_point():
            averaged[key] = torch.stack([s[key] for s in states]).mean(0)
        else:
            averaged[key] = first  # integer buffers can't be averaged; keep the first
    model.load_state_dict(averaged)
    return model.to(DEVICE).eval()


# ---------------- Step 3: write_predictions ----------------
def write_predictions(model, sp, split, out_path, method="greedy", alpha=0.0,
                      snap=True, limit=None):
    examples, tables = load_split(split)   # original questions, file order

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    written = failed = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in examples:
            if limit is not None and written == limit:
                break  # smoke tests only; the evaluator needs every line
            question = ex["question"]
            header = tables[ex["table_id"]]["header"]
            query, _ = predict(model, sp, question, header,
                               method=method, alpha=alpha, snap=snap)
            if query is None:
                fout.write(json.dumps({"error": "parse"}) + "\n")
                failed += 1
            else:
                fout.write(json.dumps({"query": query}) + "\n")
            written += 1

    rate = failed / written * 100 if written else 0.0
    print(f"written: {written}  failed: {failed}  parse-failure rate: {rate:.2f}%")
    return out_path


# ---------------- Step 4: component_accuracy ----------------
def component_accuracy(pred_path, split):
    gold_path = os.path.join(ROOT, "WikiSQL", "data", f"{split}.jsonl")
    sel_ok = agg_ok = where_ok = n = 0

    for gold_row, pred_row in zip(_read_jsonl(gold_path), _read_jsonl(pred_path)):
        n += 1
        gold = gold_row["sql"]
        if "error" in pred_row:
            continue  # a parse failure is wrong on all three
        pred = pred_row["query"]
        if pred["sel"] == gold["sel"]:
            sel_ok += 1
        if pred["agg"] == gold["agg"]:
            agg_ok += 1
        if _norm_conds(pred["conds"]) == _norm_conds(gold["conds"]):
            where_ok += 1

    if n == 0:
        return {"sel": 0.0, "agg": 0.0, "where": 0.0}
    return {"sel": 100 * sel_ok / n, "agg": 100 * agg_ok / n, "where": 100 * where_ok / n}


# ---------------- Step 5: run_official_evaluator ----------------
def run_official_evaluator(split, pred_path):
    wikisql = os.path.join(ROOT, "WikiSQL")
    result = subprocess.run(
        [sys.executable, "evaluate.py",
         f"data/{split}.jsonl", f"data/{split}.db", os.path.abspath(pred_path)],
        cwd=wikisql, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"evaluate.py failed:\n{result.stderr}")

    out = result.stdout
    scores = json.loads(out[out.index("{"):])   # JSON part: first "{" to the end
    lf = scores.get("lf_accuracy", scores.get("lf"))
    ex = scores.get("ex_accuracy", scores.get("ex"))
    return {"lf": lf * 100, "ex": ex * 100}


if __name__ == "__main__":
    snap_file = "results/dev_gold_roundtrip_snap.jsonl"
    gold_roundtrip(snap_file, snap=True)
    print("components:", component_accuracy(snap_file, "dev"))
    print("official:", run_official_evaluator("dev", snap_file))
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(STARTER_DIR, "sql_sp.model"))
    from model.transformer import Transformer
    model = Transformer(sp.get_piece_size(), pad_id=PAD_ID).to(DEVICE).eval()
    write_predictions(model, sp, "dev", "results/dev_untrained.jsonl", limit=20)
    write_predictions(model, sp, "dev", "results/dev_untrained_beam.jsonl", method="beam", limit=5)