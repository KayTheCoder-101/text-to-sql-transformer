import argparse
import json
import os
import re
import subprocess
import sys

import torch
import sentencepiece as spm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.join(ROOT, "starter")
sys.path.insert(0, STARTER_DIR)

from tokenizer import PAD_ID, EOS_ID
from data_prep import load_split, encode_source
from decode import parse_sql, snap_query, predict, greedy_decode, to_readable_sql

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SRC_SEP = " <sep> "  # separator between the question and the columns in "src"
COL_TOKEN = re.compile(r"^<c\d+>$")


# ---------------- Shared helpers ----------------
def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _norm_conds(conds):
    """Lowercase string values and sort, so condition order is ignored."""
    return sorted((int(c), int(o), str(v).lower()) for c, o, v in conds)


def _load_sp():
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(STARTER_DIR, "sql_sp.model"))
    return sp


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------- Gold round-trip ----------------
def gold_roundtrip(out_path="results/dev_gold_roundtrip.jsonl", snap=False):
    sp = _load_sp()
    written = 0
    failed = 0
    _ensure_dir(out_path)

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


# ---------------- Model loading (+ checkpoint averaging) ----------------
def load_model(ckpt_paths):
    from model.transformer import Transformer

    sp = _load_sp()
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


# ---------------- Prediction files (Task 4.3 / 5.1) ----------------
def write_predictions(model, sp, split, out_path, method="greedy", alpha=0.0,
                      snap=True, limit=None):
    examples, tables = load_split(split)   # original questions, file order
    _ensure_dir(out_path)

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


# ---------------- Component accuracy (Task 5.2) ----------------
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

    expected = sum(1 for _ in _read_jsonl(gold_path))
    if n != expected:   # zip() stops silently at the shorter file
        raise ValueError(f"{pred_path} has {n} lines but {split} has {expected}")

    if n == 0:
        return {"sel": 0.0, "agg": 0.0, "where": 0.0}
    return {"sel": 100 * sel_ok / n, "agg": 100 * agg_ok / n, "where": 100 * where_ok / n}


# ---------------- Official evaluator (Task 5.1) ----------------
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


# ---------------- Attention map (Task 5.3) ----------------
def plot_attention(model, sp, split="dev", index=0, path="results/attention_map.png"):
    """Last-layer decoder cross-attention, averaged over heads: generated tokens x source tokens."""
    examples, tables = load_split(split)
    ex = examples[index]
    question = ex["question"]
    header = tables[ex["table_id"]]["header"]

    ids = sp.encode(encode_source(question, header)) + [EOS_ID]
    device = next(model.parameters()).device
    src = torch.tensor([ids], dtype=torch.long, device=device)
    tokens, cross = greedy_decode(model, src)            # cross: (1, heads, rows, S)

    attn = cross[0].mean(0).cpu().numpy()                # (rows, S)
    rows, S = attn.shape

    src_labels = [sp.id_to_piece(i) for i in ids]
    # Row r of the last decoder call produced output token r (the final row produced </s>).
    out_labels = ([sp.id_to_piece(t) for t in tokens] + ["</s>"])[:rows]

    fig, ax = plt.subplots(figsize=(max(8, 0.25 * S), max(4, 0.35 * rows)))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(S))
    ax.set_xticklabels(src_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(out_labels, fontsize=8)
    ax.set_xlabel("source tokens")
    ax.set_ylabel("generated tokens")
    ax.set_title(question, fontsize=10)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _ensure_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    pred_sql = to_readable_sql(snap_query(parse_sql(sp.decode(tokens)), question.lower()), header)
    print(f"question : {question}")
    print(f"gold SQL : {to_readable_sql(ex['sql'], header)}")
    print(f"our SQL  : {pred_sql}")

    # The manual's check: do generated <cK> tokens attend to the matching column?
    for r, label in enumerate(out_labels):
        if COL_TOKEN.match(label):
            top3 = attn[r].argsort()[::-1][:3]
            tops = ", ".join(f"{src_labels[j]} ({attn[r, j]:.2f})" for j in top3)
            print(f"  {label:>6} attends most to: {tops}")
    print(f"saved {path}")


# ---------------- Qualitative samples (Section 4.6) ----------------
def failure_labels(gold, pred_row):
    """Return a list of failure types; an empty list means the logical form is correct."""
    if "error" in pred_row:
        return ["parse failure"]
    pred = pred_row["query"]
    labels = []
    if pred["sel"] != gold["sel"]:
        labels.append("wrong column (SELECT)")
    if pred["agg"] != gold["agg"]:
        labels.append("wrong aggregation")

    g = _norm_conds(gold["conds"])
    p = _norm_conds(pred["conds"])
    if g != p:
        if len(p) < len(g):
            labels.append("missing condition")
        elif len(p) > len(g):
            labels.append("extra condition")
        else:
            for (gc, go, gv), (pc, po, pv) in zip(g, p):
                if gc != pc:
                    labels.append("wrong condition column")
                elif go != po:
                    labels.append("wrong operator")
                elif gv != pv:
                    labels.append("wrong value")

    return list(dict.fromkeys(labels))   # remove duplicates, keep order


def write_samples(pred_path, split="dev", out_path="results/samples.md",
                  n_correct=5, n_wrong=5):
    examples, tables = load_split(split)
    correct, wrong, seen_types, spare_wrong = [], [], set(), []

    for idx, (ex, pred_row) in enumerate(zip(examples, _read_jsonl(pred_path))):
        labels = failure_labels(ex["sql"], pred_row)
        entry = (idx, ex, pred_row, labels)
        if not labels:
            if len(correct) < n_correct and ex["sql"]["conds"]:   # prefer ones with WHERE
                correct.append(entry)
        else:
            if len(wrong) < n_wrong and labels[0] not in seen_types:   # one of each failure type
                wrong.append(entry)
                seen_types.add(labels[0])
            elif len(spare_wrong) < n_wrong:
                spare_wrong.append(entry)
        if len(correct) == n_correct and len(wrong) == n_wrong:
            break

    # Not enough distinct failure types: fill up with other wrong examples
    for entry in spare_wrong:
        if len(wrong) >= n_wrong:
            break
        wrong.append(entry)

    def block(entry, show_failure):
        idx, ex, pred_row, labels = entry
        header = tables[ex["table_id"]]["header"]
        gold_sql = to_readable_sql(ex["sql"], header)
        if "error" in pred_row:
            our_sql = "(parse failure)"
        else:
            our_sql = to_readable_sql(pred_row["query"], header) or "(invalid column index)"
        lines = [f"### {split} #{idx}",
                 f"**Question:** {ex['question']}  ",
                 f"**Gold SQL:** `{gold_sql}`  ",
                 f"**Our SQL:** `{our_sql}`  "]
        if show_failure:
            lines.append(f"**Failure:** {', '.join(labels)}  ")
        return "\n".join(lines) + "\n"

    _ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Qualitative samples ({split})\n\n")
        f.write(f"Predictions: `{os.path.basename(pred_path)}`. "
                "\"Correct\" means the logical form matches the gold query.\n\n")
        f.write("## Correct\n\n")
        for e in correct:
            f.write(block(e, show_failure=False) + "\n")
        f.write("## Wrong\n\n")
        for e in wrong:
            f.write(block(e, show_failure=True) + "\n")

    print(f"saved {out_path}: {len(correct)} correct, {len(wrong)} wrong "
          f"(failure types: {', '.join(e[3][0] for e in wrong)})")
    print("correct example indices (for the attention map):", [e[0] for e in correct])


# ---------------- Command line ----------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate the Text-to-SQL Transformer.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("roundtrip", help="gold round-trip check (no model needed)")

    p = sub.add_parser("predict", help="write predictions, then score them")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--ckpt", nargs="+", default=["checkpoints/best.pt"],
                   help="one checkpoint, or several to average")
    p.add_argument("--method", default="greedy", choices=["greedy", "beam"])
    p.add_argument("--alpha", type=float, default=0.0, help="beam length penalty")
    p.add_argument("--no-snap", action="store_true", help="turn value snapping off")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke tests only")

    s = sub.add_parser("samples", help="write results/samples.md")
    s.add_argument("--pred", required=True)
    s.add_argument("--split", default="dev")
    s.add_argument("--out", default="results/samples.md")

    a = sub.add_parser("attention", help="plot the cross-attention map")
    a.add_argument("--ckpt", nargs="+", default=["checkpoints/best.pt"])
    a.add_argument("--split", default="dev")
    a.add_argument("--index", type=int, default=0)
    a.add_argument("--out", default="results/attention_map.png")

    args = ap.parse_args()

    if args.cmd == "roundtrip":
        snap_file = "results/dev_gold_roundtrip_snap.jsonl"
        gold_roundtrip(snap_file, snap=True)
        print("components:", component_accuracy(snap_file, "dev"))
        print("official:", run_official_evaluator("dev", snap_file))

    elif args.cmd == "predict":
        sp = _load_sp()
        model = load_model(args.ckpt)
        tag = args.method if args.method == "greedy" else f"beam_a{args.alpha}"
        out = args.out or f"results/{args.split}_{tag}{'_nosnap' if args.no_snap else ''}.jsonl"
        write_predictions(model, sp, args.split, out, method=args.method,
                          alpha=args.alpha, snap=not args.no_snap, limit=args.limit)
        if args.limit is None:   # the evaluator needs every line
            print("official:", run_official_evaluator(args.split, out))
            print("components:", component_accuracy(out, args.split))

    elif args.cmd == "samples":
        write_samples(args.pred, args.split, args.out)

    elif args.cmd == "attention":
        sp = _load_sp()
        model = load_model(args.ckpt)
        plot_attention(model, sp, args.split, args.index, args.out)


if __name__ == "__main__":
    main()