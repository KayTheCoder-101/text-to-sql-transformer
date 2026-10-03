import os
import re
import sys
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F

# Resolve the starter directory relative to this file, not the working directory.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter")
)
from data_prep import AGG_OPS, COND_OPS, encode_source  # single source of truth for indices
from tokenizer import BOS_ID, EOS_ID

COL_RE = re.compile(r"^<c(\d+)>$")

MAX_LEN = 64         # stop decoding after this many tokens (Task 4.1)
THRESHOLD = 0.6      # minimum similarity for snap_value to replace a value
UNK = "⁇"            # what sentencepiece decode shows for unknown characters
EDGE_PUNCT = "'\"?,.!;:"   # punctuation that question words may carry at their edges


# ---------------------------------------------------------------- decoding (4.1)

def greedy_decode(model, src, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src)
        ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=src.device)  # (1, 1)
        cross = None

        for _ in range(max_len):
            logits, cross = model.decode(ys, enc_out, src_mask)
            next_tok = logits[:, -1].argmax(dim=-1)                 # (1,)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)      # append as new column
            if next_tok.item() == EOS_ID:
                break

    tokens = ys[0].tolist()[1:]            # drop the leading BOS
    if tokens and tokens[-1] == EOS_ID:    # drop the final EOS, if one was generated
        tokens = tokens[:-1]
    return tokens, cross


def beam_search(model, src, beam_size=4, max_len=MAX_LEN, alpha=0.0):
    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src)
        device = src.device
        beams = [([BOS_ID], 0.0)]   # (tokens, total log-probability)
        finished = []

        for _ in range(max_len):
            candidates = []
            for tokens, score in beams:
                ys = torch.tensor([tokens], dtype=torch.long, device=device)
                logits, _ = model.decode(ys, enc_out, src_mask)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                top_lp, top_ids = torch.topk(log_probs, beam_size)
                for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                    candidates.append((tokens + [tok], score + lp))

            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for tokens, score in candidates:
                if tokens[-1] == EOS_ID:
                    finished.append((tokens, score))
                else:
                    beams.append((tokens, score))
                if len(beams) == beam_size:
                    break

            if len(finished) >= beam_size or not beams:
                break

        if not finished:   # hit max_len without any EOS
            finished = beams

        best_tokens, _ = max(finished, key=lambda f: f[1] / (len(f[0]) ** alpha))

    out = best_tokens[1:]                  # drop BOS
    if out and out[-1] == EOS_ID:          # drop final EOS
        out = out[:-1]
    return out


# ------------------------------------------------------------- parsing (4.2, 4.4)

def _col_num(token):
    m = COL_RE.match(token)
    return int(m.group(1)) if m else None


def parse_sql(text):
    tokens = text.split()
    if not tokens or tokens[0] != "select":
        return None

    n = len(tokens)
    i = 1
    agg = 0

    if i < n and tokens[i].upper() in AGG_OPS:
        agg = AGG_OPS.index(tokens[i].upper())
        i += 1

    if i >= n:
        return None
    sel = _col_num(tokens[i])
    if sel is None:
        return None
    i += 1

    conds = []
    if i < n:
        if tokens[i] != "where":
            return None
        i += 1

        while i < n:
            col = _col_num(tokens[i])
            if col is None:
                return None

            if i + 1 >= n or tokens[i + 1] not in COND_OPS:
                return None
            op = COND_OPS.index(tokens[i + 1])
            i += 2

            value_words = []
            while i < n:
                if (tokens[i] == "and"
                        and i + 1 < n
                        and _col_num(tokens[i + 1]) is not None):
                    break
                value_words.append(tokens[i])
                i += 1

            # Truncated output like "... where <c0> =" has no value: parse failure.
            if not value_words:
                return None

            conds.append([col, op, " ".join(value_words)])

            if i < n and tokens[i] == "and":
                i += 1

    return {"sel": sel, "agg": agg, "conds": conds}


def _valid_index(idx, seq):
    # Reject negatives explicitly: Python would otherwise wrap them around.
    return isinstance(idx, int) and 0 <= idx < len(seq)


def to_readable_sql(query, header):
    if query is None:  # parse failure upstream flows through cleanly
        return None

    if not _valid_index(query["sel"], header):
        return None
    sel_name = header[query["sel"]]

    if not _valid_index(query["agg"], AGG_OPS):
        return None
    agg = AGG_OPS[query["agg"]]
    select_part = f"{agg}({sel_name})" if agg else sel_name

    sql = f"SELECT {select_part} FROM table"

    if query["conds"]:
        parts = []
        for col, op, value in query["conds"]:
            if not _valid_index(col, header) or not _valid_index(op, COND_OPS):
                return None
            safe_value = str(value).replace("'", "''")  # str() keeps this working for int/float values
            parts.append(f"{header[col]} {COND_OPS[op]} '{safe_value}'")
        sql += " WHERE " + " AND ".join(parts)

    return sql


# ------------------------------------------------------------------ value snapping

def _trim_variants(span):
    """The span itself plus versions with edge punctuation removed (e.g. "ross'" -> "ross")."""
    variants = {span, span.strip(EDGE_PUNCT), span.lstrip(EDGE_PUNCT), span.rstrip(EDGE_PUNCT)}
    if span and span[0] in EDGE_PUNCT:
        variants.add(span[1:])           # remove just one leading character
    if span and span[-1] in EDGE_PUNCT:
        variants.add(span[:-1])          # remove just one trailing character
    return [v for v in variants if v]


def _unk_pattern(value):
    """Regex where each ⁇ matches one or more non-space characters.
    The decoder puts spaces around ⁇, so pieces are joined with optional whitespace."""
    parts = []
    for piece in value.split():
        segments = piece.split(UNK)      # "⁇" alone -> ["", ""]
        parts.append(r"\S+?".join(re.escape(s) for s in segments))
    pattern = r"\s*".join(parts)
    if pattern.endswith(r"\S+?"):        # a trailing ⁇ should run to the end of the word
        pattern = pattern[:-len(r"\S+?")] + r"\S+"
    return pattern


def snap_value(value, question):
    q = question.lower()

    # Fast path: value is already in the question exactly
    if UNK not in value and value in q:
        return value

    key = " ".join(value.replace(UNK, "").split())   # the characters we know for sure
    if not key:
        return value                                  # nothing known to match against

    # Fix 2: treat ⁇ as a wildcard and search for the pattern directly
    if UNK in value:
        m = re.search(_unk_pattern(value), q)
        if m:
            return m.group(0).rstrip("?")

    # Fuzzy search over word spans of the question
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", q)]
    max_words = len(value.split()) + 2

    best, best_score = value, 0.0
    for i in range(len(words)):
        start = words[i][0]
        for j in range(i, min(len(words), i + max_words)):
            span = q[start:words[j][1]]
            for cand in _trim_variants(span):        # Fix 1: try punctuation-trimmed versions
                score = SequenceMatcher(None, key, cand).ratio()
                if score > best_score:
                    best, best_score = cand, score

    if best_score >= THRESHOLD:
        return best
    return value  # nothing close: keep the model's value


def snap_query(query, question):
    if query is None:
        return None
    # Build a new dict and new cond lists so the unsnapped query stays intact.
    new_conds = [
        [col, op, snap_value(value, question)]
        for col, op, value in query["conds"]
    ]
    return {"sel": query["sel"], "agg": query["agg"], "conds": new_conds}


# ---------------------------------------------------------------- full pipeline

def predict(model, sp, question, header, method="greedy", snap=True,
            beam_size=4, alpha=0.0):
    src_text = encode_source(question, header)
    ids = sp.encode(src_text) + [EOS_ID]
    device = next(model.parameters()).device
    src = torch.tensor([ids], dtype=torch.long, device=device)

    if method == "greedy":
        tokens = greedy_decode(model, src)[0]
    elif method == "beam":
        tokens = beam_search(model, src, beam_size=beam_size, alpha=alpha)
    else:
        raise ValueError(f"unknown method: {method!r}")

    text = sp.decode(tokens)
    query = parse_sql(text)
    if snap:
        query = snap_query(query, question.lower())
    return query, to_readable_sql(query, header)


# --------------------------------------------------------------------------- tests

if __name__ == "__main__":
    tests = [
        "select <c2> where <c0> = terrence ross",
        "select count <c3> where <c1> = kim manners and <c4> > 5",
        "select <c1> where <c0> = rock and roll",          # value contains "and"
        "select max <c5>",                                  # no WHERE
        "select <c2> where <c0> =",                         # truncated -> None
        "garbage output",                                   # -> None
    ]
    for t in tests:
        print(t, "\n   ->", parse_sql(t))

    header = ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"]
    print(to_readable_sql(parse_sql("select <c2> where <c0> = terrence ross"), header))
    print(to_readable_sql(parse_sql("select count <c0> where <c3> = guard and <c1> > 10"), header))
    print(to_readable_sql(parse_sql("select <c9>"), header))                 # None
    print(to_readable_sql(parse_sql("select <c2> where <c0> = o'brien"), header))  # 'o''brien'
    print(to_readable_sql(None, header))                                     # None

    # Value snapping
    print(snap_value("宁 ⁇ 县", "Which county is 宁陵县?"))                          # 宁陵县
    print(snap_value("terence ross", "What is Terrence Ross' nationality"))          # terrence ross
    print(snap_value("terrence ross", "What is Terrence Ross' nationality"))         # terrence ross (fast path)
    print(snap_value("⁇ 安 ⁇ 道", "which county is in 平安北道?"))                    # 平安北道
    print(snap_value("⁇ 도", "what city is in 강원도 province"))                       # 강원도
    print(snap_value("⁇ (清明) cheongmyeong", "what is 청명 (清明) cheongmyeong"))     # 청명 (清明) cheongmyeong

    # Decoding mechanics with a fake model (no real model needed)
    class FakeModel(torch.nn.Module):
        """Always prefers tokens 10, 11, 12, then EOS: tests decoding mechanics only."""
        def encode(self, src):
            return torch.zeros(1, src.size(1), 8), torch.ones(1, 1, 1, src.size(1), dtype=torch.bool)

        def decode(self, ys, enc_out, src_mask):
            T = ys.size(1)
            logits = torch.full((ys.size(0), T, 50), -10.0)
            target = [10, 11, 12, EOS_ID]
            for t in range(T):
                logits[:, t, target[min(t, 3)]] = 10.0
            return logits, torch.zeros(ys.size(0), 4, T, src_mask.size(-1))

    fake = FakeModel()
    src = torch.tensor([[5, 6, 7, EOS_ID]])
    tokens, cross = greedy_decode(fake, src)
    print("greedy:", tokens)                                   # [10, 11, 12]
    print("beam  :", beam_search(fake, src, beam_size=4))       # [10, 11, 12]