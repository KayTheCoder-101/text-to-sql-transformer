import os
import re
import sys

# Resolve the starter directory relative to this file, not the working directory.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter")
)
from data_prep import AGG_OPS, COND_OPS  # single source of truth for indices

COL_RE = re.compile(r"^<c(\d+)>$")


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