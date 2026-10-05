"""Streamlit front end: English question + column names -> SQL, using our trained Transformer."""
import os
import sys

import torch
import sentencepiece as spm
import streamlit as st

# ---------------- Paths: repo root (for model/ and decode.py) and starter/ ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from model.transformer import Transformer   # noqa: E402
from decode import predict                  # noqa: E402
from tokenizer import PAD_ID                # noqa: E402

MAX_COLS = 64      # the model only knows <c0> ... <c63>
ALPHA = 0.6        # beam length penalty chosen on dev

# Prefer the final model (average of epochs 16-20); fall back to best.pt
_final = os.path.join(ROOT, "checkpoints", "final.pt")
_best = os.path.join(ROOT, "checkpoints", "best.pt")
MODEL_PATH = os.environ.get("MODEL_PATH", _final if os.path.exists(_final) else _best)

EXAMPLES = [
    {   # dev example
        "label": "Basketball roster: count",
        "question": "How many schools did player number 3 play at?",
        "columns": "Player, No., Nationality, Position, Years in Toronto, School/Club Team",
    },
    {   # dev example
        "label": "Basketball roster: two conditions",
        "question": "What player played guard for toronto in 1996-97?",
        "columns": "Player, No., Nationality, Position, Years in Toronto, School/Club Team",
    },
    {   # a table the model has never seen
        "label": "Your own table: countries",
        "question": "What is the capital of France?",
        "columns": "Country, Capital, Population, Area",
    },
]


# ---------------- Load the model once (Streamlit reruns the script on every click) ----------------
@st.cache_resource
def load_assets():
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(ROOT, "starter", "sql_sp.model"))
    model = Transformer(sp.get_piece_size(), pad_id=PAD_ID)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, sp


def fill_example(ex):
    """Callback: runs before the rerun, so it may safely change widget values."""
    st.session_state["question"] = ex["question"]
    st.session_state["columns"] = ex["columns"]


# ---------------- Page ----------------
st.set_page_config(page_title="Text-to-SQL", page_icon="🗄️", layout="centered")
st.title("🗄️ Text-to-SQL")
st.caption("Ask a question about a table in plain English and get the SQL query that answers it.")

with st.sidebar:
    st.header("Settings")
    choice = st.radio("Decoding", ["Beam search (best quality)", "Greedy (faster)"])
    method = "beam" if choice.startswith("Beam") else "greedy"

    st.header("About")
    st.markdown(
        "An encoder–decoder **Transformer** (*Attention Is All You Need*) built from "
        "basic PyTorch layers and trained **from scratch** on WikiSQL. "
        "No pretrained weights.\n\n"
        "- 7.6M parameters, 3 + 3 layers, d_model 256\n"
        "- Dev: **48.4%** logical form, **56.0%** execution\n"
        "- LSTM baseline (WikiSQL paper): ~36% execution\n\n"
        "[GitHub repository](https://github.com/KayTheCoder-101/text-to-sql-transformer)"
    )
    st.caption(f"Model file: `{os.path.basename(MODEL_PATH)}`")

# Make sure the input keys exist before the widgets are drawn
st.session_state.setdefault("question", "")
st.session_state.setdefault("columns", "")

st.subheader("Try an example")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex["label"], on_click=fill_example, args=(ex,), use_container_width=True)

st.subheader("Your question")
st.text_input("Question", key="question",
              placeholder="e.g. What is the nationality of Terrence Ross?")
st.text_area("Column names (comma-separated)", key="columns", height=80,
             placeholder="e.g. Player, No., Nationality, Position")

if st.button("Generate SQL", type="primary", use_container_width=True):
    question = st.session_state["question"].strip()
    header = [c.strip() for c in st.session_state["columns"].split(",") if c.strip()]

    if not question:
        st.error("Please type a question.")
    elif not header:
        st.error("Please enter at least one column name.")
    elif len(header) > MAX_COLS:
        st.error(f"Too many columns ({len(header)}). The model supports at most {MAX_COLS}.")
    else:
        model, sp = load_assets()
        with st.spinner("Generating SQL..."):
            query, sql = predict(model, sp, question, header, method=method, alpha=ALPHA)

        if sql is None:
            st.warning("The model couldn't produce a valid query for this input. "
                       "Try rephrasing the question or checking the column names.")
        else:
            st.success("Generated SQL")
            st.code(sql, language="sql")

        with st.expander("How the model read your table"):
            st.markdown("Each column is given a token `<cK>`; the model points at columns by these tokens.")
            st.table({"Token": [f"<c{i}>" for i in range(len(header))], "Column": header})
            if query is not None:
                st.markdown("Parsed query (WikiSQL format):")
                st.json(query)