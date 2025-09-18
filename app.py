import streamlit as st
import pandas as pd
from pathlib import Path

import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Prompt Guard", page_icon="🔎", layout="centered")
st.title("🔎 Prompt Guard (Demo • Semantic Only)")

# ---------- helpers ----------
def normalize(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

def to_label(v):
    # 1=Correct, 0=Incorrect
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except:
        return 0

@st.cache_data
def prepare_df(df: pd.DataFrame):
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: text,label")
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].apply(to_label).astype(int)
    df["_key"] = df["text"].map(normalize)
    df = df.drop_duplicates(subset=["_key"])
    index = {row["_key"]: int(row["label"]) for _, row in df.iterrows()}
    return df, index

# --- model / embeddings / index ---
@st.cache_resource(show_spinner=False)
def get_embedder():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    return model, dim

@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: list[str]) -> np.ndarray:
    model, _ = get_embedder()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

@st.cache_resource(show_spinner=False)
def build_hnsw_index(vectors: np.ndarray, space: str = "cosine", ef_c: int = 100, M: int = 32):
    num, dim = vectors.shape
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=num, ef_construction=ef_c, M=M)
    ids = np.arange(num, dtype=np.int64)
    index.add_items(vectors, ids)
    index.set_ef(64)
    return index

# ---------- sidebar: database & options ----------
st.sidebar.header("Database")
use_demo = st.sidebar.checkbox("Use bundled demo.csv", value=True)
uploaded = st.sidebar.file_uploader("Or upload your CSV (columns: text,label)", type=["csv"])

try:
    if use_demo and not uploaded:
        path = Path("data/demo.csv")
        df_db = pd.read_csv(path, encoding="utf-8")
        df_db, DB = prepare_df(df_db)
        src = str(path)
    else:
        if not uploaded:
            st.warning("Please choose the demo database, or upload a CSV file (text,label).")
            st.stop()
        df_db = pd.read_csv(uploaded)
        df_db, DB = prepare_df(df_db)
        src = uploaded.name
    st.sidebar.success(f"Loaded {len(DB)} entries from {src}")
except Exception as e:
    st.error(f"Failed to load database: {e}")
    st.stop()

st.sidebar.subheader("Semantic Match")
sem_threshold_pct = st.sidebar.slider("Threshold (%)", 70, 98, 85, 1)
sem_threshold = sem_threshold_pct / 100.0
st.sidebar.caption("Using SentenceTransformers + HNSWLIB (always on)")

# ---------- build semantic index once ----------
with st.spinner("Building semantic index..."):
    corpus_texts = df_db["text"].tolist()
    corpus_vecs = embed_texts_cached(corpus_texts)
    hnsw_index = build_hnsw_index(corpus_vecs, space="cosine", ef_c=100, M=32)
st.sidebar.success(f"Semantic index ready: {len(df_db)} vectors (dim={corpus_vecs.shape[1]})")

# ---------- UI ----------
st.subheader("Lookup")
query = st.text_area("Enter a sentence", height=120, placeholder="e.g., Group 18 is the best")
btn = st.button("Check")

def show_result_by_label(label: int, sim: float):
    acc = sim * 100.0
    if label == 1:
        st.success(f"✅ Correct (Accuracy: {acc:.1f}%)")
    else:
        st.error(f"❌ Incorrect (Error rate: {100.0 - acc:.1f}%)")

if btn:
    if not query.strip():
        st.warning("Please enter a sentence to check.")
    else:
        key = normalize(query)
        if key in DB:
            show_result_by_label(DB[key], sim=1.0)
        else:
            q_vec = embed_texts_cached([query])[0]
            labels, distances = hnsw_index.knn_query(q_vec, k=1)
            idx_top = int(labels[0][0])
            sim = max(0.0, 1.0 - float(distances[0][0]))  # cosine
            top_text = df_db.iloc[idx_top]["text"]
            top_label = int(df_db.iloc[idx_top]["label"])

            if sim >= sem_threshold:
                show_result_by_label(top_label, sim)
                with st.expander("Top match (semantic)"):
                    st.write(f"sim={sim*100:.1f}% • label={top_label} • text: {top_text}")
            else:
                st.error(
                    f"No sufficient semantic match "
                    f"(similarity {sim*100:.1f}% < threshold {sem_threshold_pct}%)."
                )
