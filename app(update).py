
import os
import re
import json
import time
import requests
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ---- NEW: Semantic retrieval stack (replace TF-IDF) ----
import hnswlib
from sentence_transformers import SentenceTransformer

# NLTK (for simple preprocessing / stopwords)
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Physics Fact Checker (Semantic + Ollama-ready)",
                   page_icon="🧪", layout="wide")
st.title("🧪 Physics Fact Checker — Semantic Core + (Optional) Ollama")

# ---------- Utilities ----------
def _ensure_nltk():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

_ensure_nltk()

def strip_code_fences(s: str) -> str:
    """Remove ```json ... ``` fences if model wraps JSON."""
    if s is None:
        return ""
    s = s.strip()
    if s.startswith("```"):
        # remove first fenced block
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def safe_json_loads(s: str):
    """Try to extract JSON object even if extra text surrounds it."""
    if not s:
        return None
    s = strip_code_fences(s)
    # Try to find the first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = s[start : end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            pass
    # Fallback
    try:
        return json.loads(s)
    except Exception:
        return None

# ---------- Ollama client (kept) ----------
class OllamaClient:
    """Client for interacting with local Ollama API (default http://localhost:11434)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_available_models(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return [m.get("name") for m in models]
        except Exception:
            pass
        return []

    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1):
        """Call /api/generate. Return raw 'response' text or None."""
        if not self.available:
            return None
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return None
        except Exception as e:
            st.error(f"Ollama API error: {e}")
            return None

# ---------- NEW: SentenceTransformers + HNSW helpers ----------
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

# ---------- Fact Checker (Semantic core; Ollama optional) ----------
class AdvancedPhysicsFactChecker:
    def __init__(self, dataset: pd.DataFrame, ollama_client: OllamaClient | None = None):
        self.dataset = dataset
        self.ollama_client = ollama_client

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        # physics concept keywords
        self.physics_concepts = {
            'temperature': ['celsius', 'fahrenheit', 'kelvin', 'degrees', 'hot', 'cold', 'boil', 'freeze', 'melt'],
            'energy': ['joule', 'calorie', 'kinetic', 'potential', 'thermal', 'nuclear', 'electromagnetic'],
            'waves': ['frequency', 'wavelength', 'amplitude', 'light', 'sound', 'electromagnetic', 'radio'],
            'mechanics': ['force', 'mass', 'weight', 'acceleration', 'velocity', 'momentum', 'gravity'],
            'electricity': ['current', 'voltage', 'resistance', 'charge', 'electron', 'proton', 'field'],
            'quantum': ['photon', 'quantum', 'particle', 'wave', 'uncertainty', 'probability', 'orbital'],
            'thermodynamics': ['entropy', 'heat', 'temperature', 'pressure', 'volume', 'gas', 'liquid', 'solid'],
        }

        # misconception regex patterns -> handler
        self.misconception_patterns = [
            (r'water boils at (\d+)°?[cf]', self._check_boiling_point),
            (r'sound.*faster.*light', self._check_speed_comparison),
            (r'heavier.*fall.*faster', self._check_falling_objects),
            (r'all.*orbit.*perfect.*circle', self._check_orbital_shape),
            (r'mass.*weight.*same|interchangeable', self._check_mass_weight),
            (r'all.*radiation.*man.?made', self._check_radiation_source),
            (r'electrons.*orbit.*like.*planets', self._check_atomic_model),
        ]

        # ---- NEW: build semantic index for dataset (replace TF-IDF precompute) ----
        self.corpus = self.dataset["Statement"].astype(str).tolist()
        self.corpus_vecs = embed_texts_cached(self.corpus)
        self.hnsw = build_hnsw_index(self.corpus_vecs, space="cosine", ef_c=100, M=32)

    # ------------- internals -------------
    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'°c|celsius', 'celsius', text)
        text = re.sub(r'°f|fahrenheit', 'fahrenheit', text)
        text = re.sub(r'[^a-z0-9\s\.\-]', ' ', text)
        tokens = [self.stemmer.stem(t) for t in text.split() if t not in self.stop_words]
        return " ".join(tokens)

    def _physics_concept_matching(self, text: str) -> dict:
        text_l = text.lower()
        scores = {}
        for concept, kws in self.physics_concepts.items():
            hits = sum(1 for kw in kws if kw in text_l)
            scores[concept] = min(1.0, hits / max(3, len(kws) * 0.33))
        return scores

    def _check_misconception_patterns(self, prompt: str):
        issues = []
        for pattern, handler in self.misconception_patterns:
            m = re.search(pattern, prompt, flags=re.I)
            if m:
                item = handler(m, prompt)
                if item:
                    issues.append(item)
        return issues

    # ---- NEW: semantic similarity via HNSW (no TF-IDF) ----
    def _semantic_similarity(self, prompt: str, k: int = 5) -> np.ndarray:
        q_vec = embed_texts_cached([prompt])[0]
        k = max(1, min(k, len(self.corpus)))
        labels, distances = self.hnsw.knn_query(q_vec, k=k)
        sims = 1.0 - distances[0]           # cosine similarity
        idxs = labels[0].astype(int)

        # Produce a dense vector aligned with dataset length (top-k filled, others 0)
        sim_vec = np.zeros(len(self.corpus), dtype=np.float32)
        sim_vec[idxs] = sims
        return sim_vec

    # ---- Misconception handlers ----
    def _check_boiling_point(self, match, prompt):
        temp = int(match.group(1))
        if 'celsius' in prompt.lower():
            if temp != 100:
                return {'type': 'temperature_misconception',
                        'description': f'Water boils at 100°C at standard pressure, not {temp}°C',
                        'severity': 'high'}
        elif 'fahrenheit' in prompt.lower():
            if temp != 212:
                return {'type': 'temperature_misconception',
                        'description': f'Water boils at 212°F at standard pressure, not {temp}°F',
                        'severity': 'high'}
        return None

    def _check_speed_comparison(self, match, prompt):
        if 'sound' in prompt.lower() and 'faster' in prompt.lower() and 'light' in prompt.lower():
            return {'type': 'speed_misconception',
                    'description': 'Light travels much faster than sound (~300,000 km/s vs ~343 m/s)',
                    'severity': 'high'}
        return None

    def _check_falling_objects(self, match, prompt):
        return {'type': 'gravity_misconception',
                'description': "In a vacuum, all objects fall at the same rate regardless of mass (Galileo's principle)",
                'severity': 'medium'}

    def _check_orbital_shape(self, match, prompt):
        return {'type': 'astronomy_misconception',
                'description': 'Planetary orbits are elliptical, not perfectly circular (Kepler’s laws)',
                'severity': 'medium'}

    def _check_mass_weight(self, match, prompt):
        return {'type': 'mechanics_misconception',
                'description': 'Mass is the amount of matter; weight is the gravitational force on that mass',
                'severity': 'medium'}

    def _check_radiation_source(self, match, prompt):
        return {'type': 'physics_misconception',
                'description': 'Natural radiation exists (cosmic rays, radioactive elements, solar radiation)',
                'severity': 'medium'}

    def _check_atomic_model(self, match, prompt):
        return {'type': 'quantum_misconception',
                'description': 'Electrons exist in probability clouds (orbitals), not defined orbital paths',
                'severity': 'medium'}

    def _calculate_confidence(self, similarities, concept_matches, misconceptions, ollama_result=None):
        # base from top similarity + boosts (kept same semantics as teammate’s)
        max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
        base_conf = max_similarity
        concept_boost = (sum(concept_matches.values()) * 0.1) if concept_matches else 0.0
        misconception_boost = len(misconceptions) * 0.3

        ollama_boost = 0.0
        if ollama_result and isinstance(ollama_result, dict) and "confidence" in ollama_result:
            try:
                ollama_boost = float(ollama_result["confidence"]) * 0.2
            except Exception:
                ollama_boost = 0.0

        return min(0.95, base_conf + concept_boost + misconception_boost + ollama_boost)

    # ---------- Ollama-based helpers (kept) ----------
    def _ollama_fact_check(self, prompt: str):
        if not self.ollama_client or not self.ollama_client.available:
            return None
        sys_prompt = (
            "You are a concise physics fact-checking assistant. "
            "Given a statement, respond ONLY with a compact JSON object with keys: "
            "is_correct (bool), confidence (0..1), explanation (string), corrections (string)."
        )
        user = f"Statement: {prompt}\nReturn only JSON."
        full_prompt = f"{sys_prompt}\n\n{user}"
        raw = self.ollama_client.generate_response(full_prompt, max_tokens=300, temperature=0.1)
        data = safe_json_loads(raw)
        return data

    def _ollama_explain_concepts(self, prompt: str):
        if not self.ollama_client or not self.ollama_client.available:
            return None
        p = (
            'You are a physics teacher. Explain briefly and clearly the key physics concepts in the statement below.\n'
            f'Statement: "{prompt}"\nKeep it under 120 words.'
        )
        return self.ollama_client.generate_response(p, max_tokens=200, temperature=0.3)

    # ---------- main API ----------
    def analyze_prompt(self, prompt: str, similarity_threshold: float = 0.3, use_ollama: bool = True):
        t0 = time.time()

        sim_vec = self._semantic_similarity(prompt)  # NEW semantic similarities
        concept_matches = self._physics_concept_matching(prompt)
        misconceptions = self._check_misconception_patterns(prompt)

        # (Optional) Ollama calls — kept as teammate code’s behavior
        ollama_result = None
        ollama_explanation = None
        if use_ollama and self.ollama_client:
            ollama_result = self._ollama_fact_check(prompt)
            ollama_explanation = self._ollama_explain_concepts(prompt)

        # build top-5 similar list like before
        top_idx = np.argsort(sim_vec)[::-1][:5]
        similar_statements = []
        for idx in top_idx:
            if float(sim_vec[idx]) >= similarity_threshold:
                row = self.dataset.iloc[idx]
                similar_statements.append({
                    'id': row.get('ID', idx),
                    'statement': row.get('Statement', ''),
                    'is_true': int(row.get('IsTrue', 1)),
                    'difficulty': row.get('Difficulty', 'Unknown'),
                    'category': row.get('Category', 'Unknown'),
                    'similarity_score': float(sim_vec[idx]),
                    'match_type': 'semantic'
                })

        confidence = self._calculate_confidence(sim_vec, concept_matches, misconceptions, ollama_result)

        is_flagged = False
        primary_reason = "No issues detected"

        # Decision order (kept; Ollama verdict first if says incorrect)
        if isinstance(ollama_result, dict) and (not ollama_result.get("is_correct", True)):
            is_flagged = True
            primary_reason = "AI analysis suggests statement is incorrect"
        elif len(misconceptions) > 0:
            is_flagged = True
            primary_reason = f"Detected {len(misconceptions)} physics misconception(s)"
        elif similar_statements and int(similar_statements[0]['is_true']) == 0:
            is_flagged = True
            primary_reason = "Similar to known incorrect statement"
        elif confidence < 0.2:
            primary_reason = "Insufficient information to verify"

        return {
            'flagged': is_flagged,
            'confidence': float(confidence),
            'primary_reason': primary_reason,
            'misconceptions': misconceptions,
            'physics_concepts': concept_matches,
            'similar_statements': similar_statements,
            'ollama_result': ollama_result,
            'ollama_explanation': ollama_explanation,
            'analysis_time': round(time.time() - t0, 4)
        }

# ---------- dataset loader ----------
def load_dataset():
    """
    Load dataset from ./demo_dataset.csv
    Expected columns: ID, Statement, IsTrue, Difficulty, Category
    Encoding: cp1252 (as teammate used).
    """
    csv_path = "./demo_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="cp1252")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(subset=["Statement"])
        if "IsTrue" in df.columns:
            df["IsTrue"] = df["IsTrue"].astype(int)
        if "Category" in df.columns:
            df["Category"] = df["Category"].fillna("Physics")
        if "Difficulty" in df.columns:
            df["Difficulty"] = df["Difficulty"].fillna("Unknown")
        # Basic sanity fill
        if "ID" not in df.columns:
            df["ID"] = np.arange(1, len(df) + 1)
        return df.reset_index(drop=True)
    else:
        # fallback tiny dataset
        data = [
            {"ID": 1, "Statement": "Water boils at 100 Celsius at sea level", "IsTrue": 1, "Difficulty": "Easy", "Category": "Thermodynamics"},
            {"ID": 2, "Statement": "Sound travels faster than light", "IsTrue": 0, "Difficulty": "Easy", "Category": "Waves"},
            {"ID": 3, "Statement": "Electrons orbit the nucleus like planets", "IsTrue": 0, "Difficulty": "Medium", "Category": "Quantum"},
            {"ID": 4, "Statement": "In vacuum, all objects fall at the same rate", "IsTrue": 1, "Difficulty": "Easy", "Category": "Mechanics"},
        ]
        return pd.DataFrame(data)

# ---------- UI ----------
def main():
    # Sidebar: dataset + semantic threshold + ollama config
    st.sidebar.header("Settings")

    similarity_threshold = st.sidebar.slider("Similarity Threshold (semantic)", 0.1, 0.9, 0.3, 0.05)

    st.sidebar.subheader("Ollama (optional)")
    base_url = st.sidebar.text_input("Base URL", value="http://localhost:11434")
    model = st.sidebar.text_input("Model", value="llama3.2")
    use_ai_single = st.sidebar.checkbox("Use AI in Single Analysis", value=False)
    use_ai_batch = st.sidebar.checkbox("Use AI in Batch Analysis", value=False)

    # Instantiate Ollama client
    ollama_client = OllamaClient(base_url=base_url, model=model)
    if ollama_client.available:
        st.sidebar.success("Ollama connected ✅")
        models = ", ".join(ollama_client.get_available_models()[:6]) or "(no models listed)"
        st.sidebar.caption(f"Models: {models}")
    else:
        st.sidebar.warning("Ollama not available (optional).")

    # Load dataset
    dataset = load_dataset()
    st.sidebar.info(f"Dataset loaded: {len(dataset)} rows")

    # Checker with semantic core + optional Ollama
    checker = AdvancedPhysicsFactChecker(dataset, ollama_client=ollama_client)

    # Page nav
    page = st.sidebar.radio("Pages", ["Dashboard", "Single Analysis", "Batch Analysis", "Dataset Explorer", "API Documentation"])

    if page == "Dashboard":
        st.header("Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Statements", f"{len(dataset):,}")
        with col2:
            if "IsTrue" in dataset.columns:
                st.metric("True", f"{int((dataset['IsTrue'] == 1).sum()):,}")
        with col3:
            if "IsTrue" in dataset.columns:
                st.metric("False", f"{int((dataset['IsTrue'] == 0).sum()):,}")
        with col4:
            if "IsTrue" in dataset.columns:
                acc = (dataset["IsTrue"] == 1).mean() if len(dataset) else 0.0
                st.metric("Base Accuracy", f"{acc:.1%}")

        st.write("This dashboard summarizes dataset counts and baseline accuracy. Use other tabs to analyze statements.")

    elif page == "Single Analysis":
        st.header("Single Statement Analysis")
        user_input = st.text_area("Enter a physics statement:", height=120,
                                  placeholder="e.g., Water boils at 50°C")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Analyze", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("Analyzing..."):
                        result = checker.analyze_prompt(user_input, similarity_threshold, use_ollama=use_ai_single)

                        if result['flagged']:
                            st.error(f"Flagged: {result['primary_reason']}")
                        else:
                            st.success("Approved: No issues detected")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        with c2:
                            st.metric("Time", f"{result['analysis_time']*1000:.1f} ms")
                        with c3:
                            st.metric("Issues", f"{len(result['misconceptions'])}")

                        if result['misconceptions']:
                            st.subheader("Detected Issues")
                            for i, m in enumerate(result['misconceptions'], 1):
                                with st.expander(f"Issue {i}: {m['type'].replace('_',' ').title()}"):
                                    st.write(f"**Severity:** {m['severity'].title()}")
                                    st.write(f"**Description:** {m['description']}")

                        if result['physics_concepts']:
                            st.subheader("Physics Concepts Detected")
                            cols = st.columns(max(1, len(result['physics_concepts'])))
                            for i, (k, v) in enumerate(result['physics_concepts'].items()):
                                with cols[i]:
                                    st.metric(k.title(), f"{v:.1%}")

                        if result['similar_statements']:
                            st.subheader("Top Similar (Semantic)")
                            df_sim = pd.DataFrame(result['similar_statements'])
                            st.dataframe(df_sim, use_container_width=True)

                        if result.get('ollama_result') or result.get('ollama_explanation'):
                            with st.expander("🤖 AI Details"):
                                if result.get('ollama_result'):
                                    st.write("**AI Verdict (JSON)**")
                                    st.code(json.dumps(result['ollama_result'], indent=2))
                                if result.get('ollama_explanation'):
                                    st.write("**AI Explanation**")
                                    st.write(result['ollama_explanation'])

        with col_b:
            st.write("**Threshold tip:** Higher threshold ⇒ stricter match; lower ⇒ broader match.")

    elif page == "Batch Analysis":
        st.header("Batch Statement Analysis")
        batch_input = st.text_area("Enter multiple statements (one per line):",
                                   height=200,
                                   placeholder="Water boils at 50°C\nSound travels faster than light\nE = mc²")
        if st.button("Analyze Batch", type="primary"):
            if batch_input.strip():
                statements = [s.strip() for s in batch_input.split("\n") if s.strip()]
                results_summary = []
                progress = st.progress(0)
                status = st.empty()
                for i, statement in enumerate(statements):
                    status.text(f"Analyzing {i+1}/{len(statements)}")
                    res = checker.analyze_prompt(statement, similarity_threshold, use_ollama=use_ai_batch)
                    results_summary.append({
                        "Statement": (statement[:60] + "…") if len(statement) > 60 else statement,
                        "Status": "FLAGGED" if res['flagged'] else "APPROVED",
                        "Confidence": f"{res['confidence']:.1%}",
                        "Issues": len(res['misconceptions']),
                        "Time_ms": f"{res['analysis_time']*1000:.1f}"
                    })
                    progress.progress((i + 1) / len(statements))
                status.text("Done.")
                st.subheader("Batch Results")
                st.dataframe(pd.DataFrame(results_summary), use_container_width=True)

                # quick metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", len(statements))
                with col2:
                    flagged = sum(1 for r in results_summary if r["Status"] == "FLAGGED")
                    st.metric("Flagged", flagged)
                with col3:
                    avg_conf = np.mean([float(r["Confidence"][:-1]) / 100 for r in results_summary]) if results_summary else 0.0
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                with col4:
                    avg_time = np.mean([float(r["Time_ms"]) for r in results_summary]) if results_summary else 0.0
                    st.metric("Avg Time", f"{avg_time:.1f} ms")

    elif page == "Dataset Explorer":
        st.header("Dataset Explorer")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Total Records:** {len(dataset):,}")
            st.write(f"**Columns:** {', '.join(dataset.columns)}")
            if "Category" in dataset.columns:
                st.write(f"**Categories:** {dataset['Category'].nunique()}")
            if "Difficulty" in dataset.columns:
                st.write(f"**Difficulty Levels:** {', '.join(sorted(dataset['Difficulty'].unique()))}")
        with col2:
            st.subheader("Data Quality")
            missing = dataset.isnull().sum()
            if missing.sum() > 0:
                st.write("**Missing Values:**")
                st.write(missing[missing > 0])
            else:
                st.success("No missing values detected")

        st.subheader("Sample / Filters")
        c1, c2 = st.columns(2)
        with c1:
            cat = st.selectbox("Filter by Category",
                               ["All"] + (sorted(dataset["Category"].unique()) if "Category" in dataset.columns else []))
        with c2:
            truth = st.selectbox("Filter by Truth Value",
                                 ["All"] + (["True", "False"] if "IsTrue" in dataset.columns else []))
        df_show = dataset.copy()
        if cat != "All" and "Category" in df_show.columns:
            df_show = df_show[df_show["Category"] == cat]
        if truth != "All" and "IsTrue" in df_show.columns:
            df_show = df_show[df_show["IsTrue"] == (1 if truth == "True" else 0)]
        st.write(f"Showing {len(df_show):,} rows")
        st.dataframe(df_show.head(100), use_container_width=True)

    elif page == "API Documentation":
        st.header("API Documentation")
        st.subheader("Quick Start (Python)")
        st.code("""
from physics_fact_checker_semantic_ollama import AdvancedPhysicsFactChecker
import pandas as pd

dataset = pd.read_csv('demo_dataset.csv', encoding='cp1252')
checker = AdvancedPhysicsFactChecker(dataset)  # Ollama optional

res = checker.analyze_prompt("Water boils at 50°C", similarity_threshold=0.3, use_ollama=False)
print(res)
        """, language="python")

        st.subheader("Response Format")
        st.code("""
{
  "flagged": bool,
  "confidence": float,          # 0..1
  "primary_reason": str,
  "misconceptions": [ ... ],
  "physics_concepts": { ... },
  "similar_statements": [ { id, statement, is_true, difficulty, category, similarity_score, match_type } ],
  "ollama_result": { ... } | null,
  "ollama_explanation": str | null,
  "analysis_time": float        # seconds
}
        """, language="json")

if __name__ == "__main__":
    main()
