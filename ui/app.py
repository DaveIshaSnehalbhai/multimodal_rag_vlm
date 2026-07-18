"""
ui/app.py — IndicRAG-VLM Streamlit Demo

Shows BOTH captioning modes side by side for every uploaded image:
  - No-RAG  : Qwen2-VL captions from the image alone
  - RAG     : Qwen2-VL captions grounded in retrieved context
              (BGE-M3 text retrieval + CLIP visual retrieval over the
              WIT + Sangraha + Flickr30k corpus stored in Qdrant)

Run:
    streamlit run ui/app.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.hf_auth import ensure_hf_login
from src.retrieval import VectorStore
from src.generation import Captioner, IndicRAGPipeline

# ── Defaults (edit or override via sidebar / env vars) ─────────────────────────
# Resolved to absolute paths with pathlib so this works the same on Windows
# and Linux/Mac regardless of the working directory `streamlit run` was
# launched from.
DEFAULT_ADAPTER = str(ROOT / "qwen2vl" / "content" / "qwen2vl_indic_rag_lora" / "final_adapter")
DEFAULT_QDRANT  = str(ROOT / "qdrant_db" / "content" / "qdrant_data")

# Common wrong-but-easy-to-produce shapes (e.g. unzipping qdrant_db.zip a
# second time, or dropping the archive straight into qdrant_db/) — checked
# automatically as a fallback so a path typo doesn't silently fall back to
# an empty in-memory store.
QDRANT_FALLBACKS = [
    str(ROOT / "qdrant_db" / "qdrant_db" / "content" / "qdrant_data"),
    str(ROOT / "qdrant_db" / "content" / "qdrant_data" / "content" / "qdrant_data"),
    str(ROOT / "qdrant_db"),
]

LANG_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali",
              "mr": "Marathi", "gu": "Gujarati"}
LANG_FLAGS = {"en": "🇬🇧", "hi": "🇮🇳", "bn": "🇧🇩", "mr": "🇮🇳", "gu": "🇮🇳"}

# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IndicRAG-VLM", page_icon="🖼️", layout="wide")
st.title("🖼️ IndicRAG-VLM")
st.markdown("**Multilingual Retrieval-Augmented Image Captioning** — "
            "side-by-side comparison of **No-RAG** vs **RAG** outputs · "
            "English · Hindi · Bengali · Marathi · Gujarati")
st.divider()

# ── HF login (only needed for gated/private repos) ─────────────────────────────
@st.cache_resource(show_spinner=False)
def _cached_hf_login() -> bool:
    """Streamlit reruns the whole script on every interaction — cache this
    so the login check (and its console message) runs once per process,
    not once per click."""
    return ensure_hf_login(required=False)


hf_logged_in = _cached_hf_login()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    if hf_logged_in:
        st.success("🔑 Hugging Face: logged in")
    else:
        st.info("🔑 Hugging Face: no token set (fine for public models)")
        with st.expander("How to add a token"):
            st.code(
                "export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n"
                "streamlit run ui/app.py",
                language="bash")
            st.caption(
                "Get a token at huggingface.co/settings/tokens. "
                "Needed only for gated/private models — Qwen2-VL-2B-Instruct, "
                "BAAI/bge-m3 and CLIP ViT-L/14 are public.")

    adapter_path = st.text_input(
        "LoRA Adapter Path",
        value=os.environ.get("ADAPTER_PATH", DEFAULT_ADAPTER),
        help="Directory with fine-tuned LoRA weights (adapter_config.json etc.). "
             "Leave as-is / clear it to use the base Qwen2-VL-2B model."
    ).strip()

    qdrant_path = st.text_input(
        "Qdrant DB Path",
        value=os.environ.get("QDRANT_PATH", DEFAULT_QDRANT),
        help="Path to persistent Qdrant vector store."
    ).strip()

    top_k = st.slider("Retrieved chunks (top_k)", 1, 10, 5)
    min_score = st.slider("Min retrieval score", 0.0, 1.0, 0.0, 0.05)

    langs = st.multiselect(
        "Output languages",
        options=list(LANG_NAMES.keys()),
        default=list(LANG_NAMES.keys()),
        format_func=lambda x: f"{LANG_FLAGS[x]} {LANG_NAMES[x]}")

    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- `Qwen2-VL-2B` (+ optional QLoRA adapter)\n"
        "- `CLIP ViT-L/14` (dim=768) — visual retrieval\n"
        "- `BGE-M3` (dim=1024) — text retrieval\n"
        "- `Qdrant` vector store\n"
        "- `Google Translate`")

    with st.expander("🩺 Path diagnostics"):
        st.caption(f"Project root: `{ROOT}`")
        for label, p in [("Adapter", adapter_path), ("Qdrant", qdrant_path)]:
            ok = bool(p) and os.path.exists(p)
            st.markdown(f"{'✅' if ok else '❌'} **{label}**")
            st.code(p or "(empty)", language=None)


def _resolve_qdrant_path(path: str) -> tuple[str | None, list[str]]:
    """
    Try the given path, then a few common fallback shapes (double-nested
    extraction, archive dropped straight into qdrant_db/, etc). Returns
    (working_path_or_None, list_of_every_path_checked) so the caller can
    show exactly what was tried instead of a vague "not found".
    """
    candidates = [path] + [c for c in QDRANT_FALLBACKS if c != path]
    checked = []
    for c in candidates:
        checked.append(c)
        if c and os.path.isdir(c):
            # A real Qdrant dir has either a collection/ subfolder or meta.json
            if os.path.exists(os.path.join(c, "meta.json")) or \
               os.path.isdir(os.path.join(c, "collection")):
                return c, checked
    return None, checked


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Qdrant + loading encoders ...")
def _load_store(path: str) -> VectorStore:
    resolved, checked = _resolve_qdrant_path(path)
    if resolved:
        if resolved != path:
            st.info(f"Qdrant DB Path field didn't match — found a valid "
                     f"store instead at:\n\n`{resolved}`\n\nUpdate the "
                     f"sidebar field to this path to silence this message.")
        return VectorStore(path=resolved)

    st.warning(
        "Qdrant DB not found — using an empty in-memory store "
        "(RAG will retrieve nothing). Checked these paths:\n\n"
        + "\n".join(f"- `{c}`" for c in checked)
        + "\n\nMake sure `qdrant_db/content/qdrant_data/` (containing "
          "`meta.json` and a `collection/` folder) exists under the "
          "project root, or set the Qdrant DB Path field / `QDRANT_PATH` "
          "env var to wherever you extracted it.")
    return VectorStore(path=None)


@st.cache_resource(show_spinner="Loading Qwen2-VL ...")
def _load_captioner(adapter: str) -> Captioner:
    adapter = adapter if (adapter and os.path.isdir(adapter)) else None
    return Captioner(adapter_path=adapter)


@st.cache_data(show_spinner=False)
def _load_train_df() -> pd.DataFrame | None:
    """Optional: caption lookup table for visually-similar train images."""
    csv_path = ROOT / "flickr_indic.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
if not uploaded:
    st.info("Upload an image to generate multilingual captions with and "
             "without retrieval-augmented generation.")
    st.stop()

with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    tmp.write(uploaded.read())
    img_path = tmp.name

c1, c2 = st.columns([1, 2])
with c1:
    st.image(Image.open(img_path), width="stretch")
with c2:
    st.markdown(
        "**What happens when you click Generate**\n\n"
        "Both pipelines run on the same image so you can compare them:\n"
        "1. **No-RAG** — Qwen2-VL captions the image alone.\n"
        "2. **RAG** — an initial caption is used as a retrieval query against "
        "BGE-M3 text chunks + CLIP-similar images in Qdrant, then Qwen2-VL "
        "captions again *with* that retrieved context.\n"
        "3. Both English captions are translated into hi / bn / mr / gu.")

if not st.button("▶ Generate (No-RAG + RAG)", type="primary", width="stretch"):
    st.stop()

# ── Load models / pipeline ──────────────────────────────────────────────────
store = _load_store(qdrant_path)
captioner = _load_captioner(adapter_path)
df_train = _load_train_df()
pipeline = IndicRAGPipeline(store, captioner, df_train=df_train)

with st.sidebar:
    st.success("✅ Fine-tuned LoRA loaded") if captioner.adapter_loaded \
        else st.info("ℹ️ Base Qwen2-VL-2B (no adapter found)")

# ── Run BOTH pipelines ───────────────────────────────────────────────────────
timings = {}

with st.spinner("Running No-RAG pipeline ..."):
    t0 = time.time()
    result_no_rag = pipeline.run(img_path, use_rag=False)
    timings["No-RAG"] = time.time() - t0

with st.spinner("Running RAG pipeline (retrieval + grounded captioning) ..."):
    t0 = time.time()
    result_rag = pipeline.run(img_path, use_rag=True, top_k=top_k, min_score=min_score)
    timings["RAG"] = time.time() - t0

# ── Results — side by side ──────────────────────────────────────────────────
st.divider()
st.subheader("📝 Captions — No-RAG vs RAG")

col_no, col_rag = st.columns(2)

with col_no:
    st.markdown("### 🚫 No-RAG")
    for lang in langs:
        cap = result_no_rag["captions"].get(lang, "")
        st.markdown(f"**{LANG_FLAGS[lang]} {LANG_NAMES[lang]}**")
        st.success(cap) if cap else st.warning("Not generated.")

with col_rag:
    st.markdown("### 📚 RAG")
    for lang in langs:
        cap = result_rag["captions"].get(lang, "")
        st.markdown(f"**{LANG_FLAGS[lang]} {LANG_NAMES[lang]}**")
        st.success(cap) if cap else st.warning("Not generated.")

# ── Retrieved context (RAG only) ────────────────────────────────────────────
st.divider()
st.subheader("📚 Retrieved Context (used by the RAG pipeline)")
rag_chunks = result_rag["rag_chunks"]
similar_imgs = result_rag["similar_images"]
context_str = result_rag["context_str"]

if not context_str:
    st.warning("No context was retrieved above the score threshold — "
               "RAG caption fell back to the plain caption.")
else:
    st.caption(f"Context length: {len(context_str)} chars · "
               f"Chunks: {len(rag_chunks)} · Similar images: {len(similar_imgs)}")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**Text chunks (BGE-M3)**")
        for i, ch in enumerate(rag_chunks):
            with st.expander(f"Chunk {i + 1} · {ch['source']} · score={ch['score']}"):
                st.write(ch["text"])
    with tc2:
        st.markdown("**Similar images (CLIP)**")
        for s in similar_imgs:
            st.markdown(f"`{s['image_name']}` — score={s['score']:.3f}")
            st.caption(s["caption"] or "(no caption)")

# ── Log / debug ──────────────────────────────────────────────────────────────
st.divider()
with st.expander("🔧 Log"):
    st.json({
        "adapter_path": adapter_path,
        "adapter_loaded": captioner.adapter_loaded,
        "qdrant_path": qdrant_path,
        "top_k": top_k,
        "min_score": min_score,
        "context_chars": len(context_str),
    })
    st.dataframe(
        pd.DataFrame({"Pipeline": list(timings),
                      "Seconds": [f"{v:.2f}" for v in timings.values()]}),
        hide_index=True)