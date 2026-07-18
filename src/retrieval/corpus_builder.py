"""Build WIT + Sangraha + Flickr text corpus in Qdrant."""
import pandas as pd
from typing import List


def load_wit_texts(tsv_path: str, max_rows=10000) -> List[str]:
    import os
    if not os.path.exists(tsv_path):
        print(f"WIT TSV not found: {tsv_path}"); return []
    df = pd.read_csv(tsv_path, sep="\t", nrows=max_rows, on_bad_lines="skip")
    cols = ["caption_reference_description", "context_section_description",
            "context_page_description", "section_title", "page_title"]
    texts = []
    for c in cols:
        if c in df.columns:
            texts.extend(df[c].dropna().astype(str).tolist())
    texts = [t for t in texts if len(t.strip()) > 20]
    print(f"WIT: {len(texts)} texts loaded.")
    return texts


def load_sangraha_texts(rows=2000) -> List[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets"); return []
    try:
        ds  = load_dataset("ai4bharat/sangraha", "verified",
                           split=f"train[:{rows}]", trust_remote_code=True)
        col = next((c for c in ["text","content","document","sentence"]
                    if c in ds.column_names), None)
        if not col:
            print("No text column in Sangraha"); return []
        out = [r[col] for r in ds if r[col]]
        print(f"Sangraha: {len(out)} texts loaded.")
        return out
    except Exception as e:
        print(f"Sangraha error: {e}"); return []


def build_corpus(store, df_train: pd.DataFrame,
                 wit_tsv: str, sangraha_rows=2000) -> int:
    """
    Store WIT + Sangraha + Flickr captions into Qdrant indic_rag.
    Returns total chunks stored.
    """
    pid = 0
    print("=" * 50 + "\nWIT Text\n" + "=" * 50)
    wit = load_wit_texts(wit_tsv)
    if wit:
        pid = store.store_texts(wit, "wit", start_id=pid)

    print("=" * 50 + "\nSangraha\n" + "=" * 50)
    san = load_sangraha_texts(sangraha_rows)
    if san:
        pid = store.store_texts(san, "sangraha", start_id=pid)

    print("=" * 50 + "\nFlickr30k Captions\n" + "=" * 50)
    flickr = df_train["caption_en"].dropna().tolist()
    if flickr:
        pid = store.store_texts(flickr, "flickr", start_id=pid)

    print(f"\nTotal chunks stored: {pid}")
    return pid
