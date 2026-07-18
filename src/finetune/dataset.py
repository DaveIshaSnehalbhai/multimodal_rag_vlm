"""IndicCaptioningDataset — pre-caches Qdrant context to avoid per-step queries."""
from typing import List, Dict, Any, Optional
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

LANG_NAMES = {"en":"English","hi":"Hindi","bn":"Bengali","mr":"Marathi","gu":"Gujarati"}


def build_context_cache(df: pd.DataFrame, store, top_k=3) -> Dict[str, str]:
    """
    ONE-TIME call before training.
    Queries Qdrant for every row and caches result.
    Avoids O(N × epochs) Qdrant calls inside __getitem__.
    """
    print(f"Building context cache for {len(df)} rows ...")
    cache = {}
    for _, row in df.iterrows():
        chunks = store.retrieve_context(str(row["caption_en"]), top_k=top_k)
        cache[row["image_name"]] = "\n".join(c["text"] for c in chunks)
    print(f"Context cache ready: {len(cache)} entries")
    return cache


class IndicCaptioningDataset(Dataset):
    """
    Args:
        df_split      : already-split DataFrame (train / val / test).
        context_cache : {image_name → context_str} from build_context_cache().
        languages     : list of language codes to include.
        max_samples   : cap for quick iteration.
    """
    def __init__(self, df_split: pd.DataFrame,
                 context_cache: Dict[str, str],
                 languages: Optional[List[str]] = None,
                 max_samples: Optional[int] = None):
        langs = languages or ["en", "hi", "bn", "mr", "gu"]
        self.samples: List[Dict[str, Any]] = []

        for _, row in df_split.iterrows():
            ctx = context_cache.get(row["image_name"], "")
            for lang in langs:
                col    = f"caption_{lang}"
                if col not in df_split.columns: continue
                target = str(row.get(col, "")).strip()
                if not target or target == "nan": continue
                ln = LANG_NAMES.get(lang, "English")
                prompt = (
                    f"Retrieved Context:\n{ctx}\n\n"
                    f"Generate a factual, grounded caption in {ln}."
                    if ctx.strip() else
                    f"Generate a factual caption in {ln}.")
                self.samples.append({
                    "image_path": row["image_path"],
                    "prompt":     prompt,
                    "target":     target,
                })

        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"[Dataset] {len(self.samples)} samples | langs={langs}")

    def __len__(self):          return len(self.samples)
    def __getitem__(self, i):   return self.samples[i]
