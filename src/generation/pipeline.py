"""End-to-end IndicRAG-VLM inference pipeline."""
import os
from typing import Optional, Dict, Any
import pandas as pd

from src.retrieval.vector_store import VectorStore
from src.generation.captioner   import Captioner
from src.generation.translator  import Translator


class IndicRAGPipeline:
    """
    Full pipeline:
        image
          → initial caption (Qwen2-VL, no RAG) — used as retrieval query
          → BGE-M3 text retrieval (WIT + Sangraha + Flickr corpus)
          → CLIP image retrieval (similar Flickr train captions)
          → combined context
          → Qwen2-VL grounded caption
          → Google Translate → hi / bn / mr / gu

    Args:
        store    : VectorStore pointing at pre-built Qdrant DB.
        captioner: Captioner (base or fine-tuned LoRA).
        df_train : Flickr30k train split — used to look up captions
                   of visually similar images.
    """
    def __init__(self, store: VectorStore, captioner: Captioner,
                 df_train: Optional[pd.DataFrame] = None):
        self.store      = store
        self.captioner  = captioner
        self.translator = Translator()
        self.df_train   = df_train

    def run(self, image_path: str, use_rag=True,
            top_k=5, min_score=0.0) -> Dict[str, Any]:
        """
        Run full pipeline on one image.

        Returns:
            captions      : {en, hi, bn, mr, gu}
            context_str   : combined retrieved context fed to VLM
            rag_chunks    : list of {text, source, score}
            similar_images: list of {image_name, caption, score}
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        # Step 1 — initial caption as retrieval query
        initial_cap = self.captioner.caption_no_rag(image_path, "en")

        rag_chunks, similar_imgs, context_str = [], [], ""

        if use_rag:
            # Step 2 — BGE-M3 text retrieval
            rag_chunks   = self.store.retrieve_context(
                initial_cap, top_k=top_k, min_score=min_score)

            # Step 3 — CLIP visual retrieval
            similar_imgs = self.store.retrieve_similar_images(image_path, top_k=3)

            retrieved_texts = [c["text"] for c in rag_chunks]

            # Step 4 — look up captions for similar images (train split only)
            similar_caps = []
            if self.df_train is not None:
                for img in similar_imgs:
                    rows = self.df_train[
                        self.df_train["image_name"] == img["image_name"]]
                    if len(rows):
                        similar_caps.append(rows.iloc[0]["caption_en"])

            context_str = "\n".join(retrieved_texts + similar_caps)

        # Step 5 — grounded caption
        caption_en = (
            self.captioner.caption_with_rag(image_path, context_str)
            if use_rag and context_str
            else initial_cap
        )

        # Step 6 — translate
        translations = self.translator.translate_all(caption_en)

        return {
            "captions":       {"en": caption_en, **translations},
            "context_str":    context_str,
            "rag_chunks":     rag_chunks,
            "similar_images": similar_imgs,
        }
