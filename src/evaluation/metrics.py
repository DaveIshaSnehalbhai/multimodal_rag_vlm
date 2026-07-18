"""BLEU-4, METEOR, BERTScore, CLIPScore, Hallucination Rate."""
from typing import List, Tuple
import numpy as np


class Evaluator:
    def __init__(self):
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt",   quiet=True)
        from nltk.translate.bleu_score import SmoothingFunction
        import evaluate as hf_evaluate
        import spacy
        self._smooth = SmoothingFunction().method1
        self._meteor = hf_evaluate.load("meteor")
        self._nlp    = spacy.load("en_core_web_sm")

    def bleu4(self, refs: List[str], preds: List[str]) -> float:
        from nltk.translate.bleu_score import sentence_bleu
        s = [sentence_bleu([r.split()], p.split(), weights=(.25,)*4,
                           smoothing_function=self._smooth)
             for r, p in zip(refs, preds)]
        return round(sum(s)/len(s), 4) if s else 0.0

    def meteor(self, refs: List[str], preds: List[str]) -> float:
        return round(self._meteor.compute(
            predictions=preds, references=refs)["meteor"], 4)

    def bertscore(self, refs: List[str], preds: List[str]) -> float:
        from bert_score import score as bs
        _, _, F1 = bs(preds, refs, lang="en", verbose=False)
        return round(F1.mean().item(), 4)

    def clipscore(self, image_paths: List[str],
                  captions: List[str], clip_encoder) -> float:
        scores = []
        for p, c in zip(image_paths, captions):
            try:
                scores.append(float(np.dot(
                    clip_encoder.encode_image(p),
                    clip_encoder.encode_text(c))))
            except Exception as e:
                print(f"CLIPScore error: {e}")
        return round(sum(scores)/len(scores), 4) if scores else 0.0

    def _ents(self, text: str):
        return {e.text.lower() for e in self._nlp(text).ents}

    def hallucination_rate(self, gen: str, ctx: str) -> Tuple[float, set]:
        g = self._ents(gen); c = self._ents(ctx)
        if not g: return 0.0, set()
        h = g - c
        return round(len(h)/len(g), 4), h

    def corpus_hallucination_rate(self, captions: List[str],
                                  contexts: List[str]) -> float:
        # FIX: no-RAG baseline uses refs not [] so rate isn't artificially 1.0
        s = [self.hallucination_rate(c, ctx)[0]
             for c, ctx in zip(captions, contexts)]
        return round(sum(s)/len(s), 4) if s else 0.0

    def full_report(self, refs, preds_no_rag, preds_rag,
                    contexts, image_paths, clip_encoder) -> dict:
        import pandas as pd
        results = {
            "no_rag": {
                "bleu4":        self.bleu4(refs, preds_no_rag),
                "meteor":       self.meteor(refs, preds_no_rag),
                "bertscore":    self.bertscore(refs, preds_no_rag),
                "clipscore":    self.clipscore(image_paths, preds_no_rag, clip_encoder),
                "hallucination":self.corpus_hallucination_rate(preds_no_rag, refs),
            },
            "rag": {
                "bleu4":        self.bleu4(refs, preds_rag),
                "meteor":       self.meteor(refs, preds_rag),
                "bertscore":    self.bertscore(refs, preds_rag),
                "clipscore":    self.clipscore(image_paths, preds_rag, clip_encoder),
                "hallucination":self.corpus_hallucination_rate(preds_rag, contexts),
            },
        }
        df = pd.DataFrame({
            "Model":["Qwen2-VL (No RAG)","Qwen2-VL + Hybrid RAG"],
            "BLEU-4":       [results["no_rag"]["bleu4"],         results["rag"]["bleu4"]],
            "METEOR":       [results["no_rag"]["meteor"],        results["rag"]["meteor"]],
            "BERTScore":    [results["no_rag"]["bertscore"],     results["rag"]["bertscore"]],
            "CLIPScore":    [results["no_rag"]["clipscore"],     results["rag"]["clipscore"]],
            "Hallucination":[results["no_rag"]["hallucination"], results["rag"]["hallucination"]],
        })
        print("\n=== Evaluation Results ===")
        print(df.to_string(index=False))
        return results, df
