# IndicRAG-VLM

**Multilingual Retrieval-Augmented Vision Language Model for Factually Grounded Image Captioning**

Generates factual image captions in **English · Hindi · Bengali · Marathi · Gujarati** using hybrid multimodal RAG over a multilingual knowledge corpus.

---

## What We Built

A complete multimodal RAG pipeline that:

1. Takes any image as input
2. Generates an initial caption using Qwen2-VL-2B (used as retrieval query)
3. Retrieves relevant text chunks from a 50K+ chunk corpus (WIT + Sangraha + Flickr30k) using BGE-M3
4. Retrieves visually similar image captions using CLIP ViT-L/14
5. Generates a factually grounded final caption using the retrieved context
6. Translates to Hindi, Bengali, Marathi, Gujarati via Google Translate

Fine-tuned Qwen2-VL-2B with 4-bit QLoRA on ~500 Flickr30k images × 5 languages.

---

## Results

| Model | BLEU-4 | METEOR | BERTScore | CLIPScore | Hallucination↓ |
|---|---|---|---|---|---|
| Qwen2-VL-2B (No RAG) | 0.0360 | 0.2637 | 0.9124 | 0.2553 | 0.3200 |
| Qwen2-VL-2B + Hybrid RAG | **0.0442** | **0.2734** | 0.9002 | **0.2612** | **0.2067** |

RAG improves BLEU-4 **+22.8%**, reduces hallucination **−35.4%**, improves CLIPScore **+2.3%**.

> Note: BLEU-4 values reflect a limited fine-tuning run (proof of concept). Full training on the complete dataset would yield higher scores.

---

## Architecture

```
IMAGE
  ├── CLIP ViT-L/14 ──────────────────► Qdrant image_embeddings (dim=768)
  │                                              │
  │                                     Similar image captions
  │
  └── Qwen2-VL-2B ──► Initial caption (retrieval query)
                              │
                         BGE-M3 (dim=1024)
                              │
                    Qdrant indic_rag (dim=1024)
                    [WIT + Sangraha + Flickr30k]
                              
                User Image
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
 CLIP ViT-L/14             Caption Query
        │                       │
        ▼                       ▼
 Image Embedding          BGE-M3 Embedding
        │                       │
        ▼                       ▼
Qdrant (Images)          Qdrant (Text)
        │                       │
        ▼                       ▼
Similar Images         Relevant Knowledge
        │                       │
        └───────────┬───────────┘
                    ▼
      Prompt Construction
                    │
                    ▼
              Qwen2-VL
                    │
                    ▼
         Multilingual Caption
```

---

## Datasets Used

| Dataset | Role |
|---------|------|
| **Flickr30k** (~16,000 images) | Training + evaluation captions |
| **WIT 1% sample** | Text knowledge corpus |
| **Sangraha (AI4Bharat)** | Indic multilingual text corpus |

---

## Models Used

| Component | Model | Dim |
|---|---|---|
| Image embedding | CLIP ViT-L/14 | 768 |
| Text embedding | BGE-M3 | 1024 |
| Caption generation | Qwen2-VL-2B-Instruct + QLoRA | — |
| Translation | Google Translate | — |

---

## Quick Start


```bash
# Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run UI (loads pretrained weights automatically)
streamlit run ui/app.py
```

---

