"""CLIP ViT-L/14 (dim=768) and BGE-M3 (dim=1024) encoders."""
import numpy as np
import torch
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer


class CLIPEncoder:
    DIM = 768

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model = self.model.to(self.device).eval()

    def encode_image(self, image_path: str) -> np.ndarray:
        img = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            e = self.model.encode_image(img)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy()[0]

    def encode_text(self, text: str) -> np.ndarray:
        t = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            e = self.model.encode_text(t)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy()[0]

    def image_text_score(self, image_path: str, text: str) -> float:
        return float(np.dot(self.encode_image(image_path), self.encode_text(text)))


class BGEEncoder:
    DIM = 1024

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def encode(self, texts, batch_size=32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0]
