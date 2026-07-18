"""Qdrant vector store — text corpus (BGE-M3) + image embeddings (CLIP)."""
from typing import List, Dict, Any, Optional
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.retrieval.embeddings import CLIPEncoder, BGEEncoder

TEXT_COL  = "indic_rag"
IMAGE_COL = "image_embeddings"


def _chunk(text: str, size=300, overlap=50) -> List[str]:
    w = text.split(); out = []; s = 0
    while s < len(w):
        out.append(" ".join(w[s:s+size])); s += size - overlap
    return out


class VectorStore:
    def __init__(self, path: Optional[str] = None):
        """path=None → in-memory. path='./qdrant_data' → persistent."""
        self.client = QdrantClient(path=path) if path else QdrantClient(":memory:")
        self.bge    = BGEEncoder()
        self.clip   = CLIPEncoder()

    def init_collections(self, recreate=False):
        existing = {c.name for c in self.client.get_collections().collections}
        for name, dim in [(TEXT_COL, self.bge.DIM), (IMAGE_COL, self.clip.DIM)]:
            if name in existing and not recreate:
                continue
            if name in existing and recreate:
                self.client.delete_collection(name)
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        print(f"Collections ready: {TEXT_COL}({self.bge.DIM}), {IMAGE_COL}({self.clip.DIM})")

    # ── Text ──────────────────────────────────────────────────────────────────

    def store_texts(self, texts: List[str], source: str,
                    start_id=0, batch=64) -> int:
        chunks = []
        for t in texts:
            chunks.extend(_chunk(t))
        pid = start_id
        for i in range(0, len(chunks), batch):
            b    = chunks[i:i+batch]
            vecs = self.bge.encode(b, batch_size=batch)
            self.client.upsert(TEXT_COL, points=[
                PointStruct(id=pid+j, vector=vecs[j].tolist(),
                            payload={"text": b[j], "source": source})
                for j in range(len(b))])
            pid += len(b)
            print(f"  [{source}] {pid-start_id}/{len(chunks)}", end="\r")
        print(f"\n  [{source}] {pid-start_id} chunks stored.")
        return pid

    def retrieve_context(self, query: str, top_k=5,
                         min_score=0.0) -> List[Dict[str, Any]]:
        """BGE-M3 semantic search. min_score filters low-quality chunks."""
        q = self.bge.encode_query(query)
        r = self.client.query_points(collection_name=TEXT_COL,
                                     query=q.tolist(), limit=top_k)
        return [{"text":   p.payload["text"],
                 "source": p.payload.get("source", "?"),
                 "score":  round(p.score, 4)}
                for p in r.points if p.score >= min_score]

    # ── Images ────────────────────────────────────────────────────────────────

    def store_images(self, df: pd.DataFrame) -> int:
        points, err = [], 0
        for idx, row in df.iterrows():
            try:
                e = self.clip.encode_image(row["image_path"])
                points.append(PointStruct(
                    id=int(idx), vector=e.tolist(),
                    payload={"image_name": row["image_name"],
                             "caption":    row["caption_en"],
                             "image_path": row["image_path"]}))
            except Exception:
                err += 1
            if len(points) % 200 == 0 and len(points):
                print(f"  {len(points)} images ...", end="\r")
        self.client.upsert(IMAGE_COL, points=points)
        print(f"\n  {len(points)} image embeddings stored. Errors: {err}")
        return len(points)

    def retrieve_similar_images(self, image_path: str,
                                top_k=3) -> List[Dict[str, Any]]:
        e = self.clip.encode_image(image_path)
        r = self.client.query_points(collection_name=IMAGE_COL,
                                     query=e.tolist(), limit=top_k)
        return [{"image_name": p.payload.get("image_name", ""),
                 "caption":    p.payload.get("caption", ""),
                 "image_path": p.payload.get("image_path", ""),
                 "score":      round(p.score, 4)}
                for p in r.points]
