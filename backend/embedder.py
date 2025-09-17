# backend/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name = "all-MiniLM-L6-v2"  # compact & fast
embedder = SentenceTransformer(model_name)

def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

def semantic_search(query, chunk_texts, chunk_embeddings, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, chunk_embeddings)[0]
    top_idx = np.argsort(-sims)[:top_k]
    results = [{"id": idx+1, "score": float(sims[idx]), "text": chunk_texts[idx]} for idx in top_idx]
    return results
