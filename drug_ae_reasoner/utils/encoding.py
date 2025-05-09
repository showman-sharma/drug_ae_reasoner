import numpy as np
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "local_models", "sapbert")

# Check if local model exists
if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print(f"[INFO] Loading SapBERT from local cache: {MODEL_DIR}")
    model = SentenceTransformer(MODEL_DIR)
else:
    print(f"[INFO] Downloading SapBERT model from HuggingFace...")
    model = SentenceTransformer(MODEL_NAME)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_DIR)
    print(f"[INFO] SapBERT saved locally to: {MODEL_DIR}")

# Simple cache to avoid redundant embeddings
embedding_cache = {}

def encode_text(text: str) -> np.ndarray:
    if text in embedding_cache:
        return embedding_cache[text]
    vec = model.encode([text])[0]
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    normalized = vec / (norm if norm > 0 else 1.0)
    embedding_cache[text] = normalized
    return normalized
