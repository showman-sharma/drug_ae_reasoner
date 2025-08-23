# drug_ae_reasoner/utils/similarity_search.py
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional

from ..config import OAE_INDEX_PATH, OAE_LABEL_MAP_PATH
from .encoding import model  # SentenceTransformer instance

# ─── On-demand loading for FAISS index & labels (cache by path) ─────────
_INDEX_CACHE: Dict[str, faiss.Index] = {}
_LABELS_CACHE: Dict[str, List[str]] = {}

def _load_index(path: str) -> faiss.Index:
    if path not in _INDEX_CACHE:
        _INDEX_CACHE[path] = faiss.read_index(path)
    return _INDEX_CACHE[path]

def _load_labels(path: str) -> List[str]:
    if path not in _LABELS_CACHE:
        with open(path, "rb") as f:
            _LABELS_CACHE[path] = pickle.load(f)
    return _LABELS_CACHE[path]

def _cos_from_sq_l2(d: np.ndarray) -> np.ndarray:
    # For unit-normalized vectors: cos(x,y) = 1 - 0.5 * ||x - y||^2
    return 1.0 - d / 2.0

def build_cadec_ae_oae_mapping(
    ae_cadec_list: List[str],
    n_cadec: int = 5,
    cadec_ae_threshold: float = 0.7,
    index_path: Optional[str] = None,
    label_map_path: Optional[str] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Map each CADEC AE label to up to n_cadec OAE concepts above the given threshold.
    Returns: {cadec_ae_label -> [(oae_label, sim), ...]}
    """
    if not ae_cadec_list:
        return {}

    index_path = index_path or OAE_INDEX_PATH
    label_map_path = label_map_path or OAE_LABEL_MAP_PATH

    index = _load_index(index_path)
    labels = _load_labels(label_map_path)

    # 1) Batch-encode & normalize
    vecs = model.encode(ae_cadec_list, convert_to_tensor=False, normalize_embeddings=True)
    queries = np.asarray(vecs, dtype=np.float32)

    # 2) FAISS search (IndexFlatL2 over normalized vectors)
    dists, idxs = index.search(queries, n_cadec)

    # 3) Convert to cosine and filter
    mapping: Dict[str, List[Tuple[str, float]]] = {}
    for ae_label, d_row, i_row in zip(ae_cadec_list, dists, idxs):
        sims = _cos_from_sq_l2(d_row)
        hits = [
            (labels[i], float(s))
            for s, i in zip(sims, i_row) if i != -1 and s >= cadec_ae_threshold
        ]
        mapping[ae_label] = hits
    return mapping

def build_input_ae_oae_list(
    ae_input_list: List[str],
    n_input: int = 5,
    input_ae_threshold: float = 0.7,
    index_path: Optional[str] = None,
    label_map_path: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """
    For each input AE string, returns up to `n_input` OAE concepts
    (excluding identity) with similarity ≥ `input_ae_threshold`.
    Returns: [(input_ae, oae_label, sim), ...]
    """
    if not ae_input_list:
        return []

    index_path = index_path or OAE_INDEX_PATH
    label_map_path = label_map_path or OAE_LABEL_MAP_PATH

    index = _load_index(index_path)
    labels = _load_labels(label_map_path)

    vecs = model.encode(ae_input_list, convert_to_tensor=False, normalize_embeddings=True)
    queries = np.asarray(vecs, dtype=np.float32)

    dists, idxs = index.search(queries, n_input + 1)  # +1 to allow skipping identity
    out: List[Tuple[str, str, float]] = []

    for inp_label, d_row, i_row in zip(ae_input_list, dists, idxs):
        sims = _cos_from_sq_l2(d_row)
        count = 0
        for s, i in zip(sims, i_row):
            if i == -1:
                continue
            target = labels[i]
            if target == inp_label:
                continue  # skip identity
            if s < input_ae_threshold:
                continue
            out.append((inp_label, target, float(s)))
            count += 1
            if count >= n_input:
                break
    return out
