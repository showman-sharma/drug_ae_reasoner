import pickle
import numpy as np
import faiss
from typing import List, Tuple, Dict

from ..config import OAE_INDEX_PATH, OAE_LABEL_MAP_PATH
from .encoding import model  # directly import the SentenceTransformer instance

# ─── Preload FAISS index & OAE labels ───────────────────────────────────
_INDEX = faiss.read_index(OAE_INDEX_PATH)
with open(OAE_LABEL_MAP_PATH, "rb") as f:
    _LABELS: List[str] = pickle.load(f)


def build_cadec_ae_oae_mapping(
    ae_cadec_list: List[str],
    n_cadec: int = 5,
    cadec_ae_threshold: float = 0.7
) -> Dict[str, List[Tuple[str, float]]]:
    """
    For each CADEC AE label, returns up to `n_cadec` similar OAE concepts
    whose similarity ≥ `cadec_ae_threshold`. Batch-encodes all labels in one go.
    """
    # Guard against empty input
    if not ae_cadec_list:
        return {}

    # 1) Batch-encode & normalize
    raw_vecs = model.encode(
        ae_cadec_list,
        convert_to_tensor=False,
        normalize_embeddings=True
    )
    # 2) Build a (N × D) float32 matrix
    queries = np.vstack([vec.astype("float32") for vec in raw_vecs])

    # 3) FAISS inner-product search
    distances, indices = _INDEX.search(queries, n_cadec)
    mapping: Dict[str, List[Tuple[str, float]]] = {}

    # 4) Convert to cosine-similarities & filter
    for ae_label, dists, idxs in zip(ae_cadec_list, distances, indices):
        sims = 1.0 - dists / 2.0
        hits = [
            (_LABELS[idx], float(sim))
            for sim, idx in zip(sims, idxs)
            if sim >= cadec_ae_threshold
        ]
        mapping[ae_label] = hits

    return mapping


def build_input_ae_oae_list(
    ae_input_list: List[str],
    n_input: int = 5,
    input_ae_threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """
    For each input AE text, returns up to `n_input` OAE concepts
    (excluding identity) with similarity ≥ `input_ae_threshold`.
    Batch-encodes all inputs in one go.
    """
    # Guard against empty input
    if not ae_input_list:
        return []

    # 1) Batch-encode & normalize
    raw_vecs = model.encode(
        ae_input_list,
        convert_to_tensor=False,
        normalize_embeddings=True
    )
    queries = np.vstack([vec.astype("float32") for vec in raw_vecs])

    # 2) FAISS search
    distances, indices = _INDEX.search(queries, n_input + 1)

    # 3) Collect neighbors per input AE
    oae_input_list: List[Tuple[str, str, float]] = []
    for inp_label, dists, idxs in zip(ae_input_list, distances, indices):
        sims = 1.0 - dists / 2.0
        count = 0
        for sim, idx in zip(sims, idxs):
            if sim < input_ae_threshold:
                continue
            target = _LABELS[idx]
            if target == inp_label:
                continue  # skip identity
            oae_input_list.append((inp_label, target, float(sim)))
            count += 1
            if count >= n_input:
                break

    return oae_input_list
