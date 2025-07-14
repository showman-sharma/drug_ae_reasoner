import pickle
from typing import List, Tuple, Dict

import numpy as np
import faiss

from ..config import OAE_INDEX_PATH, OAE_LABEL_MAP_PATH
from .encoding import encode_text

#─── Pre-load FAISS index & labels ──────────────────────────────────
_OAE_INDEX = faiss.read_index(OAE_INDEX_PATH)
with open(OAE_LABEL_MAP_PATH, "rb") as _f:
    _OAE_LABELS: List[str] = pickle.load(_f)


def build_cadec_ae_oae_mapping(
    ae_cadec_list: List[str],
    n_cadec: int = 5,
    cadec_ae_threshold: float = 0.7
) -> Dict[str, List[Tuple[str, float]]]:
    """
    For each CADEC AE label, return up to `n_cadec` similar OAE concepts
    whose similarity ≥ `cadec_ae_threshold`.
    """
    mapping: Dict[str, List[Tuple[str, float]]] = {}
    for ae_label in ae_cadec_list:
        # 1. Encode the CADEC AE text
        vec = encode_text(ae_label)
        q = np.array([vec.astype("float32")])

        # 2. FAISS inner-product search
        D, I = _OAE_INDEX.search(q, n_cadec)

        # 3. Convert distances to cosine sims in [0,1]
        sims = 1.0 - D[0] / 2.0

        # 4. Collect those above threshold
        results = [
            (_OAE_LABELS[idx], float(sim))
            for sim, idx in zip(sims, I[0])
            if sim >= cadec_ae_threshold
        ]

        mapping[ae_label] = results
    return mapping


def build_input_ae_oae_list(
    ae_input_list: List[str],
    n_input: int = 5,
    input_ae_threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """
    For each input AE text, return up to `n_input` OAE concepts
    (excluding identity) with similarity ≥ `input_ae_threshold`.
    """
    oae_input_list: List[Tuple[str, str, float]] = []
    for ae_label in ae_input_list:
        vec = encode_text(ae_label)
        q = np.array([vec.astype("float32")])

        # Retrieve one extra so we can skip the identical match
        D, I = _OAE_INDEX.search(q, n_input + 1)
        sims = 1.0 - D[0] / 2.0

        count = 0
        for sim, idx in zip(sims, I[0]):
            if sim < input_ae_threshold:
                continue
            oae_label = _OAE_LABELS[idx]
            if oae_label == ae_label:
                # skip the concept identical to input
                continue
            oae_input_list.append((ae_label, oae_label, float(sim)))
            count += 1
            if count >= n_input:
                break

    return oae_input_list
