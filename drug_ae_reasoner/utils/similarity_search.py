import pickle
from typing import List, Tuple, Dict
import numpy as np
import faiss
from ..utils.encoding import encode_text

def build_cadec_ae_oae_mapping(ae_cadec_list: List[str], index_path: str, label_map_path: str,
                                n_cadec: int = 5, cadec_ae_threshold: float = 0.7) -> Dict[str, List[Tuple[str, float]]]:
    index = faiss.read_index(index_path)
    with open(label_map_path, 'rb') as f:
        oae_labels = pickle.load(f)

    mapping = {}
    for ae_label in ae_cadec_list:
        vec = encode_text(ae_label)
        q = np.array([vec.astype('float32')])
        D, I = index.search(q, n_cadec)
        sims = 1.0 - D[0] / 2.0
        results = [(oae_labels[idx], float(sim)) for sim, idx in zip(sims, I[0]) if sim >= cadec_ae_threshold]
        mapping[ae_label] = results
    return mapping

def build_input_ae_oae_list(ae_input_list: List[str], index_path: str, label_map_path: str,
                            n_input: int = 5, input_ae_threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    index = faiss.read_index(index_path)
    with open(label_map_path, 'rb') as f:
        oae_labels = pickle.load(f)

    oae_input_list = []
    for ae_label in ae_input_list:
        vec = encode_text(ae_label)
        q = np.array([vec.astype('float32')])
        D, I = index.search(q, n_input + 1)
        sims = 1.0 - D[0] / 2.0
        count = 0
        for sim, idx in zip(sims, I[0]):
            if sim < input_ae_threshold:
                continue
            oae_label = oae_labels[idx]
            if oae_label == ae_label:
                continue
            oae_input_list.append((ae_label, oae_label, float(sim)))
            count += 1
            if count >= n_input:
                break
    return oae_input_list
