# drug_ae_reasoner/data/cadec_loader.py
import pickle
import logging
from typing import Any, List, Tuple, Set, Dict
import networkx as nx

logger = logging.getLogger(__name__)

# ─── Cache loaded CADEC graph(s) by path ───────────────────────────────
_CADEC_KG_CACHE: Dict[str, Any] = {}

def _load_cadec_graph(kg_path: str) -> nx.MultiDiGraph:
    if kg_path not in _CADEC_KG_CACHE:
        with open(kg_path, "rb") as f:
            _CADEC_KG_CACHE[kg_path] = pickle.load(f)
        logger.info(f"[CACHE] Loaded CADEC KG once from: {kg_path}")
    return _CADEC_KG_CACHE[kg_path]

def get_cadec_drug_nodes(
    drug_label: str,
    kg_path: str
) -> List[Tuple[str, str, Set[str]]]:
    """
    Return all CADEC *drug* nodes whose 'label' (lowercased) matches `drug_label`.
    Each tuple = (node_id, drug_label_lower, cuis_set)
    """
    G = _load_cadec_graph(kg_path)
    dl = drug_label.lower()
    out: List[Tuple[str, str, Set[str]]] = []
    for node, data in G.nodes(data=True):
        if data.get("type") != "drug":
            continue
        lbl = (data.get("label") or "").lower()
        if lbl == dl:
            cuis = set(data.get("cuis", set()))
            out.append((node, lbl, cuis))
    if not out:
        logger.warning(f"[CADEC] No drug nodes found for label='{drug_label}' in KG.")
    return out

def get_cadec_ae_pairs(
    drug_nodes: List[Tuple[str, str, Set[str]]],
    kg_path: str
) -> List[Tuple[str, str, str]]:
    """
    For each input drug node, list CADEC AE edges.
    Returns list of (drug_label_lower, ae_label_lower, cui_str)
    """
    G = _load_cadec_graph(kg_path)
    pairs: List[Tuple[str, str, str]] = []
    for node_id, drug_label, cuis in drug_nodes:
        cui_str = ", ".join(sorted(cuis)) if cuis else ""
        for _, ae_node, data in G.out_edges(node_id, data=True):
            if G.nodes[ae_node].get("type") == "adverse_effect":
                ae_label = (G.nodes[ae_node].get("label") or "").lower()
                pairs.append((drug_label, ae_label, cui_str))
    return pairs
