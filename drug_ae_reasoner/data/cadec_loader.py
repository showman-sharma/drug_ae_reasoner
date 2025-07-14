import pickle
import logging
from typing import Any, List, Tuple, Set, Dict

logger = logging.getLogger(__name__)

# ─── Cache loaded CADEC graph(s) by path ───────────────────────────────
_CADEC_KG_CACHE: Dict[str, Any] = {}


def _load_cadec_graph(kg_path: str):
    if kg_path not in _CADEC_KG_CACHE:
        with open(kg_path, "rb") as f:
            _CADEC_KG_CACHE[kg_path] = pickle.load(f)
        logger.info(f"[CACHE] Loaded CADEC KG once from: {kg_path}")
    return _CADEC_KG_CACHE[kg_path]


def get_cadec_drug_nodes(
    drug: str,
    rx_path: str,
    kg_path: str
) -> List[Tuple[str, str, Set[str]]]:
    from .rxnorm_loader import get_input_cuis

    cuis = get_input_cuis(drug, rx_path)
    G = _load_cadec_graph(kg_path)

    matches: List[Tuple[str, str, Set[str]]] = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "drug" and data.get("cuis", set()) & cuis:
            matches.append((node_id, data.get("label", "UnknownDrug"), data.get("cuis", set())))
    return matches


def get_cadec_ae_pairs(
    drug_nodes: List[Tuple[str, str, Set[str]]],
    kg_path: str
) -> List[Tuple[str, str, str]]:
    G = _load_cadec_graph(kg_path)

    pairs: List[Tuple[str, str, str]] = []
    for node_id, drug_label, cuis in drug_nodes:
        cui_str = ", ".join(sorted(cuis))
        for _, ae_node, data in G.out_edges(node_id, data=True):
            if G.nodes[ae_node].get("type") == "adverse_effect":
                ae_label = G.nodes[ae_node]["label"].lower()
                pairs.append((drug_label, ae_label, cui_str))
    return pairs
