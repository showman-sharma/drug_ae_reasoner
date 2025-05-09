import pickle
from typing import List, Tuple, Set
import logging

logger = logging.getLogger(__name__)

def get_cadec_drug_nodes(drug: str, rx_path: str, kg_path: str) -> List[Tuple[str, str, Set[str]]]:
    from .rxnorm_loader import get_input_cuis
    cuis = get_input_cuis(drug, rx_path)
    with open(kg_path, "rb") as f:
        G_cadec = pickle.load(f)

    matches = []
    for node_id, data in G_cadec.nodes(data=True):
        if data.get("type") == "drug" and data.get("cuis", set()) & cuis:
            label = data.get("label", "UnknownDrug")
            node_cuis = data.get("cuis", set())
            matches.append((node_id, label, node_cuis))
    return matches

def get_cadec_ae_pairs(drug_nodes: List[Tuple[str, str, Set[str]]], kg_path: str) -> List[Tuple[str, str, str]]:
    with open(kg_path, "rb") as f:
        G = pickle.load(f)

    pairs = []
    for node_id, drug_label, cuis in drug_nodes:
        cui_str = ", ".join(sorted(cuis))
        for _, ae_node, data in G.out_edges(node_id, data=True):
            if G.nodes[ae_node].get("type") == "adverse_effect":
                ae_label = G.nodes[ae_node]["label"].lower()
                pairs.append((drug_label, ae_label, cui_str))
    return pairs
