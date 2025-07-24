import pickle
import networkx as nx
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict

from .similarity_search import build_input_ae_oae_list, build_cadec_ae_oae_mapping
from ..data.cadec_loader import get_cadec_ae_pairs, get_cadec_drug_nodes
from .verbalizer import verbalize_drug_to_input_ae_paths
from ..config import OAE_GRAPH_PATH

# ─── Cache for loaded graphs ─────────────────────────────────────────
_GRAPH_CACHE: Dict[str, nx.Graph] = {}


def find_drug_to_input_ae_paths(
    drug_label: str,
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    oae_input_list: List[Tuple[str, str, float]],
    graph_path: str = OAE_GRAPH_PATH
) -> List[Tuple[str, str, List[str]]]:
    """
    Find all 1- or 2-hop paths in the OAE graph between each CADEC AE candidate
    and each input AE concept. Returns a list of (drug_label, input_ae, [path_nodes]).
    """
    # Load (and cache) the OAE graph
    if graph_path not in _GRAPH_CACHE:
        with open(graph_path, "rb") as f:
            _GRAPH_CACHE[graph_path] = pickle.load(f)
    G: nx.Graph = _GRAPH_CACHE[graph_path]

    # Build map from input AE -> list of OAE nodes
    input_map: Dict[str, List[str]] = defaultdict(list)
    for inp_label, oae_node, _ in oae_input_list:
        input_map[inp_label].append(oae_node)

    paths: List[Tuple[str, str, List[str]]] = []
    for cadec_ae, oae_candidates in cadec_ae_oae_dict.items():
        for oae_cand, _ in oae_candidates:
            for inp_label, oae_in_nodes in input_map.items():
                for oae_in in oae_in_nodes:
                    # direct match (0-hop in OAE graph)
                    if oae_cand == oae_in:
                        paths.append((drug_label, inp_label, [oae_cand]))
                    # 1-hop in OAE graph
                    elif G.has_edge(oae_cand, oae_in):
                        paths.append((drug_label, inp_label, [oae_cand, oae_in]))
    return paths


def rank_drug_ae_paths(
    raw_paths: List[Tuple[str, str, List[str]]],
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    oae_input_list: List[Tuple[str, str, float]],
    n_paths: int = 5
) -> List[Tuple[str, str, List[str], float]]:
    """
    Score each candidate path by averaging:
      1) similarity between the CADEC AE and the first OAE node,
      2) similarity between the input AE and the last OAE node.
    Then dedupe identical OAE‐node sequences, keeping the highest‐scoring one,
    and return the top `n_paths`.
    """
    # Build lookup dicts
    cadec_sim = {
        (ae, oae): sim
        for ae, pairs in cadec_ae_oae_dict.items()
        for oae, sim in pairs
    }
    input_sim = {
        (inp, oae): sim
        for inp, oae, sim in oae_input_list
    }

    # 1) Compute raw scored list, only non-zero
    scored: List[Tuple[str, str, List[str], float]] = []
    for drug_label, inp_label, path_nodes in raw_paths:
        oae_cand = path_nodes[0]
        oae_in = path_nodes[-1]

        # Find CADEC AE label corresponding to this OAE candidate
        ae_label = next(
            (ae for ae, lst in cadec_ae_oae_dict.items() if any(o == oae_cand for o, _ in lst)),
            None
        )

        sim1 = cadec_sim.get((ae_label, oae_cand), 0.0) if ae_label else 0.0
        sim2 = input_sim.get((inp_label, oae_in), 0.0)
        score = (sim1 + sim2) / 2.0  # average similarity in [0,1]

        if score > 0.0:
            scored.append((drug_label, inp_label, path_nodes, score))

    # 2) Deduplicate by identical path_nodes, keep highest score
    unique: Dict[Tuple[str, ...], Tuple[str, str, List[str], float]] = {}
    for d_lbl, i_lbl, path, scr in scored:
        key = tuple(path)
        prev = unique.get(key)
        if prev is None or scr > prev[3]:
            unique[key] = (d_lbl, i_lbl, path, scr)

    # 3) Sort and return top-n
    result = list(unique.values())
    result.sort(key=lambda x: x[3], reverse=True)
    return result[:n_paths]

def generate_fallback_drug_paths(
    drug_label: str,
    cadec_pairs: List[Tuple[str, str, float]],
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    n_disconnect: int = 3
) -> List[Tuple[str, str, List[str], float]]:
    """
    Fallback: If no direct path found, return top-N AEs most strongly associated
    with the given drug (based on CADEC pairs + OAE similarity).
    """
    fallback = []
    # Filter CADEC AEs caused by this drug
    ae_candidates = [(ae, score) for d, ae, score in cadec_pairs if d == drug_label]
    ae_candidates.sort(key=lambda x: x[1], reverse=True)

    for ae, drug_score in ae_candidates[:n_disconnect]:
        if ae in cadec_ae_oae_dict:
            # Take top-scoring OAE mapping for this AE
            oae, sim = max(cadec_ae_oae_dict[ae], key=lambda x: x[1])
            total_score = (drug_score + sim) / 2.0
            fallback.append((drug_label, ae, [oae], total_score))

    return fallback



def generate_fallback_ae_paths(ae_input_list: List[str],
    cadec_pairs: List[Tuple[str, str, float]],
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    oae_input_list: List[Tuple[str, str, float]],
    n_disconnect: int = 3
) -> List[Tuple[str, str, List[str], float]]:
    """
    Fallback: If no direct path from drug to AE, return top drugs that cause CADEC AEs
    semantically similar to the input AE.
    """
    fallback = []

    # Reverse index: OAE → CADEC AE
    oae_to_cadec: DefaultDict[str, List[Tuple[str, float]]] = defaultdict(list)
    for cae, lst in cadec_ae_oae_dict.items():
        for oae, sim in lst:
            oae_to_cadec[oae].append((cae, sim))

    for ae_input in ae_input_list:
        # Get top OAE nodes for this input AE
        neighbors = [
            (inp, oae, sim) for inp, oae, sim in oae_input_list if inp == ae_input
        ]
        neighbors.sort(key=lambda x: x[2], reverse=True)

        for _, oae_node, sim_input in neighbors[:n_disconnect]:
            candidates = oae_to_cadec.get(oae_node, [])
            if not candidates:
                continue

            # Pick the most similar CADEC AE and its causing drug
            cadec_ae, sim_cadec = max(candidates, key=lambda x: x[1])
            parent_drugs = [(d, s) for d, ae, s in cadec_pairs if ae == cadec_ae]
            if not parent_drugs:
                continue

            drug, drug_score = max(parent_drugs, key=lambda x: x[1])
            total_score = (sim_input + sim_cadec + drug_score) / 3.0
            fallback.append((drug, ae_input, [oae_node], total_score))

    return fallback
    
def find_top_drug_to_input_ae_paths(
    drug: str,
    ae_input_list: List[str],
    rx_path: str,
    cadec_kg_path: str,
    oae_index_path: str,
    oae_label_map_path: str,
    oae_graph_path: str,
    n_cadec: int = 5,
    cadec_ae_threshold: float = 0.7,
    n_input: int = 5,
    input_ae_threshold: float = 0.7,
    n_paths: int = 5,
    n_disconnect: int = 3
):
    """
    Orchestrates the full reasoning: from drug → CADEC AE → OAE → input AE,
    ranking paths and generating verbalizations or fallbacks.
    """
    # 1) Get CADEC drug nodes for the given drug
    drug_nodes = get_cadec_drug_nodes(drug, rx_path, cadec_kg_path)

    # 2) Extract CADEC AE pairs for those drug nodes
    cadec_pairs = get_cadec_ae_pairs(drug_nodes, cadec_kg_path)
    ae_cadec_list = sorted({ae for _, ae, _ in cadec_pairs})

    # 3) Map CADEC AEs → OAE concepts
    cadec_ae_oae = build_cadec_ae_oae_mapping(
        ae_cadec_list,
        n_cadec=n_cadec,
        cadec_ae_threshold=cadec_ae_threshold
    )

    # 4) Map input AEs → OAE concepts
    oae_input = build_input_ae_oae_list(
        ae_input_list,
        n_input=n_input,
        input_ae_threshold=input_ae_threshold
    )

    # 5) Find raw paths in the OAE graph
    raw_paths = find_drug_to_input_ae_paths(
        drug, cadec_ae_oae, oae_input, graph_path=oae_graph_path
    )

    # 6) Rank the paths
    top_paths = rank_drug_ae_paths(raw_paths, cadec_ae_oae, oae_input, n_paths)

    # 7) If we found valid paths, verbalize them
    if top_paths:
        verb = verbalize_drug_to_input_ae_paths(
            drug, cadec_pairs, cadec_ae_oae, oae_input, top_paths
        )
        return True, top_paths, [], [], verb

    # 8) Otherwise generate fallback paths & verbalizations
    fb_drug = generate_fallback_drug_paths(drug, cadec_pairs, cadec_ae_oae, n_disconnect)
    fb_ae = generate_fallback_ae_paths(ae_input_list, cadec_pairs, cadec_ae_oae, oae_input, n_disconnect)
    verb_fb = verbalize_drug_to_input_ae_paths(
        drug, cadec_pairs, cadec_ae_oae, oae_input, fb_drug + fb_ae
    )
    return False, [], fb_drug, fb_ae, verb_fb
