# drug_ae_reasoner/utils/path_reasoner.py
import pickle
import networkx as nx
from typing import List, Tuple, Dict

from .similarity_search import build_input_ae_oae_list, build_cadec_ae_oae_mapping
from ..data.cadec_loader import get_cadec_ae_pairs, get_cadec_drug_nodes
from .verbalizer import verbalize_drug_to_input_ae_paths
from ..config import (
    OAE_GRAPH_PATH,
    RX_PATH,
    OAE_INDEX_PATH,
    OAE_LABEL_MAP_PATH,
)

# ─── Simple in-proc cache for loaded OAE graph ──────────────────────────
_GRAPH_CACHE: Dict[str, nx.MultiDiGraph] = {}

def _load_oae_graph(graph_path: str) -> nx.MultiDiGraph:
    if graph_path not in _GRAPH_CACHE:
        with open(graph_path, "rb") as f:
            _GRAPH_CACHE[graph_path] = pickle.load(f)
    return _GRAPH_CACHE[graph_path]

def find_drug_to_input_ae_paths(
    drug_label: str,
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    oae_input_list: List[Tuple[str, str, float]],
    graph_path: str = OAE_GRAPH_PATH
) -> List[Tuple[str, str, List[str]]]:
    """
    Construct candidate paths in OAE space:
      - 0-hop: OAE_cand == OAE_input
      - 1-hop: OAE_cand --edge--> OAE_input
    Returns: [(drug_label, input_ae_label, [oae_nodes_along_path]), ...]
    """
    G = _load_oae_graph(graph_path)
    paths: List[Tuple[str, str, List[str]]] = []
    # build quick lookup: input_ae -> [oae_input nodes]
    input_to_oae = {}
    for inp, oae, _ in oae_input_list:
        input_to_oae.setdefault(inp, []).append(oae)

    for cadec_ae, oae_cands in cadec_ae_oae_dict.items():
        for oae_cand, _ in oae_cands:
            for inp_lbl, oae_inputs in input_to_oae.items():
                for oae_in in oae_inputs:
                    if oae_cand == oae_in:
                        paths.append((drug_label, inp_lbl, [oae_cand]))  # 0-hop
                    elif G.has_edge(oae_cand, oae_in):
                        paths.append((drug_label, inp_lbl, [oae_cand, oae_in]))  # 1-hop
    return paths

def rank_drug_ae_paths(
    raw_paths: List[Tuple[str, str, List[str]]],
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    oae_input_list: List[Tuple[str, str, float]],
    n_paths: int = 5
) -> List[Tuple[str, str, List[str], float]]:
    """
    Score a candidate path by averaging:
      (sim CADEC_AE→first_OAE) and (sim input_AE→last_OAE).
    Dedupe identical OAE-node sequences and keep highest score.
    """
    # build lookups
    cadec_sim = {(ae, oae): sim for ae, lst in cadec_ae_oae_dict.items() for oae, sim in lst}
    input_sim = {(inp, oae): sim for inp, oae, sim in oae_input_list}

    scored: List[Tuple[str, str, List[str], float]] = []
    for drug_lbl, inp_lbl, oae_path in raw_paths:
        src = oae_path[0]; dst = oae_path[-1]
        # find the CADEC AE that mapped to src (best effort)
        ae_label = None
        for ae, lst in cadec_ae_oae_dict.items():
            if any(o == src for o, _ in lst):
                ae_label = ae
                break
        s1 = cadec_sim.get((ae_label, src), 0.0) if ae_label else 0.0
        s2 = input_sim.get((inp_lbl, dst), 0.0)
        score = (s1 + s2) / 2.0
        scored.append((drug_lbl, inp_lbl, oae_path, score))

    # dedupe by path nodes
    unique: Dict[Tuple[str, ...], Tuple[str, str, List[str], float]] = {}
    for d_lbl, i_lbl, path, scr in scored:
        key = tuple(path)
        prev = unique.get(key)
        if prev is None or scr > prev[3]:
            unique[key] = (d_lbl, i_lbl, path, scr)

    res = list(unique.values())
    res.sort(key=lambda x: x[3], reverse=True)
    return res[:n_paths]

def generate_fallback_drug_paths(
    drug_label: str,
    cadec_pairs: List[Tuple[str, str, str]],
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    n_disconnect: int = 3
) -> List[Tuple[str, str, List[str], float]]:
    """
    If no direct path found, propose top-N CADEC AEs for the drug.
    Score = (1.0 + best OAE sim) / 2 (use 1.0 as presence weight).
    """
    # collect CADEC AEs for this drug (case-insensitive)
    drug_label_l = drug_label.lower()
    ae_cands = [ae for d, ae, _ in cadec_pairs if d.lower() == drug_label_l]
    fallback: List[Tuple[str, str, List[str], float]] = []
    # keep unique AEs ordered by appearance
    seen = set()
    ordered = [a for a in ae_cands if not (a in seen or seen.add(a))]
    for ae in ordered[:n_disconnect]:
        if ae in cadec_ae_oae_dict and cadec_ae_oae_dict[ae]:
            oae, sim = max(cadec_ae_oae_dict[ae], key=lambda x: x[1])
            score = (1.0 + sim) / 2.0
            fallback.append((drug_label, ae, [oae], score))
    return fallback

def generate_fallback_ae_paths(
    ae_input_list: List[str],
    cadec_pairs: List[Tuple[str, str, str]],
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    oae_input_list: List[Tuple[str, str, float]],
    n_disconnect: int = 3
) -> List[Tuple[str, str, List[str], float]]:
    """
    If no direct path found, for each input AE pick its top OAE neighbor,
    then link back to a CADEC AE that mapped there, and choose any parent drug.
    Score = mean(sim_input_oae, sim_cadec_oae, 1.0_presence)
    """
    # reverse map OAE -> list of (cadec_ae, sim)
    oae_to_cadec: Dict[str, List[Tuple[str, float]]] = {}
    for ae, lst in cadec_ae_oae_dict.items():
        for o, s in lst:
            oae_to_cadec.setdefault(o, []).append((ae, s))

    # pregroup input neighbors
    grouped_in: Dict[str, List[Tuple[str, float]]] = {}
    for inp, oae, sim in oae_input_list:
        grouped_in.setdefault(inp, []).append((oae, sim))

    out: List[Tuple[str, str, List[str], float]] = []
    for ae_in in ae_input_list:
        neigh = sorted(grouped_in.get(ae_in, []), key=lambda x: x[1], reverse=True)[:n_disconnect]
        for oae_node, s_in in neigh:
            cand_cadec = oae_to_cadec.get(oae_node, [])
            if not cand_cadec:
                continue
            ae_cadec, s_ca = max(cand_cadec, key=lambda x: x[1])
            # find any parent drug for that CADEC AE
            parents = [d for d, ae, _ in cadec_pairs if ae == ae_cadec]
            if not parents:
                continue
            drug = parents[0]
            score = (s_in + s_ca + 1.0) / 3.0
            out.append((drug, ae_in, [oae_node], score))
    return out

def find_top_drug_to_input_ae_paths(
    drug: str,
    ae_input_list: List[str],
    rx_path: str,
    cadec_kg_path: str,
    oae_index_path: str = OAE_INDEX_PATH,
    oae_label_map_path: str = OAE_LABEL_MAP_PATH,
    oae_graph_path: str = OAE_GRAPH_PATH,
    n_paths: int = 5,
    n_cadec: int = 5,
    n_input: int = 5,
    cadec_ae_threshold: float = 0.7,
    input_ae_threshold: float = 0.7,
    n_disconnect: int = 3
):
    """
    Orchestrates:
      1) CADEC drug node(s) -> CADEC AE pairs
      2) CADEC AEs -> OAE candidates (FAISS)
      3) input AEs -> OAE candidates (FAISS)
      4) OAE graph 0/1-hop paths, ranking, fallback, verbalization
    """
    # 1) CADEC drug nodes & pairs
    # Respect the caller-provided RxNorm path instead of always using the
    # package-level default.  Previously this function ignored the `rx_path`
    # argument and unconditionally used `RX_PATH`, making it impossible to run
    # against alternative RxNorm files (e.g., in tests or custom deployments).
    drug_nodes = get_cadec_drug_nodes(drug, cadec_kg_path, rx_path)
    cadec_pairs = get_cadec_ae_pairs(drug_nodes, cadec_kg_path)
    ae_cadec_list = sorted({ae for _, ae, _ in cadec_pairs})

    # 2) map CADEC AEs -> OAE (threshold)
    cadec_ae_oae = build_cadec_ae_oae_mapping(
        ae_cadec_list,
        n_cadec=n_cadec,
        cadec_ae_threshold=cadec_ae_threshold,
        index_path=oae_index_path,
        label_map_path=oae_label_map_path,
    )

    # 3) map input AEs -> OAE (threshold)
    oae_input = build_input_ae_oae_list(
        ae_input_list,
        n_input=n_input,
        input_ae_threshold=input_ae_threshold,
        index_path=oae_index_path,
        label_map_path=oae_label_map_path,
    )

    # 4) candidate paths (0/1 hop)
    raw_paths = find_drug_to_input_ae_paths(
        drug, cadec_ae_oae, oae_input, graph_path=oae_graph_path
    )

    # 5) rank
    top_paths = rank_drug_ae_paths(raw_paths, cadec_ae_oae, oae_input, n_paths)

    if top_paths:
        verb = verbalize_drug_to_input_ae_paths(drug, cadec_pairs, cadec_ae_oae, oae_input, top_paths)
        return True, top_paths, [], [], verb

    # 6) fallbacks
    fb_drug = generate_fallback_drug_paths(drug, cadec_pairs, cadec_ae_oae, n_disconnect)
    fb_ae = generate_fallback_ae_paths(ae_input_list, cadec_pairs, cadec_ae_oae, oae_input, n_disconnect)
    verb_fb = verbalize_drug_to_input_ae_paths(drug, cadec_pairs, cadec_ae_oae, oae_input, fb_drug + fb_ae)
    return False, [], fb_drug, fb_ae, verb_fb
