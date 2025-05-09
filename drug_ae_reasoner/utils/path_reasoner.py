import pickle
import networkx as nx
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict
from .similarity_search import build_input_ae_oae_list, build_cadec_ae_oae_mapping
from ..data.cadec_loader import get_cadec_ae_pairs, get_cadec_drug_nodes
from .verbalizer import verbalize_drug_to_input_ae_paths

def find_drug_to_input_ae_paths(drug_label: str,
                                cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
                                oae_input_list: List[Tuple[str, str, float]],
                                graph_path: str) -> List[Tuple[str, str, List[str]]]:
    with open(graph_path, "rb") as f:
        G: nx.Graph = pickle.load(f)

    paths = []
    input_map: Dict[str, List[str]] = {}
    for inp_label, oae_node, _ in oae_input_list:
        input_map.setdefault(inp_label, []).append(oae_node)

    for cadec_ae, oae_candidates in cadec_ae_oae_dict.items():
        for oae_cand, _ in oae_candidates:
            for inp_label, oae_in_nodes in input_map.items():
                for oae_in in oae_in_nodes:
                    if oae_cand == oae_in:
                        paths.append((drug_label, inp_label, [oae_cand]))
                    elif G.has_edge(oae_cand, oae_in):
                        paths.append((drug_label, inp_label, [oae_cand, oae_in]))
    return paths

def rank_drug_ae_paths(raw_paths, cadec_ae_oae_dict, oae_input_list, n_paths=5):
    cadec_sim = {(ae, oae): sim for ae, lst in cadec_ae_oae_dict.items() for oae, sim in lst}
    input_sim = {(inp, oae): sim for inp, oae, sim in oae_input_list}
    scored = []
    for drug_label, inp_label, path_nodes in raw_paths:
        oae_cand = path_nodes[0]
        oae_in = path_nodes[-1]
        sim1 = cadec_sim.get((path_nodes[0], oae_cand), 0.0)
        sim2 = input_sim.get((inp_label, oae_in), 0.0)
        score = sim1 + sim2
        scored.append((drug_label, inp_label, path_nodes, score))
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:n_paths]

def generate_fallback_drug_paths(drug_label, cadec_pairs, cadec_ae_oae_dict, n_disconnect):
    edges = [(ae_c, oae, sim) for ae_c, neigh in cadec_ae_oae_dict.items() for oae, sim in neigh]
    edges.sort(key=lambda x: x[2], reverse=True)
    fallback = []
    for ae_c, oae, sim in edges[:n_disconnect]:
        fallback_label = f"__fallback_ae__::{oae}"
        fallback.append((drug_label, fallback_label, [oae], sim))
    return fallback

def generate_fallback_ae_paths(ae_input_list, cadec_pairs, cadec_ae_oae_dict, oae_input_list, n_disconnect):
    rev: DefaultDict[str, List[Tuple[str, float]]] = defaultdict(list)
    for ca, neigh in cadec_ae_oae_dict.items():
        for oae, s in neigh:
            rev[oae].append((ca, s))

    fallback = []
    for inp_ae in ae_input_list:
        neighbors = [(i, o, s) for i, o, s in oae_input_list if i == inp_ae]
        neighbors.sort(key=lambda x: x[2], reverse=True)
        for _, oae, sim_inp in neighbors[:n_disconnect]:
            candidates = rev.get(oae, [])
            if candidates:
                cae, _ = max(candidates, key=lambda x: x[1])
                parents = [d for d, ae, _ in cadec_pairs if ae == cae]
                drug2 = parents[0] if parents else "__no_drug2__"
            else:
                cae, drug2 = "__no_cadec__", "__no_drug2__"
            fallback.append((drug2, inp_ae, [oae], sim_inp))
    return fallback

def find_top_drug_to_input_ae_paths(drug, ae_input_list, rx_path, cadec_kg_path,
                                    oae_index_path, oae_label_map_path, oae_graph_path,
                                    n_cadec=5, cadec_ae_threshold=0.7,
                                    n_input=5, input_ae_threshold=0.7,
                                    n_paths=5, n_disconnect=3):
    drug_nodes = get_cadec_drug_nodes(drug, rx_path, cadec_kg_path)
    cadec_pairs = get_cadec_ae_pairs(drug_nodes, cadec_kg_path)
    ae_cadec_list = sorted({ae for _, ae, _ in cadec_pairs})
    cadec_ae_oae = build_cadec_ae_oae_mapping(ae_cadec_list, oae_index_path, oae_label_map_path,
                                              n_cadec, cadec_ae_threshold)
    oae_input = build_input_ae_oae_list(ae_input_list, oae_index_path, oae_label_map_path,
                                        n_input, input_ae_threshold)
    raw_paths = find_drug_to_input_ae_paths(drug, cadec_ae_oae, oae_input, oae_graph_path)
    top_paths = rank_drug_ae_paths(raw_paths, cadec_ae_oae, oae_input, n_paths)
    if top_paths:
        verb = verbalize_drug_to_input_ae_paths(drug, cadec_pairs, cadec_ae_oae, oae_input, top_paths)
        return True, top_paths, [], [], verb

    fb_drug = generate_fallback_drug_paths(drug, cadec_pairs, cadec_ae_oae, n_disconnect)
    fb_ae = generate_fallback_ae_paths(ae_input_list, cadec_pairs, cadec_ae_oae, oae_input, n_disconnect)
    all_fb = fb_drug + fb_ae
    verb_fb = verbalize_drug_to_input_ae_paths(drug, cadec_pairs, cadec_ae_oae, oae_input, all_fb)
    return False, [], fb_drug, fb_ae, verb_fb
