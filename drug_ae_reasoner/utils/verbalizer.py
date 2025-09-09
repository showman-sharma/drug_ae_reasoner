# drug_ae_reasoner/utils/verbalizer.py
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict

def verbalize_drug_to_input_ae_paths(
    drug_input: str,
    cadec_pairs: List[Tuple[str, str, str]],  # (drug_label, cadec_ae, cui_str)
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    oae_input_list: List[Tuple[str, str, float]],  # (input_ae, oae, sim)
    top_paths: List[Tuple[str, str, List[str], float]]  # (drug_label, input_ae, [oae...], score)
) -> List[str]:
    """
    Build human-readable explanations for each path, e.g.:
    DRUG (CUIs: ...) -> CADEC_AE -> OAE_x -> ... -> OAE_y ~ similar to INPUT_AE
    """
    # AE -> CUI-strings (collected from all drug pairs)
    cui_map: DefaultDict[str, str] = defaultdict(str)
    for d, ae, cui_str in cadec_pairs:
        if cui_str and not cui_map[ae]:
            cui_map[ae] = cui_str

    # convenience lookups
    cadec_sim = {(ae, oae): sim for ae, lst in cadec_ae_oae_dict.items() for oae, sim in lst}
    input_sim = {(inp, oae): sim for inp, oae, sim in oae_input_list}

    narr: List[str] = []
    for drug_lbl, inp_lbl, path_nodes, score in top_paths:
        # Always try to recover the CADEC-AE label that mapped to the first OAE node
        oae_from, oae_to = path_nodes[0], path_nodes[-1]
        middle = path_nodes[1:-1]

        ae_cadec = None
        for ae, lst in cadec_ae_oae_dict.items():
            if any(o == oae_from for o, _ in lst):
                ae_cadec = ae
                break

        sim1 = cadec_sim.get((ae_cadec, oae_from), 0.0) if ae_cadec else 0.0
        sim2 = input_sim.get((inp_lbl, oae_to), 0.0)

        lines = [f"{drug_lbl} → {ae_cadec or 'unknown AE'}"]
        if middle:
            for n in middle:
                lines.append(f"→ {n} (OAE)")
        lines.extend([
            f"→ {oae_to} (OAE) ~ '{inp_lbl}'",
            f"[sim(cadec→oae_from)={sim1:.2f}; sim(oae_to→input)={sim2:.2f}; score={score:.2f}]"
        ])
        narr.append(" ".join(lines))

    return narr
