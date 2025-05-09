from typing import List, Tuple, Dict

def verbalize_drug_to_input_ae_paths(drug_input: str,
                                     cadec_pairs: List[Tuple[str, str, str]],
                                     cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
                                     oae_input_list: List[Tuple[str, str, float]],
                                     top_paths: List[Tuple[str, str, List[str], float]]) -> List[str]:
    cui_map = {ae: cui for _, ae, cui in cadec_pairs}
    cadec_sim = {(ae, oae): sim for ae, lst in cadec_ae_oae_dict.items() for oae, sim in lst}
    input_sim = {(inp, oae): sim for inp, oae, sim in oae_input_list}
    narratives = []
    for drug_lbl, inp_lbl, path, score in top_paths:
        if len(path) < 2:
            ae_cadec = next((ae for ae, lst in cadec_ae_oae_dict.items() if any(o == path[0] for o, _ in lst)),"__fallback_ae__")
            cui_str = cui_map.get(ae_cadec, "N/A")
            
            sim_to_input = input_sim.get((inp_lbl, path[0]), 0.0)
            narratives.append("; ".join([
                f"{drug_input} normalizes_to CADEC_drug {drug_lbl} via CUI(s)({cui_str})",
                f"{drug_lbl} causes {ae_cadec}",
                f"{ae_cadec} is_similar_to {path[0]} (sim={cadec_sim[(ae_cadec, path[0])]:.2f})",
                f"{path[0]} is_similar_to {inp_lbl} (sim={sim_to_input:.2f})",
                f"# total path score = {score:.2f}"
            ]))
            continue
        oae_from, oae_to = path[0], path[-1]
        middle = path[1:-1]
        ae_cadec = next(ae for ae, lst in cadec_ae_oae_dict.items() if any(o == oae_from for o, _ in lst))
        cui_str = cui_map[ae_cadec]
        sim1 = cadec_sim.get((ae_cadec, path[0]), 0.0)
        sim2 = input_sim.get((inp_lbl, oae_to), 0.0)
        lines = [
            f"{drug_input} normalizes_to CADEC_drug {drug_lbl} via CUI(s)({cui_str})",
            f"{drug_lbl} causes {ae_cadec}",
            f"{ae_cadec} is_similar_to {oae_from} (sim={sim1:.2f})"
        ]
        prev = oae_from
        for nxt in middle + [oae_to]:
            lines.append(f"{prev} relates_to {nxt} (in OAE)")
            prev = nxt
        lines.append(f"{oae_to} is_similar_to {inp_lbl} (sim={sim2:.2f})")
        lines.append(f"# total path score = {score:.2f}")
        narratives.append("; ".join(lines))
    return narratives
