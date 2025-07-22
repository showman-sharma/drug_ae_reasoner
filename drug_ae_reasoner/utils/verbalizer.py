from typing import List, Tuple, Dict

def verbalize_drug_to_input_ae_paths(
    drug_input: str,
    cadec_pairs: List[Tuple[str, str, str]],
    cadec_ae_oae_dict: Dict[str, List[Tuple[str, float]]],
    oae_input_list: List[Tuple[str, str, float]],
    top_paths: List[Tuple[str, str, List[str], float]]
) -> List[str]:
    """
    Build human-readable explanations for each path:
      - drug_input: the original drug string
      - cadec_pairs: list of (node_id, ae_label, cui_str)
      - cadec_ae_oae_dict: CADEC-AE → [(OAE, sim), ...]
      - oae_input_list: input-AE → [(OAE, sim), ...]
      - top_paths: list of (drug_node, input_ae, path_nodes, score)
    """
    # Map CADEC AE → its CUI string
    cui_map = {ae: cui for _, ae, cui in cadec_pairs}

    # Similarities: (CADEC AE, OAE) and (input AE, OAE)
    cadec_sim = {
        (ae, oae): sim
        for ae, lst in cadec_ae_oae_dict.items()
        for oae, sim in lst
    }
    input_sim = {
        (inp, oae): sim
        for inp, oae, sim in oae_input_list
    }

    narratives: List[str] = []

    for drug_lbl, inp_lbl, path, score in top_paths:
        # ── direct (0-hop) paths or fallbacks ─────────────────────────────
        if len(path) < 2:
            # find the CADEC-AE label that maps to this OAE, or mark fallback
            ae_cadec = next(
                (
                    ae
                    for ae, lst in cadec_ae_oae_dict.items()
                    if any(o == path[0] for o, _ in lst)
                ),
                "__fallback_ae__"
            )
            cui_str = cui_map.get(ae_cadec, "N/A")

            # safely get similarities (defaults to 0.0)
            sim_val = cadec_sim.get((ae_cadec, path[0]), 0.0)
            sim_to_input = input_sim.get((inp_lbl, path[0]), 0.0)

            narratives.append("; ".join([
                f"{drug_input} normalizes_to CADEC_drug {drug_lbl} via CUI(s)({cui_str})",
                f"{drug_lbl} causes {ae_cadec}",
                f"{ae_cadec} is_similar_to {path[0]} (sim={sim_val:.2f})",
                f"{path[0]} is_similar_to {inp_lbl} (sim={sim_to_input:.2f})",
                f"# total path score = {score:.2f}"
            ]))
            continue

        # ── 1-hop+ paths ───────────────────────────────────────────────────
        oae_from, oae_to = path[0], path[-1]
        middle = path[1:-1]

        # CADEC-AE label mapping
        ae_cadec = next(
            ae
            for ae, lst in cadec_ae_oae_dict.items()
            if any(o == oae_from for o, _ in lst)
        )
        cui_str = cui_map.get(ae_cadec, "N/A")

        sim1 = cadec_sim.get((ae_cadec, oae_from), 0.0)
        sim2 = input_sim.get((inp_lbl, oae_to), 0.0)

        # build the narrative lines
        lines = [
            f"{drug_input} normalizes_to CADEC_drug {drug_lbl} via CUI(s)({cui_str})",
            f"{drug_lbl} causes {ae_cadec}",
            f"{ae_cadec} is_similar_to {oae_from} (sim={sim1:.2f})"
        ]

        prev = oae_from
        for nxt in (*middle, oae_to):
            lines.append(f"{prev} relates_to {nxt} (in OAE)")
            prev = nxt

        lines.extend([
            f"{oae_to} is_similar_to {inp_lbl} (sim={sim2:.2f})",
            f"# total path score = {score:.2f}"
        ])

        narratives.append("; ".join(lines))

    return narratives
