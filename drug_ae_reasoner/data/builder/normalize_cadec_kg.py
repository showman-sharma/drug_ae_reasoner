import os
import pickle
from collections import defaultdict


def load_rxnorm(rrf_dir):
    """Load RxNorm CUI -> names mapping including brand synonyms."""
    cui_to_names = defaultdict(set)
    with open(os.path.join(rrf_dir, "RXNCONSO.RRF"), encoding="utf-8") as f:
        for ln in f:
            parts = ln.split("|")
            if len(parts) < 17:
                continue
            cui, lang, supp, name = parts[0], parts[1], parts[16], parts[14]
            if lang == "ENG" and supp != "Y":
                cui_to_names[cui].add(name.lower())

    # Map generic CUI -> brand CUIs
    rel_path = os.path.join(rrf_dir, "RXNREL.RRF")
    generic_to_brand = defaultdict(set)
    if os.path.exists(rel_path):
        with open(rel_path, encoding="utf-8") as f:
            for ln in f:
                parts = ln.split("|")
                if len(parts) < 8:
                    continue
                cui1, cui2, rela = parts[0], parts[4], parts[7]
                if rela == "has_tradename":
                    generic_to_brand[cui1].add(cui2)
                elif rela == "tradename_of":
                    generic_to_brand[cui2].add(cui1)

    for gen, brands in generic_to_brand.items():
        for b in brands:
            cui_to_names[gen].update(cui_to_names.get(b, []))

    return cui_to_names

def normalize():
    cadec_dir = os.path.join("drug_ae_reasoner", "data", "cadec")
    rx_dir = os.path.join("drug_ae_reasoner", "data", "rxnorm")
    in_kg = os.path.join(cadec_dir, "cadec_verbalizer_kg.gpickle")
    out_kg = os.path.join(cadec_dir, "cadec_normalized_kg.gpickle")

    G = pickle.load(open(in_kg, "rb"))
    rx_map = load_rxnorm(rx_dir)

    for n, data in G.nodes(data=True):
        if data.get("type") == "drug":
            label = data.get("label", "")
            cuis = {cui for cui, names in rx_map.items() if any(label.lower() in name for name in names)}
            data["cuis"] = cuis
            syn = set()
            for cui in cuis:
                syn.update(rx_map.get(cui, set()))
            if syn:
                data["synonyms"] = sorted(syn)

    with open(out_kg, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved normalized KG to: {out_kg}")

if __name__ == "__main__":
    normalize()
