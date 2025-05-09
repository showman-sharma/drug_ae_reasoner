import os
import pickle

def load_rxnorm(rrf_dir):
    cui_to_names = {}
    with open(os.path.join(rrf_dir, "RXNCONSO.RRF"), encoding="utf-8") as f:
        for ln in f:
            parts = ln.split("|")
            cui, lang, supp, name = parts[0], parts[1], parts[16], parts[14]
            if lang == "ENG" and supp != "Y":
                cui_to_names.setdefault(cui, set()).add(name.lower())
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

    with open(out_kg, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved normalized KG to: {out_kg}")

if __name__ == "__main__":
    normalize()
