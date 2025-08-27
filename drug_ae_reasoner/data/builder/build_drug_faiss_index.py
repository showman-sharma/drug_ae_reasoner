import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from drug_ae_reasoner.data.cadec_loader import _load_cadec_graph
from drug_ae_reasoner.utils.encoding import encode_text

def main():
    kg_path = os.path.join("drug_ae_reasoner", "data", "cadec", "cadec_normalized_kg.gpickle")
    out_dir = os.path.join("drug_ae_reasoner", "data", "cadec")
    os.makedirs(out_dir, exist_ok=True)
    G = _load_cadec_graph(kg_path)

    drug_texts = []
    node_ids = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "drug":
            names = [data.get("label") or ""] + list(data.get("synonyms", []))
            for nm in names:
                nm_l = nm.lower()
                drug_texts.append(nm_l)
                node_ids.append(node)

    print(f"Encoding {len(drug_texts)} drug names/synonyms...")
    vecs = []
    for nm in tqdm(drug_texts, desc="Encoding drug names"):
        vecs.append(encode_text(nm).astype(np.float32))
    xb = np.vstack(vecs)

    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)

    index_path = os.path.join(out_dir, "drug_faiss_index.faiss")
    names_path = os.path.join(out_dir, "drug_faiss_names.pkl")
    nodes_path = os.path.join(out_dir, "drug_faiss_nodes.pkl")

    faiss.write_index(index, index_path)
    with open(names_path, "wb") as f:
        pickle.dump(drug_texts, f)
    with open(nodes_path, "wb") as f:
        pickle.dump(node_ids, f)

    print(f"FAISS drug index saved: {index_path}")
    print(f"Names saved: {names_path}")
    print(f"Node ids saved: {nodes_path}")

if __name__ == "__main__":
    main()
