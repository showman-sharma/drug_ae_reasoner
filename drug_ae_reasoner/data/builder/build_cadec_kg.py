import os
import pickle
import random
from drug_ae_reasoner.utils.verbalizer_utils import (
    read_cadec_documents, build_cadec_kg_from_docs,
    dedupe_cadec, list_all_unique_drugs
)

def main():
    raw_path = os.path.join("drug_ae_reasoner", "data", "cadec", "train.conll")
    out_dir = os.path.join("drug_ae_reasoner", "data", "cadec")
    os.makedirs(out_dir, exist_ok=True)

    documents = read_cadec_documents(raw_path)
    print(f"Total documents: {len(documents)}")

    # Build KG from all documents (no split)
    G = build_cadec_kg_from_docs(documents)
    G = dedupe_cadec(G)
    kg_out = os.path.join(out_dir, "cadec_verbalizer_kg.gpickle")
    with open(kg_out, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved KG (full CADEC): {kg_out}")

    print("All drug node labels in KG:")
    all_drugs = list_all_unique_drugs(G)
    for drug in all_drugs:
        print(" •", drug)
    print(f"Total unique drugs: {len(all_drugs)}")

if __name__ == "__main__":
    main()
