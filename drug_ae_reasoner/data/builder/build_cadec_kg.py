import os
import pickle
import random
from drug_ae_reasoner.utils.verbalizer_utils import (
    read_cadec_documents, build_cadec_kg_from_docs,
    dedupe_cadec, list_all_unique_drugs
)

def main():
    random.seed(42)
    raw_path = os.path.join("drug_ae_reasoner", "data", "cadec", "train.conll")
    out_dir = os.path.join("drug_ae_reasoner", "data", "cadec")
    os.makedirs(out_dir, exist_ok=True)

    documents = read_cadec_documents(raw_path)
    print(f"Total documents: {len(documents)}")

    random.shuffle(documents)
    split_idx = int(0.3 * len(documents))
    train_docs = documents[:split_idx]
    test_docs = documents[split_idx:]

    train_out = os.path.join(out_dir, "train_30.jsonl")
    with open(train_out, "w", encoding="utf-8") as f:
        for doc_id, tokens in train_docs:
            f.write(f"{doc_id}\n")
            for parts in tokens:
                f.write("\t".join(parts) + "\n")
            f.write("\n")
    print(f"Saved 30% training split to: {train_out}")

    G = build_cadec_kg_from_docs(test_docs)
    G = dedupe_cadec(G)
    kg_out = os.path.join(out_dir, "cadec_verbalizer_kg.gpickle")
    with open(kg_out, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved KG: {kg_out}")

    for drug in list_all_unique_drugs(G)[:10]:
        print(" •", drug)

if __name__ == "__main__":
    main()
