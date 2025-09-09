import os
import csv
import json
from tqdm import tqdm
from drug_ae_reasoner.data.cadec_loader import get_cadec_drug_nodes, get_cadec_ae_pairs
from drug_ae_reasoner.config import CADEC_KG_PATH, RX_PATH
from drug_ae_reasoner.utils.similarity_search import build_input_ae_oae_list

ADR_PATH = "drug_ae_reasoner/data/adr/adr_dataset_train_test.jsonl"
OUTPUT_CSV = "drug_ae_reasoner/data/adr/adr_paths.csv"

print("Loading ADR dataset and checking paths in CADEC KG...")
with open(ADR_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    records = data["train"] if isinstance(data, dict) and "train" in data else data

results = []
paths_found = 0
for idx, obj in enumerate(tqdm(records, desc="Checking ADR paths in KG")):
    text = obj.get("text", "")
    drug_spans = obj.get("Medicine_list", [])
    ae_spans = obj.get("ADE_list", [])
    drugs = [text[start:end].strip().lower() for start, end in drug_spans if isinstance(start, int) and isinstance(end, int)]
    aes = [text[start:end].strip().lower() for start, end in ae_spans if isinstance(start, int) and isinstance(end, int)]
    found_path = False
    for drug in drugs:
        drug_nodes = get_cadec_drug_nodes(drug, CADEC_KG_PATH, RX_PATH)
        if not drug_nodes:
            continue
        ae_pairs = get_cadec_ae_pairs(drug_nodes, CADEC_KG_PATH)
        ae_labels = set([ae for _, ae, _ in ae_pairs])
        # Use MEL for AE matching (embedding-based)
        if ae_labels:
            ae_mel_matches = build_input_ae_oae_list(
                aes,
                n_input=1,
                input_ae_threshold=0.6,  # Same threshold as entity coverage
                index_path=os.path.join(os.path.dirname(CADEC_KG_PATH), "ae_faiss_index.faiss"),
                label_map_path=os.path.join(os.path.dirname(CADEC_KG_PATH), "ae_faiss_names.pkl")
            )
            matched_ae_set = set([ae for _, ae, _ in ae_mel_matches])
            for ae in matched_ae_set:
                if ae in ae_labels:
                    results.append({
                        "id": obj.get("id", ""),
                        "drug": drug,
                        "ae": ae,
                        "path_found": True
                    })
                    found_path = True
    if not found_path:
        results.append({
            "id": obj.get("id", ""),
            "drug": ",".join(drugs),
            "ae": ",".join(aes),
            "path_found": False
        })
    if found_path:
        paths_found += 1
    if idx % 100 == 0 and idx > 0:
        print(f"Processed {idx} ADR records...")

# Write results to CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["id", "drug", "ae", "path_found"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Print stats
total = len(records)
print(f"ADR sentences with legit paths in KG: {paths_found}/{total} ({100.0 * paths_found / total:.2f}%)")
print(f"Path dataset saved to: {OUTPUT_CSV}")
