
import os
import csv
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from drug_ae_reasoner.data.cadec_loader import get_cadec_drug_nodes
from drug_ae_reasoner.config import CADEC_KG_PATH, RX_PATH, OAE_INDEX_PATH, OAE_LABEL_MAP_PATH
from drug_ae_reasoner.utils.similarity_search import build_input_ae_oae_list
from drug_ae_reasoner.utils.path_reasoner import find_top_drug_to_input_ae_paths
from drug_ae_reasoner.utils.verbalizer import verbalize_drug_to_input_ae_paths


ADR_PATH = "drug_ae_reasoner/data/adr/adr_dataset_train_test.jsonl"

OUTPUT_CSV = "drug_ae_reasoner/data/adr/adr_paths.csv"
OUTPUT_JSONL = "drug_ae_reasoner/data/adr/adr_paths.jsonl"
CHECKPOINT_JSONL = "drug_ae_reasoner/data/adr/adr_paths_checkpoint.jsonl"


print("Loading ADR dataset and checking paths in CADEC KG...")
with open(ADR_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    records = data["train"] if isinstance(data, dict) and "train" in data else data

# Load checkpoint if exists
import os
processed_ids = set()
if os.path.exists(CHECKPOINT_JSONL):
    with open(CHECKPOINT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                processed_ids.add(obj.get("id"))
            except Exception:
                continue

# Precompute MEL for all unique drugs and AEs
all_aes = set()
all_drugs = set()
for obj in records:
    text = obj.get("text", "")
    ae_spans = obj.get("ADE_list", [])
    drug_spans = obj.get("Medicine_list", [])
    all_aes.update([text[start:end].strip().lower() for start, end in ae_spans if isinstance(start, int) and isinstance(end, int)])
    all_drugs.update([text[start:end].strip().lower() for start, end in drug_spans if isinstance(start, int) and isinstance(end, int)])

print(f"Running batch MEL for {len(all_aes)} unique AE mentions...")
ae_mel_matches = build_input_ae_oae_list(
    list(all_aes),
    n_input=1,
    input_ae_threshold=0.6,
    index_path=OAE_INDEX_PATH,
    label_map_path=OAE_LABEL_MAP_PATH
)
ae_mel_map = {ae: matched for ae, matched, _ in ae_mel_matches}

# Cache all drug node lookups in memory
print(f"Running batch MEL for {len(all_drugs)} unique drug mentions...")
drug_mel_map = {}
for drug in tqdm(all_drugs, desc="Drug MEL"):
    if drug not in drug_mel_map:
        nodes = get_cadec_drug_nodes(drug, CADEC_KG_PATH, RX_PATH, mel_top_k=1, mel_threshold=0.6, use_embedding=True, mel_require_confirmation=False)
        drug_mel_map[drug] = nodes

# Load CADEC KG once and keep in memory
from drug_ae_reasoner.data.cadec_loader import _load_cadec_graph
G = _load_cadec_graph(CADEC_KG_PATH)

def process_record(obj):
    text = obj.get("text", "")
    drug_spans = obj.get("Medicine_list", [])
    ae_spans = obj.get("ADE_list", [])
    drugs = [text[start:end].strip().lower() for start, end in drug_spans if isinstance(start, int) and isinstance(end, int)]
    aes = [text[start:end].strip().lower() for start, end in ae_spans if isinstance(start, int) and isinstance(end, int)]
    paths = []
    found_path = False
    for drug in drugs:
        drug_nodes = drug_mel_map.get(drug, [])
        if not drug_nodes:
            continue
        for ae in aes:
            input_ae = ae_mel_map.get(ae, None)
            if not input_ae:
                continue
            # Only search for paths if both drug and AE are mapped
            top_paths = find_top_drug_to_input_ae_paths(drug, ae, kg=G, rx_path=RX_PATH, mel_top_k=1, mel_threshold=0.6)
            verbalized = verbalize_drug_to_input_ae_paths(drug, ae, top_paths)
            if verbalized:
                found_path = True
                for v in verbalized:
                    paths.append({
                        "drug": drug,
                        "ae": ae,
                        "verbalized_path": v
                    })
    # Only document found paths
    if found_path:
        return {
            "id": obj.get("id", ""),
            "paths": paths,
            "path_found": True
        }
    return None


results = []
paths_found = 0
to_process = [obj for obj in records if obj.get("id") not in processed_ids]
print(f"Resuming from checkpoint: {len(processed_ids)} already processed, {len(to_process)} to process.")
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_record, obj) for obj in to_process]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel ADR path checking"):
        result = future.result()
        if result:
            results.append(result)
            paths_found += 1
            # Append to checkpoint file immediately
            with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")


# Merge checkpoint and new results for final output
all_results = []
if os.path.exists(CHECKPOINT_JSONL):
    with open(CHECKPOINT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                all_results.append(obj)
            except Exception:
                continue
all_results.extend(results)

# Write results to CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["id", "drug", "ae", "verbalized_path", "path_found"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in all_results:
        for p in r["paths"]:
            writer.writerow({
                "id": r["id"],
                "drug": p["drug"],
                "ae": p["ae"],
                "verbalized_path": p["verbalized_path"],
                "path_found": r["path_found"]
            })

# Write results to JSONL
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for r in all_results:
        f.write(json.dumps(r) + "\n")

# Print stats
total = len(records)
print(f"ADR sentences with legit paths in KG: {paths_found}/{total} ({100.0 * paths_found / total:.2f}%)")
print(f"Path dataset saved to: {OUTPUT_CSV}")
print(f"Full path details saved to: {OUTPUT_JSONL}")
