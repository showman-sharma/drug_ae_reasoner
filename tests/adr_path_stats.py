

import os
import json
from tqdm import tqdm
from drug_ae_reasoner.utils.path_reasoner import find_top_drug_to_input_ae_paths

DATA_DIR = os.path.join(os.path.dirname(__file__), '../drug_ae_reasoner/data/adr')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '../drug_ae_reasoner/data/adr/adr_path_stats.json')
THRESHOLD = 0  # Low threshold as requested
TOP_K = 3

def read_jsonl(file_path):
    # Special loader for the main ADR dataset
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.read().strip()
        if first_line.startswith('{') and '"train"' in first_line:
            # JSON object with 'train' key
            obj = json.loads(first_line)
            return obj.get("train", [])
        else:
            # Fallback: standard JSONL
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]

def main():


    # Only process the main ADR dataset file
    input_file = os.path.join(DATA_DIR, "adr_dataset_train_test.jsonl")
    data = read_jsonl(input_file)
    # Checkpointing: resume from file if exists
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    with tqdm(total=len(data), desc="Processing ADR dataset entries") as pbar:
        for entry in data:
            sent_id = entry.get("id")
            if sent_id in results:
                pbar.update(1)
                continue
            # Try to extract drug and AE from text using Medicine_list and ADE_list indices
            text = entry.get("text", "")
            med_spans = entry.get("Medicine_list", [])
            ade_spans = entry.get("ADE_list", [])
            paths_for_pairs = []

            # For each drug–AE pair in the entry
            import random
            for m_idx, m_span in enumerate(med_spans):
                drug = text[m_span[0]:m_span[1]].strip() if m_span else None
                for a_idx, a_span in enumerate(ade_spans):
                    ae = text[a_span[0]:a_span[1]].strip() if a_span else None
                    if drug and ae:
                        from drug_ae_reasoner.config import RX_PATH, CADEC_KG_PATH, OAE_INDEX_PATH, OAE_LABEL_MAP_PATH, OAE_GRAPH_PATH
                        connected, top_paths, fb_drug, fb_ae, verb = find_top_drug_to_input_ae_paths(
                            drug=drug,
                            ae_input_list=[ae],
                            rx_path=RX_PATH,
                            cadec_kg_path=CADEC_KG_PATH,
                            oae_index_path=OAE_INDEX_PATH,
                            oae_label_map_path=OAE_LABEL_MAP_PATH,
                            oae_graph_path=OAE_GRAPH_PATH,
                            mel_top_k=TOP_K,
                            mel_threshold=-1,
                            cadec_ae_threshold=-1,
                            input_ae_threshold=-1
                        )
                        def classify_path_type(path):
                            score = path[-1] if len(path) > 3 else None
                            if score is not None and score >= 0.6:
                                return "positive"
                            elif score is not None and score > 0:
                                return "negative"
                            else:
                                return "random"

                        relation = "ADR" if connected else "No-ADR"
                        if top_paths:
                            for i, path in enumerate(top_paths):
                                paths_for_pairs.append({
                                    "drug": drug,
                                    "adr": ae,
                                    "mapped_drug_concept": path[0],
                                    "mapped_ae_concept": path[1],
                                    "oae_nodes": path[2],
                                    "path_type": classify_path_type(path),
                                    "path_weight": path[3] if len(path) > 3 else None,
                                    "relation": relation,
                                    "verbalization": verb[i] if i < len(verb) else ""
                                })
                        else:
                            # Sample a real random path from the KG
                            from drug_ae_reasoner.data.cadec_loader import _load_cadec_graph
                            kg_path = CADEC_KG_PATH
                            G = _load_cadec_graph(kg_path)
                            drug_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "drug"]
                            ae_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "adverse_effect"]
                            if drug_nodes and ae_nodes:
                                # Try up to 5 times to find valid nodes
                                for _ in range(5):
                                    rand_drug_node = random.choice(drug_nodes)
                                    rand_ae_node = random.choice(ae_nodes)
                                    if rand_drug_node not in G.nodes or rand_ae_node not in G.nodes:
                                        continue  # skip missing nodes
                                    drug_label = G.nodes[rand_drug_node].get("label", drug)
                                    ae_label = G.nodes[rand_ae_node].get("label", ae)
                                    # Try to find a direct edge, else just use the two nodes
                                    oae_nodes = []
                                    for _, _, data in G.out_edges(rand_drug_node, data=True):
                                        if data.get("type") == "adverse_effect" and _ == rand_ae_node:
                                            oae_nodes = [ae_label]
                                            break
                                    if not oae_nodes:
                                        oae_nodes = [ae_label]
                                    score = round(random.uniform(0.1, 0.3), 2)
                                    # Build a CLI-style verbalization: drug → [intermediate nodes] → ae (OAE) ~ 'ae' [sim(cadec→oae_from)=X; sim(oae_to→input)=Y; score=Z]
                                    path_nodes = [drug_label] + oae_nodes
                                    # For random, we don't have real similarity scores, so use placeholders
                                    sim1 = round(random.uniform(0.7, 0.9), 2)
                                    sim2 = round(random.uniform(0.7, 0.9), 2)
                                    # Build node traversal string
                                    node_str = " → ".join(path_nodes)
                                    verbalization = f"{node_str} (OAE) ~ '{ae}' [sim(cadec→oae_from)={sim1}; sim(oae_to→input)={sim2}; score={score}]"
                                    random_path = {
                                        "drug": drug,
                                        "adr": ae,
                                        "mapped_drug_concept": drug_label,
                                        "mapped_ae_concept": ae_label,
                                        "oae_nodes": oae_nodes,
                                        "path_type": "random",
                                        "path_weight": score,
                                        "relation": "No-ADR",
                                        "verbalization": verbalization
                                    }
                                    paths_for_pairs.append(random_path)
                                    break
                                else:
                                    # If no valid nodes found after retries, fallback to synthetic
                                    random_path = {
                                        "drug": drug,
                                        "adr": ae,
                                        "mapped_drug_concept": drug,
                                        "mapped_ae_concept": ae,
                                        "oae_nodes": ["random_node_" + str(random.randint(1, 1000))],
                                        "path_type": "random",
                                        "path_weight": None,
                                        "relation": "No-ADR",
                                        "verbalization": f"Random path for {drug} → {ae} (missing KG nodes)"
                                    }
                                    paths_for_pairs.append(random_path)
                            else:
                                # Fallback to synthetic random path if KG is empty
                                random_path = {
                                    "drug": drug,
                                    "adr": ae,
                                    "mapped_drug_concept": drug,
                                    "mapped_ae_concept": ae,
                                    "oae_nodes": ["random_node_" + str(random.randint(1, 1000))],
                                    "path_type": "random",
                                    "path_weight": None,
                                    "relation": "No-ADR",
                                    "verbalization": f"Random path for {drug} → {ae}"
                                }
                                paths_for_pairs.append(random_path)
            results[sent_id] = paths_for_pairs
            # Checkpoint after every entry
            with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, indent=2, ensure_ascii=False)
            pbar.update(1)

if __name__ == "__main__":
    main()