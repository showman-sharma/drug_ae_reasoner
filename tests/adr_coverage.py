import os
import json
import argparse
from tqdm import tqdm
from drug_ae_reasoner.data.cadec_loader import get_cadec_drug_nodes
from drug_ae_reasoner.utils.similarity_search import build_cadec_ae_oae_mapping, build_input_ae_oae_list
from drug_ae_reasoner.config import CADEC_KG_PATH, RX_PATH, OAE_INDEX_PATH, OAE_LABEL_MAP_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="ADR drug/AE match coverage against CADEC KG and OAE.")
    parser.add_argument("--adr_path", type=str, default="drug_ae_reasoner/data/adr/adr_dataset_train_test.jsonl", help="Path to ADR JSONL file")
    parser.add_argument("--cadec_kg_path", type=str, default=CADEC_KG_PATH, help="Path to CADEC KG")
    parser.add_argument("--rxn_rrf_path", type=str, default=RX_PATH, help="Path to RxNorm RRF")
    parser.add_argument("--oae_index_path", type=str, default=OAE_INDEX_PATH, help="Path to OAE FAISS index")
    parser.add_argument("--oae_label_map_path", type=str, default=OAE_LABEL_MAP_PATH, help="Path to OAE label map")
    parser.add_argument("--mel_top_k", type=int, default=5)
    parser.add_argument("--mel_threshold", type=float, default=0.6)
    return parser.parse_args()

def load_adr_drugs_aes(adr_path):
    drugs = set()
    aes = set()
    with open(adr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # If top-level is a dict with 'train', use that
        records = data["train"] if isinstance(data, dict) and "train" in data else data
        for obj in records:
            text = obj.get("text", "")
            # Extract drugs
            for span in obj.get("Medicine_list", []):
                if isinstance(span, list) and len(span) == 2:
                    start, end = span
                    drug = text[start:end].strip().lower()
                    if drug:
                        drugs.add(drug)
            # Extract AEs
            for span in obj.get("ADE_list", []):
                if isinstance(span, list) and len(span) == 2:
                    start, end = span
                    ae = text[start:end].strip().lower()
                    if ae:
                        aes.add(ae)
    return sorted(drugs), sorted(aes)

def main():
    args = parse_args()
    drugs, aes = load_adr_drugs_aes(args.adr_path)
    print(f"Found {len(drugs)} unique drugs and {len(aes)} unique AEs in ADR data.")

    # Check drug match in CADEC KG
    matched_drugs = []
    import sys
    import contextlib
    from functools import lru_cache
    class DummyFile:
        def write(self, x):
            pass
        def flush(self):
            pass

    # Wrap get_cadec_drug_nodes with lru_cache for efficiency
    @lru_cache(maxsize=2048)
    def cached_get_cadec_drug_nodes(drug, kg_path, rxn_rrf_path, mel_top_k, mel_threshold):
        # Force use_embedding=True and skip CUI/label passes for speed
        return tuple(get_cadec_drug_nodes(
            drug,
            kg_path,
            rxn_rrf_path,
            mel_top_k=mel_top_k,
            mel_threshold=mel_threshold,
            use_embedding=True,
            mel_require_confirmation=False
        ))

    # Only redirect stdout so tqdm's progress bar (on stderr) is visible
    with contextlib.redirect_stdout(DummyFile()):
        for drug in tqdm(drugs, desc="Matching drugs in CADEC KG"):
            nodes = cached_get_cadec_drug_nodes(drug, args.cadec_kg_path, args.rxn_rrf_path, args.mel_top_k, args.mel_threshold)
            if nodes:
                matched_drugs.append(drug)
    print(f"Drugs matched in CADEC KG: {len(matched_drugs)}/{len(drugs)}")

    # Check AE match in OAE (via FAISS index)
    matched_aes_oae = []
    oae_matches = build_input_ae_oae_list(aes, n_input=1, input_ae_threshold=args.mel_threshold, index_path=args.oae_index_path, label_map_path=args.oae_label_map_path)
    oae_matched_set = set([ae for _, ae, _ in oae_matches])
    for ae in aes:
        if any(ae in match for match in oae_matched_set):
            matched_aes_oae.append(ae)
    print(f"AEs matched in OAE: {len(matched_aes_oae)}/{len(aes)}")

    # Check AE match in CADEC KG (string match)
    from drug_ae_reasoner.data.cadec_loader import get_cadec_ae_pairs
    cadec_ae_set = set()
    pairs = get_cadec_ae_pairs([], args.cadec_kg_path)
    for _, ae, _ in pairs:
        cadec_ae_set.add(ae.strip().lower())
    matched_aes_cadec = [ae for ae in aes if ae in cadec_ae_set]
    print(f"AEs matched in CADEC KG: {len(matched_aes_cadec)}/{len(aes)}")

    # Optionally, print lists
    # print("Matched drugs:", matched_drugs)
    # print("Matched AEs (OAE):", matched_aes_oae)
    # print("Matched AEs (CADEC):", matched_aes_cadec)

if __name__ == "__main__":
    main()
