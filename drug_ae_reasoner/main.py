import argparse
from .config import RX_PATH, CADEC_KG_PATH, OAE_INDEX_PATH, OAE_LABEL_MAP_PATH, OAE_GRAPH_PATH
from .data.cadec_loader import get_cadec_drug_nodes, get_cadec_ae_pairs
from .utils.similarity_search import build_cadec_ae_oae_mapping, build_input_ae_oae_list
from .utils.path_reasoner import find_top_drug_to_input_ae_paths

def main():
    parser = argparse.ArgumentParser(description="Trace semantic paths from a drug to adverse effects using CADEC and OAE KGs.")
    parser.add_argument("--drug", type=str, required=True, help="Drug name (e.g., 'metformin')")
    parser.add_argument("--aes", type=str, required=True, nargs='+', help="List of adverse effect labels (e.g., 'nausea' 'vomiting')")
    args = parser.parse_args()

    print(f"[INFO] Running Drug-AE Path Reasoning for drug: {args.drug}")
    print(f"[INFO] Input AE terms: {args.aes}")

    connected, top_paths, fb_drug, fb_ae, verbalizations = find_top_drug_to_input_ae_paths(
        drug=args.drug,
        ae_input_list=args.aes,
        rx_path=RX_PATH,
        cadec_kg_path=CADEC_KG_PATH,
        oae_index_path=OAE_INDEX_PATH,
        oae_label_map_path=OAE_LABEL_MAP_PATH,
        oae_graph_path=OAE_GRAPH_PATH
    )

    if connected:
        print(f"[INFO] Found {len(top_paths)} real paths.")
    else:
        print(f"[INFO] No real paths found. Showing fallback paths instead.")
        print(f"[INFO] Fallbacks from Drug→AE: {len(fb_drug)} | AE→Drug: {len(fb_ae)}")

    print("\\n--- Verbalized Reasoning Paths ---\\n")
    for v in verbalizations:
        print(v)

if __name__ == "__main__":
    main()
